"""TTS generation: edge-tts, NGHI-TTS/Piper, VoxCPM, and OmniVoice."""

import asyncio
import importlib.util
import importlib.metadata
import subprocess
import json
import re
import shutil
import sys
import tempfile
import threading
import time
import unicodedata
from pathlib import Path
from typing import Callable, Optional

from src.utils.file_utils import ensure_dir, get_safe_temp_path
from src.utils.logger import logger

# ─── NGHI-TTS / Piper ───────────────────────────────────────────────────────

NGHITTS_VOICE_NAME = "Ngọc Huyền (mới)"
NGHITTS_VOICES = (
    "Ban Mai", "Chiếu Thành", "Duy Onyx (mới)", "Duy Oryx", "Lạc Phi",
    "Mai Phương", "Minh Khang", "Minh Quang", "Mạnh Dũng", "Mỹ Tâm",
    "Mỹ Tâm Real", "Ngọc Huyền (mới)", "Ngọc Ngạn", "Phương Trang",
    "Thanh Phương Viettel", "Thiện Tâm", "Trấn Thành", "Tài An",
    "Việt Thảo", "adam",
)
from urllib.parse import quote

_NGHITTS_CACHE_DIR = Path("data") / "tts_models" / "nghitts"
_nghitts_voices: dict[str, object] = {}
_nghitts_lock = threading.Lock()


def _nghitts_slug(voice_name: str) -> str:
    value = voice_name.replace("Đ", "D").replace("đ", "d")
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "voice"


def nghitts_model_urls(voice_name: str) -> tuple[str, str]:
    base = "https://nghitts.app/api/model/"
    return (
        base + quote(voice_name + ".onnx", safe=""),
        base + quote(voice_name + ".onnx.json", safe=""),
    )


def is_nghitts_available() -> bool:
    """Return whether the local Piper runtime is installed."""
    return importlib.util.find_spec("piper") is not None


def _download_nghitts_file(url: str, destination: Path, min_bytes: int) -> None:
    import requests

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".download")
    temporary.unlink(missing_ok=True)
    try:
        with requests.get(url, stream=True, timeout=(20, 180)) as response:
            response.raise_for_status()
            content_type = response.headers.get("content-type", "").lower()
            if "text/html" in content_type:
                raise RuntimeError("Máy chủ trả về trang web thay vì file model.")
            with temporary.open("wb") as output:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        output.write(chunk)
        if temporary.stat().st_size < min_bytes:
            raise RuntimeError("File tải về không đầy đủ.")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def ensure_nghitts_model(
    voice_name: str = NGHITTS_VOICE_NAME,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> tuple[Path, Path]:
    """Download one NGHI-TTS model once and return local model/config paths."""
    if voice_name not in NGHITTS_VOICES:
        raise ValueError(f"Giọng NGHI-TTS không hợp lệ: {voice_name}")
    slug = _nghitts_slug(voice_name)
    # ONNX Runtime on Windows can fail to open a model whose local path has
    # Vietnamese characters, so keep ASCII cache filenames.
    model_path = _NGHITTS_CACHE_DIR / f"{slug}.onnx"
    config_path = _NGHITTS_CACHE_DIR / f"{slug}.onnx.json"
    model_url, config_url = nghitts_model_urls(voice_name)
    if not model_path.is_file() or model_path.stat().st_size < 1_000_000:
        if progress_cb:
            progress_cb(f"⏳ Lần đầu sử dụng: đang tải model {voice_name}…")
        _download_nghitts_file(model_url, model_path, 1_000_000)
    if not config_path.is_file() or config_path.stat().st_size < 100:
        if progress_cb:
            progress_cb(f"⏳ Đang tải cấu hình giọng {voice_name}…")
        _download_nghitts_file(config_url, config_path, 100)
    return model_path, config_path


def generate_nghitts(
    text: str,
    output_path: str,
    voice_name: str = NGHITTS_VOICE_NAME,
    rate_factor: float = 1.0,
    volume_percent: float = 0.0,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> tuple[bool, str]:
    """Synthesize the bundled NGHI-TTS voice locally through Piper."""
    if not is_nghitts_available():
        return False, "Thiếu piper-tts. Hãy chạy lại run.bat để cài thư viện."
    cleaned = _clean_tts_text(text)
    if not cleaned:
        return False, "Nội dung đọc đang trống."
    try:
        from piper import PiperVoice, SynthesisConfig
        import wave

        ensure_dir(Path(output_path).parent)
        with _nghitts_lock:
            model_path, config_path = ensure_nghitts_model(voice_name, progress_cb)
            voice = _nghitts_voices.get(voice_name)
            if voice is None:
                if progress_cb:
                    progress_cb(f"⚙️ Đang nạp model {voice_name} vào bộ nhớ…")
                voice = PiperVoice.load(
                    model_path, config_path=config_path, use_cuda=False
                )
                _nghitts_voices[voice_name] = voice
            config = SynthesisConfig(
                length_scale=1.0 / max(0.5, min(2.0, float(rate_factor))),
                volume=max(0.0, min(2.0, 1.0 + float(volume_percent) / 100.0)),
            )
            if progress_cb:
                progress_cb(f"🎤 Đang tạo giọng {voice_name} trên máy…")
            with wave.open(str(output_path), "wb") as wav_file:
                voice.synthesize_wav(cleaned, wav_file, syn_config=config)
        if not _audio_file_ok(output_path, min_bytes=1000):
            raise RuntimeError("Piper không tạo được file audio hợp lệ.")
        return True, str(output_path)
    except Exception as exc:
        logger.exception("NGHI-TTS generation failed")
        return False, str(exc)

# ─── edge-tts ─────────────────────────────────────────────────────────────────

def is_edge_tts_available() -> bool:
    try:
        import edge_tts  # noqa
        return True
    except ImportError:
        return False


def get_edge_tts_voices(
    lang_filter: str = "", *, raise_errors: bool = False
) -> list[dict]:
    """Return list of edge-tts voices, optionally filtered by language code."""
    if not is_edge_tts_available():
        if raise_errors:
            raise RuntimeError(
                "edge-tts chưa được cài trong môi trường của tool."
            )
        return []
    try:
        import edge_tts

        async def _fetch():
            return await edge_tts.list_voices()

        voices = asyncio.run(_fetch())
        if lang_filter:
            voices = [v for v in voices if lang_filter.lower() in v["Locale"].lower()]
        return sorted(voices, key=lambda v: v["Locale"])
    except Exception as e:
        logger.error(f"edge-tts list voices error: {e}")
        if raise_errors:
            raise RuntimeError(str(e)) from e
        return []


# edge-tts already splits oversized UTF-8 payloads inside one WebSocket. Keep a
# normal narration in one connection because the public service can reject
# several synthesis connections opened back-to-back.
_CHUNK_MAX_CHARS = 3500


def _clean_tts_text(text: str) -> str:
    """Remove hidden/control characters that can make edge-tts return empty audio."""
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\u200b-\u200f\u202a-\u202e\ufeff]", "", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_timed_dubbing_script(text: str, video_duration: float) -> list[dict]:
    """Parse editable ``[12.3s] sentence`` blocks into dubbing segments."""
    source = (text or "").replace("&#x20;", " ").strip()
    pattern = re.compile(
        r"(?m)^\s*\[(?:(\d+):)?(\d+(?:\.\d+)?)s?\]\s*"
    )
    matches = list(pattern.finditer(source))
    if not matches:
        raise ValueError("Không tìm thấy mốc dạng [0.0s], [13.4s]…")
    rows: list[tuple[float, str]] = []
    for index, match in enumerate(matches):
        minutes = int(match.group(1) or 0)
        seconds = float(match.group(2))
        start = minutes * 60 + seconds
        end_pos = matches[index + 1].start() if index + 1 < len(matches) else len(source)
        content = re.sub(r"\s+", " ", source[match.end():end_pos]).strip()
        if not content:
            raise ValueError(f"Mốc [{start:.1f}s] chưa có lời thoại.")
        if rows and start <= rows[-1][0]:
            raise ValueError("Các mốc thời gian phải tăng dần.")
        rows.append((start, content))
    duration = max(float(video_duration or 0), rows[-1][0] + 0.5)
    return [
        {
            "start": start,
            "end": rows[index + 1][0] if index + 1 < len(rows) else duration,
            "text_zh": "",
            "text_vi": content,
        }
        for index, (start, content) in enumerate(rows)
    ]


def _audio_file_ok(path: str, min_bytes: int = 100) -> bool:
    try:
        p = Path(path)
        return p.exists() and p.stat().st_size >= min_bytes
    except OSError:
        return False


def _split_text_to_chunks(text: str, max_chars: int = _CHUNK_MAX_CHARS) -> list[str]:
    """Split text into chunks at part, paragraph, sentence, then word boundaries."""
    import re as _re

    text = _clean_tts_text(text)

    def split_long_text(value: str) -> list[str]:
        pieces: list[str] = []
        current = ""
        for token in _re.split(r"(\s+)", value.strip()):
            if not token:
                continue
            if len(current) + len(token) > max_chars and current.strip():
                pieces.append(current.strip())
                current = token.strip()
            else:
                current += token
        if current.strip():
            pieces.append(current.strip())
        return pieces

    def add_piece(piece: str, out: list[str]):
        piece = piece.strip()
        if not piece:
            return
        if len(piece) <= max_chars:
            out.append(piece)
        else:
            sentences = _re.split(r"(?<=[.!?。…])\s+", piece)
            current = ""
            for sent in [s.strip() for s in sentences if s.strip()]:
                if len(sent) > max_chars:
                    if current:
                        out.append(current)
                        current = ""
                    out.extend(split_long_text(sent))
                elif current and len(current) + len(sent) + 1 > max_chars:
                    out.append(current)
                    current = sent
                else:
                    current = (current + " " + sent).strip() if current else sent
            if current:
                out.append(current)

    raw_parts = _re.split(r"(?=={3,}\s*PART\s+\d+)", text)
    raw_parts = [p.strip() for p in raw_parts if p.strip()]
    source_blocks = raw_parts if raw_parts else [text]

    chunks: list[str] = []
    current = ""
    for block in source_blocks:
        paragraphs = _re.split(r"\n\s*\n", block)
        for para in [p.strip() for p in paragraphs if p.strip()]:
            if len(para) > max_chars:
                if current:
                    chunks.append(current)
                    current = ""
                add_piece(para, chunks)
            elif current and len(current) + len(para) + 2 > max_chars:
                chunks.append(current)
                current = para
            else:
                current = (current + "\n\n" + para).strip() if current else para

    if current:
        chunks.append(current)

    return chunks if chunks else ([text] if text else [])


def generate_edge_tts(
    text: str,
    voice: str,
    output_path: str,
    rate: str = "+0%",
    volume: str = "+0%",
    progress_cb: Optional[Callable[[str], None]] = None,
    progress_pct_cb: Optional[Callable[[int], None]] = None,
) -> tuple[bool, str]:
    """Generate TTS audio using edge-tts. Returns (success, path_or_error).

    Automatically splits long text into chunks and concatenates audio.
    progress_cb: text status messages (for log display)
    progress_pct_cb: integer 0-100 percentage (for progress bar)
    """
    if not is_edge_tts_available():
        return False, "edge-tts chưa được cài. Chạy: pip install edge-tts"

    try:
        import edge_tts

        ensure_dir(Path(output_path).parent)

        total_chars = len(text)
        chunks = _split_text_to_chunks(text)
        num_chunks = len(chunks)

        if progress_cb:
            progress_cb(f"🎤 Đang tạo giọng đọc bằng {voice}...")
            progress_cb(
                f"📝 Tổng: {total_chars:,} ký tự"
                + (f" → chia thành {num_chunks} đoạn" if num_chunks > 1 else "")
            )

        async def _generate_chunk(chunk_text: str, out_file: str):
            """Generate audio for one chunk using stream()."""
            communicate = edge_tts.Communicate(
                chunk_text,
                voice,
                rate=rate,
                volume=volume,
                connect_timeout=20,
                receive_timeout=90,
            )
            bytes_written = 0
            with open(out_file, "wb") as f:
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        data = chunk.get("data")
                        if data:
                            f.write(data)
                            bytes_written += len(data)
            if bytes_written <= 0:
                raise RuntimeError("edge-tts không trả về audio cho đoạn này.")

        def _generate_with_retry(chunk_text: str, out_file: str, label: str):
            last_error = None
            # Edge's public endpoint occasionally returns an empty stream even
            # though the same request succeeds immediately afterwards.  Keep
            # retries short so the UI does not look as if it has frozen.
            retry_delays = (2, 5)
            for attempt in range(1, 4):
                try:
                    Path(out_file).unlink(missing_ok=True)
                except OSError:
                    pass
                try:
                    asyncio.run(_generate_chunk(chunk_text, out_file))
                    if _audio_file_ok(out_file):
                        if attempt > 1 and progress_cb:
                            progress_cb(f"✅ {label} đã tạo được audio ở lần {attempt}/3.")
                        return
                    raise RuntimeError("File audio đoạn quá nhỏ hoặc rỗng.")
                except Exception as exc:
                    last_error = exc
                    if attempt < 3:
                        delay = retry_delays[attempt - 1]
                        if progress_cb:
                            progress_cb(
                                f"⚠️ {label} lỗi: {exc}. "
                                f"Chờ {delay}s rồi thử lần {attempt + 1}/3..."
                            )
                        time.sleep(delay)
            raise RuntimeError(str(last_error or "Không tạo được audio."))

        def _generate_resilient(chunk_text: str, base_file: str, label: str) -> list[str]:
            try:
                _generate_with_retry(chunk_text, base_file, label)
                return [base_file]
            except Exception as exc:
                if "No audio was received" in str(exc):
                    raise RuntimeError(f"{label}: {exc}") from exc
                if len(chunk_text) <= 700:
                    raise RuntimeError(f"{label}: {exc}") from exc
                subchunks = _split_text_to_chunks(chunk_text, max_chars=700)
                if len(subchunks) <= 1:
                    raise RuntimeError(f"{label}: {exc}") from exc
                if progress_cb:
                    progress_cb(
                        f"🧩 {label} vẫn lỗi → chờ dịch vụ ổn định rồi chia nhỏ "
                        f"thành {len(subchunks)} đoạn..."
                    )
                time.sleep(8)
                files: list[str] = []
                base = Path(base_file)
                for sub_idx, sub_text in enumerate(subchunks, start=1):
                    sub_file = str(base.with_name(f"{base.stem}_{sub_idx:02d}{base.suffix}"))
                    _generate_with_retry(
                        sub_text,
                        sub_file,
                        f"{label}.{sub_idx}",
                    )
                    if progress_cb:
                        progress_cb(
                            f"✅ {label}.{sub_idx}/{len(subchunks)} xong "
                            f"({len(sub_text):,} ký tự)."
                        )
                    files.append(sub_file)
                return files

        chunk_files: list[str] = []
        out_dir = Path(output_path).parent
        for i, chunk_text in enumerate(chunks):
            chunk_path = str(out_dir / f"_tts_chunk_{i:03d}.mp3")

            pct = int((i / num_chunks) * 100)
            if progress_pct_cb:
                progress_pct_cb(pct)
            if progress_cb:
                progress_cb(
                    f"⏳ Đoạn {i+1}/{num_chunks} ({pct}%) — "
                    f"{len(chunk_text):,} ký tự..."
                )

            generated = _generate_resilient(
                chunk_text,
                chunk_path,
                f"Đoạn {i+1}/{num_chunks}",
            )
            if progress_cb and len(generated) == 1:
                progress_cb(f"✅ Đoạn {i+1}/{num_chunks} xong.")
            chunk_files.extend(generated)
            if i + 1 < num_chunks:
                time.sleep(5)

        # Concatenate all chunk files using binary append (MP3 is concatenatable)
        if progress_cb:
            progress_cb(f"🔗 Đang ghép {len(chunk_files)} đoạn audio...")
        with open(output_path, "wb") as fout:
            for cf in chunk_files:
                with open(cf, "rb") as fin:
                    fout.write(fin.read())

        # Clean up temp chunk files
        for cf in chunk_files:
            try:
                Path(cf).unlink()
            except OSError:
                pass

        if progress_pct_cb:
            progress_pct_cb(100)

        if Path(output_path).exists():
            file_size = Path(output_path).stat().st_size
            if file_size < 100:
                return False, "File audio quá nhỏ — có thể edge-tts không trả về audio."
            if progress_cb:
                progress_cb(
                    f"✅ Tạo giọng xong: {Path(output_path).name} "
                    f"({file_size / 1024:.0f} KB)"
                )
            return True, output_path
        return False, "Không tạo được file audio."
    except Exception as e:
        logger.error(f"edge-tts error: {e}")
        if progress_cb:
            progress_cb(f"❌ Lỗi: {e}")
        return False, str(e)


# ─── OmniVoice Studio (REST API) ──────────────────────────────────────────────

OMNIVOICE_BASE = "http://localhost:8000"


def is_omnivoice_running() -> bool:
    try:
        import urllib.request
        urllib.request.urlopen(f"{OMNIVOICE_BASE}/health", timeout=2)
        return True
    except Exception:
        return False


def omnivoice_generate(
    text: str,
    voice_id: str,
    output_path: str,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> tuple[bool, str]:
    """Generate TTS via OmniVoice Studio REST API (must be running locally)."""
    if not is_omnivoice_running():
        return False, "OmniVoice Studio chưa chạy. Mở app lên trước."
    try:
        import urllib.request, urllib.parse
        ensure_dir(Path(output_path).parent)
        if progress_cb:
            progress_cb("🎤 Đang gọi OmniVoice Studio API...")
        payload = json.dumps({"text": text, "voice_id": voice_id}).encode()
        req = urllib.request.Request(
            f"{OMNIVOICE_BASE}/tts/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            audio_data = resp.read()
        with open(output_path, "wb") as f:
            f.write(audio_data)
        if progress_cb:
            progress_cb(f"✅ OmniVoice xong: {Path(output_path).name}")
        return True, output_path
    except Exception as e:
        logger.error(f"OmniVoice error: {e}")
        if progress_cb:
            progress_cb(f"❌ OmniVoice lỗi: {e}")
        return False, str(e)


# ─── VoxCPM 2 (local Python API) ──────────────────────────────────────────────

VOXCPM_DEFAULT_MODEL = "openbmb/VoxCPM2"
_VOXCPM_MODELS: dict[tuple, object] = {}
_VOXCPM_MODEL_LOCK = threading.RLock()


def _add_external_voxcpm_path() -> str:
    """Add a sibling ``venv`` site-packages path when one contains VoxCPM.

    This also lets a packaged app reuse the optional runtime installed beside
    the source tree instead of incorrectly claiming that VoxCPM is absent.
    """
    seeds = [
        Path.cwd(),
        Path(sys.executable).resolve().parent,
        Path(__file__).resolve().parent,
    ]
    checked: set[Path] = set()
    for seed in seeds:
        for root in (seed, *list(seed.parents)[:5]):
            site_packages = root / "venv" / "Lib" / "site-packages"
            if site_packages in checked:
                continue
            checked.add(site_packages)
            if (site_packages / "voxcpm").is_dir():
                value = str(site_packages)
                if value not in sys.path:
                    sys.path.insert(0, value)
                return value
    return ""


def get_voxcpm_runtime_status() -> tuple[bool, str]:
    """Return availability and reader-friendly runtime diagnostics."""
    _add_external_voxcpm_path()
    importlib.invalidate_caches()
    try:
        spec = importlib.util.find_spec("voxcpm")
    except (ImportError, ValueError):
        spec = None
    if spec is not None:
        try:
            version = importlib.metadata.version("voxcpm")
        except importlib.metadata.PackageNotFoundError:
            version = "đã cài"
        origin = getattr(spec, "origin", "") or ""
        return True, f"VoxCPM {version} — {origin}"
    return (
        False,
        f"Python hiện tại không thấy VoxCPM: {sys.executable} "
        f"(prefix: {sys.prefix})",
    )


def is_voxcpm_available() -> bool:
    """Return whether the optional VoxCPM Python package is installed."""
    return get_voxcpm_runtime_status()[0]


def resolve_voxcpm_device(
    requested_device: str = "auto",
    model_id: str = VOXCPM_DEFAULT_MODEL,
) -> tuple[str, str]:
    """Resolve ``auto`` safely on GPUs that are too small for VoxCPM2.

    Upstream's automatic order is CUDA -> MPS -> CPU. VoxCPM2 needs roughly
    8 GB VRAM, so blindly selecting a smaller NVIDIA GPU normally ends in an
    out-of-memory error. Explicit device selections are left untouched.
    """
    requested = (requested_device or "auto").strip().lower()
    if requested != "auto":
        return requested, ""

    try:
        import torch

        if torch.cuda.is_available():
            total_gb = (
                torch.cuda.get_device_properties(0).total_memory
                / (1024 ** 3)
            )
            is_voxcpm2 = "voxcpm2" in (model_id or "").replace("-", "").lower()
            if is_voxcpm2 and total_gb < 7.5:
                return (
                    "cpu",
                    f"GPU chỉ có {total_gb:.1f} GB VRAM; VoxCPM2 cần khoảng "
                    "8 GB nên app tự chuyển sang CPU.",
                )
            return "cuda", ""

        mps = getattr(getattr(torch, "backends", None), "mps", None)
        if mps is not None and mps.is_available():
            return "mps", ""
        return (
            "cpu",
            "PyTorch hiện không có CUDA/MPS; app dùng CPU cho VoxCPM.",
        )
    except Exception:
        # Let the upstream automatic selection handle unknown runtimes.
        pass

    return "auto", ""


def _get_voxcpm_model(
    model_id: str,
    device: str,
    load_denoiser: bool,
    optimize: bool,
    progress_cb: Optional[Callable[[str], None]] = None,
):
    """Load and cache a VoxCPM model for all scene generations in this run."""
    key = (model_id, device, bool(load_denoiser), bool(optimize))
    with _VOXCPM_MODEL_LOCK:
        model = _VOXCPM_MODELS.get(key)
        if model is not None:
            return model

        if progress_cb:
            progress_cb(
                "⏳ Đang nạp VoxCPM; lần đầu có thể tải vài GB trọng số model..."
            )
        from voxcpm import VoxCPM

        model = VoxCPM.from_pretrained(
            model_id,
            load_denoiser=load_denoiser,
            device=device,
            optimize=optimize,
        )
        _VOXCPM_MODELS[key] = model
        return model


def generate_voxcpm(
    text: str,
    output_path: str,
    model_id: str = VOXCPM_DEFAULT_MODEL,
    device: str = "auto",
    reference_wav_path: str = "",
    prompt_text: str = "",
    voice_description: str = "",
    cfg_value: float = 2.0,
    inference_timesteps: int = 10,
    normalize: bool = True,
    denoise: bool = False,
    optimize: bool = True,
    volume_percent: float = 0.0,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> tuple[bool, str]:
    """Generate a WAV file with VoxCPM2.

    With only ``reference_wav_path`` this performs controllable cloning. When
    ``prompt_text`` is also provided, the same clip is used for Hi-Fi cloning.
    Without a reference clip, ``voice_description`` enables Voice Design.
    """
    if not is_voxcpm_available():
        return (
            False,
            "VoxCPM chưa được cài. Chạy install_voxcpm.bat rồi mở lại app.",
        )

    clean_text = _clean_tts_text(text)
    if not clean_text:
        return False, "Nội dung đọc VoxCPM đang trống."

    model_id = (model_id or VOXCPM_DEFAULT_MODEL).strip()
    reference = (reference_wav_path or "").strip()
    transcript = (prompt_text or "").strip()
    description = (voice_description or "").strip().strip("()")

    if reference and not Path(reference).is_file():
        return False, f"Không tìm thấy audio tham chiếu: {reference}"
    if transcript and not reference:
        return False, "Hi-Fi clone cần cả audio tham chiếu và transcript chính xác."

    # Hi-Fi mode ignores style control upstream. Other modes accept a natural
    # language instruction in parentheses at the beginning of the target text.
    generation_text = clean_text
    if description and not transcript:
        generation_text = f"({description}){clean_text}"

    resolved_device, device_note = resolve_voxcpm_device(device, model_id)
    if device_note and progress_cb:
        progress_cb(f"⚠️ {device_note}")
    # torch.compile is primarily a CUDA optimization and may fail on CPU/MPS.
    effective_optimize = bool(optimize and resolved_device not in {"cpu", "mps"})

    try:
        ensure_dir(Path(output_path).parent)
        if progress_cb:
            mode = (
                "Hi-Fi clone"
                if transcript
                else ("clone giọng" if reference else "Voice Design")
            )
            progress_cb(
                f"🎙 VoxCPM2: {mode} | device={resolved_device} | "
                f"CFG={cfg_value:.1f} | {int(inference_timesteps)} bước"
            )

        model = _get_voxcpm_model(
            model_id,
            resolved_device,
            load_denoiser=bool(denoise),
            optimize=effective_optimize,
            progress_cb=progress_cb,
        )
        kwargs = {
            "text": generation_text,
            "cfg_value": float(cfg_value),
            "inference_timesteps": int(inference_timesteps),
            "normalize": bool(normalize),
            "denoise": bool(denoise),
            "retry_badcase": True,
        }
        if reference:
            kwargs["reference_wav_path"] = str(Path(reference).resolve())
        if transcript:
            kwargs["prompt_wav_path"] = str(Path(reference).resolve())
            kwargs["prompt_text"] = transcript

        with _VOXCPM_MODEL_LOCK:
            wav = model.generate(**kwargs)

        import soundfile as sf

        waveform = wav.reshape(-1) if hasattr(wav, "reshape") else wav
        try:
            sample_count = int(getattr(waveform, "size", len(waveform)))
        except TypeError:
            sample_count = 0
        if sample_count == 0:
            return False, "VoxCPM không trả về dữ liệu audio."
        gain = max(0.0, 1.0 + float(volume_percent) / 100.0)
        if abs(gain - 1.0) > 0.0001:
            import numpy as np

            waveform = np.asarray(waveform, dtype=np.float32)
            waveform = np.clip(waveform * gain, -1.0, 1.0)

        sample_rate = int(model.tts_model.sample_rate)
        sf.write(output_path, waveform, sample_rate, subtype="PCM_16")
        if not _audio_file_ok(output_path):
            return False, "VoxCPM không tạo được file WAV hợp lệ."
        if progress_cb:
            progress_cb(f"✅ VoxCPM xong: {Path(output_path).name}")
        return True, output_path
    except Exception as exc:
        message = str(exc)
        lowered = message.lower()
        if "out of memory" in lowered or "cuda" in lowered and "memory" in lowered:
            message = (
                "Không đủ VRAM cho VoxCPM. Chọn device CPU hoặc dùng GPU có "
                "ít nhất khoảng 8 GB VRAM. Chi tiết: " + message
            )
        elif isinstance(exc, ModuleNotFoundError):
            message = (
                f"VoxCPM thiếu phụ thuộc '{getattr(exc, 'name', '')}'. "
                "Chạy lại install_voxcpm.bat. Chi tiết: " + message
            )
        logger.error(f"VoxCPM error: {exc}")
        if progress_cb:
            progress_cb(f"❌ VoxCPM lỗi: {message}")
        return False, message


# ─── FFmpeg: merge voice + video ──────────────────────────────────────────────

def _ffprobe_path(ffmpeg_bin: str) -> str:
    ff = Path(ffmpeg_bin)
    exe = "ffprobe.exe" if ff.name.lower().endswith(".exe") else "ffprobe"
    candidate = ff.with_name(exe)
    return str(candidate) if candidate.exists() else "ffprobe"


def _probe_duration(path: str, ffmpeg_bin: str) -> float:
    try:
        proc = subprocess.run(
            [
                _ffprobe_path(ffmpeg_bin),
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                path,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return float((proc.stdout or "0").strip() or 0)
    except Exception:
        return 0.0


def probe_media_duration(path: str, ffmpeg_bin: str) -> float:
    """Public wrapper for UI duration checks."""
    return _probe_duration(path, ffmpeg_bin)


def split_voiceover_sections(text: str) -> list[str]:
    """Split ``=== title ===`` narration into ordered scene scripts.

    AI-generated titles do not always contain a Part number, so chronological
    order is the reliable mapping between script sections and review clips.
    """
    value = (text or "").strip()
    if not value:
        return []

    headers = list(re.finditer(r"(?m)^\s*={2,}\s*[^=\r\n]+?\s*={2,}\s*$", value))
    if not headers:
        return [value]

    sections: list[str] = []
    for pos, header in enumerate(headers):
        start = header.end()
        end = headers[pos + 1].start() if pos + 1 < len(headers) else len(value)
        section = value[start:end].strip()
        if section:
            sections.append(section)
    return sections


def strip_voiceover_part_labels(text: str) -> str:
    """Remove visual/editor Part labels so TTS never reads them aloud."""
    value = re.sub(
        r"(?m)^\s*={2,}\s*[^=\r\n]+?\s*={2,}\s*$",
        "",
        text or "",
    )
    value = re.sub(
        r"(?im)^\s*(?:PART|PHẦN)\s*\d+\s*(?::[^\r\n]*)?\s*$",
        "",
        value,
    )
    return re.sub(r"\n{3,}", "\n\n", value).strip()


def _atempo_chain(factor: float) -> str:
    """Return an FFmpeg atempo chain for any positive speed factor."""
    remaining = max(0.01, float(factor or 1.0))
    values: list[float] = []
    while remaining > 2.0:
        values.append(2.0)
        remaining /= 2.0
    while remaining < 0.5:
        values.append(0.5)
        remaining /= 0.5
    if not values or abs(remaining - 1.0) > 0.0005:
        values.append(remaining)
    return ",".join(f"atempo={value:.6f}" for value in values)


def align_voice_segments(
    segment_paths: list[tuple[str, float, int]],
    output_path: str,
    ffmpeg_bin: str,
    progress_cb: Optional[Callable[[str], None]] = None,
    allow_slowdown: bool = True,
    max_tempo: float | None = None,
) -> tuple[bool, str, list[dict]]:
    """Fit each narration file to its scene and join them in timeline order."""
    if not segment_paths:
        return False, "Không có đoạn giọng đọc để căn theo phân cảnh.", []

    inputs: list[str] = []
    filters: list[str] = []
    labels: list[str] = []
    adjustments: list[dict] = []

    for pos, (audio_path, target_seconds, part_index) in enumerate(segment_paths):
        if not Path(audio_path).exists():
            return False, f"Không tìm thấy audio Part {part_index}: {audio_path}", adjustments
        actual = _probe_duration(audio_path, ffmpeg_bin)
        target = max(0.1, float(target_seconds or 0.0))
        if actual <= 0:
            return False, f"Không đo được thời lượng audio Part {part_index}.", adjustments

        requested_tempo = actual / target
        # Timed dubbing should sound like one interpreter speaking at a stable
        # pace. A short sentence ends naturally and the remaining slot stays
        # silent; only an overlong sentence is accelerated to meet the next cue.
        tempo = requested_tempo if allow_slowdown else max(1.0, requested_tempo)
        adjustments.append({
            "part_index": int(part_index),
            "source_duration": actual,
            "target_duration": target,
            "tempo_factor": tempo,
        })
        if max_tempo is not None and tempo > float(max_tempo) + 0.001:
            fitted_seconds = actual / float(max_tempo)
            overflow_seconds = max(0.0, fitted_seconds - target)
            if progress_cb:
                progress_cb(
                    f"❌ Part {part_index}: voice {actual:.2f}s vượt khung "
                    f"{target:.2f}s; cần hệ số căn {tempo:.2f}x nhưng chỉ còn "
                    f"cho phép tối đa {float(max_tempo):.2f}x. Sau khi căn vẫn "
                    f"dư {overflow_seconds:.2f}s."
                )
            return (
                False,
                f"Voice Part {part_index}: ở tốc độ tối đa vẫn dài "
                f"{fitted_seconds:.2f}s / video {target:.2f}s, dư "
                f"{overflow_seconds:.2f}s. Hãy rút ngắn nội dung tương ứng; "
                "ứng dụng không tăng quá giới hạn hoặc cắt mất câu cuối.",
                adjustments,
            )
        inputs.extend(["-i", audio_path])
        label = f"a{pos}"
        labels.append(f"[{label}]")
        filters.append(
            f"[{pos}:a:0]aresample=48000,{_atempo_chain(tempo)},"
            f"apad=whole_dur={target:.6f},atrim=duration={target:.6f},"
            f"asetpts=N/SR/TB[{label}]"
        )
        if progress_cb:
            if not allow_slowdown and requested_tempo < 1.0:
                progress_cb(
                    f"🎬 Đoạn {part_index}: giữ tốc độ gốc {actual:.2f}s, "
                    f"bù {target - actual:.2f}s im lặng đến mốc kế tiếp"
                )
            else:
                progress_cb(
                    f"🎬 Part {part_index}: voice {actual:.2f}s → cảnh {target:.2f}s "
                    f"(nhịp {tempo:.2f}x)"
                )

    if len(labels) == 1:
        output_label = labels[0]
    else:
        output_label = "[voiceout]"
        filters.append(
            "".join(labels)
            + f"concat=n={len(labels)}:v=0:a=1{output_label}"
        )

    ensure_dir(Path(output_path).parent)
    cmd = [
        ffmpeg_bin, "-hide_banner", "-y",
        *inputs,
        "-filter_complex", ";".join(filters),
        "-map", output_label,
        "-c:a", "libmp3lame",
        "-b:a", "128k",
        output_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=600)
        if result.returncode != 0:
            err = result.stderr.decode(errors="replace")[-800:]
            logger.error(f"Voice scene alignment failed: {err}")
            return False, err, adjustments
        if not _audio_file_ok(output_path):
            return False, "FFmpeg không tạo được track voice đã căn cảnh.", adjustments
        if progress_cb:
            total = sum(item["target_duration"] for item in adjustments)
            progress_cb(f"✅ Đã căn {len(adjustments)} đoạn voice theo timeline ({total:.2f}s).")
        return True, output_path, adjustments
    except Exception as exc:
        logger.error(f"Voice scene alignment error: {exc}")
        return False, str(exc), adjustments


def trim_dubbing_edges(source, output, ffmpeg_bin):
    """Trim only outer silence, retaining 50ms margins and internal pauses."""
    edge = 'silenceremove=start_periods=1:start_duration=0.02:start_threshold=-50dB:start_silence=0.05'
    result = subprocess.run(
        [ffmpeg_bin, '-y', '-i', source, '-af',
         f'{edge},areverse,{edge},areverse', '-c:a', 'pcm_s16le', output],
        capture_output=True, timeout=120,
    )
    if result.returncode != 0 or _probe_duration(output, ffmpeg_bin) <= 0:
        raise RuntimeError('Không xử lý được khoảng lặng đầu/cuối giọng đọc.')
    return output


def align_interpreter_segments(
    segment_paths: list[tuple[str, float, float, int]],
    output_path: str,
    ffmpeg_bin: str,
    total_duration: float,
    progress_cb: Optional[Callable[[str], None]] = None,
    min_tempo: float = 1.0,
    max_tempo: float = 1.0,
    allow_overflow: bool = False,
) -> tuple[bool, str, list[dict]]:
    """Place timestamped speech like one interpreter with one stable tempo.

    Each sentence starts at its requested cue when the previous sentence has
    finished.  Dense translations continue immediately after the preceding
    sentence instead of being independently squeezed into tiny timestamp
    slots. Keep the selected TTS speed unchanged; reject an overflowing
    timeline before writing audio rather than speeding up or cutting speech.
    """
    if not segment_paths:
        return False, "Không có câu lồng tiếng để căn timeline.", []

    probed: list[dict] = []
    for audio_path, start, end, part_index in segment_paths:
        if not Path(audio_path).exists():
            return False, f"Không tìm thấy audio câu {part_index}: {audio_path}", []
        actual = _probe_duration(audio_path, ffmpeg_bin)
        if actual <= 0:
            return False, f"Không đo được thời lượng audio câu {part_index}.", []
        probed.append({
            "path": audio_path,
            "start": max(0.0, float(start or 0.0)),
            "end": max(float(start or 0.0) + 0.1, float(end or 0.0)),
            "part_index": int(part_index),
            "source_duration": actual,
        })

    target_total = max(
        0.1,
        float(total_duration or 0.0),
        max(item["end"] for item in probed),
    )

    def _schedule(tempo: float) -> tuple[float, list[float]]:
        cursor = 0.0
        starts: list[float] = []
        for item in probed:
            placed = max(item["start"], cursor)
            starts.append(placed)
            cursor = placed + item["source_duration"] / tempo
        return cursor, starts

    if not 0 < min_tempo <= max_tempo:
        return False, "Khoảng tốc độ lồng tiếng không hợp lệ.", []
    tempo = min_tempo
    if _schedule(tempo)[0] > target_total and max_tempo > min_tempo:
        low, high = min_tempo, max_tempo
        for _ in range(40):
            mid = (low + high) / 2.0
            if _schedule(mid)[0] > target_total:
                low = mid
            else:
                high = mid
        tempo = high
    finish, placed_starts = _schedule(tempo)
    if finish > target_total + 0.005 and not allow_overflow:
        return False, (
            f"Lời đọc ở mức tối đa {max_tempo:.2f}x kết thúc tại {finish:.2f}s, "
            f"vượt video {finish - target_total:.2f}s. "
            "Hãy rút gọn lời dịch rồi tạo lại. "
            "Tool không vượt giới hạn tốc độ và không cắt lời đọc."
        ), []
    if allow_overflow and finish > target_total:
        if progress_cb:
            progress_cb(
                f"⚠️ Voice dư {finish - target_total:.2f}s. Vẫn ghép ở tốc độ đã chọn; "
                "video kết thúc sẽ cắt phần lời dư. File audio riêng giữ đầy đủ."
            )
        target_total = finish
    inputs: list[str] = []
    filters: list[str] = []
    labels: list[str] = []
    adjustments: list[dict] = []
    for pos, (item, placed_start) in enumerate(zip(probed, placed_starts)):
        inputs.extend(["-i", item["path"]])
        label = f"interp_{pos}"
        labels.append(f"[{label}]")
        delay_ms = max(0, int(round(placed_start * 1000.0)))
        speed_filter = f"{_atempo_chain(tempo)}," if tempo != 1.0 else ""
        filters.append(
            f"[{pos}:a:0]aresample=48000,{speed_filter}asetpts=N/SR/TB,"
            f"adelay=delays={delay_ms}:all=1[{label}]"
        )
        adjustments.append({
            "part_index": item["part_index"],
            "source_duration": item["source_duration"],
            "target_duration": item["end"] - item["start"],
            "tempo_factor": tempo,
            "requested_start": item["start"],
            "placed_start": placed_start,
            "start_delay": placed_start - item["start"],
        })

    output_label = "[interpreter_out]"
    filters.append(
        "".join(labels)
        + f"amix=inputs={len(labels)}:duration=longest:normalize=0,"
        f"apad=whole_dur={target_total:.6f},"
        f"atrim=duration={target_total:.6f}{output_label}"
    )
    ensure_dir(Path(output_path).parent)
    cmd = [
        ffmpeg_bin, "-hide_banner", "-y",
        *inputs,
        "-filter_complex", ";".join(filters),
        "-map", output_label,
        "-c:a", "libmp3lame",
        "-b:a", "128k",
        output_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=600)
        if result.returncode != 0:
            err = result.stderr.decode(errors="replace")[-800:]
            logger.error(f"Interpreter timeline alignment failed: {err}")
            return False, err, adjustments
        if not _audio_file_ok(output_path):
            return False, "FFmpeg không tạo được track phiên dịch.", adjustments
        if progress_cb:
            delayed = sum(item["start_delay"] > 0.03 for item in adjustments)
            progress_cb(
                f"✅ Nhịp đọc chung {tempo:.3f}x cho "
                f"{len(adjustments)} câu; {delayed} câu nối sau câu trước."
            )
        return True, output_path, adjustments
    except Exception as exc:
        logger.error(f"Interpreter timeline alignment error: {exc}")
        return False, str(exc), adjustments


def build_clean_review_video(
    project,
    output_path: str,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> tuple[bool, str]:
    """Render enabled review clips with every subtitle source disabled."""
    from copy import deepcopy
    from src.core.video_processor import concatenate_clips, export_clip

    clips = [deepcopy(clip) for clip in getattr(project, "clips", []) if clip.enabled]
    current_source = getattr(project, "source_video", "") or ""
    original_source = (
        getattr(project, "original_source_video", "")
        or getattr(project, "video_metadata", {}).get("path", "")
        or current_source
    )
    if not Path(original_source).exists():
        original_source = current_source

    if not clips:
        if not original_source or not Path(original_source).exists():
            return False, "Không tìm thấy video nguồn để dựng bản review sạch subtitle."
        ensure_dir(Path(output_path).parent)
        shutil.copy2(original_source, output_path)
        return True, output_path

    clean_project = deepcopy(project)
    clean_project.source_video = original_source
    clean_cfg = deepcopy(project.export_config)
    clean_cfg.subtitle_enabled = False
    clean_cfg.subtitle_style = "none"
    clean_cfg.global_subtitle_file = ""
    clean_cfg.part_text_enabled = False
    clean_cfg.watermark_text = ""
    # This is an intermediate clean master, not the final delivery render.
    # Normalize every scene to the raw source canvas and defer crop/aspect,
    # watermark treatment, typography and subtitle burn-in to the final step.
    clean_cfg.aspect_mode = "keep_ratio"
    clean_cfg.pre_crop_enabled = False
    clean_cfg.remove_watermark_enabled = False
    clean_cfg.background_image = ""
    try:
        from src.core.video_manager import get_video_metadata
        raw_meta = get_video_metadata(original_source) or {}
        raw_width = int(raw_meta.get("width", 0) or 0)
        raw_height = int(raw_meta.get("height", 0) or 0)
        if raw_width > 1 and raw_height > 1:
            clean_cfg.width = raw_width if raw_width % 2 == 0 else raw_width - 1
            clean_cfg.height = raw_height if raw_height % 2 == 0 else raw_height - 1
    except Exception:
        pass

    work_parent = Path(project.output_dir) / "audio"
    ensure_dir(work_parent)
    work_dir = Path(tempfile.mkdtemp(prefix="_voice_review_", dir=str(work_parent)))
    rendered: list[str] = []
    try:
        if progress_cb:
            progress_cb(
                f"🎞 Đang dựng lại {len(clips)} phân cảnh, tắt hoàn toàn subtitle..."
            )
        for pos, clip in enumerate(clips, start=1):
            clip.subtitle_file = ""
            clip.custom_subtitle = ""
            clip.custom_header = ""
            clip.part_text = ""
            clip.export_path = ""
            part_path = str(work_dir / f"scene_{pos:03d}.mp4")
            if progress_cb:
                progress_cb(f"⏳ Dựng cảnh {pos}/{len(clips)} (không subtitle)...")
            ok, error = export_clip(
                clean_project,
                clip,
                clean_cfg,
                part_path,
            )
            if not ok:
                return False, f"Dựng cảnh {clip.index} thất bại: {error}"
            rendered.append(part_path)

        ok, result = concatenate_clips(rendered, output_path, progress_cb=progress_cb)
        if ok and progress_cb:
            progress_cb(f"✅ Video nền sạch subtitle: {Path(output_path).name}")
        return ok, result
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def _has_audio_stream(path: str, ffmpeg_bin: str) -> bool:
    try:
        proc = subprocess.run(
            [
                _ffprobe_path(ffmpeg_bin),
                "-v", "error",
                "-select_streams", "a:0",
                "-show_entries", "stream=index",
                "-of", "csv=p=0",
                path,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return bool((proc.stdout or "").strip())
    except Exception:
        return False


def _srt_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    if ms >= 1000:
        s += 1
        ms -= 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _strip_script_for_subtitles(text: str) -> str:
    first_part = re.search(r"={2,}\s*PART\s+\d+", text, flags=re.I)
    if first_part:
        text = text[first_part.start():]
    text = re.sub(r"={2,}\s*PART\s+\d+[^=]*={2,}", " ", text, flags=re.I)
    text = re.sub(r"^\s*PART\s+\d+\s*:.*$", " ", text, flags=re.I | re.M)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _subtitle_chunks(text: str, max_words: int = 8, max_chars: int = 52) -> list[str]:
    text = _strip_script_for_subtitles(text)
    if not text:
        return []

    sentences = re.split(r"(?<=[.!?。…])\s+", text)
    chunks: list[str] = []
    current: list[str] = []

    def flush():
        nonlocal current
        if current:
            chunks.append(" ".join(current).strip())
            current = []

    for sentence in sentences:
        words = sentence.split()
        for word in words:
            candidate = " ".join(current + [word])
            if current and (len(current) >= max_words or len(candidate) > max_chars):
                flush()
            current.append(word)
        if current and len(" ".join(current)) >= max_chars * 0.7:
            flush()
    flush()
    return chunks


def build_voiceover_subtitle(
    script_text: str,
    voice_path: str,
    output_path: str,
    ffmpeg_bin: str,
    style: str = "plain",
) -> tuple[bool, str]:
    """Create a simple timed SRT from the narration script and voice length."""
    cleaned_text = _strip_script_for_subtitles(script_text)
    if not cleaned_text:
        return False, "Script trống nên không tạo được subtitle."

    voice_duration = _probe_duration(voice_path, ffmpeg_bin)
    if voice_duration <= 0:
        # Fallback: about 2.5 words/second.
        total_words = len(cleaned_text.split())
        voice_duration = max(1.0, total_words / 2.5)

    if style in ("karaoke", "word"):
        from src.core.subtitle_builder import build_ass_karaoke, build_ass_word_by_word

        words = cleaned_text.split()
        total_words = max(1, len(words))
        step = voice_duration / total_words
        transcript = {
            "segments": [{
                "start": 0.0,
                "end": voice_duration,
                "text": cleaned_text,
                "words": [
                    {
                        "word": word,
                        "start": i * step,
                        "end": (i + 1) * step,
                    }
                    for i, word in enumerate(words)
                ],
            }]
        }
        ass_path = str(Path(output_path).with_suffix(".ass"))
        if style == "word":
            build_ass_word_by_word(transcript, ass_path)
        else:
            build_ass_karaoke(transcript, ass_path)
        return True, ass_path

    chunks = _subtitle_chunks(cleaned_text)
    if not chunks:
        return False, "Script trống nên không tạo được subtitle."

    total_words = max(1, sum(len(c.split()) for c in chunks))
    cursor = 0.0
    lines = []
    for idx, chunk in enumerate(chunks, start=1):
        words = max(1, len(chunk.split()))
        dur = max(1.15, voice_duration * words / total_words)
        end = min(voice_duration, cursor + dur)
        if idx == len(chunks):
            end = voice_duration
        lines.append(str(idx))
        lines.append(f"{_srt_time(cursor)} --> {_srt_time(end)}")
        lines.append(chunk)
        lines.append("")
        cursor = end
        if cursor >= voice_duration:
            break

    ensure_dir(Path(output_path).parent)
    with open(output_path, "w", encoding="utf-8-sig") as f:
        f.write("\n".join(lines).strip() + "\n")
    return True, output_path


def _color_to_ass(name: str) -> str:
    colors = {
        "white": "&H00FFFFFF",
        "black": "&H00000000",
        "yellow": "&H0000FFFF",
        "red": "&H000000FF",
        "green": "&H0000FF00",
        "blue": "&H00FF0000",
        "cyan": "&H00FFFF00",
        "magenta": "&H00FF00FF",
    }
    return colors.get((name or "white").lower(), "&H00FFFFFF")


def _subtitle_force_style(cfg=None) -> str:
    fs = getattr(cfg, "subtitle_fontsize", 54) if cfg else 54
    font = getattr(cfg, "subtitle_font", "Arial") if cfg else "Arial"
    bold = 1 if (getattr(cfg, "subtitle_bold", True) if cfg else True) else 0
    italic = 1 if (getattr(cfg, "subtitle_italic", False) if cfg else False) else 0
    underline = 1 if (getattr(cfg, "subtitle_underline", False) if cfg else False) else 0
    primary = _color_to_ass(getattr(cfg, "subtitle_color", "white") if cfg else "white")
    secondary = _color_to_ass(getattr(cfg, "subtitle_highlight_color", "yellow") if cfg else "yellow")
    position = getattr(cfg, "subtitle_position", "bottom") if cfg else "bottom"
    alignment = {"bottom": 2, "top": 8, "center": 5}.get(position or "bottom", 2)
    margin_v = getattr(cfg, "subtitle_margin_v", 0) if cfg else 0
    if not margin_v:
        margin_v = 60
    return (
        f"Fontname={font},Fontsize={fs},Bold={bold},Italic={italic},"
        f"Underline={underline},PrimaryColour={primary},SecondaryColour={secondary},"
        f"OutlineColour=&H00000000,BackColour=&H80000000,"
        f"BorderStyle=1,Outline=2,Shadow=0,Alignment={alignment},MarginV={margin_v}"
    )


def _escape_subtitle_filter_path(path: str) -> str:
    return (
        str(Path(path).resolve())
        .replace("\\", "/")
        .replace(":", "\\:")
        .replace("'", "\\'")
        .replace(" ", "\\ ")
    )

def merge_voice_with_video(
    video_path: str,
    voice_path: str,
    output_path: str,
    ffmpeg_bin: str,
    original_audio_volume: float = 0.15,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> tuple[bool, str]:
    """
    Merge voice-over with video.
    original_audio_volume: 0.0 = mute original, 1.0 = keep full, 0.15 = duck under voice.
    """
    ensure_dir(Path(output_path).parent)
    if progress_cb:
        progress_cb("🎞 Đang ghép giọng vào video...")

    video_has_audio = _has_audio_stream(video_path, ffmpeg_bin)

    if original_audio_volume <= 0.0 or not video_has_audio:
        # Replace audio entirely
        audio_filter = "[1:a]anull[aout]"
        audio_map = "[aout]"
    else:
        # Mix: duck original audio, keep voice on top
        audio_filter = (
            f"[0:a]volume={original_audio_volume}[orig];"
            f"[orig][1:a]amix=inputs=2:duration=first:dropout_transition=3[aout]"
        )
        audio_map = "[aout]"

    duration = _probe_duration(video_path, ffmpeg_bin)
    cmd = [
        ffmpeg_bin, "-y",
        "-i", video_path,
        "-i", voice_path,
        "-filter_complex", audio_filter,
        "-map", "0:v:0",
        "-map", audio_map,
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
    ]
    if duration > 0:
        cmd.extend(["-t", f"{duration:.3f}"])
    cmd.extend([
        output_path,
    ])

    try:
        result = subprocess.run(cmd, capture_output=True, timeout=600)
        if result.returncode != 0:
            err = result.stderr.decode(errors="replace")[-500:]
            logger.error(f"FFmpeg merge error: {err}")
            if progress_cb:
                progress_cb(f"❌ Ghép thất bại: {err}")
            return False, err
        if progress_cb:
            progress_cb(f"✅ Ghép xong: {Path(output_path).name}")
        return True, output_path
    except Exception as e:
        logger.error(f"Merge error: {e}")
        if progress_cb:
            progress_cb(f"❌ Lỗi: {e}")
        return False, str(e)


# ─── Text pre-processing for natural-sounding narration ──────────────────────

def preprocess_narration_text(text: str) -> str:
    """Add natural pauses and breathing marks for better TTS narration.

    Inserts short pauses at sentence boundaries and paragraph breaks
    so edge-tts sounds more like a human narrator.
    """
    import re as _re

    # Older AI-script saves could contain the overview before the first Part.
    # The overview is editing context, not narration for the video.
    first_part = _re.search(r"={2,}\s*PART\s+\d+", text, flags=_re.I)
    if first_part:
        text = text[first_part.start():]

    # Remove "=== PART N: ... ===" headers (they sound bad when read aloud)
    text = _re.sub(r"={2,}\s*PART\s+\d+[^=]*={2,}", "", text)

    # Remove excess whitespace from header removal
    text = _re.sub(r"\n{3,}", "\n\n", text)

    # Add pause after paragraph breaks (double newline → period + pause)
    # Edge-tts respects periods and commas for natural pausing
    lines = text.strip().split("\n\n")
    processed_parts = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Ensure each paragraph ends with punctuation
        if line and line[-1] not in ".!?。…":
            line += "."
        processed_parts.append(line)

    # Join with double newline (edge-tts will pause between paragraphs)
    result = "\n\n".join(processed_parts)

    # Add micro-pauses: replace " — " and " – " with comma (natural pause)
    result = _re.sub(r"\s*[—–]\s*", ", ", result)

    # Normalize multiple spaces
    result = _re.sub(r"  +", " ", result)

    return result.strip()


# ─── Convenience: popular Vietnamese edge-tts voices ─────────────────────────

VI_VOICES = [
    ("vi-VN-HoaiMyNeural",  "Hoài My (Nữ - Miền Nam) ⭐"),
    ("vi-VN-NamMinhNeural", "Nam Minh (Nam - Miền Nam) ⭐"),
]

EN_VOICES = [
    ("en-US-AriaNeural",      "Aria (Nữ - tường thuật)"),
    ("en-US-DavisNeural",     "Davis (Nam - tường thuật)"),
    ("en-US-JennyNeural",     "Jenny (Nữ - Anh Mỹ)"),
    ("en-US-GuyNeural",       "Guy (Nam - Anh Mỹ)"),
    ("en-GB-SoniaNeural",     "Sonia (Nữ - Anh Anh)"),
    ("en-US-ChristopherNeural", "Christopher (Nam - trầm)"),
]
