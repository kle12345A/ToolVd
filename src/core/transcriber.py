"""Speech-to-text transcription using faster-whisper."""

import json
import subprocess
from pathlib import Path
from typing import Callable, Optional

from src.utils.file_utils import ensure_dir
from src.utils.logger import logger

WHISPER_MODELS = ["tiny", "base", "small", "medium", "large-v2", "large-v3"]
DEFAULT_MODEL = "base"


def is_faster_whisper_available() -> bool:
    try:
        import faster_whisper  # noqa
        return True
    except ImportError:
        return False


def is_cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def extract_audio(video_path: str, audio_path: str, ffmpeg_bin: str) -> bool:
    """Extract audio track from video to WAV for Whisper."""
    cmd = [
        ffmpeg_bin, "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        audio_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=300)
        if result.returncode != 0:
            logger.error(f"Audio extract failed: {result.stderr.decode(errors='replace')}")
            return False
        return True
    except Exception as e:
        logger.error(f"Audio extract error: {e}")
        return False


def transcribe_video(
    video_path: str,
    output_dir: str,
    model_size: str = DEFAULT_MODEL,
    language: Optional[str] = None,
    use_gpu: bool = False,
    ffmpeg_bin: str = "",
    progress_cb: Optional[Callable[[str], None]] = None,
    cancel_event=None,   # threading.Event – set it to stop early
    beam_size: int = 5,  # 1=greedy/fastest … 5=beam/most accurate
) -> Optional[dict]:
    """
    Transcribe video audio using faster-whisper.
    Returns dict with segments + word timestamps, saves to transcript.json.
    """
    if not is_faster_whisper_available():
        if progress_cb:
            progress_cb("❌ faster-whisper chưa được cài. Chạy: pip install faster-whisper")
        return None

    from faster_whisper import WhisperModel

    ensure_dir(output_dir)
    audio_path = str(Path(output_dir) / "audio_16k.wav")

    if progress_cb:
        progress_cb("🔊 Đang trích xuất audio từ video...")

    if ffmpeg_bin and not extract_audio(video_path, audio_path, ffmpeg_bin):
        audio_path = video_path  # fallback: pass video directly to whisper

    if progress_cb:
        progress_cb(f"🤖 Đang tải model Whisper '{model_size}'...")

    try:
        device = "cuda" if use_gpu and is_cuda_available() else "cpu"
        compute = "float16" if device == "cuda" else "int8"
        model = WhisperModel(model_size, device=device, compute_type=compute)

        if progress_cb:
            progress_cb(f"📝 Đang phiên âm {'(GPU)' if device == 'cuda' else '(CPU)'}...")

        segments_gen, info = model.transcribe(
            audio_path,
            language=language or None,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
            beam_size=beam_size,
        )

        detected_lang = info.language
        duration = info.duration
        if progress_cb:
            progress_cb(f"🌐 Ngôn ngữ phát hiện: {detected_lang} | Thời lượng: {duration:.1f}s")

        segments_data = []
        cancelled = False
        for seg in segments_gen:
            if cancel_event and cancel_event.is_set():
                if progress_cb:
                    progress_cb("⛔ Phiên âm đã bị hủy.")
                cancelled = True
                break
            words = []
            if seg.words:
                for w in seg.words:
                    words.append({
                        "word": w.word,
                        "start": round(w.start, 3),
                        "end": round(w.end, 3),
                        "probability": round(w.probability, 3),
                    })
            seg_dict = {
                "id": seg.id,
                "start": round(seg.start, 3),
                "end": round(seg.end, 3),
                "text": seg.text.strip(),
                "words": words,
            }
            segments_data.append(seg_dict)
            if progress_cb:
                progress_cb(f"  [{seg.start:.1f}s] {seg.text.strip()[:60]}")

        transcript = {
            "language": detected_lang,
            "duration": duration,
            "model": model_size,
            "video": video_path,
            "segments": segments_data,
        }

        if cancelled:
            return None

        transcript_path = str(Path(output_dir) / "transcript.json")
        with open(transcript_path, "w", encoding="utf-8") as f:
            json.dump(transcript, f, ensure_ascii=False, indent=2)

        if progress_cb:
            progress_cb(f"✅ Phiên âm xong! {len(segments_data)} đoạn → {transcript_path}")

        return transcript

    except Exception as e:
        logger.error(f"Transcription error: {e}")
        if progress_cb:
            progress_cb(f"❌ Lỗi phiên âm: {e}")
        return None


def load_transcript(transcript_path: str) -> Optional[dict]:
    try:
        with open(transcript_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Load transcript error: {e}")
        return None
