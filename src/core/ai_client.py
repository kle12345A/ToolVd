"""Free AI API client — Gemini, Groq, OpenRouter, Ollama.

All HTTP calls use stdlib urllib only (no extra deps).
"""

from __future__ import annotations

import json
import re
import base64
import subprocess
import socket
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

from src.utils.logger import logger
from src.core.dependency_manager import ffmpeg_path

_SCRIPT_RATES = {
    # Vietnamese tokens are mostly short syllables separated by spaces.
    "vi": (3.4, 4.6, 5.2),
    # English words are longer and naturally read at a lower token rate.
    "en": (1.5, 2.2, 2.7),
}


def _script_word_budget(
    seconds: float,
    output_language: str = "vi",
) -> tuple[int, int, int]:
    """Return (min, target, max) word budget for readable TTS narration."""
    sec = max(1.0, float(seconds or 0.0))
    min_wps, target_wps, max_wps = _SCRIPT_RATES.get(
        output_language, _SCRIPT_RATES["vi"]
    )
    target = max(5, int(round(sec * target_wps)))
    max_words = max(target, int(round(sec * max_wps)))
    min_words = max(3, int(round(sec * min_wps)))
    return min_words, target, max_words


def _count_words(text: str) -> int:
    return len(re.findall(r"\S+", text or ""))


def _trim_to_word_limit(text: str, max_words: int) -> str:
    """Trim narration to a hard word limit, preferring whole sentences."""
    words = re.findall(r"\S+", text or "")
    if len(words) <= max_words:
        return (text or "").strip()

    sentences = re.split(r"(?<=[.!?。！？])\s+", (text or "").strip())
    kept: list[str] = []
    used = 0
    for sentence in sentences:
        sentence_words = re.findall(r"\S+", sentence)
        if not sentence_words:
            continue
        if used and used + len(sentence_words) > max_words:
            break
        if not used and len(sentence_words) > max_words:
            trimmed = " ".join(sentence_words[:max_words]).strip()
            break
        kept.append(sentence.strip())
        used += len(sentence_words)
    else:
        trimmed = " ".join(kept).strip()

    if not kept and "trimmed" not in locals():
        trimmed = " ".join(words[:max_words]).strip()
    elif kept and "trimmed" not in locals():
        trimmed = " ".join(kept).strip()

    if trimmed and trimmed[-1] not in ".!?。！？":
        trimmed += "."
    return trimmed


def _enforce_part_script_budgets(
    result: dict,
    clip_durations: list[float] | None,
    duration_per_part_sec: int,
    output_language: str = "vi",
    status_cb=None,
) -> dict:
    parts = result.get("parts") or []
    trimmed_count = 0

    for pos, part in enumerate(parts):
        if clip_durations and pos < len(clip_durations):
            seconds = clip_durations[pos]
        else:
            seconds = duration_per_part_sec
        _min_words, target_words, max_words = _script_word_budget(
            seconds, output_language
        )

        script = (part.get("script") or "").strip()
        word_count = _count_words(script)
        if word_count > max_words:
            part["script"] = _trim_to_word_limit(script, max_words)
            trimmed_count += 1
        part["estimated_seconds"] = int(round(seconds))
        part["target_words"] = target_words
        part["max_words"] = max_words

    if trimmed_count and status_cb:
        status_cb(
            f"✂️ Đã tự rút gọn {trimmed_count} Part vì vượt số từ tối đa theo thời lượng video."
        )
    return result

# ── API key store ──────────────────────────────────────────────────────────────

_API_KEYS_FILE = Path("data/api_keys.json")


def load_api_key(provider: str) -> str:
    """Return stored API key for *provider*, or empty string."""
    if _API_KEYS_FILE.exists():
        try:
            with open(_API_KEYS_FILE, "r", encoding="utf-8") as f:
                return json.load(f).get(provider, "")
        except Exception:
            pass
    return ""


# ── Low-level HTTP helper ──────────────────────────────────────────────────────

def _post_json(url: str, payload: dict, headers: dict | None = None,
               timeout: int = 120) -> dict:
    """POST *payload* as JSON to *url*; return parsed response dict."""
    body = json.dumps(payload).encode("utf-8")
    req_headers = {"Content-Type": "application/json"}
    if headers:
        req_headers.update(headers)
    req = urllib.request.Request(url, data=body, headers=req_headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        msg = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code}: {msg}") from e
    except urllib.error.URLError as e:
        reason = getattr(e, "reason", e)
        if "localhost:11434" in url or "127.0.0.1:11434" in url:
            raise RuntimeError(
                "Ollama chưa chạy ở http://localhost:11434.\n\n"
                "Mở ứng dụng Ollama trước, sau đó chạy lệnh:\n"
                f"  ollama pull {payload.get('model', 'phi3:mini')}\n\n"
                "Nếu Ollama đã mở, kiểm tra service có đang listen port 11434 không."
            ) from e
        if isinstance(reason, (ConnectionRefusedError, TimeoutError, socket.timeout)):
            raise RuntimeError(f"Cannot connect to API server: {reason}") from e
        raise RuntimeError(f"Network error: {reason}") from e


# ── Provider implementations ───────────────────────────────────────────────────

def call_gemini(prompt: str, api_key: str,
                model: str = "gemini-2.5-flash-lite") -> str:
    """Call Google Gemini REST API (free tier: 15 RPM)."""
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 8192,
        },
    }
    data = _post_json(url, payload)
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"Gemini response parse error: {data}") from e


def call_gemini_vision(prompt: str, image_paths: list[str], api_key: str,
                       model: str = "gemini-2.5-flash-lite") -> str:
    """Call Gemini with text + local image frames."""
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )
    parts = [{"text": prompt}]
    for image_path in image_paths:
        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("ascii")
        parts.append({
            "inlineData": {
                "mimeType": "image/jpeg",
                "data": image_b64,
            }
        })
    payload = {
        "contents": [{"parts": parts}],
        "generationConfig": {
            "temperature": 0.35,
            "maxOutputTokens": 1024,
        },
    }
    data = _post_json(url, payload, timeout=180)
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"Gemini vision response parse error: {data}") from e


def call_groq(prompt: str, api_key: str,
              model: str = "llama-3.3-70b-versatile") -> str:
    """Call Groq API (free tier, OpenAI-compatible)."""
    url = "https://api.groq.com/openai/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Respond in JSON.\n\n" + prompt}],
        "max_tokens": 8192,
        "temperature": 0.7,
    }
    data = _post_json(url, payload, headers={"Authorization": f"Bearer {api_key}"})
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"Groq response parse error: {data}") from e


def call_openrouter(prompt: str, api_key: str,
                    model: str = "google/gemini-flash-1.5:free") -> str:
    """Call OpenRouter (free models available).

    Good free models:
      - google/gemini-flash-1.5:free
      - meta-llama/llama-3.1-8b-instruct:free
      - mistralai/mistral-7b-instruct:free
    """
    url = "https://openrouter.ai/api/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 8192,
    }
    data = _post_json(url, payload, headers={
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://github.com/amsr-studio",
    })
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"OpenRouter response parse error: {data}") from e


def call_ollama(prompt: str, model: str = "llama3.1:8b",
                host: str = "http://localhost:11434") -> str:
    """Call local Ollama instance (completely free, no API key)."""
    url = f"{host}/api/chat"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }
    data = _post_json(url, payload, timeout=300)
    try:
        return data["message"]["content"]
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"Ollama response parse error: {data}") from e


def call_text_provider(provider: str, model: str, prompt: str) -> str:
    """Call one configured text provider with a plain prompt."""
    if provider == "gemini":
        key = load_api_key("gemini")
        if not key:
            raise RuntimeError("Chưa có Gemini API key. Vào ⚙ Cài đặt → API Keys.")
        return call_gemini(prompt, key, model or "gemini-2.5-flash-lite")
    if provider == "groq":
        key = load_api_key("groq")
        if not key:
            raise RuntimeError("Chưa có Groq API key. Vào ⚙ Cài đặt → API Keys.")
        return call_groq(prompt, key, model or "llama-3.3-70b-versatile")
    if provider == "openrouter":
        key = load_api_key("openrouter")
        if not key:
            raise RuntimeError("Chưa có OpenRouter API key. Vào ⚙ Cài đặt → API Keys.")
        return call_openrouter(prompt, key, model or "google/gemini-2.0-flash-exp:free")
    if provider == "ollama":
        return call_ollama(prompt, model or "llama3.1:8b")
    raise RuntimeError(f"Provider dịch không hợp lệ: {provider}")


def translate_timed_segments(
    segments: list[dict], provider: str, model: str, status_cb=None,
) -> list[dict]:
    """Translate timestamped Chinese speech to concise Vietnamese per segment."""
    source = [s for s in segments if (s.get("text") or "").strip()]
    translated: list[dict] = []
    for offset in range(0, len(source), 30):
        batch = source[offset:offset + 30]
        payload = [
            {
                "id": i,
                "duration": round(max(0.2, float(s["end"]) - float(s["start"])), 2),
                "text_zh": s["text"].strip(),
            }
            for i, s in enumerate(batch)
        ]
        prompt = (
            "Bạn là biên dịch viên lồng tiếng Trung-Việt. Dịch từng câu sang tiếng Việt "
            "tự nhiên, giữ đúng ý, xưng hô nhất quán và đủ ngắn để đọc trong duration. "
            "Không gộp, tách hoặc đổi id. Chỉ trả JSON dạng "
            '{"segments":[{"id":0,"text_vi":"..."}]}.\n\n'
            + json.dumps(payload, ensure_ascii=False)
        )
        if status_cb:
            status_cb(f"🌐 Đang dịch câu {offset + 1}–{offset + len(batch)}/{len(source)}…")
        raw = call_text_provider(provider, model, prompt)
        match = re.search(r"\{[\s\S]*\}", raw)
        if not match:
            raise RuntimeError(f"AI không trả JSON dịch hợp lệ: {raw[:300]}")
        rows = json.loads(match.group()).get("segments") or []
        by_id = {
            int(row.get("id", -1)): (row.get("text_vi") or "").strip()
            for row in rows
        }
        for i, segment in enumerate(batch):
            text_vi = by_id.get(i, "")
            if not text_vi:
                raise RuntimeError(f"Thiếu bản dịch cho câu {offset + i + 1}.")
            translated.append({
                "start": float(segment["start"]),
                "end": float(segment["end"]),
                "text_zh": segment["text"].strip(),
                "text_vi": text_vi,
            })
    return translated


# ── Transcript loader ──────────────────────────────────────────────────────────

def preferred_transcript_path(project) -> str:
    """Return the source transcript when available, then the active transcript.

    Review workflows may create a second transcript for the narrated review.
    AI scene/script generation must keep using the transcript that belongs to
    the source footage and its clip timestamps.
    """
    candidates = (
        getattr(project, "source_transcript_file", "") or "",
        getattr(project, "transcript_file", "") or "",
    )
    for path in candidates:
        if path and Path(path).exists():
            return path
    return ""

def _load_transcript_segments(project) -> list[dict]:
    """Load transcript segments with timestamps from project JSON.

    Returns list of dicts: [{start, end, text}, ...]
    """
    tf = preferred_transcript_path(project)
    if not tf:
        return []
    try:
        with open(tf, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.warning(f"Could not read transcript JSON: {e}")
        return []

    segments = data.get("segments", [])
    return [
        {
            "start": seg.get("start", 0.0),
            "end": seg.get("end", 0.0),
            "text": seg.get("text", "").strip(),
        }
        for seg in segments
        if seg.get("text", "").strip()
    ]


def load_transcript_text(project) -> str:
    """Extract full plain text from the project's transcript JSON or SRT files.

    Returns a string of ≤ 12 000 characters (trimmed from middle if too long).
    """
    text = ""

    # 1) Try transcript JSON (Whisper output)
    tf = preferred_transcript_path(project)
    if tf:
        try:
            with open(tf, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Whisper JSON: {"text": "...", "segments": [...]}
            if "text" in data:
                text = data["text"]
            elif "segments" in data:
                text = " ".join(s.get("text", "") for s in data["segments"])
        except Exception as e:
            logger.warning(f"Could not read transcript JSON: {e}")

    # 2) Fallback: concatenate subtitle files from clips
    if not text:
        parts = []
        for clip in (project.clips or []):
            if not getattr(clip, "enabled", True):
                continue
            sf = clip.subtitle_file
            if sf and Path(sf).exists():
                try:
                    with open(sf, "r", encoding="utf-8") as f:
                        raw = f.read()
                    # Strip SRT timestamps (lines like "00:00:01,000 --> 00:00:04,000")
                    raw = re.sub(r"\d+:\d+:\d+[,.]\d+ --> \d+:\d+:\d+[,.]\d+", "", raw)
                    raw = re.sub(r"^\d+\s*$", "", raw, flags=re.MULTILINE)
                    parts.append(raw.strip())
                except Exception:
                    pass
        text = "\n".join(parts)

    if not text:
        return ""

    # Trim to ~12 000 chars: keep first 5000, last 5000, indicate cut
    MAX = 12_000
    if len(text) <= MAX:
        return text.strip()
    half = MAX // 2
    return (
        text[:half].strip()
        + "\n\n[... phần giữa phim được lược bỏ để giảm kích thước ...]\n\n"
        + text[-half:].strip()
    )


def _extract_clip_transcript(
    segments: list[dict],
    start_time: float,
    end_time: float,
    clip_id: str = "",
    max_chars: int = 800,
) -> str:
    """Extract transcript text that falls within [start_time, end_time].

    Returns plain text trimmed to max_chars.
    """
    parts = []
    total = 0
    has_clip_tags = bool(
        clip_id and any(seg.get("source_clip_id") for seg in segments)
    )
    for seg in segments:
        if has_clip_tags:
            if seg.get("source_clip_id") != clip_id:
                continue
        else:
            # Include segment if it overlaps with the clip time range
            if seg["end"] < start_time:
                continue
            if seg["start"] > end_time:
                break
        txt = seg["text"]
        total += len(txt) + 1
        if total > max_chars:
            parts.append("...")
            break
        parts.append(txt)
    return " ".join(parts).strip()


def _build_clip_context_block(
    project,
    clips: list,
    clip_durations: list[float] | None,
    segments: list[dict],
    output_language: str = "vi",
) -> str:
    """Build a detailed per-clip context block for the AI prompt.

    Includes: clip index, time range, duration, scene description,
    and the transcript excerpt for that clip's time range.
    """
    lines = []
    for i, clip in enumerate(clips):
        dur = clip_durations[i] if clip_durations and i < len(clip_durations) else clip.duration
        _min_words, words_target, max_words = _script_word_budget(
            dur, output_language
        )

        # Extract transcript for this clip's time range
        clip_transcript = _extract_clip_transcript(
            segments, clip.start_time, clip.end_time,
            clip_id=getattr(clip, "id", ""),
        )

        # Scene description from AI scene analysis (stored in custom_subtitle)
        scene_desc = getattr(clip, "custom_subtitle", "") or ""

        header = (
            f"--- PART {clip.index} [{_fmt_ts(clip.start_time)} → {_fmt_ts(clip.end_time)}] "
            f"| {int(dur)}s | mục tiêu ~{words_target} từ | tối đa {max_words} từ ---"
        )
        lines.append(header)
        if scene_desc:
            lines.append(f"  Mô tả cảnh: {scene_desc}")
        if clip_transcript:
            lines.append(f"  Lời thoại: {clip_transcript}")
        else:
            lines.append("  (Không có lời thoại trong đoạn này)")
        lines.append("")

    return "\n".join(lines)


def _extract_clip_frames(project, clip, max_frames: int = 3) -> list[str]:
    """Extract a few representative JPEG frames for visual-only script context."""
    ff = ffmpeg_path()
    source = getattr(clip, "source_video", "") or getattr(project, "source_video", "")
    if not ff or not source or not Path(source).exists():
        return []

    out_dir = Path(getattr(project, "output_dir", "") or "output") / "analysis" / "script_frames"
    out_dir.mkdir(parents=True, exist_ok=True)

    start = float(getattr(clip, "start_time", 0.0) or 0.0)
    end = float(getattr(clip, "end_time", start) or start)
    duration = max(0.1, end - start)
    offsets = [0.18, 0.50, 0.82][:max(1, max_frames)]
    frame_paths = []

    for idx, offset in enumerate(offsets, start=1):
        timestamp = start + duration * offset
        timestamp = min(max(start, timestamp), max(start, end - 0.2))
        out_path = out_dir / f"part{int(getattr(clip, 'index', 0) or 0):02d}_frame{idx}.jpg"
        cmd = [
            ff, "-hide_banner", "-y",
            "-ss", f"{timestamp:.3f}",
            "-i", source,
            "-frames:v", "1",
            "-q:v", "4",
            "-vf", "scale='min(768,iw)':-2",
            str(out_path),
        ]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=45,
            )
            if proc.returncode == 0 and out_path.exists() and out_path.stat().st_size > 0:
                frame_paths.append(str(out_path))
            else:
                logger.warning(f"Frame extraction failed: {proc.stderr[-500:]}")
        except Exception as exc:
            logger.warning(f"Frame extraction error: {exc}")
    return frame_paths


def _describe_clip_frames(project, clip, image_paths: list[str], api_key: str, model: str) -> str:
    """Ask Gemini to describe what happens in a silent/process clip."""
    if not image_paths:
        return ""
    prompt = f"""Describe these video frames for a voice-over writer.

Video/project name: "{getattr(project, 'name', '')}"
Part: {getattr(clip, 'index', 0)}
Time range: {_fmt_ts(getattr(clip, 'start_time', 0.0))} to {_fmt_ts(getattr(clip, 'end_time', 0.0))}

Focus on visible actions, objects, machines, materials, workflow steps, factory/process details, and changes between frames.
Do not invent unseen technical facts. If uncertain, say it as visual observation.
Return 3-5 concise bullet points in Vietnamese."""
    try:
        return call_gemini_vision(prompt, image_paths, api_key, model).strip()
    except Exception as exc:
        logger.warning(f"Gemini visual description failed for Part {getattr(clip, 'index', '?')}: {exc}")
        return ""


def _build_visual_clip_context_block(
    project,
    clips: list,
    clip_durations: list[float] | None,
    provider: str,
    model: str,
    output_language: str = "vi",
    status_cb=None,
) -> str:
    """Build per-clip context when no transcript is available."""
    gemini_key = load_api_key("gemini")
    vision_model = GEMINI_MODEL_FALLBACKS.get(model, model) if provider == "gemini" else "gemini-2.5-flash-lite"
    lines = []

    for i, clip in enumerate(clips):
        dur = clip_durations[i] if clip_durations and i < len(clip_durations) else clip.duration
        _min_words, words_target, max_words = _script_word_budget(
            dur, output_language
        )
        scene_desc = (getattr(clip, "custom_subtitle", "") or "").strip()
        visual_desc = ""

        if not scene_desc and gemini_key:
            if status_cb:
                status_cb(f"Đang đọc hình ảnh Part {clip.index} bằng Gemini Vision...")
            frames = _extract_clip_frames(project, clip)
            visual_desc = _describe_clip_frames(project, clip, frames, gemini_key, vision_model)

        header = (
            f"--- PART {clip.index} [{_fmt_ts(clip.start_time)} -> {_fmt_ts(clip.end_time)}] "
            f"| {int(dur)}s | mục tiêu ~{words_target} từ | tối đa {max_words} từ ---"
        )
        lines.append(header)
        if scene_desc:
            lines.append(f"  Mô tả cảnh có sẵn: {scene_desc}")
        if visual_desc:
            lines.append(f"  Mô tả hình ảnh từ frame: {visual_desc}")
        if not scene_desc and not visual_desc:
            lines.append(
                "  Không có transcript. Hãy viết theo hướng quan sát quy trình, "
                "dựa vào tên video, Part number và mốc thời gian; không bịa chi tiết kỹ thuật."
            )
        lines.append("")

    return "\n".join(lines)


def _fmt_ts(seconds: float) -> str:
    """Format seconds to MM:SS."""
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"


# ── Main generation function ───────────────────────────────────────────────────

REVIEW_STYLES = {
    "recap_complete": "Recap hoan chinh tu timeline (ke lai nhu mot cau chuyen moi)",
    "narrate":    "Tường thuật kể chuyện (giống kênh Tóm Tắt Phim)",
    "summary":    "Tóm tắt ngắn gọn, súc tích",
    "detailed":   "Review chi tiết, phân tích nhân vật & cốt truyện",
    "hype":       "Highlight kịch tính, gây tò mò (hook người xem)",
    "spoiler_free": "Spoiler-free — chỉ gợi ý, không tiết lộ kết thúc",
    "critique":   "Phê bình chuyên sâu, đánh giá điểm mạnh/yếu",
}

OPENROUTER_FREE_MODELS = [
    ("google/gemini-flash-1.5:free",          "Gemini Flash 1.5 (OR)"),
    ("meta-llama/llama-3.1-8b-instruct:free", "Llama 3.1 8B (OR)"),
    ("mistralai/mistral-7b-instruct:free",    "Mistral 7B (OR)"),
    ("microsoft/phi-3-mini-128k-instruct:free","Phi-3 Mini (OR)"),
]

GROQ_FREE_MODELS = [
    ("llama-3.3-70b-versatile",   "Llama 3.3 70B"),
    ("llama-3.1-8b-instant",      "Llama 3.1 8B (fast)"),
    ("mixtral-8x7b-32768",        "Mixtral 8x7B"),
    ("gemma2-9b-it",              "Gemma2 9B"),
]

GEMINI_FREE_MODELS = [
    ("gemini-2.5-flash-lite", "Gemini 2.5 Flash-Lite (free/test)"),
    ("gemini-2.5-flash",      "Gemini 2.5 Flash"),
    ("gemini-2.0-flash-lite", "Gemini 2.0 Flash-Lite"),
    ("gemini-2.0-flash",      "Gemini 2.0 Flash"),
]

GEMINI_MODEL_FALLBACKS = {
    "gemini-1.5-flash": "gemini-2.5-flash-lite",
    "gemini-1.5-flash-8b": "gemini-2.5-flash-lite",
}

OLLAMA_SUGGESTED = [
    ("llama3.1:8b",   "Llama 3.1 8B"),
    ("llama3.2:3b",   "Llama 3.2 3B (nhẹ)"),
    ("mistral:7b",    "Mistral 7B"),
    ("qwen2.5:7b",    "Qwen2.5 7B"),
    ("phi3:mini",     "Phi-3 Mini"),
]


def _build_prompt(
    transcript: str,
    movie_name: str,
    num_parts: int,
    duration_per_part_sec: int,
    style_key: str,
    output_language: str,
    clip_durations: list[float] | None = None,
    clip_context: str = "",
    extra_prompt: str = "",
) -> str:
    style_desc = REVIEW_STYLES.get(style_key, style_key)
    lang_instruction = (
        "Viết hoàn toàn bằng TIẾNG VIỆT."
        if output_language == "vi"
        else "Write entirely in ENGLISH."
    )

    # ── Per-part duration info ──
    if clip_durations and len(clip_durations) == num_parts:
        total_dur = sum(clip_durations)
        _min_example, example_words, max_example = _script_word_budget(
            clip_durations[0], output_language
        )
        example_seconds = int(clip_durations[0])
        budget_lines = []
        for i, seconds in enumerate(clip_durations[:num_parts], start=1):
            min_words, target_words, max_words = _script_word_budget(
                seconds, output_language
            )
            budget_lines.append(
                f"- Part {i}: {int(seconds)}s -> {min_words}-{max_words} từ, mục tiêu {target_words} từ"
            )
    else:
        total_dur = duration_per_part_sec * num_parts
        _min_example, example_words, max_example = _script_word_budget(
            duration_per_part_sec, output_language
        )
        example_seconds = duration_per_part_sec
        min_words, target_words, max_words = _script_word_budget(
            duration_per_part_sec, output_language
        )
        budget_lines = [
            f"- Mỗi Part: {duration_per_part_sec}s -> {min_words}-{max_words} từ, mục tiêu {target_words} từ"
        ]
    budget_block = "\n".join(budget_lines)

    has_transcript = bool((transcript or "").strip())

    # ── Clip context or full transcript ──
    if clip_context:
        content_block = (
            f"=== CÁC CẢNH ĐÃ CẮT (theo thứ tự thời gian) ===\n"
            f"{clip_context}\n"
            f"=== KẾT THÚC CÁC CẢNH ==="
        )
        if has_transcript:
            content_block += (
                f"\n\n=== BẢN PHIÊN ÂM TOÀN BỘ PHIM (tham khảo thêm) ===\n"
                f"{transcript}\n"
                f"=== KẾT THÚC PHIÊN ÂM ==="
            )
    else:
        content_block = (
            f"=== BẢN PHIÊN ÂM ===\n"
            f"{transcript}\n"
            f"=== KẾT THÚC PHIÊN ÂM ==="
        )

    # ── Style-specific writing instructions ──
    style_writing_guide = {
        "recap_complete": (
            "PHONG CACH RECAP HOAN CHINH TU TIMELINE:\n"
            "- Doc hoi thoai va mo ta canh de viet lai thanh mot cau chuyen moi, lien mach.\n"
            "- Moi Part phai khop voi canh video, nhung cau chu la loi ke moi, khong copy loi thoai goc.\n"
            "- Neu co mo ta canh tu AI cat canh, dung no lam xuong song timeline.\n"
            "- Nhan manh nhan vat, cam xuc, dong co, bien co va he qua cua tung su kien.\n"
            "- Giong review/recap tu nhien, ro nghia, co hook nhung khong lam lech noi dung."
        ),
        "narrate": (
            "PHONG CÁCH TƯỜNG THUẬT KỂ CHUYỆN (giống kênh 'Tóm Tắt Phim'):\n"
            "- Kể lại cốt truyện bằng NGÔI THỨ BA, giọng người dẫn chuyện.\n"
            "- Viết THUẦN câu TRẦN THUẬT — TUYỆT ĐỐI KHÔNG dùng câu hỏi.\n"
            "  ❌ SAI: 'Liệu anh ta có thoát được?', 'Chuyện gì xảy ra tiếp theo?'\n"
            "  ✅ ĐÚNG: 'Anh ta lao vào cứu cô gái mà không biết cái bẫy phía sau.'\n"
            "- Mục đích: Người nghe HIỂU TOÀN BỘ CỐT TRUYỆN mà không cần xem phim.\n"
            "- Câu ngắn, nhịp nhanh, không dài dòng.\n"
            "- Dùng từ nối tự nhiên: 'Ngay lúc đó', 'Thế nhưng', 'Không ngờ', 'Cuối cùng'..."
        ),
        "summary": (
            "PHONG CÁCH TÓM TẮT NGẮN GỌN:\n"
            "- Kể lại cốt truyện chính, bỏ chi tiết phụ.\n"
            "- Giọng trung lập, rõ ràng.\n"
            "- Tập trung vào sự kiện chính và kết quả."
        ),
        "detailed": (
            "PHONG CÁCH REVIEW CHI TIẾT:\n"
            "- Phân tích nhân vật, động cơ, diễn biến tâm lý.\n"
            "- Kể lại cốt truyện kèm nhận xét sâu.\n"
            "- Có thể xen lẫn ý kiến cá nhân của người review."
        ),
        "hype": (
            "PHONG CÁCH KỊCH TÍNH, GÂY TÒ MÒ:\n"
            "- Kể chuyện với nhịp điệu căng thẳng.\n"
            "- Nhấn mạnh xung đột, bất ngờ.\n"
            "- Có thể dùng câu hỏi tu từ VỪA PHẢI (tối đa 1-2 câu/part)."
        ),
        "spoiler_free": (
            "PHONG CÁCH SPOILER-FREE:\n"
            "- Giới thiệu bối cảnh và nhân vật, gợi ý diễn biến.\n"
            "- KHÔNG tiết lộ kết thúc hay twist chính.\n"
            "- Tạo hứng thú để người xem muốn xem phim."
        ),
        "critique": (
            "PHONG CÁCH PHÊ BÌNH:\n"
            "- Phân tích điểm mạnh/yếu của phim.\n"
            "- Kể tóm tắt cốt truyện kết hợp đánh giá.\n"
            "- Giọng chuyên nghiệp, có chiều sâu."
        ),
    }
    writing_guide = style_writing_guide.get(style_key, style_writing_guide["narrate"])
    extra_prompt_block = ""
    if (extra_prompt or "").strip():
        extra_prompt_block = f"""
YÊU CẦU THÊM TỪ NGƯỜI DÙNG:
{extra_prompt.strip()}

Lưu ý: Yêu cầu thêm này được ưu tiên về phong cách/nội dung, nhưng vẫn KHÔNG được phá các luật thời lượng, số từ, JSON output và không bịa chi tiết ngoài video.
"""

    return f"""Bạn là biên kịch tường thuật phim chuyên nghiệp cho YouTube/TikTok.

{lang_instruction}

NHIỆM VỤ: Viết kịch bản TƯỜNG THUẬT LẠI bộ phim "{movie_name}".
Video đã được cắt thành {num_parts} đoạn cảnh chính theo thứ tự thời gian.
Kịch bản sẽ được đọc lên TRÊN các đoạn video này.
Mục tiêu: Người nghe HIỂU ĐƯỢC TOÀN BỘ CỐT TRUYỆN chỉ qua lời kể.
Tổng thời lượng: {total_dur/60:.1f} phút ({int(total_dur)}s).

NGÂN SÁCH CHỮ THEO THỜI LƯỢNG ĐỌC THẬT:
{budget_block}

{writing_guide}

{content_block}

{extra_prompt_block}

QUY TẮC BẮT BUỘC:
1. TƯỜNG THUẬT LIÊN TỤC — Toàn bộ {num_parts} part là MỘT câu chuyện liền mạch.
   Khi ghép lại, phải đọc như một bài kể chuyện hoàn chỉnh từ đầu đến cuối.

2. MỖI PART KHỚP NỘI DUNG VIDEO — Part N chỉ kể những gì xảy ra trong đoạn video Part N.
   Không kể trước, không kể lệch.

3. NỐI TIẾP TỰ NHIÊN — Cuối Part N chuyển mượt sang đầu Part N+1.
   Dùng từ nối: "Ngay lúc đó", "Thế nhưng", "Về phía bên kia", "Không lâu sau"...
   KHÔNG lặp lại nội dung đã kể ở part trước.

4. CẤU TRÚC:
   Part 1 → Giới thiệu bối cảnh, nhân vật, tình huống mở đầu.
   Parts giữa → Diễn biến chính, xung đột, bước ngoặt.
   Part cuối → Cao trào, kết cục, kết thúc câu chuyện.

5. SỐ TỪ PHẢI KHỚP THỜI LƯỢNG ĐỌC THẬT.
   Viết theo ngân sách chữ ở trên, ưu tiên câu ngắn và đọc vừa nhịp.
   TUYỆT ĐỐI không vượt số từ tối đa của từng Part.
   Ví dụ Part 14s chỉ nên khoảng 25 từ, không viết thành đoạn 50-80 từ.

6. KHÔNG copy lời thoại gốc — diễn đạt lại bằng lời kể tự nhiên.

7. GIỌNG VĂN: Như đang KỂ CHUYỆN cho bạn bè nghe.
   Câu ngắn gọn, dễ hiểu, nhịp đều đặn.
   Tránh câu dài quá 25 từ. Tránh liệt kê.

TRẢ VỀ JSON (KHÔNG thêm text nào ngoài JSON):
{{
  "movie_name": "{movie_name}",
  "overview": "Tóm tắt cốt truyện 2-3 câu",
  "style": "{style_key}",
  "parts": [
    {{
      "index": 1,
      "title": "PART 1: [Tên ngắn]",
      "script": "Kịch bản tường thuật Part 1 (~{example_words} từ, tối đa {max_example} từ)...",
      "estimated_seconds": {example_seconds}
    }}
  ]
}}"""


def generate_review_scripts(
    project,
    provider: str,
    model: str,
    style_key: str,
    output_language: str,
    duration_per_part_sec: int,
    num_parts: int,
    clip_durations: list[float] | None = None,
    extra_prompt: str = "",
    status_cb=None,
) -> dict:
    """Generate voiceover review scripts for all parts.

    Returns parsed dict with keys: overview, parts (list of dicts).
    Raises RuntimeError on failure.
    """
    if status_cb:
        status_cb("📖 Đang đọc transcript từ project...")

    transcript = load_transcript_text(project)
    has_transcript = bool(transcript)

    # ── Build per-clip context (transcript + scene description) ──
    clip_context = ""
    enabled_clips = [
        clip for clip in (project.clips or [])
        if getattr(clip, "enabled", True)
    ]
    clips = enabled_clips[:num_parts]
    if project.clips and not clips:
        raise RuntimeError(
            "Khong co Part nao dang duoc bat de tao kich ban."
        )
    if clips:
        # Keep the requested count and optional duration vector aligned with
        # the exact enabled clips used to build context.
        num_parts = len(clips)
        if clip_durations is not None:
            clip_durations = list(clip_durations[:num_parts])
    if clips:
        if status_cb:
            status_cb(f"🎬 Đang trích xuất nội dung cho {len(clips)} clips...")
        segments = _load_transcript_segments(project)
        if segments:
            clip_context = _build_clip_context_block(
                project, clips, clip_durations, segments, output_language,
            )
            if status_cb:
                status_cb(
                    f"✅ Đã map transcript cho {len(clips)} clips "
                    f"({len(clip_context):,} ký tự context)"
                )
        else:
            if status_cb:
                if has_transcript:
                    status_cb("⚠️ Không có timestamp segments — dùng transcript gộp")
                else:
                    status_cb("⚠️ Không có transcript — sẽ dùng mô tả hình ảnh/visual")

    if status_cb:
        if has_transcript:
            status_cb(f"📝 Transcript: {len(transcript):,} ký tự. Đang xây dựng prompt...")
        else:
            status_cb("📝 Đang xây dựng prompt từ mô tả hình ảnh/visual...")

    if not has_transcript and clips and not clip_context:
        clip_context = _build_visual_clip_context_block(
            project=project,
            clips=clips,
            clip_durations=clip_durations,
            provider=provider,
            model=model,
            output_language=output_language,
            status_cb=status_cb,
        )
        if status_cb:
            status_cb(
                f"Đã tạo context hình ảnh/visual cho {len(clips)} clips "
                f"({len(clip_context):,} ký tự)."
            )

    if not has_transcript and not clip_context:
        raise RuntimeError(
            "Project chua co transcript va chua co Parts de tao context hinh anh."
        )

    prompt = _build_prompt(
        transcript=transcript,
        movie_name=project.name,
        num_parts=num_parts,
        duration_per_part_sec=duration_per_part_sec,
        style_key=style_key,
        output_language=output_language,
        clip_durations=clip_durations,
        clip_context=clip_context,
        extra_prompt=extra_prompt,
    )

    if provider == "gemini":
        model = GEMINI_MODEL_FALLBACKS.get(model, model) or "gemini-2.5-flash-lite"

    if status_cb:
        status_cb(f"🤖 Gửi yêu cầu đến {provider} ({model})...")

    # Call the chosen provider
    if provider == "gemini":
        key = load_api_key("gemini")
        if not key:
            raise RuntimeError("Chưa có Gemini API key. Vào ⚙ Cài đặt → API Keys.")
        candidates = [
            model,
            "gemini-2.5-flash",
            "gemini-2.0-flash-lite",
            "gemini-2.0-flash",
        ]
        tried = []
        last_error = None
        for candidate in dict.fromkeys(candidates):
            if not candidate:
                continue
            tried.append(candidate)
            if status_cb and candidate != model:
                status_cb(f"⚠️ Gemini đang quá tải, thử model dự phòng: {candidate}...")
            try:
                raw = call_gemini(prompt, key, candidate)
                model = candidate
                break
            except RuntimeError as exc:
                last_error = exc
                msg = str(exc)
                if "HTTP 503" not in msg and "UNAVAILABLE" not in msg:
                    raise
        else:
            raise RuntimeError(
                "Gemini đang quá tải hoặc tạm thời không khả dụng.\n"
                f"Đã thử: {', '.join(tried)}\n\n"
                f"Lỗi cuối: {last_error}"
            )

    elif provider == "groq":
        key = load_api_key("groq")
        if not key:
            raise RuntimeError("Chưa có Groq API key. Vào ⚙ Cài đặt → API Keys.")
        raw = call_groq(prompt, key, model)

    elif provider == "openrouter":
        key = load_api_key("openrouter")
        if not key:
            raise RuntimeError("Chưa có OpenRouter API key. Vào ⚙ Cài đặt → API Keys.")
        raw = call_openrouter(prompt, key, model)

    elif provider == "ollama":
        raw = call_ollama(prompt, model)

    else:
        raise RuntimeError(f"Provider không hợp lệ: {provider}")

    if status_cb:
        status_cb("🔍 Đang parse kết quả JSON...")

    # Extract JSON from response (model may wrap in ```json ... ```)
    json_match = re.search(r"\{[\s\S]*\}", raw)
    if not json_match:
        raise RuntimeError(f"Không tìm thấy JSON trong phản hồi:\n{raw[:500]}")
    try:
        result = json.loads(json_match.group())
    except json.JSONDecodeError as e:
        raise RuntimeError(f"JSON không hợp lệ: {e}\n\nRaw:\n{raw[:800]}") from e

    if "parts" not in result:
        raise RuntimeError(f"JSON thiếu trường 'parts': {list(result.keys())}")

    result = _enforce_part_script_budgets(
        result,
        clip_durations=clip_durations,
        duration_per_part_sec=duration_per_part_sec,
        output_language=output_language,
        status_cb=status_cb,
    )

    if status_cb:
        status_cb(f"✅ Đã tạo {len(result['parts'])} kịch bản Part.")

    return result


def generate_clip_titles(
    project,
    provider: str,
    model: str,
    use_bottom_title: bool = True,
    status_cb=None,
) -> dict:
    """Generate short top/bottom title overlays for each clip."""
    clips = project.clips or []
    if not clips:
        raise RuntimeError("Project chua co Parts.")

    segments = _load_transcript_segments(project)
    clip_lines = []
    for clip in clips:
        transcript = _extract_clip_transcript(
            segments, clip.start_time, clip.end_time,
            clip_id=getattr(clip, "id", ""), max_chars=600
        ) if segments else ""
        clip_lines.append(
            {
                "index": clip.index,
                "part_text": clip.part_text or f"PART {clip.index}",
                "duration": round(clip.duration, 1),
                "source": Path(getattr(clip, "source_video", "") or project.source_video).name,
                "existing_top": getattr(clip, "custom_header", "") or "",
                "existing_bottom": getattr(clip, "custom_subtitle", "") or "",
                "transcript": transcript,
            }
        )

    prompt = f"""
You are a short-form video editor writing English overlay titles.

Create 2 overlay text lines for each video Part:
- top_title: short hook/title shown at the PART text position.
- bottom_title: fixed subtitle-style line shown near the bottom when the video has no real subtitles.

Rules:
- English only.
- Titles must match the actual content of each Part.
- Prefer the transcript when available; otherwise infer carefully from the source filename and Part number.
- top_title: 3-8 words, punchy, no hashtags, no quotes.
- bottom_title: 5-14 words, clear context/hook, no hashtags, no quotes.
- Do not invent names if context is missing.
- If context is weak, keep the title generic but still relevant to the visible topic.
- Return valid JSON only.

Project: {project.name}
Use bottom_title: {use_bottom_title}

Parts:
{json.dumps(clip_lines, ensure_ascii=False, indent=2)}

Return JSON:
{{
  "parts": [
    {{"index": 1, "top_title": "...", "bottom_title": "..."}}
  ]
}}
"""
    if provider == "gemini":
        model = GEMINI_MODEL_FALLBACKS.get(model, model) or "gemini-2.5-flash-lite"

    if status_cb:
        status_cb(f"Gui AI tao title ({provider}/{model})...")

    if provider == "gemini":
        key = load_api_key("gemini")
        if not key:
            raise RuntimeError("Chua co Gemini API key.")
        raw = call_gemini(prompt, key, model)
    elif provider == "groq":
        key = load_api_key("groq")
        if not key:
            raise RuntimeError("Chua co Groq API key.")
        raw = call_groq(prompt, key, model)
    elif provider == "openrouter":
        key = load_api_key("openrouter")
        if not key:
            raise RuntimeError("Chua co OpenRouter API key.")
        raw = call_openrouter(prompt, key, model)
    elif provider == "ollama":
        raw = call_ollama(prompt, model)
    else:
        raise RuntimeError(f"Provider khong hop le: {provider}")

    json_match = re.search(r"\{[\s\S]*\}", raw)
    if not json_match:
        raise RuntimeError(f"Khong tim thay JSON trong phan hoi:\n{raw[:500]}")
    data = json.loads(json_match.group())
    if "parts" not in data:
        raise RuntimeError("JSON AI thieu truong parts.")
    return data
