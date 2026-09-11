"""AI-powered key scene extraction from long movies.

Analyses the full transcript (with timestamps) and asks AI to identify
the most important scenes, automatically creating clips that total a
user-specified duration (e.g. 6–7 minutes for a review video).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from src.core.ai_client import (
    load_api_key,
    call_gemini,
    call_groq,
    call_openrouter,
    call_ollama,
    GEMINI_MODEL_FALLBACKS,
)
from src.core.recap_analyzer import build_recap_timeline, format_timeline_for_prompt
from src.utils.logger import logger


# ── Transcript loader (with timestamps) ──────────────────────────────────────

def _load_transcript_segments(project) -> list[dict]:
    """Load transcript segments with timestamps from the project.

    Returns a list of dicts: [{start, end, text}, ...]
    """
    tf = getattr(project, "transcript_file", "")
    if not tf or not Path(tf).exists():
        return []

    try:
        with open(tf, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.warning(f"Could not read transcript JSON: {e}")
        return []

    segments = data.get("segments", [])
    if not segments:
        return []

    return [
        {
            "start": seg.get("start", 0.0),
            "end": seg.get("end", 0.0),
            "text": seg.get("text", "").strip(),
        }
        for seg in segments
        if seg.get("text", "").strip()
    ]


def _format_timestamp(seconds: float) -> str:
    """Format seconds into MM:SS for compact display."""
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"


def _format_segments_for_prompt(segments: list[dict], max_chars: int = 30_000) -> str:
    """Format transcript segments into a timestamped text block for AI.

    Groups segments into ~30s chunks to reduce token count while
    preserving enough detail for scene identification.
    """
    if not segments:
        return ""

    lines = []
    total_chars = 0
    chunk_start = segments[0]["start"]
    chunk_texts = []
    chunk_end = segments[0]["end"]

    for seg in segments:
        chunk_texts.append(seg["text"])
        chunk_end = seg["end"]

        # Group into ~30 second chunks
        if chunk_end - chunk_start >= 30.0 or seg == segments[-1]:
            line = (
                f"[{_format_timestamp(chunk_start)} - {_format_timestamp(chunk_end)}] "
                f"{' '.join(chunk_texts)}"
            )
            total_chars += len(line) + 1
            if total_chars > max_chars:
                lines.append("...[transcript quá dài, đã lược bỏ phần giữa]...")
                # Skip to last 20% of segments
                skip_to = int(len(segments) * 0.8)
                remaining = segments[skip_to:]
                for rseg in remaining:
                    rline = (
                        f"[{_format_timestamp(rseg['start'])} - "
                        f"{_format_timestamp(rseg['end'])}] {rseg['text']}"
                    )
                    lines.append(rline)
                break
            lines.append(line)
            chunk_start = chunk_end
            chunk_texts = []

    return "\n".join(lines)


# ── Prompt builder ────────────────────────────────────────────────────────────

def _build_scene_prompt(
    transcript_text: str,
    movie_name: str,
    target_minutes: float,
    max_scenes: int,
    total_duration_sec: float,
    output_language: str = "vi",
) -> str:
    """Build the AI prompt for key scene identification."""
    lang_instruction = (
        "Trả lời hoàn toàn bằng TIẾNG VIỆT."
        if output_language == "vi"
        else "Respond entirely in ENGLISH."
    )
    total_min = total_duration_sec / 60
    target_seconds = int(target_minutes * 60)
    avg_scene_sec = int(target_seconds / max_scenes)

    return f"""Bạn là chuyên gia cắt phim. Nhiệm vụ: chọn ra ĐÚNG những cảnh quan trọng nhất
từ bộ phim dài để ghép thành video tóm tắt NGẮN cho TikTok/YouTube.

{lang_instruction}

THÔNG TIN:
- Tên phim: "{movie_name}"
- Tổng thời lượng phim gốc: {total_min:.0f} phút ({total_duration_sec:.0f}s)
- ⚠️ THỜI LƯỢNG MỤC TIÊU: CHỈ {target_minutes:.0f} PHÚT ({target_seconds}s)
- ⚠️ SỐ CẢNH TỐI ĐA: {max_scenes} cảnh
- ⚠️ MỖI CẢNH: trung bình {avg_scene_sec}s (tối thiểu 15s, tối đa 45s)

=== BẢN PHIÊN ÂM (có timestamp) ===
{transcript_text}
=== KẾT THÚC PHIÊN ÂM ===

RÀNG BUỘC NGHIÊM NGẶT — BẮT BUỘC TUÂN THỦ:
1. TỔNG thời lượng tất cả cảnh PHẢI xấp xỉ {target_seconds} giây (±30s).
   VÍ DỤ: nếu mục tiêu = 420s, tổng phải nằm trong 390-450s. KHÔNG ĐƯỢC vượt quá.
2. Số cảnh: tối đa {max_scenes} cảnh. Có thể ít hơn nhưng KHÔNG nhiều hơn.
3. Mỗi cảnh dài {avg_scene_sec}s trung bình. Tối đa 45 giây cho 1 cảnh.
4. Phải RẤT CHỌN LỌC — chỉ giữ những bước ngoặt, xung đột chính, cao trào.
   Bỏ qua hết đoạn chuyển cảnh, im lặng, hội thoại phụ, lặp lại.
5. Các cảnh phải kể được câu chuyện chính khi ghép lại theo thứ tự thời gian.

CÁCH TÍNH: Hãy tính total_selected_seconds TRƯỚC khi trả về.
Nếu tổng > {target_seconds + 30}, hãy XÓA BỚT cảnh ít quan trọng nhất.

TRẢ VỀ JSON (KHÔNG thêm text nào ngoài JSON):
{{
  "movie_name": "{movie_name}",
  "summary": "Tóm tắt cốt truyện 2-3 câu",
  "total_selected_seconds": <tổng giây của tất cả cảnh, PHẢI ≈ {target_seconds}>,
  "scenes": [
    {{
      "index": 1,
      "start_time": <số giây float>,
      "end_time": <số giây float>,
      "duration": <end - start>,
      "description": "Mô tả ngắn 1 câu",
      "importance": "high hoặc medium"
    }}
  ]
}}"""


# ── Post-processing: trim scenes to fit target ───────────────────────────────

def _build_recap_timeline_prompt(
    timeline_text: str,
    timeline_stats: dict,
    movie_name: str,
    target_minutes: float,
    max_scenes: int,
    total_duration_sec: float,
    output_language: str = "vi",
) -> str:
    """Build a recap prompt from visual-shot + dialogue aligned timeline."""
    lang_instruction = (
        "Tra loi hoan toan bang TIENG VIET."
        if output_language == "vi"
        else "Respond entirely in ENGLISH."
    )
    total_min = total_duration_sec / 60
    target_seconds = int(target_minutes * 60)
    avg_scene_sec = max(1, int(target_seconds / max(1, max_scenes)))

    return f"""You are a professional movie recap editor.

{lang_instruction}

TASK:
Select the best scenes from a long movie to create a short review/recap video.
The input timeline was built by:
1. Detecting visual shot boundaries with FFmpeg.
2. Aligning every dialogue segment to the matching visual shot.
3. Merging micro-shots into story beats.

MOVIE:
- Name: "{movie_name}"
- Original duration: {total_min:.0f} minutes ({total_duration_sec:.0f}s)
- Target recap duration: {target_minutes:.1f} minutes ({target_seconds}s)
- Maximum selected scenes: {max_scenes}
- Average selected scene length: about {avg_scene_sec}s

ANALYSIS STATS:
- Visual shots detected: {timeline_stats.get('visual_scene_count', 0)}
- Story beats: {timeline_stats.get('story_beat_count', 0)}
- Transcript segments: {timeline_stats.get('transcript_segment_count', 0)}

=== ALIGNED TIMELINE ===
{timeline_text}
=== END TIMELINE ===

STRICT RULES:
1. The selected scenes must tell one complete chronological story.
2. Total duration must be close to {target_seconds}s, tolerance +/- 30s.
3. Do not exceed {max_scenes} scenes.
4. Prefer plot turns, conflict, reveals, emotional decisions, action climaxes, and ending payoffs.
5. Skip repeated dialogue, filler, transitions, silent empty shots, credits, and low-value exposition.
6. If a beat has no dialogue, treat it as visual action. Select it only when the action is essential.
7. Scene descriptions must be rewritten as recap narration: what happens, who changes, and why it matters.
8. Do not copy original dialogue verbatim.

Return valid JSON only:
{{
  "movie_name": "{movie_name}",
  "summary": "2-3 sentence story overview",
  "recap_timeline": "one concise chronological timeline sentence",
  "total_selected_seconds": <sum of all selected scene durations>,
  "scenes": [
    {{
      "index": 1,
      "start_time": <seconds float>,
      "end_time": <seconds float>,
      "duration": <end - start>,
      "description": "recap-style scene description",
      "importance": "high or medium"
    }}
  ]
}}"""


def _trim_scenes_to_target(
    scenes: list[dict],
    target_seconds: float,
    tolerance: float = 30.0,
) -> list[dict]:
    """Remove lowest-importance scenes until total fits within target ± tolerance.

    If AI returned too many scenes (e.g. 31 min instead of 7 min),
    this function trims them down by:
    1. First removing all "medium" importance scenes from the end
    2. Then removing "medium" scenes from the middle
    3. Finally trimming "high" scenes if still over budget
    """
    total = sum(s["duration"] for s in scenes)
    max_allowed = target_seconds + tolerance

    if total <= max_allowed:
        return scenes

    logger.info(
        f"Scene trimming: {total:.0f}s → target {target_seconds:.0f}s "
        f"(max {max_allowed:.0f}s). Removing excess scenes..."
    )

    # Sort by importance (medium first) then by duration (longest first)
    # to prefer removing long, less important scenes
    indexed = list(enumerate(scenes))
    indexed.sort(key=lambda x: (
        0 if x[1].get("importance") == "medium" else 1,
        -x[1]["duration"],
    ))

    removed = set()
    current_total = total

    for orig_idx, scene in indexed:
        if current_total <= max_allowed:
            break
        removed.add(orig_idx)
        current_total -= scene["duration"]

    # Rebuild list preserving original order, re-index
    result = []
    for i, scene in enumerate(scenes):
        if i not in removed:
            scene_copy = dict(scene)
            scene_copy["index"] = len(result) + 1
            result.append(scene_copy)

    logger.info(
        f"Trimmed: {len(scenes)} → {len(result)} scenes, "
        f"{total:.0f}s → {current_total:.0f}s"
    )
    return result


# ── Main analysis function ────────────────────────────────────────────────────

def analyze_key_scenes(
    project,
    provider: str,
    model: str,
    target_minutes: float = 7.0,
    max_scenes: int = 12,
    output_language: str = "vi",
    use_visual_timeline: bool = True,
    scene_threshold: float = 0.32,
    max_detected_scenes: int = 5000,
    status_cb=None,
) -> dict:
    """Analyze transcript and identify key scenes using AI.

    Returns dict with keys: summary, scenes (list of scene dicts).
    Raises RuntimeError on failure.
    """
    if status_cb:
        status_cb("📖 Đang đọc transcript với timestamps...")

    segments = _load_transcript_segments(project)
    if not segments:
        raise RuntimeError(
            "Không tìm thấy transcript với timestamps.\n\n"
            "Hãy chạy phiên âm (Tab 3) trước.\n"
            "Đảm bảo chọn 'Word timestamps' khi phiên âm."
        )

    total_duration = segments[-1]["end"] if segments else 0
    if total_duration < 60:
        raise RuntimeError(
            f"Video quá ngắn ({total_duration:.0f}s). "
            "Tính năng này dành cho phim dài (30+ phút)."
        )

    if status_cb:
        status_cb(
            f"📊 {len(segments)} đoạn phiên âm | "
            f"Phim: {total_duration/60:.0f} phút | "
            f"Mục tiêu: ~{target_minutes:.0f} phút ({max_scenes} cảnh tối đa)"
        )

    timeline = None
    if use_visual_timeline:
        try:
            if status_cb:
                status_cb("Dang phat hien phan canh hinh anh va map loi thoai...")
            timeline = build_recap_timeline(
                project,
                threshold=scene_threshold,
                max_detected_scenes=max_detected_scenes,
                status_cb=status_cb,
            )
        except Exception as exc:
            logger.warning(f"Visual recap timeline failed, falling back to transcript: {exc}")
            if status_cb:
                status_cb(f"Khong tao duoc visual timeline, fallback transcript: {exc}")

    # Format segments for prompt
    transcript_text = _format_segments_for_prompt(segments)
    if status_cb:
        status_cb(f"📝 Đã format {len(transcript_text):,} ký tự transcript")

    # Build prompt
    prompt = _build_scene_prompt(
        transcript_text=transcript_text,
        movie_name=project.name,
        target_minutes=target_minutes,
        max_scenes=max_scenes,
        total_duration_sec=total_duration,
        output_language=output_language,
    )
    if timeline:
        transcript_text = format_timeline_for_prompt(timeline)
        if status_cb:
            status_cb(
                f"Formatted recap timeline: {len(transcript_text):,} chars "
                f"from {timeline.get('story_beat_count', 0):,} story beats"
            )
        prompt = _build_recap_timeline_prompt(
            timeline_text=transcript_text,
            timeline_stats=timeline,
            movie_name=project.name,
            target_minutes=target_minutes,
            max_scenes=max_scenes,
            total_duration_sec=timeline.get("duration") or total_duration,
            output_language=output_language,
        )

    if provider == "gemini":
        model = GEMINI_MODEL_FALLBACKS.get(model, model) or "gemini-2.5-flash-lite"

    if status_cb:
        status_cb(f"🤖 Gửi yêu cầu đến {provider} ({model})...")

    # ── Call AI provider ──────────────────────────────────────────────────
    raw = _call_provider(provider, model, prompt, status_cb)

    if status_cb:
        status_cb("🔍 Đang phân tích kết quả...")

    # ── Parse JSON response ──────────────────────────────────────────────
    json_match = re.search(r"\{[\s\S]*\}", raw)
    if not json_match:
        raise RuntimeError(f"Không tìm thấy JSON trong phản hồi:\n{raw[:500]}")

    try:
        result = json.loads(json_match.group())
    except json.JSONDecodeError as e:
        raise RuntimeError(f"JSON không hợp lệ: {e}\n\nRaw:\n{raw[:800]}") from e

    if "scenes" not in result:
        raise RuntimeError(f"JSON thiếu trường 'scenes': {list(result.keys())}")

    # ── Validate and clean scenes ────────────────────────────────────────
    scenes = result["scenes"]
    valid_scenes = []
    total_selected = 0

    for scene in scenes:
        start = float(scene.get("start_time", 0))
        end = float(scene.get("end_time", 0))
        if end <= start:
            continue
        if start < 0:
            start = 0
        if end > total_duration:
            end = total_duration
        duration = end - start
        if duration < 3:  # skip very short scenes
            continue

        valid_scenes.append({
            "index": len(valid_scenes) + 1,
            "start_time": round(start, 1),
            "end_time": round(end, 1),
            "duration": round(duration, 1),
            "description": scene.get("description", f"Cảnh {len(valid_scenes)+1}"),
            "importance": scene.get("importance", "medium"),
            "enabled": True,
        })
        total_selected += duration

    if status_cb:
        status_cb(
            f"📋 AI trả về {len(valid_scenes)} cảnh, "
            f"tổng {total_selected/60:.1f} phút"
        )

    # ── Post-process: trim if AI exceeded target ─────────────────────────
    target_seconds = target_minutes * 60
    if total_selected > target_seconds + 30:
        if status_cb:
            status_cb(
                f"✂️ Tổng {total_selected/60:.1f} phút > mục tiêu "
                f"{target_minutes:.0f} phút. Đang cắt bớt cảnh phụ..."
            )
        valid_scenes = _trim_scenes_to_target(
            valid_scenes, target_seconds, tolerance=30.0,
        )
        total_selected = sum(s["duration"] for s in valid_scenes)

    result["scenes"] = valid_scenes
    result["total_selected_seconds"] = round(total_selected, 1)

    if status_cb:
        status_cb(
            f"✅ Kết quả: {len(valid_scenes)} cảnh | "
            f"Tổng: {total_selected/60:.1f} phút"
        )

    return result


def _call_provider(provider: str, model: str, prompt: str, status_cb=None) -> str:
    """Call the chosen AI provider and return raw text response."""
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
        last_error = None
        for candidate in dict.fromkeys(candidates):
            if not candidate:
                continue
            if status_cb and candidate != model:
                status_cb(f"⚠️ Thử model dự phòng: {candidate}...")
            try:
                return call_gemini(prompt, key, candidate)
            except RuntimeError as exc:
                last_error = exc
                if "HTTP 503" not in str(exc) and "UNAVAILABLE" not in str(exc):
                    raise
        raise RuntimeError(
            f"Gemini không khả dụng. Lỗi cuối: {last_error}"
        )

    elif provider == "groq":
        key = load_api_key("groq")
        if not key:
            raise RuntimeError("Chưa có Groq API key. Vào ⚙ Cài đặt → API Keys.")
        return call_groq(prompt, key, model)

    elif provider == "openrouter":
        key = load_api_key("openrouter")
        if not key:
            raise RuntimeError("Chưa có OpenRouter API key.")
        return call_openrouter(prompt, key, model)

    elif provider == "ollama":
        return call_ollama(prompt, model)

    else:
        raise RuntimeError(f"Provider không hợp lệ: {provider}")
