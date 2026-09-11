"""Recap-oriented movie analysis helpers.

This module builds a richer intermediate timeline than the older
transcript-only scene picker:

1. Detect visual shot boundaries with FFmpeg scene scoring.
2. Align Whisper transcript segments to each visual shot.
3. Merge tiny shots into recap-friendly story beats.
4. Format the timeline for text AI selection and recap writing.

The visual understanding is intentionally conservative: without a
multimodal model we do not claim to identify faces or emotions from pixels.
We do preserve keyframe timestamps and visual boundary metadata so the same
timeline can be extended later with frame captions/vision labels.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Callable

from src.core.dependency_manager import ffmpeg_path
from src.utils.file_utils import ensure_dir
from src.utils.logger import logger


DEFAULT_SCENE_THRESHOLD = 0.32
DEFAULT_MIN_SCENE_SEC = 1.0
DEFAULT_BEAT_SEC = 8.0


def load_transcript_segments(project) -> list[dict]:
    """Load Whisper transcript segments with timestamps."""
    tf = getattr(project, "transcript_file", "")
    if not tf or not Path(tf).exists():
        return []
    try:
        with open(tf, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        logger.warning(f"Could not read transcript JSON: {exc}")
        return []

    segments = []
    for seg in data.get("segments", []):
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        segments.append({
            "start": float(seg.get("start", 0.0) or 0.0),
            "end": float(seg.get("end", 0.0) or 0.0),
            "text": text,
        })
    segments.sort(key=lambda item: item["start"])
    return segments


def detect_visual_scenes(
    video_path: str,
    duration_sec: float,
    threshold: float = DEFAULT_SCENE_THRESHOLD,
    min_scene_sec: float = DEFAULT_MIN_SCENE_SEC,
    max_scenes: int = 5000,
    status_cb: Callable[[str], None] | None = None,
) -> list[dict]:
    """Detect visual scene cuts using FFmpeg's `scene` score.

    Returns normalized shots: [{index, start, end, duration, keyframe_time}, ...].
    FFmpeg emits one selected frame per boundary; we convert those boundary
    timestamps into continuous ranges covering the full video.
    """
    ff = ffmpeg_path()
    if not ff:
        raise RuntimeError("FFmpeg chua san sang. Hay tai FFmpeg trong Settings.")
    if not video_path or not Path(video_path).exists():
        raise RuntimeError("Khong tim thay video goc de phan tich canh.")
    if duration_sec <= 0:
        duration_sec = _probe_duration(video_path, ff)
    if duration_sec <= 0:
        raise RuntimeError("Khong doc duoc thoi luong video.")

    threshold = max(0.05, min(0.95, float(threshold)))
    min_scene_sec = max(0.2, float(min_scene_sec))

    if status_cb:
        status_cb(
            f"Detecting visual cuts with FFmpeg scene score "
            f"(threshold={threshold:.2f})..."
        )

    # showinfo writes pts_time to stderr. We avoid output files by using null muxer.
    vf = f"select='gt(scene,{threshold})',showinfo"
    cmd = [ff, "-hide_banner", "-i", video_path, "-vf", vf, "-an", "-f", "null", "-"]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=None,
        )
    except Exception as exc:
        raise RuntimeError(f"FFmpeg scene detection failed: {exc}") from exc

    if proc.returncode != 0:
        logger.warning(proc.stderr[-2000:])

    cut_times = []
    for match in re.finditer(r"pts_time:([0-9]+(?:\.[0-9]+)?)", proc.stderr):
        t = float(match.group(1))
        if 0 < t < duration_sec:
            cut_times.append(round(t, 3))

    cut_times = _dedupe_sorted(cut_times, min_gap=min_scene_sec)
    if len(cut_times) > max_scenes:
        cut_times = cut_times[:max_scenes]

    boundaries = [0.0] + cut_times + [float(duration_sec)]
    scenes = []
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        if end - start < min_scene_sec and scenes:
            scenes[-1]["end"] = end
            scenes[-1]["duration"] = round(scenes[-1]["end"] - scenes[-1]["start"], 3)
            continue
        scenes.append({
            "index": len(scenes) + 1,
            "start": round(start, 3),
            "end": round(end, 3),
            "duration": round(end - start, 3),
            "keyframe_time": round(start + min(0.5, max(0.0, (end - start) / 2)), 3),
        })

    if status_cb:
        status_cb(f"Detected {len(scenes):,} visual shots.")
    return scenes


def align_dialogue_to_scenes(scenes: list[dict], transcript_segments: list[dict]) -> list[dict]:
    """Attach overlapping transcript text to each visual shot."""
    if not scenes:
        return []
    enriched = []
    seg_idx = 0
    for scene in scenes:
        start = scene["start"]
        end = scene["end"]
        while seg_idx < len(transcript_segments) and transcript_segments[seg_idx]["end"] <= start:
            seg_idx += 1

        texts = []
        probe = seg_idx
        while probe < len(transcript_segments):
            seg = transcript_segments[probe]
            if seg["start"] >= end:
                break
            if seg["end"] > start:
                texts.append(seg["text"])
            probe += 1

        item = dict(scene)
        item["dialogue"] = " ".join(texts).strip()
        item["has_dialogue"] = bool(item["dialogue"])
        enriched.append(item)
    return enriched


def build_story_beats(
    scenes: list[dict],
    min_beat_sec: float = DEFAULT_BEAT_SEC,
    max_beat_sec: float = 35.0,
) -> list[dict]:
    """Merge micro-shots into recap-friendly timeline beats."""
    if not scenes:
        return []
    beats = []
    current = None
    for scene in scenes:
        if current is None:
            current = _new_beat(scene)
            continue

        current_dur = current["end"] - current["start"]
        should_merge = current_dur < min_beat_sec or (
            current_dur < max_beat_sec and not current.get("dialogue") and not scene.get("dialogue")
        )
        if should_merge:
            _merge_scene_into_beat(current, scene)
        else:
            beats.append(_finalize_beat(current, len(beats) + 1))
            current = _new_beat(scene)

    if current is not None:
        beats.append(_finalize_beat(current, len(beats) + 1))
    return beats


def build_recap_timeline(
    project,
    threshold: float = DEFAULT_SCENE_THRESHOLD,
    min_scene_sec: float = DEFAULT_MIN_SCENE_SEC,
    min_beat_sec: float = DEFAULT_BEAT_SEC,
    max_detected_scenes: int = 5000,
    status_cb: Callable[[str], None] | None = None,
) -> dict:
    """Build and persist the recap analysis timeline for a project."""
    duration = float(project.video_metadata.get("duration", 0) or 0)
    transcript_segments = load_transcript_segments(project)
    if status_cb:
        status_cb(f"Loaded {len(transcript_segments):,} transcript segments.")

    scenes = detect_visual_scenes(
        project.source_video,
        duration,
        threshold=threshold,
        min_scene_sec=min_scene_sec,
        max_scenes=max_detected_scenes,
        status_cb=status_cb,
    )
    scenes = align_dialogue_to_scenes(scenes, transcript_segments)
    beats = build_story_beats(scenes, min_beat_sec=min_beat_sec)

    data = {
        "version": 1,
        "movie_name": project.name,
        "source_video": project.source_video,
        "duration": duration or (scenes[-1]["end"] if scenes else 0),
        "scene_threshold": threshold,
        "visual_scene_count": len(scenes),
        "story_beat_count": len(beats),
        "transcript_segment_count": len(transcript_segments),
        "scenes": scenes,
        "beats": beats,
    }

    out_dir = Path(project.output_dir or "output") / "analysis"
    ensure_dir(str(out_dir))
    out_path = out_dir / "recap_timeline.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    data["timeline_file"] = str(out_path)

    if status_cb:
        status_cb(
            f"Timeline ready: {len(scenes):,} shots -> {len(beats):,} story beats "
            f"({out_path})"
        )
    return data


def format_timeline_for_prompt(timeline: dict, max_chars: int = 45_000) -> str:
    """Format story beats for an LLM prompt while preserving timestamps."""
    beats = timeline.get("beats") or []
    lines = []
    total = 0
    for beat in beats:
        dialogue = (beat.get("dialogue") or "").strip()
        if len(dialogue) > 600:
            dialogue = dialogue[:580].rstrip() + "..."
        line = (
            f"[{_fmt_ts(beat['start'])} - {_fmt_ts(beat['end'])}] "
            f"{beat['duration']:.1f}s | shots {beat.get('scene_start_index')}.."
            f"{beat.get('scene_end_index')} | "
            f"{dialogue if dialogue else '(no dialogue / visual action)'}"
        )
        if total + len(line) > max_chars:
            lines.append(
                f"... timeline truncated: {len(beats) - beat['index'] + 1} later beats omitted ..."
            )
            break
        lines.append(line)
        total += len(line) + 1
    return "\n".join(lines)


def _probe_duration(video_path: str, ff: str) -> float:
    ffprobe = ff.replace("ffmpeg", "ffprobe") if "ffmpeg" in ff.lower() else "ffprobe"
    cmd = [
        ffprobe, "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return float((proc.stdout or "0").strip() or 0)
    except Exception:
        return 0.0


def _dedupe_sorted(values: list[float], min_gap: float) -> list[float]:
    result = []
    last = None
    for value in sorted(set(values)):
        if last is None or value - last >= min_gap:
            result.append(value)
            last = value
    return result


def _new_beat(scene: dict) -> dict:
    return {
        "start": scene["start"],
        "end": scene["end"],
        "scene_start_index": scene["index"],
        "scene_end_index": scene["index"],
        "dialogue_parts": [scene.get("dialogue", "")] if scene.get("dialogue") else [],
        "has_dialogue": bool(scene.get("dialogue")),
    }


def _merge_scene_into_beat(beat: dict, scene: dict) -> None:
    beat["end"] = scene["end"]
    beat["scene_end_index"] = scene["index"]
    if scene.get("dialogue"):
        beat["dialogue_parts"].append(scene["dialogue"])
        beat["has_dialogue"] = True


def _finalize_beat(beat: dict, index: int) -> dict:
    dialogue = " ".join(part for part in beat.pop("dialogue_parts", []) if part).strip()
    beat["index"] = index
    beat["duration"] = round(beat["end"] - beat["start"], 3)
    beat["dialogue"] = dialogue
    return beat


def _fmt_ts(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"
