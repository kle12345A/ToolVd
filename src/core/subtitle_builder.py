"""
Build SRT (plain) and ASS (karaoke) subtitle files from transcript.
Supports both full-video and clip-specific (time-shifted) subtitles.
"""

import re
from pathlib import Path
from typing import Optional

from src.utils.file_utils import ensure_dir
from src.utils.logger import logger


# ─── Time format helpers ──────────────────────────────────────────────────────

def _srt_time(seconds: float) -> str:
    """00:01:23,456"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _ass_time(seconds: float) -> str:
    """0:01:23.45"""
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int(round((seconds - int(seconds)) * 100))
    if cs >= 100:
        s += 1
        cs = 0
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


# ─── Word grouping ────────────────────────────────────────────────────────────

def _group_words_into_lines(
    words: list[dict],
    max_words: int = 6,
    max_chars: int = 40,
    gap_threshold: float = 0.8,
) -> list[list[dict]]:
    """Group word-level timestamps into subtitle lines."""
    if not words:
        return []

    lines: list[list[dict]] = []
    current: list[dict] = []

    for i, w in enumerate(words):
        if current:
            gap = w["start"] - current[-1]["end"]
            chars = sum(len(x["word"]) for x in current) + len(w["word"])
            if (
                gap > gap_threshold
                or len(current) >= max_words
                or chars > max_chars
            ):
                lines.append(current)
                current = []
        current.append(w)

    if current:
        lines.append(current)
    return lines


def _group_segments_into_lines(
    segments: list[dict],
    max_chars: int = 45,
) -> list[dict]:
    """
    For plain SRT: use segment text directly, splitting long lines.
    Returns list of {start, end, text} dicts.
    """
    result = []
    for seg in segments:
        text = seg["text"].strip()
        if not text:
            continue
        # Split long segments into smaller chunks
        words = text.split()
        chunk_words: list[str] = []
        seg_duration = seg["end"] - seg["start"]
        total_words = len(words)
        chunk_start = seg["start"]

        for i, w in enumerate(words):
            chunk_words.append(w)
            chunk_text = " ".join(chunk_words)
            is_last = (i == len(words) - 1)
            if len(chunk_text) >= max_chars or is_last:
                # Estimate end time proportionally
                chunk_end = seg["start"] + seg_duration * (i + 1) / total_words
                result.append({
                    "start": round(chunk_start, 3),
                    "end": round(chunk_end, 3),
                    "text": chunk_text,
                })
                chunk_start = chunk_end
                chunk_words = []
    return result


# ─── SRT builder ─────────────────────────────────────────────────────────────

def build_srt(
    transcript: dict,
    output_path: str,
    time_offset: float = 0.0,
) -> str:
    """
    Build a plain SRT file from transcript segments.
    time_offset: subtract this from all timestamps (for clip-specific subs).
    """
    segments = transcript.get("segments", [])
    lines = _group_segments_into_lines(segments)

    srt_lines = []
    idx = 1
    for line in lines:
        start = line["start"] - time_offset
        end = line["end"] - time_offset
        if end <= 0:
            continue
        start = max(0.0, start)
        srt_lines.append(str(idx))
        srt_lines.append(f"{_srt_time(start)} --> {_srt_time(end)}")
        srt_lines.append(line["text"])
        srt_lines.append("")
        idx += 1

    content = "\n".join(srt_lines)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    logger.info(f"SRT saved: {output_path} ({idx - 1} lines)")
    return output_path


# ─── ASS karaoke builder ──────────────────────────────────────────────────────

_ASS_COLORS = {
    "white":  "&H00FFFFFF",
    "yellow": "&H0000FFFF",
    "cyan":   "&H00FFFF00",
    "green":  "&H0000FF00",
    "red":    "&H000000FF",
    "orange": "&H000080FF",
    "pink":   "&H00FF00FF",
    "black":  "&H00000000",
}

_ASS_HEADER = """\
[Script Info]
ScriptType: v4.00+
PlayResX: {w}
PlayResY: {h}
Timer: 100.0000
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Plain,Arial,{fontsize},{primary},&H00000000,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,4,1,2,50,50,{margin_v},1
Style: Karaoke,Arial,{fontsize},{secondary},{primary},&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,4,1,2,50,50,{margin_v},1
Style: Word,Arial,{fontsize},{primary},&H00000000,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,4,1,2,50,50,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def _color(name: str) -> str:
    return _ASS_COLORS.get(name.lower(), "&H00FFFFFF")


def _escape_ass_text(text: str) -> str:
    return (
        text.replace("\\", r"\\")
            .replace("{", r"\{")
            .replace("}", r"\}")
            .replace("\n", r"\N")
    )


def build_ass_karaoke(
    transcript: dict,
    output_path: str,
    time_offset: float = 0.0,
    width: int = 1080,
    height: int = 1920,
    fontsize: int = 65,
    text_color: str = "white",
    highlight_color: str = "yellow",
    position: str = "bottom",
    max_words_per_line: int = 6,
) -> str:
    """
    Build an ASS subtitle file with karaoke word-by-word highlight.
    Primary = highlight color, Secondary = normal (pre-highlight) color.
    """
    segments = transcript.get("segments", [])

    primary = _color(highlight_color)
    secondary = _color(text_color)
    margin_v = 80 if position == "bottom" else height - 200

    header = _ASS_HEADER.format(
        w=width, h=height,
        fontsize=fontsize,
        primary=primary,
        secondary=secondary,
        margin_v=margin_v,
    )

    dialogue_lines = []

    for seg in segments:
        words = seg.get("words", [])
        if not words:
            # Fallback: no word timestamps — treat segment as single line
            start = max(0.0, seg["start"] - time_offset)
            end = max(0.0, seg["end"] - time_offset)
            if end <= 0:
                continue
            text = seg["text"].strip()
            duration_cs = max(1, int((seg["end"] - seg["start"]) * 100))
            dialogue_lines.append(
                f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},"
                f"Karaoke,,0,0,0,,{{\\kf{duration_cs}}}{text}"
            )
            continue

        # Group words into lines
        line_groups = _group_words_into_lines(
            words, max_words=max_words_per_line
        )
        for group in line_groups:
            if not group:
                continue
            line_start = group[0]["start"] - time_offset
            line_end = group[-1]["end"] - time_offset
            if line_end <= 0:
                continue
            line_start = max(0.0, line_start)

            tagged_words = []
            for w in group:
                dur_cs = max(1, int(round((w["end"] - w["start"]) * 100)))
                word_text = w["word"].strip()
                if not word_text:
                    continue
                tagged_words.append(f"{{\\kf{dur_cs}}}{word_text}")

            if not tagged_words:
                continue

            text = " ".join(tagged_words)
            dialogue_lines.append(
                f"Dialogue: 0,{_ass_time(line_start)},{_ass_time(line_end)},"
                f"Karaoke,,0,0,0,,{text}"
            )

    content = header + "\n".join(dialogue_lines)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    logger.info(f"ASS saved: {output_path} ({len(dialogue_lines)} lines)")
    return output_path


def _words_from_segment(seg: dict) -> list[dict]:
    """Return word timings; estimate evenly if the transcript lacks word data."""
    words = [
        {
            "word": (w.get("word") or "").strip(),
            "start": float(w.get("start", seg.get("start", 0.0))),
            "end": float(w.get("end", seg.get("end", 0.0))),
        }
        for w in seg.get("words", [])
        if (w.get("word") or "").strip()
    ]
    if words:
        return words

    text_words = (seg.get("text") or "").strip().split()
    if not text_words:
        return []
    start = float(seg.get("start", 0.0))
    end = float(seg.get("end", start))
    duration = max(0.1, end - start)
    step = duration / max(1, len(text_words))
    return [
        {
            "word": word,
            "start": start + i * step,
            "end": start + (i + 1) * step,
        }
        for i, word in enumerate(text_words)
    ]


def build_ass_word_by_word(
    transcript: dict,
    output_path: str,
    time_offset: float = 0.0,
    width: int = 1080,
    height: int = 1920,
    fontsize: int = 65,
    text_color: str = "white",
    highlight_color: str = "yellow",
    position: str = "bottom",
) -> str:
    """
    Build ASS subtitles where only one word is visible at a time.
    Example: "have many" appears as "have", then "many".
    """
    segments = transcript.get("segments", [])

    primary = _color(text_color)
    secondary = _color(highlight_color)
    margin_v = 80 if position == "bottom" else height - 200

    header = _ASS_HEADER.format(
        w=width, h=height,
        fontsize=fontsize,
        primary=primary,
        secondary=secondary,
        margin_v=margin_v,
    )

    dialogue_lines = []
    for seg in segments:
        for word in _words_from_segment(seg):
            text = (word.get("word") or "").strip()
            if not text:
                continue
            start = max(0.0, float(word["start"]) - time_offset)
            end = max(0.0, float(word["end"]) - time_offset)
            if end <= 0:
                continue
            if end <= start:
                end = start + 0.15
            dialogue_lines.append(
                f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},"
                f"Word,,0,0,0,,{_escape_ass_text(text)}"
            )

    content = header + "\n".join(dialogue_lines)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    logger.info(f"Word-by-word ASS saved: {output_path} ({len(dialogue_lines)} words)")
    return output_path


# ─── Full-video subtitle generation ──────────────────────────────────────────

def _clamp_transcript_to_duration(
    transcript: dict,
    duration: Optional[float],
) -> dict:
    """Return a copy whose segment/word timestamps stay inside the video."""
    max_duration = None
    if duration is not None:
        try:
            max_duration = max(0.0, float(duration))
        except (TypeError, ValueError):
            max_duration = None

    clamped_segments: list[dict] = []
    for original in transcript.get("segments", []):
        try:
            start = max(0.0, float(original.get("start", 0.0)))
            end = max(0.0, float(original.get("end", start)))
        except (TypeError, ValueError):
            continue

        if max_duration is not None:
            start = min(start, max_duration)
            end = min(end, max_duration)
        if end <= start:
            continue

        segment = dict(original)
        segment["start"] = round(start, 3)
        segment["end"] = round(end, 3)

        clamped_words: list[dict] = []
        for original_word in original.get("words", []):
            try:
                word_start = max(
                    start,
                    float(original_word.get("start", start)),
                )
                word_end = min(
                    end,
                    float(original_word.get("end", end)),
                )
            except (TypeError, ValueError):
                continue
            if max_duration is not None:
                word_start = min(word_start, max_duration)
                word_end = min(word_end, max_duration)
            if word_end <= word_start:
                continue
            word = dict(original_word)
            word["start"] = round(word_start, 3)
            word["end"] = round(word_end, 3)
            clamped_words.append(word)

        segment["words"] = clamped_words
        clamped_segments.append(segment)

    result = dict(transcript)
    result["segments"] = clamped_segments
    if max_duration is not None:
        result["duration"] = max_duration
    return result


def build_subtitle_for_full_video(
    transcript: dict,
    output_dir: str,
    style: str = "plain",
    width: int = 1080,
    height: int = 1920,
    fontsize: int = 65,
    text_color: str = "white",
    highlight_color: str = "yellow",
    position: str = "bottom",
    duration: Optional[float] = None,
    filename_stem: str = "review_subtitle",
) -> Optional[str]:
    """Build one subtitle file for an entire rendered review timeline.

    Unlike :func:`build_subtitle_for_clip`, timestamps remain relative to the
    full video and the filename contains no PART number.  When ``duration`` is
    supplied, every segment and word is clamped to ``0..duration``.
    """
    ensure_dir(output_dir)
    full_transcript = _clamp_transcript_to_duration(transcript, duration)
    if not full_transcript.get("segments"):
        return None

    safe_stem = (filename_stem or "review_subtitle").strip() or "review_subtitle"
    if style == "karaoke":
        out_path = str(Path(output_dir) / f"{safe_stem}.ass")
        return build_ass_karaoke(
            full_transcript,
            out_path,
            width=width,
            height=height,
            fontsize=fontsize,
            text_color=text_color,
            highlight_color=highlight_color,
            position=position,
        )
    if style == "word":
        out_path = str(Path(output_dir) / f"{safe_stem}.ass")
        return build_ass_word_by_word(
            full_transcript,
            out_path,
            width=width,
            height=height,
            fontsize=fontsize,
            text_color=text_color,
            highlight_color=highlight_color,
            position=position,
        )

    out_path = str(Path(output_dir) / f"{safe_stem}.srt")
    return build_srt(full_transcript, out_path)


# ─── Clip-specific subtitle generation ───────────────────────────────────────

def build_subtitle_for_clip(
    transcript: dict,
    clip_start: float,
    clip_end: float,
    output_dir: str,
    clip_index: int,
    style: str = "plain",
    width: int = 1080,
    height: int = 1920,
    fontsize: int = 65,
    text_color: str = "white",
    highlight_color: str = "yellow",
    position: str = "bottom",
) -> Optional[str]:
    """
    Build subtitle file for a specific clip, with times shifted to clip-local.
    Returns path to generated subtitle file, or None on failure.
    """
    ensure_dir(output_dir)

    # Filter transcript segments that overlap with this clip
    all_segs = transcript.get("segments", [])
    clip_segs = []
    for seg in all_segs:
        if seg["end"] <= clip_start or seg["start"] >= clip_end:
            continue
        # Clamp to clip bounds
        filtered_words = [
            w for w in seg.get("words", [])
            if w["end"] > clip_start and w["start"] < clip_end
        ]
        clipped = dict(seg)
        clipped["words"] = filtered_words
        clipped["start"] = max(seg["start"], clip_start)
        clipped["end"] = min(seg["end"], clip_end)
        clip_segs.append(clipped)

    if not clip_segs:
        return None

    clip_transcript = dict(transcript)
    clip_transcript["segments"] = clip_segs

    if style == "karaoke":
        out_path = str(Path(output_dir) / f"sub_karaoke_part{clip_index:02d}.ass")
        return build_ass_karaoke(
            clip_transcript, out_path,
            time_offset=clip_start,
            width=width, height=height,
            fontsize=fontsize,
            text_color=text_color,
            highlight_color=highlight_color,
            position=position,
        )
    if style == "word":
        out_path = str(Path(output_dir) / f"sub_word_part{clip_index:02d}.ass")
        return build_ass_word_by_word(
            clip_transcript, out_path,
            time_offset=clip_start,
            width=width, height=height,
            fontsize=fontsize,
            text_color=text_color,
            highlight_color=highlight_color,
            position=position,
        )
    else:
        out_path = str(Path(output_dir) / f"sub_plain_part{clip_index:02d}.srt")
        return build_srt(clip_transcript, out_path, time_offset=clip_start)


# ─── Shift existing SRT ───────────────────────────────────────────────────────

def shift_srt(srt_path: str, offset_seconds: float, out_path: str) -> str:
    """Shift all timestamps in an SRT file by -offset_seconds."""
    time_re = re.compile(
        r"(\d{2}):(\d{2}):(\d{2}),(\d{3}) --> (\d{2}):(\d{2}):(\d{2}),(\d{3})"
    )

    def shift_match(m: re.Match) -> str:
        def to_s(h, m_, s, ms):
            return int(h)*3600 + int(m_)*60 + int(s) + int(ms)/1000
        s1 = max(0.0, to_s(*m.groups()[:4]) - offset_seconds)
        s2 = max(0.0, to_s(*m.groups()[4:]) - offset_seconds)
        return f"{_srt_time(s1)} --> {_srt_time(s2)}"

    with open(srt_path, "r", encoding="utf-8") as f:
        content = f.read()
    shifted = time_re.sub(shift_match, content)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(shifted)
    return out_path
