"""Pure workflow-state helpers for the review-video pipeline.

This module deliberately has no PyQt or media-processing dependencies.  It
only evaluates persisted project state, which makes it safe to use from the
UI, project migration code, and tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


PRIMARY_STEPS = (
    "source",
    "voice",
    "narration_subtitle",
    "export",
)


_PREREQUISITE_REASON = {
    "voice": "Cần chọn video và tạo project trước.",
    "narration_subtitle": "Cần video review đã ghép giọng đọc trước.",
    "export": "Cần hoàn tất subtitle của giọng đọc trước khi xuất.",
}


def _value(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _nonempty_path(value: Any) -> str:
    try:
        return str(value).strip() if value is not None else ""
    except (TypeError, ValueError):
        return ""


def _file_exists(value: Any) -> bool:
    path = _nonempty_path(value)
    return bool(path and Path(path).is_file())


def _enabled_clips(project: Any) -> list[Any]:
    clips = _value(project, "clips", None) or []
    return [clip for clip in clips if bool(_value(clip, "enabled", True))]


def _valid_scene(clip: Any, project: Any) -> bool:
    try:
        start = float(_value(clip, "start_time", 0.0))
        end = float(_value(clip, "end_time", 0.0))
    except (TypeError, ValueError):
        return False
    if start < 0 or end <= start:
        return False

    # A non-empty per-clip source is authoritative.  Falling back only when it
    # is empty prevents a broken/moved clip source from silently using another
    # project's main video.
    source = _nonempty_path(_value(clip, "source_video", ""))
    if not source:
        source = _nonempty_path(_value(project, "source_video", ""))
    return _file_exists(source)


def _source_complete(project: Any) -> bool:
    # Prefer explicit immutable-source fields when a migrated project has
    # them, while still accepting the legacy source_video-only model.
    for name in ("raw_source_video", "original_source_video", "source_video"):
        if _file_exists(_value(project, name, "")):
            return True
    return False


def _source_transcript_complete(project: Any) -> bool:
    if _file_exists(_value(project, "source_transcript_file", "")):
        return True

    # In old projects transcript_file meant the source transcript.  Once a
    # narration transcript field exists, however, transcript_file may point at
    # the newly voiced master and must never unlock source analysis.
    if _file_exists(_value(project, "narration_transcript_file", "")):
        return False
    return _file_exists(_value(project, "transcript_file", ""))


def _script_complete(project: Any, clips: list[Any]) -> bool:
    if not clips:
        return False
    scene_revision = _value(project, "scene_revision", None)
    script_revision = _value(project, "script_scene_revision", None)
    if (
        scene_revision is not None
        and script_revision is not None
        and int(script_revision) >= 0
        and int(script_revision) != int(scene_revision)
    ):
        return False
    if str(_value(project, "voiceover_script", "") or "").strip():
        return True
    return all(
        bool(str(_value(clip, "voiceover_script", "") or "").strip())
        for clip in clips
    )


def _positive_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result > 0 else None


def _metadata_duration(metadata: Any, keys: tuple[str, ...]) -> float | None:
    if metadata is None:
        return None
    for key in keys:
        result = _positive_float(_value(metadata, key, None))
        if result is not None:
            return result
    return None


def _voice_duration_pair(project: Any) -> tuple[float | None, float | None]:
    """Return ``(audio_duration, video_duration)`` when metadata provides it.

    Several names are accepted intentionally: workflow.py is also the
    migration boundary for projects created before duration fields were
    standardized.
    """

    direct_pairs = (
        ("review_voice_duration", "review_video_duration"),
        ("voiceover_audio_duration", "review_voice_video_duration"),
        ("narration_audio_duration", "review_video_duration"),
        ("voiceover_duration", "review_video_duration"),
        ("voice_duration", "review_duration"),
    )
    for audio_name, video_name in direct_pairs:
        audio = _positive_float(_value(project, audio_name, None))
        video = _positive_float(_value(project, video_name, None))
        if audio is not None or video is not None:
            return audio, video

    metadata = _value(project, "review_voice_metadata", None)
    audio = _metadata_duration(
        metadata,
        ("audio_duration", "voice_duration", "narration_duration"),
    )
    video = _metadata_duration(
        metadata,
        ("video_duration", "duration"),
    )
    if audio is not None or video is not None:
        return audio, video

    voice_metadata = _value(project, "voice_metadata", None)
    review_metadata = _value(project, "review_video_metadata", None)
    audio = _metadata_duration(
        voice_metadata,
        ("duration", "audio_duration", "voice_duration"),
    )
    video = _metadata_duration(
        review_metadata,
        ("duration", "video_duration"),
    )
    return audio, video


def _voice_complete(project: Any) -> bool:
    if not _file_exists(_value(project, "review_voice_video", "")):
        return False
    scene_revision = _value(project, "scene_revision", None)
    voice_revision = _value(project, "voice_scene_revision", None)
    if (
        scene_revision is not None
        and voice_revision is not None
        and int(voice_revision) >= 0
        and int(voice_revision) != int(scene_revision)
    ):
        return False

    audio_duration, video_duration = _voice_duration_pair(project)
    # Duration validation is optional for legacy projects.  When both values
    # are available, enforce a tight A/V synchronization tolerance.
    if audio_duration is None or video_duration is None:
        return True
    if _value(project, "workflow_mode", "") == "dubbing_timed_manual":
        # The merged video already clips the permitted voice overflow.
        audio_duration = min(audio_duration, video_duration)
    tolerance = max(0.15, video_duration * 0.005)
    return abs(audio_duration - video_duration) <= tolerance


def workflow_facts(project: Any) -> dict[str, bool]:
    """Return completion facts for each primary workflow step.

    Facts describe files/data that really exist; they do not imply that a step
    is currently accessible.  ``step_access`` adds prerequisite gating.
    """

    if project is None:
        return {step: False for step in PRIMARY_STEPS}

    subtitle_complete = _file_exists(
        _value(project, "narration_subtitle_file", "")
    )
    subtitle_revision = _value(project, "subtitle_voice_revision", None)
    voice_revision = _value(project, "voice_scene_revision", None)
    if (
        subtitle_complete
        and subtitle_revision is not None
        and voice_revision is not None
        and int(subtitle_revision) >= 0
        and int(voice_revision) >= 0
        and int(subtitle_revision) != int(voice_revision)
    ):
        subtitle_complete = False

    return {
        "source": _source_complete(project),
        "voice": _voice_complete(project),
        "narration_subtitle": subtitle_complete,
        "export": _file_exists(_value(project, "final_video", "")),
    }


def step_access(project: Any) -> dict[str, dict[str, Any]]:
    """Return enabled/completed state and a blocking reason for every step."""

    facts = workflow_facts(project)
    result: dict[str, dict[str, Any]] = {}

    for index, step in enumerate(PRIMARY_STEPS):
        enabled = index == 0
        if index > 0:
            previous = result[PRIMARY_STEPS[index - 1]]
            enabled = bool(previous["enabled"] and previous["complete"])
        result[step] = {
            "enabled": enabled,
            "complete": facts[step],
            "reason": "" if enabled else _PREREQUISITE_REASON[step],
        }
    return result


def resume_step(project: Any) -> str:
    """Return the earliest accessible incomplete step.

    A fully completed workflow resumes at the terminal export step.
    """

    access = step_access(project)
    for step in PRIMARY_STEPS:
        state = access[step]
        if state["enabled"] and not state["complete"]:
            return step
    return PRIMARY_STEPS[-1]


def next_step(current_id: str, project: Any) -> str:
    """Return the next accessible incomplete step after ``current_id``.

    If the current step is incomplete it remains selected.  At the end of a
    completed workflow the terminal export step remains selected.
    """

    if current_id not in PRIMARY_STEPS:
        raise ValueError(f"Unknown workflow step: {current_id!r}")

    access = step_access(project)
    current = access[current_id]
    if not current["complete"]:
        return current_id

    current_index = PRIMARY_STEPS.index(current_id)
    for step in PRIMARY_STEPS[current_index + 1 :]:
        state = access[step]
        if state["enabled"] and not state["complete"]:
            return step
    return resume_step(project)
