"""Create, save, and load projects."""

import json
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.models.project import Project, ExportConfig
from src.models.clip import Clip
from src.utils.file_utils import sanitize_filename, ensure_dir
from src.utils.logger import logger

OUTPUT_DIR = Path("output")


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _write_json_atomic(path: Path, data) -> None:
    """Write JSON as one replace operation so interrupted saves stay readable."""
    ensure_dir(path.parent)
    temp_path = path.with_name(f".{path.name}.tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush()
    temp_path.replace(path)


def create_project(name: str, source_video: str, metadata: dict) -> Project:
    pid = str(uuid.uuid4())[:8]
    safe_name = sanitize_filename(name)
    out_dir = OUTPUT_DIR / f"{safe_name}_{pid}"
    ensure_dir(out_dir)
    for sub in ("captions", "subtitles", "previews", "exports", "audio"):
        ensure_dir(out_dir / sub)

    project = Project(
        id=pid,
        name=name,
        created_at=_now(),
        updated_at=_now(),
        source_video=source_video,
        original_source_video=source_video,
        video_metadata=metadata,
        output_dir=str(out_dir),
    )
    duration = float(metadata.get("duration", 0) or 0)
    if duration > 0:
        project.clips = [Clip(
            id=str(uuid.uuid4())[:8],
            index=1,
            start_time=0.0,
            end_time=round(duration, 3),
            part_text="",
            source_video=source_video,
        )]
        project.scene_revision = 1
    save_project(project)
    logger.info(f"Project created: {name} -> {out_dir}")
    return project


def save_project(project: Project) -> None:
    project.updated_at = _now()
    out = Path(project.output_dir)
    ensure_dir(out)

    _write_json_atomic(out / "project.json", project.to_dict())

    clips_data = [c.to_dict() for c in project.clips]
    # Compatibility cache for older builds. project.json is the canonical
    # snapshot so a crash can no longer combine project state from one save
    # with clips from another.
    _write_json_atomic(out / "clips.json", clips_data)

    logger.debug(f"Project saved: {project.name}")


def load_project(project_dir: str) -> Optional[Project]:
    p = Path(project_dir) / "project.json"
    if not p.exists():
        logger.error(f"project.json not found in {project_dir}")
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        project = Project.from_dict(data)

        clips_path = Path(project_dir) / "clips.json"
        if "clips" not in data and clips_path.exists():
            with open(clips_path, "r", encoding="utf-8") as f:
                clips_data = json.load(f)
            project.clips = [Clip.from_dict(c) for c in clips_data]
        project.output_dir = str(Path(project_dir).resolve())

        logger.info(f"Project loaded: {project.name}")
        return project
    except Exception as e:
        logger.error(f"Load project error: {e}")
        return None


def list_projects() -> list[dict]:
    OUTPUT_DIR.mkdir(exist_ok=True)
    projects = []
    for d in OUTPUT_DIR.iterdir():
        if d.is_dir():
            pj = d / "project.json"
            if pj.exists():
                try:
                    with open(pj, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    projects.append({
                        "name": data.get("name", d.name),
                        "created_at": data.get("created_at", ""),
                        "updated_at": data.get("updated_at", ""),
                        "dir": str(d),
                        "clips_count": len(data.get("clips", [])),
                    })
                except Exception:
                    pass
    return sorted(projects, key=lambda x: x["updated_at"], reverse=True)


def delete_project(project_dir: str) -> bool:
    """Permanently delete a project folder and all its contents."""
    try:
        d = Path(project_dir)
        # Safety: only delete folders that actually contain a project.json
        if d.is_dir() and (d / "project.json").exists():
            shutil.rmtree(d)
            logger.info(f"Project deleted: {project_dir}")
            return True
        logger.warning(f"Refused to delete (not a project dir): {project_dir}")
        return False
    except Exception as e:
        logger.error(f"Delete project error: {e}")
        return False


def generate_clips(project: Project) -> list:
    """Auto-generate clip segments based on clip_duration."""
    duration = project.video_metadata.get("duration", 0)
    if duration <= 0 or project.clip_duration <= 0:
        return []

    clips = []
    start = 0.0
    idx = 1
    while start < duration:
        end = min(start + project.clip_duration, duration)
        clip = Clip(
            id=str(uuid.uuid4())[:8],
            index=idx,
            start_time=round(start, 3),
            end_time=round(end, 3),
            part_text=f"CẢNH {idx:02d}",
        )
        clips.append(clip)
        start = end
        idx += 1

    project.clips = clips
    save_project(project)
    logger.info(f"Generated {len(clips)} clips for project {project.name}")
    return clips


def save_export_history(project: Project, exported: list[str]) -> None:
    out = Path(project.output_dir) / "export_history.json"
    history = []
    if out.exists():
        try:
            with open(out, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            pass
    history.append({"timestamp": _now(), "files": exported})
    with open(out, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def save_captions(project: Project, captions: list[dict]) -> None:
    captions_dir = Path(project.output_dir) / "captions"
    ensure_dir(captions_dir)
    for item in captions:
        idx = item.get("index", 1)
        platform = item.get("platform", "TikTok")
        fname = f"part{idx:02d}_{platform}.txt"
        with open(captions_dir / fname, "w", encoding="utf-8") as f:
            f.write(item.get("caption", ""))
            f.write("\n\n")
            f.write(item.get("hashtags", ""))
    logger.info(f"Saved {len(captions)} captions")
