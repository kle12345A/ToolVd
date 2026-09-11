import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Optional


def sanitize_filename(name: str) -> str:
    name = re.sub(r'[\\/:*?"<>|]', "_", name)
    name = name.strip(". ")
    return name or "project"


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_safe_temp_path(suffix: str = ".srt") -> str:
    """Return ASCII-only, space-free temp path for FFmpeg subtitle filter.

    FFmpeg's ``subtitles=`` filter cannot handle spaces or non-ASCII characters
    in the file path even when the path is single-quoted inside filter_complex.
    We try several candidate directories in order of preference.
    """
    # Directories that are guaranteed to have no spaces
    candidates: list[str] = [
        r"C:\Temp",
        r"C:\tmp",
        r"D:\Temp",
    ]
    # Also accept system temp if it happens to be space-free
    sys_tmp = tempfile.gettempdir()
    if " " not in sys_tmp:
        candidates.append(sys_tmp)

    for temp_dir in candidates:
        try:
            os.makedirs(temp_dir, exist_ok=True)
            # Double-check: no spaces anywhere in the chosen directory
            if " " in temp_dir:
                continue
            tmp = tempfile.NamedTemporaryFile(
                suffix=suffix,
                delete=False,
                dir=temp_dir,
                prefix="amsr_sub_",
            )
            tmp.close()
            return tmp.name
        except Exception:
            continue

    # Last resort – system default (may have spaces; caller should log a warning)
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False, prefix="amsr_sub_")
    tmp.close()
    return tmp.name


def copy_to_ascii_temp(src: str, suffix: str = "") -> str:
    """Copy file to temp location with ASCII path."""
    if not suffix:
        suffix = Path(src).suffix
    tmp_path = get_safe_temp_path(suffix)
    shutil.copy2(src, tmp_path)
    return tmp_path


def format_duration(seconds: float) -> str:
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:05.2f}"
    return f"{m:02d}:{s:05.2f}"


def seconds_to_hms(seconds: float) -> str:
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def hms_to_seconds(hms: str) -> float:
    """Convert HH:MM:SS.ms or MM:SS or SS string to float seconds."""
    try:
        hms = hms.strip()
        parts = hms.replace(",", ".").split(":")
        if len(parts) == 3:
            return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
        if len(parts) == 2:
            return float(parts[0]) * 60 + float(parts[1])
        return float(parts[0])
    except (ValueError, IndexError):
        return 0.0


def file_size_str(path: str) -> str:
    try:
        size = os.path.getsize(path)
        for unit in ["B", "KB", "MB", "GB"]:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"
    except OSError:
        return "N/A"


def find_windows_font(names: list[str] = None) -> Optional[str]:
    """Find a font file in Windows Fonts directory."""
    if names is None:
        names = ["arial.ttf", "ArialMT.ttf", "calibri.ttf", "segoeui.ttf"]
    fonts_dir = Path("C:/Windows/Fonts")
    if not fonts_dir.exists():
        return None
    for name in names:
        p = fonts_dir / name
        if p.exists():
            return str(p)
    return None
