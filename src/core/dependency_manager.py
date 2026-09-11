"""Auto-download FFmpeg, yt-dlp, and gallery-dl into tools/."""

import os
import sys
import zipfile
import shutil
import urllib.request
from pathlib import Path
from typing import Callable, Optional

from src.utils.logger import logger

TOOLS_DIR = Path("tools")

FFMPEG_URL = (
    "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/"
    "ffmpeg-master-latest-win64-gpl.zip"
)
YTDLP_URL = "https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp.exe"
GALLERYDL_URL = (
    "https://github.com/gdl-org/builds/releases/latest/download/"
    "gallery-dl_windows.exe"
)


def _tools_dir() -> Path:
    TOOLS_DIR.mkdir(exist_ok=True)
    return TOOLS_DIR


def ffmpeg_path() -> str:
    p = _tools_dir() / "ffmpeg.exe"
    return str(p) if p.exists() else ""


def ffprobe_path() -> str:
    p = _tools_dir() / "ffprobe.exe"
    return str(p) if p.exists() else ""


def ytdlp_path() -> str:
    p = _tools_dir() / "yt-dlp.exe"
    return str(p) if p.exists() else ""


def gallerydl_path() -> str:
    p = _tools_dir() / "gallery-dl.exe"
    return str(p) if p.exists() else ""


def is_ffmpeg_available() -> bool:
    return bool(ffmpeg_path()) and bool(ffprobe_path())


def is_ytdlp_available() -> bool:
    return bool(ytdlp_path())


def download_gallerydl(
    progress_cb: Optional[Callable[[int, int], None]] = None,
    status_cb: Optional[Callable[[str], None]] = None,
) -> bool:
    tools = _tools_dir()
    dest = tools / "gallery-dl.exe"
    try:
        if status_cb:
            status_cb("Đang tải bộ hỗ trợ Instagram gallery-dl...")
        _download_with_progress(GALLERYDL_URL, str(dest), progress_cb)
        if status_cb:
            status_cb("gallery-dl sẵn sàng.")
        return True
    except Exception as e:
        logger.error(f"gallery-dl download failed: {e}")
        if status_cb:
            status_cb(f"Lỗi tải gallery-dl: {e}")
        return False


def _download_with_progress(
    url: str,
    dest: str,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> None:
    logger.info(f"Downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as response:
        total = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        chunk = 65536
        with open(dest, "wb") as f:
            while True:
                data = response.read(chunk)
                if not data:
                    break
                f.write(data)
                downloaded += len(data)
                if progress_cb and total:
                    progress_cb(downloaded, total)
    logger.info(f"Downloaded to {dest}")


def _replace_tool_file(src_path: Path, dest_path: Path) -> tuple[bool, str]:
    """Replace a tool executable, tolerating Windows locks when old file exists."""
    try:
        os.replace(str(src_path), str(dest_path))
        return True, ""
    except PermissionError as e:
        if dest_path.exists():
            msg = (
                f"{dest_path.name} đang được Windows khóa, giữ lại bản hiện có. "
                "Hãy đóng export/preview hoặc tiến trình ffmpeg rồi cập nhật lại nếu cần."
            )
            logger.warning(f"{msg} ({e})")
            src_path.unlink(missing_ok=True)
            return True, msg
        return False, str(e)
    except Exception as e:
        return False, str(e)


def download_ffmpeg(
    progress_cb: Optional[Callable[[int, int], None]] = None,
    status_cb: Optional[Callable[[str], None]] = None,
) -> bool:
    tools = _tools_dir()
    zip_path = tools / "ffmpeg_tmp.zip"
    try:
        if status_cb:
            status_cb("Đang tải FFmpeg...")
        _download_with_progress(FFMPEG_URL, str(zip_path), progress_cb)

        if status_cb:
            status_cb("Đang giải nén FFmpeg...")
        warnings = []
        with zipfile.ZipFile(zip_path, "r") as zf:
            members = zf.namelist()
            for member in members:
                name = Path(member).name
                if name in ("ffmpeg.exe", "ffprobe.exe", "ffplay.exe"):
                    tmp_exe = tools / f"{name}.download"
                    with zf.open(member) as src, open(tmp_exe, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    ok, msg = _replace_tool_file(tmp_exe, tools / name)
                    if not ok:
                        raise PermissionError(msg)
                    if msg:
                        warnings.append(msg)
                    logger.info(f"Extracted {name}")

        zip_path.unlink(missing_ok=True)
        if status_cb:
            status_cb("FFmpeg sẵn sàng.")
        if status_cb and warnings:
            status_cb("FFmpeg sẵn sàng, nhưng có file đang được dùng nên chưa cập nhật hết.")
        return True
    except Exception as e:
        logger.error(f"FFmpeg download failed: {e}")
        zip_path.unlink(missing_ok=True)
        if status_cb:
            status_cb(f"Lỗi tải FFmpeg: {e}")
        return False


def download_ytdlp(
    progress_cb: Optional[Callable[[int, int], None]] = None,
    status_cb: Optional[Callable[[str], None]] = None,
) -> bool:
    tools = _tools_dir()
    dest = tools / "yt-dlp.exe"
    try:
        if status_cb:
            status_cb("Đang tải yt-dlp...")
        _download_with_progress(YTDLP_URL, str(dest), progress_cb)
        if status_cb:
            status_cb("yt-dlp sẵn sàng.")
        return True
    except Exception as e:
        logger.error(f"yt-dlp download failed: {e}")
        if status_cb:
            status_cb(f"Lỗi tải yt-dlp: {e}")
        return False


def check_and_report() -> dict:
    return {
        "ffmpeg": is_ffmpeg_available(),
        "ytdlp": is_ytdlp_available(),
        "gallerydl": bool(gallerydl_path()),
    }
