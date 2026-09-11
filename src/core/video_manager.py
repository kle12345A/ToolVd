"""Read video metadata using ffprobe."""

import json
import subprocess
from pathlib import Path
from typing import Optional

from src.core.dependency_manager import ffprobe_path
from src.utils.logger import logger


def get_video_metadata(video_path: str) -> Optional[dict]:
    probe = ffprobe_path()
    if not probe:
        logger.error("ffprobe not found")
        return None

    cmd = [
        probe,
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        video_path,
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
        if result.returncode != 0:
            logger.error(f"ffprobe error: {result.stderr}")
            return None

        data = json.loads(result.stdout)
        return _parse_metadata(data, video_path)
    except subprocess.TimeoutExpired:
        logger.error("ffprobe timed out")
        return None
    except Exception as e:
        logger.error(f"Metadata read error: {e}")
        return None


def _parse_metadata(data: dict, video_path: str) -> dict:
    fmt = data.get("format", {})
    streams = data.get("streams", [])

    video_stream = next(
        (s for s in streams if s.get("codec_type") == "video"), {}
    )
    audio_stream = next(
        (s for s in streams if s.get("codec_type") == "audio"), {}
    )

    duration = float(fmt.get("duration", 0) or video_stream.get("duration", 0) or 0)
    width = int(video_stream.get("width", 0) or 0)
    height = int(video_stream.get("height", 0) or 0)

    fps = 0.0
    fps_str = video_stream.get("r_frame_rate", "0/1")
    try:
        num, den = fps_str.split("/")
        if float(den) > 0:
            fps = float(num) / float(den)
    except Exception:
        pass

    size_bytes = int(fmt.get("size", 0) or 0)
    size_mb = size_bytes / (1024 * 1024)

    return {
        "path": video_path,
        "filename": Path(video_path).name,
        "duration": duration,
        "width": width,
        "height": height,
        "fps": round(fps, 2),
        "video_codec": video_stream.get("codec_name", "unknown"),
        "audio_codec": audio_stream.get("codec_name", "unknown"),
        "size_bytes": size_bytes,
        "size_mb": round(size_mb, 2),
        "bitrate": int(fmt.get("bit_rate", 0) or 0),
        "format": fmt.get("format_long_name", ""),
        "has_audio": bool(audio_stream),
    }
