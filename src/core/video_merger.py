"""Merge multiple videos into one with transition effects (FFmpeg xfade)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Callable, Optional

from src.core.dependency_manager import ffmpeg_path, ffprobe_path
from src.core.video_processor import _run_ffmpeg
from src.utils.file_utils import ensure_dir
from src.utils.logger import logger


# xfade transition presets: ffmpeg id -> Vietnamese label
TRANSITIONS = {
    "fade":        "Mờ dần (Fade)",
    "fadeblack":   "Qua đen (Fade Black)",
    "fadewhite":   "Qua trắng (Fade White)",
    "dissolve":    "Hòa tan (Dissolve)",
    "smoothleft":  "Mượt sang trái",
    "smoothright": "Mượt sang phải",
    "wipeleft":    "Gạt trái (Wipe)",
    "wiperight":   "Gạt phải (Wipe)",
    "slideup":     "Trượt lên",
    "slidedown":   "Trượt xuống",
    "slideleft":   "Trượt trái",
    "slideright":  "Trượt phải",
    "circleopen":  "Mở vòng tròn",
    "circleclose": "Đóng vòng tròn",
    "pixelize":    "Vỡ điểm ảnh (Pixelize)",
    "radial":      "Quét tròn (Radial)",
    "zoomin":      "Phóng to (Zoom In)",
}


def probe_video(path: str) -> dict:
    """Return {duration, width, height, fps, has_audio} for *path*."""
    info = {"duration": 0.0, "width": 0, "height": 0, "fps": 30.0, "has_audio": False}
    fp = ffprobe_path()
    if not fp:
        return info
    try:
        out = subprocess.run(
            [fp, "-v", "error", "-print_format", "json",
             "-show_format", "-show_streams", path],
            capture_output=True, text=True, timeout=30,
        )
        data = json.loads(out.stdout or "{}")
        fmt = data.get("format", {})
        info["duration"] = float(fmt.get("duration", 0) or 0)
        for s in data.get("streams", []):
            if s.get("codec_type") == "video" and info["width"] == 0:
                info["width"] = int(s.get("width", 0) or 0)
                info["height"] = int(s.get("height", 0) or 0)
                rate = s.get("r_frame_rate", "30/1")
                try:
                    num, den = rate.split("/")
                    info["fps"] = round(float(num) / float(den), 3) if float(den) else 30.0
                except Exception:
                    info["fps"] = 30.0
            elif s.get("codec_type") == "audio":
                info["has_audio"] = True
    except Exception as e:
        logger.warning(f"probe_video error for {path}: {e}")
    return info


def _even(n: int) -> int:
    return int(n) - (int(n) % 2)


def merge_videos(
    paths: list[str],
    transition: str,
    trans_dur: float,
    output_path: str,
    target_w: int = 1080,
    target_h: int = 1920,
    fps: int = 30,
    progress_cb: Optional[Callable[[float, float], None]] = None,
    clip_max_durations: Optional[list[float | None]] = None,
) -> tuple[bool, str]:
    """Merge *paths* into one video with the chosen *transition* between each.

    Each clip is normalised to target_w×target_h @ fps (letter-boxed) so xfade
    can chain them. Audio is cross-faded; clips without audio get silence.
    Returns (ok, error_or_warning).
    """
    ff = ffmpeg_path()
    if not ff:
        return False, "FFmpeg không tìm thấy."
    paths = [p for p in paths if p and Path(p).exists()]
    if len(paths) < 2:
        return False, "Cần ít nhất 2 video để ghép."

    W, H = _even(target_w), _even(target_h)
    infos = [probe_video(p) for p in paths]
    durs = []
    for i, info in enumerate(infos):
        dur = max(0.1, float(info["duration"] or 0))
        if clip_max_durations and i < len(clip_max_durations):
            cap = clip_max_durations[i]
            if cap and float(cap) > 0:
                dur = max(0.1, min(dur, float(cap)))
        durs.append(dur)

    # Clamp transition duration so it never exceeds the shortest clip.
    max_trans = max(0.1, min(durs) - 0.1)
    td = max(0.1, min(float(trans_dur), max_trans))
    warn = ""
    if td < trans_dur:
        warn = f"Thời lượng chuyển cảnh giảm còn {td:.1f}s (clip ngắn nhất chỉ {min(durs):.1f}s)."

    fp = []  # filter_complex segments

    # 1) Normalise every video stream
    for i in range(len(paths)):
        fp.append(
            f"[{i}:v]trim=duration={durs[i]:.3f},setpts=PTS-STARTPTS,"
            f"scale={W}:{H}:force_original_aspect_ratio=decrease,"
            f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:black,setsar=1,"
            f"fps={fps},format=yuv420p[v{i}]"
        )

    # 2) Normalise / synthesise audio for every input
    for i in range(len(paths)):
        if infos[i]["has_audio"]:
            fp.append(
                f"[{i}:a]atrim=0:{durs[i]:.3f},asetpts=PTS-STARTPTS,"
                f"aresample=44100,"
                f"aformat=sample_fmts=fltp:channel_layouts=stereo[a{i}]"
            )
        else:
            fp.append(
                f"anullsrc=r=44100:cl=stereo,atrim=0:{durs[i]:.3f},"
                f"asetpts=PTS-STARTPTS[a{i}]"
            )

    # 3) Chain xfade (video) + acrossfade (audio)
    acc = durs[0]
    vp, ap = "v0", "a0"
    for i in range(1, len(paths)):
        offset = max(0.05, acc - td)
        vo, ao = f"vx{i}", f"ax{i}"
        fp.append(
            f"[{vp}][v{i}]xfade=transition={transition}:"
            f"duration={td:.3f}:offset={offset:.3f}[{vo}]"
        )
        fp.append(f"[{ap}][a{i}]acrossfade=d={td:.3f}[{ao}]")
        acc = acc + durs[i] - td
        vp, ap = vo, ao

    fc = ";".join(fp)
    total = acc

    ensure_dir(Path(output_path).parent)
    cmd = [
        ff, "-y",
        *sum([["-i", p] for p in paths], []),
        "-filter_complex", fc,
        "-map", f"[{vp}]",
        "-map", f"[{ap}]",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-r", str(fps), "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        output_path,
    ]
    ok, err = _run_ffmpeg(cmd, total, progress_cb)
    if ok and warn:
        return True, warn
    return ok, err


def concatenate_videos(
    paths: list[str],
    output_path: str,
    target_w: int | None = None,
    target_h: int | None = None,
    fps: int = 30,
    progress_cb: Optional[Callable[[float, float], None]] = None,
    clip_max_durations: Optional[list[float | None]] = None,
) -> tuple[bool, str]:
    """Join videos end-to-end without transitions.

    Inputs are normalized before concat so short videos from different sources
    can be used as one project source video.
    """
    ff = ffmpeg_path()
    if not ff:
        return False, "FFmpeg khong tim thay."
    paths = [p for p in paths if p and Path(p).exists()]
    if len(paths) < 2:
        return False, "Can it nhat 2 video de ghep."

    infos = [probe_video(p) for p in paths]
    first_video = next((i for i in infos if i["width"] and i["height"]), None)
    if target_w is None:
        target_w = first_video["width"] if first_video else 1080
    if target_h is None:
        target_h = first_video["height"] if first_video else 1920

    W, H = _even(target_w), _even(target_h)
    durs = []
    for i, info in enumerate(infos):
        dur = max(0.1, float(info["duration"] or 0))
        if clip_max_durations and i < len(clip_max_durations):
            cap = clip_max_durations[i]
            if cap and float(cap) > 0:
                dur = max(0.1, min(dur, float(cap)))
        durs.append(dur)
    total = sum(durs)

    filters = []
    concat_inputs = []
    for i, info in enumerate(infos):
        filters.append(
            f"[{i}:v]trim=duration={durs[i]:.3f},setpts=PTS-STARTPTS,"
            f"scale={W}:{H}:force_original_aspect_ratio=decrease,"
            f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:black,setsar=1,"
            f"fps={fps},format=yuv420p[v{i}]"
        )
        if info["has_audio"]:
            filters.append(
                f"[{i}:a]atrim=0:{durs[i]:.3f},asetpts=PTS-STARTPTS,"
                f"aresample=44100,"
                f"aformat=sample_fmts=fltp:channel_layouts=stereo[a{i}]"
            )
        else:
            filters.append(
                f"anullsrc=r=44100:cl=stereo,atrim=0:{durs[i]:.3f},"
                f"asetpts=PTS-STARTPTS[a{i}]"
            )
        concat_inputs.append(f"[v{i}][a{i}]")

    filters.append(
        "".join(concat_inputs) + f"concat=n={len(paths)}:v=1:a=1[v][a]"
    )

    ensure_dir(Path(output_path).parent)
    cmd = [
        ff, "-y",
        *sum([["-i", p] for p in paths], []),
        "-filter_complex", ";".join(filters),
        "-map", "[v]",
        "-map", "[a]",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-r", str(fps), "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        output_path,
    ]
    return _run_ffmpeg(cmd, total, progress_cb)
