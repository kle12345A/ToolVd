"""FFmpeg-based video processing: cut, scale, PART text, subtitle burn."""

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Optional

from src.core.dependency_manager import ffmpeg_path
from src.core.video_manager import get_video_metadata
from src.models.clip import Clip
from src.models.project import Project, ExportConfig
from src.utils.file_utils import (
    ensure_dir,
    seconds_to_hms,
    copy_to_ascii_temp,
)
from src.utils.logger import logger


def is_nvenc_available() -> bool:
    """Check if NVIDIA NVENC encoder actually works (not just compiled in).

    Does a quick 1-frame test encode to verify:
    - FFmpeg has h264_nvenc compiled in
    - NVIDIA driver is installed and supports required NVENC API version
    - GPU hardware encoder is functional
    """
    try:
        ff = ffmpeg_path()
        # Quick test: encode 1 black frame with h264_nvenc
        result = subprocess.run(
            [
                ff, "-hide_banner", "-loglevel", "error",
                "-f", "lavfi", "-i", "color=black:s=64x64:d=0.04",
                "-c:v", "h264_nvenc", "-frames:v", "1",
                "-f", "null", "-",
            ],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0:
            logger.info("NVENC test encode succeeded — GPU encoding available")
            return True
        else:
            logger.warning(f"NVENC test encode failed: {result.stderr.strip()}")
            return False
    except Exception as e:
        logger.warning(f"NVENC availability check error: {e}")
        return False


_NVENC_TO_X264 = {
    "p1": "ultrafast", "p2": "superfast", "p3": "veryfast",
    "p4": "fast", "p5": "medium", "p6": "slow", "p7": "slower",
}
_X264_TO_NVENC = {
    "ultrafast": "p1", "superfast": "p1", "veryfast": "p2", "faster": "p3",
    "fast": "p4", "medium": "p4", "slow": "p5", "slower": "p6",
    "veryslow": "p7", "placebo": "p7",
}
_X264_PRESETS = {
    "ultrafast", "superfast", "veryfast", "faster", "fast",
    "medium", "slow", "slower", "veryslow", "placebo",
}
_NVENC_PRESETS = {"p1", "p2", "p3", "p4", "p5", "p6", "p7"}


def _cpu_thread_count(low_memory: bool = False) -> int:
    if low_memory:
        return 1
    try:
        return max(1, min(4, os.cpu_count() or 2))
    except Exception:
        return 2


def _is_x264_memory_error(err: str) -> bool:
    msg = (err or "").lower()
    return (
        "x264" in msg
        and (
            "malloc" in msg
            or "error while opening encoder" in msg
            or "could not open encoder" in msg
        )
    )


def _normalize_preset(preset: str, gpu: bool) -> str:
    """Translate a preset name to one valid for the target encoder.

    The UI swaps preset lists when toggling GPU, but a saved config may still
    hold an NVENC preset (p1-p7) while exporting with libx264 (or vice-versa).
    This guarantees the encoder always gets a preset it understands.
    """
    p = (preset or "").strip().lower()
    if gpu:
        if p in _NVENC_PRESETS:
            return p
        return _X264_TO_NVENC.get(p, "p4")
    else:
        if p in _X264_PRESETS:
            return p
        return _NVENC_TO_X264.get(p, "fast")


def _ff() -> str:
    p = ffmpeg_path()
    if not p:
        raise RuntimeError("FFmpeg không tìm thấy. Vui lòng tải FFmpeg trước.")
    return p


def _run_ffmpeg(
    cmd: list[str],
    duration: float,
    progress_cb: Optional[Callable[[float, float], None]] = None,
) -> tuple[bool, str]:
    """Run FFmpeg command with progress tracking via stderr."""
    logger.debug("FFmpeg cmd: " + " ".join(cmd))
    try:
        proc = subprocess.Popen(
            cmd,
            stderr=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        stderr_lines = []
        time_pattern = re.compile(r"time=(\d+):(\d+):([\d.]+)")
        for line in proc.stderr:
            stderr_lines.append(line)
            if progress_cb and duration > 0:
                m = time_pattern.search(line)
                if m:
                    elapsed = (
                        float(m.group(1)) * 3600
                        + float(m.group(2)) * 60
                        + float(m.group(3))
                    )
                    progress_cb(elapsed, duration)

        proc.wait()
        if proc.returncode != 0:
            err = "".join(stderr_lines[-20:])
            logger.error(f"FFmpeg failed (rc={proc.returncode}): {err}")
            return False, err
        return True, ""
    except Exception as e:
        logger.error(f"FFmpeg exception: {e}")
        return False, str(e)


def _escape_text(text: str) -> str:
    """Escape special characters for FFmpeg drawtext ``text=`` value.

    Single-line only — newlines are stripped.  Strips characters that
    are unsafe for the filter-graph parser (apostrophes, curly quotes).
    Use ``_drawtext_expr`` for text that must preserve all characters.
    """
    text = text.replace("\r\n", " ").replace("\n", " ")
    text = text.replace("\\", "\\\\")
    text = text.replace(":", "\\:")
    text = text.replace(";", "\\;")
    # Strip quote chars that break filter-graph parser
    text = text.replace("’", "")
    text = text.replace("‘", "").replace("’", "")
    text = text.replace("“", "").replace("”", "")
    return text


def _drawtext_expr(text: str, in_pad: str = "", out_pad: str = "", **kw) -> str:
    """Build a drawtext filter string, supporting multi-line text.

    When ``in_pad`` and ``out_pad`` are provided, returns a complete
    filterchain string — e.g. ``[in_pad]drawtext=...[out_pad]`` for a
    single line, or semicolon-separated chains for multiple lines.  This
    form avoids FFmpeg’s filter-graph parser misreading the output pad
    label as part of a drawtext option when comma-chained drawtext is
    followed by another filterchain.

    When omitted (legacy callers), returns just the drawtext expression(s)
    as a comma-chained string.

    The ``y`` kwarg is treated as the **centre** of the text block.
    Each line is offset from that centre so the block as a whole stays
    vertically centred.

    Extra keyword args are forwarded to every drawtext instance.
    """
    def _esc_drawtext(t: str) -> str:
        t = t.replace("\\", "\\\\")
        t = t.replace(":", "\:")
        t = t.replace(";", "\;")
        t = t.replace("'", "")          # U+0027 ASCII apostrophe
        t = t.replace("’", "")     # right single quotation
        return t

    def _needs_textfile(t: str) -> bool:
        # Any apostrophe/quote variant that confuses the filter-graph parser
        return any(c in t for c in (
            "'",        # U+0027 ASCII apostrophe
            '"',        # U+0022 ASCII double quote
            "‘",   # left single quotation
            "’",   # right single quotation
            "ʼ",   # modifier letter apostrophe
            "“",   # left double quotation
            "”",   # right double quotation
        ))

    from src.utils.file_utils import get_safe_temp_path

    def _text_source(line_text: str) -> str:
        if _needs_textfile(line_text):
            tmp = get_safe_temp_path(".txt")
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(line_text)
            # Wrap the path in single quotes (like the subtitles= filter) so the
            # Windows drive colon doesn't break the filter-graph parser.
            safe = (tmp.replace("\\", "/")
                       .replace(":", "\\:")
                       .replace(" ", "\\ "))
            return f"textfile='{safe}'"
        return f"text={_esc_drawtext(line_text)}"

    underline = bool(kw.pop("underline", False))
    bold = bool(kw.pop("bold", False))
    italic = bool(kw.pop("italic", False))
    if bold:
        kw.setdefault("borderw", 2)
        kw.setdefault("bordercolor", "black")

    def _underline_text(line_text: str) -> str:
        return "_" * max(2, len(line_text.strip()))

    def _underline_y(y_expr: str) -> str:
        offset = max(2, int(int(kw.get("fontsize", 48)) * 0.62))
        return f"({y_expr})+{offset}"

    def _build_expr(line_text: str, y_expr: str) -> str:
        src = _text_source(line_text)
        line_parts = [src]
        for k, v in kw.items():
            if k == "y":
                continue
            line_parts.append(f"{k}={v}")
        if "x" not in kw:
            line_parts.append("x=(w-tw)/2")
        line_parts.append(f"y={y_expr}")
        expr = "drawtext=" + ":".join(line_parts)
        if underline and line_text.strip():
            u_parts = [_text_source(_underline_text(line_text))]
            for k, v in kw.items():
                if k == "y":
                    continue
                u_parts.append(f"{k}={v}")
            if "x" not in kw:
                u_parts.append("x=(w-tw)/2")
            u_parts.append(f"y={_underline_y(y_expr)}")
            expr += ",drawtext=" + ":".join(u_parts)
        return expr

    text = text.replace("\r\n", "\n")
    lines = [l for l in text.split("\n")]

    # ── Single line ──────────────────────────────────────────────
    if len(lines) <= 1:
        line_text = lines[0] if lines else ""
        src = _text_source(line_text)
        parts = [src]
        for k, v in kw.items():
            parts.append(f"{k}={v}")
        expr = "drawtext=" + ":".join(parts)
        if underline and line_text.strip():
            u_parts = [_text_source(_underline_text(line_text))]
            for k, v in kw.items():
                if k == "y":
                    continue
                u_parts.append(f"{k}={v}")
            if "x" not in kw:
                u_parts.append("x=(w-tw)/2")
            u_parts.append(f"y={_underline_y(kw.get('y', '(h-th)/2'))}")
            expr += ",drawtext=" + ":".join(u_parts)
        if in_pad and out_pad:
            return f"[{in_pad}]{expr}[{out_pad}]"
        return expr

    # ── Multi-line ───────────────────────────────────────────────
    fontsize = int(kw.get("fontsize", 48))
    line_h = int(fontsize * 1.35)
    n = len(lines)
    total_h = n * line_h
    y_base = kw.get("y", "(h-th)/2")

    exprs = []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        if y_base == "(h-th)/2":
            ly = f"(h-{total_h})/2+{i * line_h}"
        else:
            offset = i * line_h - (total_h - line_h) // 2
            ly = f"{y_base}+{offset}" if offset >= 0 else f"{y_base}-{abs(offset)}"
        exprs.append(_build_expr(line, ly))

    if in_pad and out_pad:
        # Use semicolon-separated filterchains so the output pad label is
        # never ambiguous to FFmpeg’s filter-graph parser.
        chains = []
        for i, expr in enumerate(exprs):
            cur_in = in_pad if i == 0 else f"_{in_pad}{i - 1}"
            cur_out = out_pad if i == len(exprs) - 1 else f"_{in_pad}{i}"
            chains.append(f"[{cur_in}]{expr}[{cur_out}]")
        return ";".join(chains)

    return ",".join(exprs)


def _drawtext_font_value(font_name: str) -> str:
    return (font_name or "Arial").replace("\\", "\\\\").replace(":", "\\:")


def _fontfile_value(path: Path) -> str:
    safe = str(path).replace("\\", "/").replace(":", "\\:").replace(" ", "\\ ")
    return f"'{safe}'"


def _windows_fontfile(font_name: str, bold: bool = False, italic: bool = False) -> str:
    """Return a drawtext-safe Windows font file for common UI choices."""
    fonts_dir = Path("C:/Windows/Fonts")
    key = (font_name or "Arial").lower()
    table = {
        "arial": {
            (False, False): "arial.ttf", (True, False): "arialbd.ttf",
            (False, True): "ariali.ttf", (True, True): "arialbi.ttf",
        },
        "times new roman": {
            (False, False): "times.ttf", (True, False): "timesbd.ttf",
            (False, True): "timesi.ttf", (True, True): "timesbi.ttf",
        },
        "verdana": {
            (False, False): "verdana.ttf", (True, False): "verdanab.ttf",
            (False, True): "verdanai.ttf", (True, True): "verdanaz.ttf",
        },
        "tahoma": {
            (False, False): "tahoma.ttf", (True, False): "tahomabd.ttf",
        },
        "calibri": {
            (False, False): "calibri.ttf", (True, False): "calibrib.ttf",
            (False, True): "calibrii.ttf", (True, True): "calibriz.ttf",
        },
        "segoe ui": {
            (False, False): "segoeui.ttf", (True, False): "segoeuib.ttf",
            (False, True): "segoeuii.ttf", (True, True): "segoeuiz.ttf",
        },
        "impact": {
            (False, False): "impact.ttf", (True, False): "impact.ttf",
            (False, True): "impact.ttf", (True, True): "impact.ttf",
        },
        "georgia": {
            (False, False): "georgia.ttf", (True, False): "georgiab.ttf",
            (False, True): "georgiai.ttf", (True, True): "georgiaz.ttf",
        },
        "trebuchet ms": {
            (False, False): "trebuc.ttf", (True, False): "trebucbd.ttf",
            (False, True): "trebucit.ttf", (True, True): "trebucbi.ttf",
        },
        "comic sans ms": {
            (False, False): "comic.ttf", (True, False): "comicbd.ttf",
            (False, True): "comic.ttf", (True, True): "comicbd.ttf",
        },
        "courier new": {
            (False, False): "cour.ttf", (True, False): "courbd.ttf",
            (False, True): "couri.ttf", (True, True): "courbi.ttf",
        },
    }
    family = table.get(key, table["arial"])
    candidates = [
        family.get((bold, italic)),
        family.get((bold, False)),
        family.get((False, italic)),
        family.get((False, False)),
    ]
    for name in candidates:
        if not name:
            continue
        path = fonts_dir / name
        if path.exists():
            return _fontfile_value(path)
    return ""


def _part_text_style_args(cfg: "ExportConfig") -> dict:
    bold = getattr(cfg, "part_text_bold", True)
    italic = getattr(cfg, "part_text_italic", False)
    font_name = getattr(cfg, "part_text_font", "Arial") or "Arial"
    fontfile = _windows_fontfile(font_name, bold=bold, italic=italic)
    font_args = {"fontfile": fontfile} if fontfile else {"font": _drawtext_font_value(font_name)}
    return {
        **font_args,
        "bold": bold,
        "italic": italic,
        "underline": getattr(cfg, "part_text_underline", False),
    }


def _subtitle_drawtext_style_args(cfg: "ExportConfig") -> dict:
    bold = getattr(cfg, "subtitle_bold", True)
    italic = getattr(cfg, "subtitle_italic", False)
    font_name = getattr(cfg, "subtitle_font", "Arial") or "Arial"
    fontfile = _windows_fontfile(font_name, bold=bold, italic=italic)
    font_args = {"fontfile": fontfile} if fontfile else {"font": _drawtext_font_value(font_name)}
    return {
        **font_args,
        "bold": bold,
        "italic": italic,
        "underline": getattr(cfg, "subtitle_underline", False),
    }


def _color_to_ass(name: str) -> str:
    """Convert a color name or #RRGGBB hex string to ASS &HAABBGGRR format.

    ASS stores colors in BGRA order (little-endian): &H<AA><BB><GG><RR>
    where AA=alpha (00=opaque).
    """
    _named = {
        "white":    "&H00FFFFFF",
        "black":    "&H00000000",
        "yellow":   "&H0000FFFF",
        "#ff6b6b":  "&H006B6BFF",
        "#7aa2f7":  "&H00F7A27A",
        "#a6e3a1":  "&H00A1E3A6",
    }
    if name in _named:
        return _named[name]
    # Try #RRGGBB
    if name.startswith("#") and len(name) == 7:
        try:
            r = int(name[1:3], 16)
            g = int(name[3:5], 16)
            b = int(name[5:7], 16)
            return f"&H00{b:02X}{g:02X}{r:02X}"
        except ValueError:
            pass
    return "&H00FFFFFF"


def _even(value: int, minimum: int = 2) -> int:
    value = max(minimum, int(value))
    return value if value % 2 == 0 else value - 1


def _get_ass_play_res_y(path: str) -> int:
    """Read PlayResY from the [Script Info] header of an ASS file.

    Returns 288 (Whisper default) if not found or if the file cannot be read.
    """
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if line.startswith("[Events]"):
                    break   # stop before event section
                m = re.match(r"PlayResY\s*:\s*(\d+)", line, re.IGNORECASE)
                if m:
                    return int(m.group(1))
    except Exception:
        pass
    return 288


def _ass_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int(round((seconds - int(seconds)) * 100))
    if cs >= 100:
        s += 1
        cs = 0
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _parse_srt_time(value: str) -> float:
    h, m, rest = value.strip().split(":")
    s, ms = rest.replace(".", ",").split(",", 1)
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms[:3].ljust(3, "0")) / 1000


def _escape_ass_text(text: str) -> str:
    text = text.replace("\\", "\\\\")
    text = text.replace("{", r"\{").replace("}", r"\}")
    return text.replace("\r\n", r"\N").replace("\n", r"\N")


def _srt_to_ass_events(src: str) -> str:
    with open(src, "r", encoding="utf-8-sig", errors="replace") as f:
        raw = f.read().replace("\r\n", "\n").replace("\r", "\n")

    time_re = re.compile(
        r"(?P<start>\d{2}:\d{2}:\d{2}[,.]\d{1,3})\s*-->\s*"
        r"(?P<end>\d{2}:\d{2}:\d{2}[,.]\d{1,3})"
    )
    events = [
        "[Events]\n",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n",
    ]
    for block in re.split(r"\n\s*\n", raw.strip()):
        lines = [line.strip() for line in block.split("\n") if line.strip()]
        if not lines:
            continue
        time_idx = next((i for i, line in enumerate(lines) if time_re.search(line)), -1)
        if time_idx < 0:
            continue
        match = time_re.search(lines[time_idx])
        if not match:
            continue
        text_lines = lines[time_idx + 1:]
        if not text_lines:
            continue
        start = _ass_time(_parse_srt_time(match.group("start")))
        end = _ass_time(_parse_srt_time(match.group("end")))
        text = _escape_ass_text("\n".join(text_lines))
        events.append(f"Dialogue: 0,{start},{end},Plain,,0,0,0,,{text}\n")
    return "".join(events)


def _prepare_subtitle_for_export(
    src: str,
    cfg: "ExportConfig",
    actual_height: int | None = None,
) -> str:
    """Return a temp subtitle file path ready for FFmpeg burn-in.

    SRT  → copied to an ASCII/space-free temp path (FFmpeg auto-sets
           PlayResY = frame height, so force_style Fontsize = pixels 1:1).

    ASS  → a BRAND-NEW ASS file is written with:
           • PlayResY = actual_height  (the pixel height of the video stream
             that the subtitles= filter will run on — NOT the final output)
           • Fontsize  = cfg.subtitle_fontsize  (renders 1:1 because
             PlayResY == frame height → no libass scaling)
           • The [Events] section is copied verbatim from the source file so
             all \\kf karaoke timing tags are preserved unchanged.
           • Any stray inline \\fsN tags in the event text are stripped.

    Because we own PlayResY we never need the old "scale formula" at all:
        rendered_px = Fontsize × frame_height / PlayResY
                    = user_fs  × actual_height / actual_height
                    = user_fs   ✓
    """
    from src.utils.file_utils import get_safe_temp_path

    ext = Path(src).suffix.lower()

    # ── SRT: simple safe-copy ─────────────────────────────────────────────
    if ext != ".ass":
        try:
            events_section = _srt_to_ass_events(src)
        except Exception as exc:
            logger.warning(f"SRT to ASS conversion failed ({exc}) - falling back to raw copy")
            return copy_to_ascii_temp(src)
    else:
        events_section = ""

    # ── ASS: rebuild header, keep [Events] intact ────────────────────────
    try:
        if ext == ".ass":
            with open(src, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        else:
            content = ""

        # Extract [Events] section (everything from the [Events] heading onward)
        events_match = (
            re.search(
            r"\[Events\].*", content, re.DOTALL | re.IGNORECASE
            )
            if ext == ".ass" else None
        ) 
        if ext == ".ass" and not events_match:
            logger.warning("ASS missing [Events] section — falling back to raw copy")
            return copy_to_ascii_temp(src)

        if events_match:
            events_section = events_match.group()

        # Strip inline \fsN overrides so they can't fight the style fontsize
        events_section = re.sub(r"\\fs\d+", "", events_section)
        events_section = re.sub(r"\{\}",    "", events_section)

        # ── Build style values ────────────────────────────────────────────
        # actual_height = the pixel height of the stream the filter runs on.
        # Setting PlayResY = actual_height means Fontsize IS the pixel size.
        target_h    = max(1, actual_height or cfg.height)
        target_w    = max(1, cfg.width)
        user_fs     = cfg.subtitle_fontsize or 65
        font_name   = getattr(cfg, "subtitle_font",   "Arial") or "Arial"
        # ASS bold convention: -1 = bold, 0 = normal
        bold        = -1 if getattr(cfg, "subtitle_bold",   True)  else 0
        italic      = 1  if getattr(cfg, "subtitle_italic", False) else 0
        underline   = 1  if getattr(cfg, "subtitle_underline", False) else 0
        text_col    = _color_to_ass(cfg.subtitle_color            or "white")
        hilite_col  = _color_to_ass(cfg.subtitle_highlight_color  or "yellow")
        outline_col = "&H00000000"
        back_col    = "&H80000000"
        sub_y_pct   = getattr(cfg, "subtitle_y_pct", -1.0)

        # When user has dragged subtitle position, compute MarginV from y_pct
        if sub_y_pct >= 0:
            # Place text at y_pct of frame height — use bottom alignment (2)
            # and calculate MarginV = distance from bottom
            target_y_px = int(sub_y_pct * target_h)
            margin_v = max(0, target_h - target_y_px)
            alignment = 2  # bottom-centre — MarginV pushes up from bottom
        else:
            pos_map     = {"bottom": 2, "top": 8, "center": 5}
            alignment   = pos_map.get(cfg.subtitle_position or "bottom", 2)
            manual_mv   = getattr(cfg, "subtitle_margin_v", 0)
            margin_v    = manual_mv if manual_mv > 0 else max(10, round(target_h * 0.04))

        # ASS karaoke colour semantics:
        #   PrimaryColour   = colour the syllable animates INTO  (highlight)
        #   SecondaryColour = colour before that syllable starts (normal text)
        style_fmt = (
            "Style: {name},{font},{fs},{pri},{sec},"
            "{out},{back},{bold},{ital},{under},0,100,100,0,0,1,4,1,"
            "{align},50,50,{mv},1\n"
        )
        style_plain = style_fmt.format(
            name="Plain", font=font_name, fs=user_fs,
            pri=text_col, sec=hilite_col,
            out=outline_col, back=back_col,
            bold=bold, ital=italic, under=underline,
            align=alignment, mv=margin_v,
        )
        style_karaoke = style_fmt.format(
            name="Karaoke", font=font_name, fs=user_fs,
            pri=hilite_col, sec=text_col,   # swapped for karaoke
            out=outline_col, back=back_col,
            bold=bold, ital=italic, under=underline,
            align=alignment, mv=margin_v,
        )
        style_word = style_fmt.format(
            name="Word", font=font_name, fs=user_fs,
            pri=text_col, sec=hilite_col,
            out=outline_col, back=back_col,
            bold=bold, ital=italic, under=underline,
            align=alignment, mv=margin_v,
        )

        new_header = (
            "[Script Info]\n"
            "ScriptType: v4.00+\n"
            f"PlayResX: {target_w}\n"
            f"PlayResY: {target_h}\n"
            "Timer: 100.0000\n"
            "WrapStyle: 0\n"
            "\n"
            "[V4+ Styles]\n"
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
            "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
            "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
            "Alignment, MarginL, MarginR, MarginV, Encoding\n"
            + style_plain
            + style_karaoke
            + style_word
            + "\n"
        )

        new_content = new_header + events_section

        tmp_path = get_safe_temp_path(".ass")
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(new_content)

        logger.debug(
            f"ASS rebuilt: PlayResY={target_h}, Fontsize={user_fs}px, "
            f"font={font_name}, align={alignment}, margin_v={margin_v}"
        )
        return tmp_path

    except Exception as exc:
        logger.warning(f"ASS rebuild failed ({exc}) — falling back to raw copy")
        return copy_to_ascii_temp(src)


def _subtitle_force_style_srt(
    cfg: "ExportConfig",
    actual_height: int | None = None,
) -> str:
    """force_style string for SRT files only.

    For SRT, FFmpeg sets its internal PlayResY = frame_height, so
    Fontsize here equals rendered pixels 1:1 — no scaling required.
    actual_height should be the pixel height of the stream the filter runs on
    (may differ from cfg.height in 'shrink' mode).
    """
    h           = actual_height or cfg.height
    fs          = cfg.subtitle_fontsize or 65
    font_name   = getattr(cfg, "subtitle_font",   "Arial") or "Arial"
    bold        = 1 if getattr(cfg, "subtitle_bold",   True)  else 0
    italic      = 1 if getattr(cfg, "subtitle_italic", False) else 0
    underline   = 1 if getattr(cfg, "subtitle_underline", False) else 0
    primary     = _color_to_ass(cfg.subtitle_color           or "white")
    secondary   = _color_to_ass(cfg.subtitle_highlight_color or "yellow")
    pos_map     = {"bottom": 2, "top": 8, "center": 5}
    alignment   = pos_map.get(cfg.subtitle_position or "bottom", 2)
    manual_mv   = getattr(cfg, "subtitle_margin_v", 0)
    margin_v    = manual_mv if manual_mv > 0 else max(20, round(h * 0.03))

    return (
        f"Fontname={font_name},"
        f"Fontsize={fs},"
        f"Bold={bold},"
        f"Italic={italic},"
        f"Underline={underline},"
        f"PrimaryColour={primary},"
        f"SecondaryColour={secondary},"
        f"OutlineColour=&H00000000,"
        f"BackColour=&H80000000,"
        f"Outline=3,"
        f"Shadow=1,"
        f"Alignment={alignment},"
        f"MarginV={margin_v}"
    )


def _generate_custom_subtitle_ass(
    text: str,
    duration: float,
    cfg: "ExportConfig",
    actual_height: int | None = None,
) -> str:
    """Generate an ASS file with a single fixed-text dialogue line.

    Uses PlayResY = actual_height so Fontsize renders 1:1 in pixels.
    This avoids the SRT force_style bug where FFmpeg uses a default
    PlayResY (~384) causing enormous subtitle rendering.

    When subtitle_y_pct >= 0 (user dragged position), uses ASS \\pos
    override to place text at the exact vertical pixel coordinate.
    """
    from src.utils.file_utils import get_safe_temp_path

    target_h    = max(1, actual_height or cfg.height)
    target_w    = max(1, cfg.width)
    user_fs     = cfg.subtitle_fontsize or 65
    font_name   = getattr(cfg, "subtitle_font",   "Arial") or "Arial"
    bold        = -1 if getattr(cfg, "subtitle_bold",   True)  else 0
    italic      = 1  if getattr(cfg, "subtitle_italic", False) else 0
    underline   = 1  if getattr(cfg, "subtitle_underline", False) else 0
    text_col    = _color_to_ass(cfg.subtitle_color            or "white")
    hilite_col  = _color_to_ass(cfg.subtitle_highlight_color  or "yellow")
    outline_col = "&H00000000"
    back_col    = "&H80000000"
    sub_y_pct   = getattr(cfg, "subtitle_y_pct", -1.0)

    # When y_pct is set, use centre alignment (5) and \pos override
    if sub_y_pct >= 0:
        alignment = 5  # centre alignment — \pos will override placement
    else:
        pos_map = {"bottom": 2, "top": 8, "center": 5}
        alignment = pos_map.get(cfg.subtitle_position or "bottom", 2)

    manual_mv   = getattr(cfg, "subtitle_margin_v", 0)
    margin_v    = manual_mv if manual_mv > 0 else max(10, round(target_h * 0.04))

    is_word_style = (getattr(cfg, "subtitle_style", "") or "").lower() == "word"
    style_name = "Word" if is_word_style else "Plain"
    style_line = (
        f"Style: {style_name},{font_name},{user_fs},{text_col},{hilite_col},"
        f"{outline_col},{back_col},{bold},{italic},{underline},0,100,100,0,0,1,4,1,"
        f"{alignment},50,50,{margin_v},1\n"
    )

    duration = max(0.1, float(duration or 0.0))
    end_stamp = _ass_time(duration)
    pos_tag = ""
    if sub_y_pct >= 0:
        pos_x = target_w // 2
        pos_y = max(0, min(target_h, int(sub_y_pct * target_h)))
        pos_tag = f"{{\\pos({pos_x},{pos_y})}}"

    if is_word_style:
        words = [w for w in text.split() if w.strip()]
        step = duration / max(1, len(words))
        dialogue = []
        for idx, word in enumerate(words):
            start = idx * step
            end = duration if idx == len(words) - 1 else (idx + 1) * step
            if end <= start:
                end = start + 0.15
            dialogue.append(
                f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},"
                f"{style_name},,0,0,0,,{pos_tag}{_escape_ass_text(word)}\n"
            )
        if not dialogue:
            dialogue.append(
                f"Dialogue: 0,0:00:00.00,{end_stamp},"
                f"{style_name},,0,0,0,,{pos_tag}{_escape_ass_text(text)}\n"
            )
        event_lines = "".join(dialogue)
    else:
        escaped = f"{pos_tag}{_escape_ass_text(text)}"
        event_lines = (
            f"Dialogue: 0,0:00:00.00,{end_stamp},"
            f"{style_name},,0,0,0,,{escaped}\n"
        )

    content = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {target_w}\n"
        f"PlayResY: {target_h}\n"
        "Timer: 100.0000\n"
        "WrapStyle: 0\n"
        "\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        + style_line
        + "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        + event_lines
    )

    tmp_path = get_safe_temp_path(".ass")
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(content)

    logger.debug(
        f"Custom subtitle ASS: PlayResY={target_h}, Fontsize={user_fs}px, "
        f"text='{text[:40]}', duration={duration:.1f}s"
    )
    return tmp_path


def _refine_delogo_region(
    rx: float,
    ry: float,
    rw: float,
    rh: float,
    strength: int,
) -> tuple[float, float, float, float]:
    """Return the exact normalized region selected by the user.

    Older builds silently reduced selections wider than 85%, which made it
    impossible to remove subtitles spanning the whole frame. FFmpeg already
    receives a one-pixel safety margin below, so no extra shrinking is needed.
    """
    del strength  # Kept in the signature for compatibility with callers.
    return (
        max(0.0, min(1.0, float(rx))),
        max(0.0, min(1.0, float(ry))),
        max(0.05, min(1.0, float(rw))),
        max(0.05, min(1.0, float(rh))),
    )


def _watermark_filter_segments(
    cfg: "ExportConfig",
    in_pad: str = "0:v",
    out_pad: str = "wm_src",
    source_size: tuple[int, int] | None = None,
) -> list[str]:
    """Build source-space filters for every selected watermark rectangle."""
    if not getattr(cfg, "remove_watermark_enabled", False):
        return []
    regions = getattr(cfg, "remove_watermark_regions", None) or []
    if not regions:
        regions = [{
            "x": getattr(cfg, "remove_watermark_x", 0.82),
            "y": getattr(cfg, "remove_watermark_y", 0.88),
            "w": getattr(cfg, "remove_watermark_w", 0.16),
            "h": getattr(cfg, "remove_watermark_h", 0.08),
        }]
    mode = getattr(cfg, "remove_watermark_mode", "blur") or "blur"
    strength = max(1, min(5, int(getattr(cfg, "remove_watermark_strength", 3))))
    source_w, source_h = source_size or (
        int(getattr(cfg, "width", 1080) or 1080),
        int(getattr(cfg, "height", 1920) or 1920),
    )
    if getattr(cfg, "pre_crop_enabled", False):
        source_w = max(
            2, _even(source_w * float(getattr(cfg, "pre_crop_w", 1.0)))
        )
        source_h = max(
            2, _even(source_h * float(getattr(cfg, "pre_crop_h", 1.0)))
        )
    segments: list[str] = []
    current_pad = in_pad
    for index, region in enumerate(regions):
        if not isinstance(region, dict):
            continue
        rx = max(0.0, min(1.0, float(region.get("x", 0.82))))
        ry = max(0.0, min(1.0, float(region.get("y", 0.88))))
        rw = max(0.05, min(1.0, float(region.get("w", 0.16))))
        rh = max(0.05, min(1.0, float(region.get("h", 0.08))))
        x = f"trunc((iw-iw*{rw:.4f})*{rx:.4f}/2)*2"
        y = f"trunc((ih-ih*{rh:.4f})*{ry:.4f}/2)*2"
        ow = f"trunc(iw*{rw:.4f}/2)*2"
        oh = f"trunc(ih*{rh:.4f}/2)*2"
        ox = f"trunc((W-W*{rw:.4f})*{rx:.4f}/2)*2"
        oy = f"trunc((H-H*{rh:.4f})*{ry:.4f}/2)*2"
        next_pad = out_pad if index == len(regions) - 1 else f"wm_step_{index}"
        if mode == "cover":
            color = getattr(cfg, "remove_watermark_color", "black") or "black"
            segments.append(
                f"[{current_pad}]drawbox=x={x}:y={y}:w={ow}:h={oh}:"
                f"color={color}@1:t=fill[{next_pad}]"
            )
        elif mode == "delogo":
            # Reconstruct the selected rectangle from the pixels around its
            # edges.  Because this runs before subtitle rendering, translated
            # captions are drawn cleanly over the repaired moving background.
            # This FFmpeg build requires literal pixel coordinates for delogo;
            # unlike crop/overlay it does not expose iw/ih expression vars.
            rx, ry, rw, rh = _refine_delogo_region(
                rx, ry, rw, rh, strength
            )
            pixel_w = min(source_w - 2, max(4, _even(source_w * rw)))
            pixel_h = min(source_h - 2, max(4, _even(source_h * rh)))
            pixel_x = int(round((source_w - pixel_w) * rx))
            pixel_y = int(round((source_h - pixel_h) * ry))
            pixel_x = max(1, min(source_w - pixel_w - 1, pixel_x))
            pixel_y = max(1, min(source_h - pixel_h - 1, pixel_y))
            segments.append(
                f"[{current_pad}]delogo=x={pixel_x}:y={pixel_y}:"
                f"w={pixel_w}:h={pixel_h}:"
                f"show=0[{next_pad}]"
            )
        else:
            base = f"wm_base_{index}"
            area = f"wm_area_{index}"
            patch = f"wm_patch_{index}"
            segments.append(f"[{current_pad}]split[{base}][{area}]")
            if mode == "mosaic":
                block = {1: 6, 2: 8, 3: 12, 4: 16, 5: 24}[strength]
                segments.append(
                    f"[{area}]crop=w={ow}:h={oh}:x={x}:y={y},"
                    f"scale=trunc(iw/{block}/2)*2:trunc(ih/{block}/2)*2,"
                    f"scale=iw*{block}:ih*{block}:flags=neighbor[{patch}]"
                )
            else:
                blur = ",".join(["boxblur=lr=4:lp=1:cr=4:cp=1"] * strength)
                segments.append(
                    f"[{area}]crop=w={ow}:h={oh}:x={x}:y={y},{blur}[{patch}]"
                )
            segments.append(
                f"[{base}][{patch}]overlay=x={ox}:y={oy}[{next_pad}]"
            )
        current_pad = next_pad
    return segments


def _source_precrop_segments(cfg: "ExportConfig", in_pad: str = "0:v", out_pad: str = "pre_src") -> list[str]:
    """Crop/zoom the source before all export aspect processing."""
    if not getattr(cfg, "pre_crop_enabled", False):
        return []
    cx = max(0.0, min(1.0, getattr(cfg, "pre_crop_x", 0.5)))
    cy = max(0.0, min(1.0, getattr(cfg, "pre_crop_y", 0.5)))
    cw = max(0.05, min(1.0, getattr(cfg, "pre_crop_w", 1.0)))
    ch = max(0.05, min(1.0, getattr(cfg, "pre_crop_h", 1.0)))
    return [
        f"[{in_pad}]crop=w=trunc(iw*{cw:.4f}/2)*2:h=trunc(ih*{ch:.4f}/2)*2"
        f":x=trunc((iw-iw*{cw:.4f})*{cx:.4f})"
        f":y=trunc((ih-ih*{ch:.4f})*{cy:.4f})[{out_pad}]"
    ]


def _build_image_bg_filter(
    cfg: "ExportConfig",
    clip: "Clip",
    has_subtitle: bool,
    subtitle_path: str,
    bg_input: str = "1:v",
    source_size: tuple[int, int] | None = None,
) -> str:
    """Compose the video over a full-frame custom background image.

    The image is input index 1 (added by export_clip as `-loop 1 -i <img>`).
    Layout:
      • shrink part-text  → title in a top band (header_h), video below it
      • other part modes  → video centred, title overlaid on the video
    Subtitles (non-bar) are burned onto the video region; a "bar" subtitle
    is overlaid as a coloured strip at the bottom.
    """
    w = _even(cfg.width)
    h = _even(cfg.height)
    header_h = min(_even(cfg.header_height), h - 2)
    sub_style = cfg.subtitle_style or "none"
    is_bar = sub_style in ("bar", "shrink_bar")
    bar_h = _even(min(getattr(cfg, "subtitle_bar_height", 200), h // 2)) if is_bar else 0
    part_mode = cfg.part_text_mode if cfg.part_text_enabled else None
    text_color = cfg.part_text_color or "white"
    fs = max(2, int(cfg.part_text_fontsize))

    parts = []
    pre_segments = _source_precrop_segments(cfg)
    if pre_segments:
        parts.extend(pre_segments)
        src = "pre_src"
    else:
        src = "0:v"
    wm_segments = _watermark_filter_segments(
        cfg, in_pad=src, source_size=source_size
    )
    if wm_segments:
        parts.extend(wm_segments)
        src = "wm_src"

    # 1) Full-frame background from the image (cover + crop, square pixels)
    parts.append(
        f"[{bg_input}]scale={w}:{h}:force_original_aspect_ratio=increase,"
        f"crop={w}:{h},setsar=1[bg]"
    )

    # 2) Determine the video drawing area and vertical offset
    if part_mode == "shrink":
        vid_area_h = _even(h - header_h - bar_h)
        # centre the scaled video within the band below the header
        # ('h' here is FFmpeg's overlay-input height variable)
        y_off = f"{header_h}+({vid_area_h}-h)/2"
    else:
        vid_area_h = _even(h - bar_h)
        # centre the scaled video within the area above the bottom bar
        y_off = f"(({h}-{bar_h})-h)/2"
    vid_area_h = max(2, vid_area_h)

    # 3) Scale video to fit inside (w × vid_area_h), keeping aspect
    parts.append(
        f"[{src}]scale={w}:{vid_area_h}:force_original_aspect_ratio=decrease[fgv]"
    )
    fg = "fgv"

    # 4) Overlay video onto background
    parts.append(f"[bg][{fg}]overlay=(W-w)/2:{y_off}:shortest=1[comp]")
    cur = "comp"

    # 5) Title text — draggable (part_text_y_pct) anywhere on the frame,
    #    otherwise centred in the header band (shrink) or by preset position.
    raw_header = getattr(clip, "custom_header", "").strip() or clip.part_text
    if cfg.part_text_enabled and raw_header:
        part_y_pct = getattr(cfg, "part_text_y_pct", -1.0)
        if part_y_pct >= 0:
            y_expr = f"trunc({part_y_pct:.4f}*h-th/2)"
        elif part_mode == "shrink":
            y_expr = f"({header_h}-th)/2"
        else:
            pos = cfg.part_text_position
            y_expr = "50" if pos == "top" else (
                "h-th-50" if pos == "bottom" else "(h-th)/2")
        parts.append(_drawtext_expr(
            raw_header, in_pad=cur, out_pad="titled",
            fontsize=fs, fontcolor=text_color,
            **_part_text_style_args(cfg),
            x="(w-tw)/2", y=y_expr,
            borderw=4, bordercolor="black",
        ))
        cur = "titled"

    # 6) Non-bar subtitles composited over the FULL frame so they can be
    #    positioned in the empty area below the video (matches preview).
    if has_subtitle and not is_bar:
        safe_path = (subtitle_path
                     .replace("\\", "/")
                     .replace(":", "\\:")
                     .replace(" ", "\\ "))
        if subtitle_path.lower().endswith(".srt"):
            fstyle = _subtitle_force_style_srt(cfg, actual_height=h)
            parts.append(
                f"[{cur}]subtitles='{safe_path}':force_style='{fstyle}'[subbed]"
            )
        else:
            parts.append(f"[{cur}]subtitles='{safe_path}'[subbed]")
        cur = "subbed"

    # 7) Bar subtitle strip at bottom
    if is_bar and bar_h > 0:
        sub_bg = getattr(cfg, "subtitle_bg_color", "white") or "white"
        sub_fs = max(2, cfg.subtitle_fontsize or 65)
        sub_tc = cfg.subtitle_color or "black"
        custom_sub = getattr(clip, "custom_subtitle", "") or ""
        if custom_sub:
            parts.append(f"color=c={sub_bg}:s={w}x{bar_h}:r=30[bar_bg]")
            parts.append(_drawtext_expr(
                custom_sub, in_pad="bar_bg", out_pad="subbar",
                fontsize=sub_fs, fontcolor=sub_tc,
                **_subtitle_drawtext_style_args(cfg),
                x="(w-tw)/2", y="(h-th)/2",
            ))
        elif has_subtitle:
            safe_path = (subtitle_path
                         .replace("\\", "/")
                         .replace(":", "\\:")
                         .replace(" ", "\\ "))
            parts.append(f"color=c={sub_bg}:s={w}x{bar_h}:r=30[bar_bg0]")
            parts.append(f"[bar_bg0]subtitles='{safe_path}'[subbar]")
        else:
            parts.append(f"color=c={sub_bg}:s={w}x{bar_h}:r=30[subbar]")
        parts.append(f"[{cur}][subbar]overlay=0:{h - bar_h}:shortest=1[withbar]")
        cur = "withbar"

    # 8) Ensure final output pad is [out]
    if cur != "out":
        parts.append(f"[{cur}]copy[out]")

    graph = ";".join(parts)
    logger.debug(f"image_bg filter_complex: {graph}")
    return graph


def _build_video_filter(
    cfg: "ExportConfig",
    clip: "Clip",
    has_subtitle: bool,
    subtitle_path: str,
    source_size: tuple[int, int] | None = None,
) -> str:
    """Build the -filter_complex filter graph string for a clip export."""
    # Custom background image mode — delegate to a dedicated builder that
    # composes the video over a full-frame image (input index 1).
    if cfg.aspect_mode == "image_bg":
        bg_path = getattr(cfg, "background_image", "") or ""
        if bg_path and Path(bg_path).exists():
            return _build_image_bg_filter(
                cfg, clip, has_subtitle, subtitle_path,
                source_size=source_size,
            )
        # No valid image → fall back to black background behaviour below.

    w = _even(cfg.width)
    h = _even(cfg.height)
    header_h = _even(cfg.header_height)
    header_h = min(header_h, h - 2)
    video_h = _even(h - header_h)  # height for video in shrink mode
    header_h = h - video_h         # keep final vstack height exactly h

    # Subtitle bar mode: reserve space for a coloured bar
    sub_style = cfg.subtitle_style or "none"
    is_bar_sub = sub_style in ("bar", "shrink_bar")
    bar_h = 0
    if is_bar_sub:
        bar_h = _even(min(getattr(cfg, "subtitle_bar_height", 200), h // 2))

    mode = cfg.aspect_mode
    part_mode = cfg.part_text_mode if cfg.part_text_enabled else None
    # Raw text — _drawtext_expr handles escaping / multiline internally
    text_color = cfg.part_text_color or "white"
    fs = max(2, int(cfg.part_text_fontsize))

    parts = []  # filter graph segments, joined with ";"
    pre_segments = _source_precrop_segments(cfg)
    if pre_segments:
        parts.extend(pre_segments)
        src = "pre_src"
    else:
        src = "0:v"
    wm_segments = _watermark_filter_segments(
        cfg, in_pad=src, source_size=source_size
    )
    if wm_segments:
        parts.extend(wm_segments)
        src = "wm_src"

    if part_mode == "shrink":
        # Adjust video_h if bar subtitle eats space too
        actual_vid_h = _even(video_h - bar_h) if is_bar_sub else video_h

        # Scale video to fill the video portion (actual_vid_h height)
        if mode == "keep_ratio":
            parts.append(
                f"[{src}]scale={w}:{actual_vid_h}:force_original_aspect_ratio=decrease,"
                f"pad={w}:{actual_vid_h}:(ow-iw)/2:(oh-ih)/2:black[vid_scaled]"
            )
        elif mode == "center_crop":
            parts.append(
                f"[{src}]scale={w}:{actual_vid_h}:force_original_aspect_ratio=increase,"
                f"crop={w}:{actual_vid_h}[vid_scaled]"
            )
        elif mode == "blur_bg":
            parts.append(f"[{src}]split[va][vb]")
            parts.append(
                f"[va]scale={w}:{actual_vid_h}:force_original_aspect_ratio=increase,"
                f"crop={w}:{actual_vid_h},boxblur=25:5[blurred]"
            )
            parts.append(
                f"[vb]scale={w}:{actual_vid_h}:force_original_aspect_ratio=decrease[fg]"
            )
            parts.append(f"[blurred][fg]overlay=(W-w)/2:(H-h)/2[vid_scaled]")
        elif mode == "custom_crop":
            cx = max(0.0, min(1.0, getattr(cfg, "crop_x", 0.5)))
            cy = max(0.0, min(1.0, getattr(cfg, "crop_y", 0.5)))
            cw = max(0.05, min(1.0, getattr(cfg, "crop_w", 1.0)))
            ch = max(0.05, min(1.0, getattr(cfg, "crop_h", 1.0)))
            parts.append(
                f"[{src}]crop=w=trunc(iw*{cw:.4f}/2)*2:h=trunc(ih*{ch:.4f}/2)*2"
                f":x=trunc((iw-iw*{cw:.4f})*{cx:.4f})"
                f":y=trunc((ih-ih*{ch:.4f})*{cy:.4f}),"
                f"scale={w}:{actual_vid_h}:force_original_aspect_ratio=increase,"
                f"crop={w}:{actual_vid_h}[vid_scaled]"
            )
        else:  # black_bg
            parts.append(
                f"[{src}]scale={w}:{actual_vid_h}:force_original_aspect_ratio=decrease,"
                f"pad={w}:{actual_vid_h}:(ow-iw)/2:(oh-ih)/2:black[vid_scaled]"
            )

        vid_out = "vid_scaled"
        # For non-bar subtitles, burn directly onto video
        if has_subtitle and not is_bar_sub:
            safe_path = (subtitle_path
                         .replace("\\", "/")
                         .replace(":", "\\:")
                         .replace(" ", "\\ "))
            if subtitle_path.lower().endswith(".srt"):
                fstyle = _subtitle_force_style_srt(cfg, actual_height=actual_vid_h)
                parts.append(
                    f"[{vid_out}]subtitles='{safe_path}':force_style='{fstyle}'[vid_subbed]"
                )
            else:
                parts.append(
                    f"[{vid_out}]subtitles='{safe_path}'[vid_subbed]"
                )
            vid_out = "vid_subbed"

        # Build header band (solid colour only). The title is overlaid AFTER
        # stacking so it can be dragged anywhere via part_text_y_pct.
        bg_color = cfg.part_text_bg_color or "black"
        raw_header = getattr(clip, "custom_header", "").strip() or clip.part_text
        has_header_text = bool(cfg.part_text_enabled and raw_header)
        hdr_col = bg_color if has_header_text else "black"
        parts.append(f"color=c={hdr_col}:s={w}x{header_h}:r=30[header]")

        # Build subtitle bar (if bar mode)
        if is_bar_sub and bar_h > 0:
            sub_bg = getattr(cfg, "subtitle_bg_color", "white") or "white"
            sub_fs = max(2, cfg.subtitle_fontsize or 65)
            sub_tc = cfg.subtitle_color or "black"
            custom_sub = getattr(clip, "custom_subtitle", "") or ""
            if custom_sub:
                parts.append(f"color=c={sub_bg}:s={w}x{bar_h}:r=30[bar_bg]")
                parts.append(_drawtext_expr(
                    custom_sub, in_pad="bar_bg", out_pad="subbar",
                    fontsize=sub_fs, fontcolor=sub_tc,
                    **_subtitle_drawtext_style_args(cfg),
                    x="(w-tw)/2", y="(h-th)/2",
                ))
            elif has_subtitle:
                # Burn transcript subtitles onto the coloured bar
                safe_path = (subtitle_path
                             .replace("\\", "/")
                             .replace(":", "\\:")
                             .replace(" ", "\\ "))
                parts.append(f"color=c={sub_bg}:s={w}x{bar_h}:r=30[bar_bg0]")
                parts.append(
                    f"[bar_bg0]subtitles='{safe_path}'[subbar]"
                )
            else:
                parts.append(f"color=c={sub_bg}:s={w}x{bar_h}:r=30[subbar]")

            # Stack: header + video + bar (3 inputs)
            parts.append(
                f"[header][{vid_out}][subbar]vstack=inputs=3:shortest=1[stacked]"
            )
        else:
            # Stack: header + video (2 inputs)
            parts.append(f"[header][{vid_out}]vstack=inputs=2:shortest=1[stacked]")

        # Overlay the title on the stacked frame (draggable). Default position
        # is centred inside the header band; a dragged y_pct places it freely.
        if has_header_text:
            part_y_pct = getattr(cfg, "part_text_y_pct", -1.0)
            if part_y_pct >= 0:
                y_expr = f"trunc({part_y_pct:.4f}*h-th/2)"
            else:
                y_expr = f"({header_h}-th)/2"
            parts.append(_drawtext_expr(
                raw_header, in_pad="stacked", out_pad="out",
                fontsize=fs, fontcolor=text_color,
                **_part_text_style_args(cfg),
                x="(w-tw)/2", y=y_expr,
            ))
        else:
            parts.append("[stacked]copy[out]")

    else:  # overlay / watermark / lower_third / outline — all use full-height video
        # When bar subtitle is active, shrink video to make room for bar
        full_h = _even(h - bar_h) if is_bar_sub else h

        if mode == "keep_ratio":
            parts.append(
                f"[{src}]scale={w}:{full_h}:force_original_aspect_ratio=decrease,"
                f"pad={w}:{full_h}:(ow-iw)/2:(oh-ih)/2:black[vid_full]"
            )
        elif mode == "center_crop":
            parts.append(
                f"[{src}]scale={w}:{full_h}:force_original_aspect_ratio=increase,"
                f"crop={w}:{full_h}[vid_full]"
            )
        elif mode == "blur_bg":
            parts.append(f"[{src}]split[va][vb]")
            parts.append(
                f"[va]scale={w}:{full_h}:force_original_aspect_ratio=increase,"
                f"crop={w}:{full_h},boxblur=25:5[blurred]"
            )
            parts.append(
                f"[vb]scale={w}:{full_h}:force_original_aspect_ratio=decrease[fg]"
            )
            parts.append(f"[blurred][fg]overlay=(W-w)/2:(H-h)/2[vid_full]")
        elif mode == "custom_crop":
            cx = max(0.0, min(1.0, getattr(cfg, "crop_x", 0.5)))
            cy = max(0.0, min(1.0, getattr(cfg, "crop_y", 0.5)))
            cw = max(0.05, min(1.0, getattr(cfg, "crop_w", 1.0)))
            ch = max(0.05, min(1.0, getattr(cfg, "crop_h", 1.0)))
            parts.append(
                f"[{src}]crop=w=trunc(iw*{cw:.4f}/2)*2:h=trunc(ih*{ch:.4f}/2)*2"
                f":x=trunc((iw-iw*{cw:.4f})*{cx:.4f})"
                f":y=trunc((ih-ih*{ch:.4f})*{cy:.4f}),"
                f"scale={w}:{full_h}:force_original_aspect_ratio=increase,"
                f"crop={w}:{full_h}[vid_full]"
            )
        else:
            parts.append(
                f"[{src}]scale={w}:{full_h}:force_original_aspect_ratio=decrease,"
                f"pad={w}:{full_h}:(ow-iw)/2:(oh-ih)/2:black[vid_full]"
            )

        vid_out = "vid_full"
        # For non-bar subtitles, burn directly onto video
        if has_subtitle and not is_bar_sub:
            safe_path = (subtitle_path
                         .replace("\\", "/")
                         .replace(":", "\\:")
                         .replace(" ", "\\ "))
            if subtitle_path.lower().endswith(".srt"):
                fstyle = _subtitle_force_style_srt(cfg, actual_height=full_h)
                parts.append(
                    f"[{vid_out}]subtitles='{safe_path}':force_style='{fstyle}'[vid_subbed]"
                )
            else:
                parts.append(
                    f"[{vid_out}]subtitles='{safe_path}'[vid_subbed]"
                )
            vid_out = "vid_subbed"

        # Per-clip custom header text
        raw_custom = getattr(clip, "custom_header", "").strip() or clip.part_text

        # Apply part text overlay modes (on the video portion)
        if not cfg.part_text_enabled or not raw_custom:
            parts.append(f"[{vid_out}]copy[vid_txt]")

        elif part_mode == "lower_third":
            lt_h = min(160, int(full_h * 0.085))
            bg_color = cfg.part_text_bg_color or "black"
            small_fs = int(fs * 0.85)
            text_y = int(lt_h * 0.65)
            # drawbox and drawtext must be in separate filterchains so the
            # output pad label is unambiguous to FFmpeg's parser.
            parts.append(
                f"[{vid_out}]"
                f"drawbox=x=0:y=ih-{lt_h}:w=iw:h={lt_h}:color={bg_color}@0.82:t=fill"
                f"[_lt_bg]"
            )
            parts.append(_drawtext_expr(
                raw_custom, in_pad="_lt_bg", out_pad="vid_txt",
                fontsize=small_fs, fontcolor=text_color,
                **_part_text_style_args(cfg),
                x="(w-tw)/2", y=f"h-{text_y}",
            ))

        elif part_mode == "watermark":
            opacity = getattr(cfg, "watermark_opacity", 0.35)
            shadow_op = round(min(opacity * 0.8, 0.9), 2)
            part_y_pct = getattr(cfg, "part_text_y_pct", -1.0)
            if part_y_pct >= 0:
                y_expr = f"trunc({part_y_pct:.4f}*h-th/2)"
            else:
                pos = cfg.part_text_position
                y_expr = "60" if pos == "top" else ("h-th-60" if pos == "bottom" else "(h-th)/2")
            parts.append(_drawtext_expr(
                raw_custom, in_pad=vid_out, out_pad="vid_txt",
                fontsize=fs,
                fontcolor=f"{text_color}@{opacity}",
                **_part_text_style_args(cfg),
                x="(w-tw)/2", y=y_expr,
                shadowx=3, shadowy=3,
                shadowcolor=f"black@{shadow_op}",
            ))

        elif part_mode == "outline":
            part_y_pct = getattr(cfg, "part_text_y_pct", -1.0)
            if part_y_pct >= 0:
                y_expr = f"trunc({part_y_pct:.4f}*h-th/2)"
            else:
                pos = cfg.part_text_position
                y_expr = "60" if pos == "top" else ("h-th-60" if pos == "bottom" else "(h-th)/2")
            parts.append(_drawtext_expr(
                raw_custom, in_pad=vid_out, out_pad="vid_txt",
                fontsize=fs, fontcolor=text_color,
                **_part_text_style_args(cfg),
                x="(w-tw)/2", y=y_expr,
                borderw=5, bordercolor="black",
            ))

        else:  # overlay (default)
            part_y_pct = getattr(cfg, "part_text_y_pct", -1.0)
            if part_y_pct >= 0:
                y_expr = f"trunc({part_y_pct:.4f}*h-th/2)"
            else:
                pos = cfg.part_text_position
                y_expr = "50" if pos == "top" else ("h-th-50" if pos == "bottom" else "(h-th)/2")
            parts.append(_drawtext_expr(
                raw_custom, in_pad=vid_out, out_pad="vid_txt",
                fontsize=fs, fontcolor=text_color,
                **_part_text_style_args(cfg),
                x="(w-tw)/2", y=y_expr,
                box=1, boxcolor="black@0.5", boxborderw=10,
            ))

        # If bar subtitle mode, stack video + bar
        if is_bar_sub and bar_h > 0:
            sub_bg = getattr(cfg, "subtitle_bg_color", "white") or "white"
            sub_fs = max(2, cfg.subtitle_fontsize or 65)
            sub_tc = cfg.subtitle_color or "black"
            custom_sub = getattr(clip, "custom_subtitle", "") or ""
            if custom_sub:
                parts.append(f"color=c={sub_bg}:s={w}x{bar_h}:r=30[bar_bg]")
                parts.append(_drawtext_expr(
                    custom_sub, in_pad="bar_bg", out_pad="subbar",
                    fontsize=sub_fs, fontcolor=sub_tc,
                    **_subtitle_drawtext_style_args(cfg),
                    x="(w-tw)/2", y="(h-th)/2",
                ))
            elif has_subtitle:
                safe_path = (subtitle_path
                             .replace("\\", "/")
                             .replace(":", "\\:")
                             .replace(" ", "\\ "))
                parts.append(f"color=c={sub_bg}:s={w}x{bar_h}:r=30[bar_bg0]")
                parts.append(
                    f"[bar_bg0]subtitles='{safe_path}'[subbar]"
                )
            else:
                parts.append(f"color=c={sub_bg}:s={w}x{bar_h}:r=30[subbar]")
            parts.append(f"[vid_txt][subbar]vstack=inputs=2:shortest=1[out]")
        else:
            # No bar — rename vid_txt to out
            # Replace the last [vid_txt] tag with [out]
            parts[-1] = parts[-1].replace("[vid_txt]", "[out]")

    graph = ";".join(parts)
    logger.debug(f"filter_complex: {graph}")
    return graph


def extract_thumbnail(
    video_path: str,
    out_path: str,
    time_offset: float = 30.0,
) -> bool:
    """Extract a single 270×480 portrait frame for in-app preview."""
    ff = ffmpeg_path()
    if not ff:
        return False
    ensure_dir(Path(out_path).parent)
    cmd = [
        ff, "-y",
        "-ss", str(time_offset),
        "-i", video_path,
        "-vframes", "1",
        "-vf", "scale=270:480:force_original_aspect_ratio=increase,crop=270:480",
        out_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=30)
        return result.returncode == 0 and Path(out_path).exists()
    except Exception as e:
        logger.warning(f"Thumbnail extraction failed: {e}")
        return False


def extract_wide_thumbnail(
    video_path: str,
    out_path: str,
    time_offset: float = 30.0,
) -> bool:
    """Extract a wide frame at original aspect ratio (480px wide) for the crop tool."""
    ff = ffmpeg_path()
    if not ff:
        return False
    ensure_dir(Path(out_path).parent)
    cmd = [
        ff, "-y",
        "-ss", str(time_offset),
        "-i", video_path,
        "-vframes", "1",
        "-vf", "scale=480:-2",
        out_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=30)
        return result.returncode == 0 and Path(out_path).exists()
    except Exception as e:
        logger.warning(f"Wide thumbnail extraction failed: {e}")
        return False


def export_clip(
    project: Project,
    clip: Clip,
    cfg: ExportConfig,
    output_path: str,
    progress_cb: Optional[Callable[[float, float], None]] = None,
) -> tuple[bool, str]:
    """Export a single clip with all settings applied."""
    ff = _ff()
    source_video = getattr(clip, "source_video", "") or project.source_video
    music_path = getattr(cfg, "background_music_path", "") or ""
    use_background_music = bool(
        getattr(cfg, "background_music_enabled", False)
        and music_path
        and Path(music_path).exists()
    )

    # Resolve subtitle file
    # subtitle_style == "none"  → user explicitly disabled subtitle burn-in
    # Otherwise use per-clip subtitle file (set by Tab 3) or global fallback
    has_subtitle = False
    subtitle_path = ""
    tmp_srt = ""

    # Determine the ACTUAL height of the video stream that the subtitle filter
    # will be applied to.  In "shrink" mode the header bar eats some pixels;
    # in "bar" subtitle mode the bar eats pixels from the bottom;
    # in all other modes the whole frame is available.
    _part_mode = cfg.part_text_mode if cfg.part_text_enabled else None
    _sub_style = cfg.subtitle_style or "none"
    _base_h = _even(cfg.height)
    if cfg.aspect_mode == "image_bg":
        # image_bg composites the subtitle over the FULL frame so it can sit
        # in the empty area below the video — don't subtract the header.
        if _sub_style in ("bar", "shrink_bar"):
            _bar_h = _even(getattr(cfg, "subtitle_bar_height", 200))
            _base_h = _even(_base_h - _bar_h)
    else:
        if _part_mode == "shrink":
            _base_h = _even(cfg.height) - _even(cfg.header_height)
        if _sub_style in ("bar", "shrink_bar"):
            _bar_h = _even(getattr(cfg, "subtitle_bar_height", 200))
            _base_h = _even(_base_h - _bar_h)
    subtitle_canvas_height = _even(max(100, _base_h))

    # Render subtitles when: style is not "none" OR clip has custom text
    custom_sub_text = getattr(clip, "custom_subtitle", "") or ""
    want_sub = (_sub_style != "none") or bool(custom_sub_text)
    logger.info(
        f"Subtitle check: style={_sub_style}, custom_text='{custom_sub_text}', "
        f"want_sub={want_sub}, canvas_h={subtitle_canvas_height}"
    )
    if want_sub:
        if custom_sub_text:
            # User entered fixed subtitle text for this clip.
            # Generate an ASS file directly (NOT SRT) with proper PlayResY
            # so Fontsize = rendered pixels exactly (no scaling).
            tmp_srt = _generate_custom_subtitle_ass(
                custom_sub_text, clip.duration, cfg,
                actual_height=subtitle_canvas_height,
            )
            has_subtitle = True
            subtitle_path = tmp_srt
            logger.debug(f"Custom subtitle ASS: '{custom_sub_text}' → {tmp_srt}")
        else:
            srt_src = clip.subtitle_file or (
                cfg.global_subtitle_file if cfg.subtitle_enabled else ""
            )
            if srt_src and Path(srt_src).exists():
                # _prepare_subtitle_for_export:
                #   • SRT  → converted to ASS with PlayResY = actual_height
                #   • ASS  → rebuilds header with PlayResY = actual_height
                #   → Fontsize = cfg.subtitle_fontsize renders 1:1 pixels
                tmp_srt = _prepare_subtitle_for_export(
                    srt_src, cfg, actual_height=subtitle_canvas_height
                )
                if " " in tmp_srt:
                    logger.warning(
                        f"Subtitle temp path contains spaces: {tmp_srt}. "
                        "Subtitle burn-in may fail. Set TEMP=C:\\Temp to fix."
                    )
                has_subtitle = True
                subtitle_path = tmp_srt

    try:
        source_meta = get_video_metadata(source_video) or {}
        source_size = (
            int(source_meta.get("width", 0) or 0),
            int(source_meta.get("height", 0) or 0),
        )
        if source_size[0] <= 0 or source_size[1] <= 0:
            source_size = None
        fc = _build_video_filter(
            cfg, clip, has_subtitle, subtitle_path,
            source_size=source_size,
        )
        ensure_dir(Path(output_path).parent)

        use_gpu = getattr(cfg, "use_gpu", False)

        # Custom background image → add as a looped second input ([1:v])
        bg_path = getattr(cfg, "background_image", "") or ""
        use_image_bg = (
            cfg.aspect_mode == "image_bg"
            and bg_path and Path(bg_path).exists()
        )
        if use_image_bg:
            bg_safe = copy_to_ascii_temp(bg_path) if not bg_path.isascii() else bg_path
        else:
            bg_safe = ""

        source_has_audio = bool(
            source_meta.get(
                "has_audio",
                getattr(project, "video_metadata", {}).get("has_audio", True),
            )
        )
        music_volume = max(
            0.0,
            min(1.0, float(getattr(cfg, "background_music_volume", 15)) / 100.0),
        )
        music_input_index = 2 if use_image_bg else 1
        if use_background_music:
            music_chain = (
                f"[{music_input_index}:a]volume={music_volume:.3f},"
                f"atrim=duration={clip.duration:.3f},"
                "asetpts=PTS-STARTPTS[bg_music]"
            )
            if source_has_audio:
                music_chain += (
                    ";[0:a]asetpts=PTS-STARTPTS[main_audio]"
                    ";[main_audio][bg_music]amix=inputs=2:duration=first:"
                    "dropout_transition=2:normalize=0,alimiter=limit=0.95[audio_out]"
                )
            else:
                music_chain += ";[bg_music]anull[audio_out]"
            fc = f"{fc};{music_chain}"

        def _build_cmd(gpu: bool, low_memory: bool = False) -> list[str]:
            preset = _normalize_preset(cfg.preset, gpu)
            if gpu:
                enc = "h264_nvenc"
                q_args = [
                    "-preset", preset,
                    "-cq", str(cfg.crf),
                    "-profile:v", "high",
                    "-pix_fmt", "yuv420p",
                ]
            else:
                enc = "libx264"
                if low_memory:
                    q_args = [
                        "-preset", "ultrafast",
                        "-tune", "zerolatency",
                        "-crf", str(cfg.crf),
                        "-threads", str(_cpu_thread_count(low_memory=True)),
                        "-profile:v", "high",
                        "-pix_fmt", "yuv420p",
                    ]
                else:
                    q_args = [
                        "-preset", preset,
                        "-crf", str(cfg.crf),
                        "-threads", str(_cpu_thread_count()),
                        "-profile:v", "high",
                        "-pix_fmt", "yuv420p",
                    ]
            bg_input = ["-loop", "1", "-i", bg_safe] if use_image_bg else []
            music_input = (
                ["-stream_loop", "-1", "-i", music_path]
                if use_background_music else []
            )
            audio_map = (
                ["-map", "[audio_out]"]
                if use_background_music else ["-map", "0:a?"]
            )
            return [
                ff,
                "-y",
                "-ss", str(clip.start_time),
                "-to", str(clip.end_time),
                "-i", source_video,
                *bg_input,
                *music_input,
                "-filter_complex", fc,
                "-map", "[out]",
                *audio_map,
                "-c:v", enc,
                *q_args,
                "-r", "30",              # force CFR 30fps (fixes VFR sources
                                         # that make x264 see a huge framerate)
                "-c:a", "aac",
                "-b:a", cfg.audio_bitrate,
                "-movflags", "+faststart",
                output_path,
            ]

        duration = clip.duration
        cmd = _build_cmd(gpu=use_gpu)
        ok, err = _run_ffmpeg(cmd, duration, progress_cb)

        # Auto-fallback: if GPU failed, retry with CPU encoder
        if not ok and use_gpu:
            logger.warning(
                "NVENC export failed, falling back to CPU (libx264). "
                f"Error: {err[:200]}"
            )
            cmd = _build_cmd(gpu=False)
            ok, err = _run_ffmpeg(cmd, duration, progress_cb)
            if ok:
                err = "[GPU→CPU fallback] Xuất thành công bằng CPU (libx264). " \
                      "Cập nhật driver NVIDIA ≥570.0 để dùng GPU."
        if not ok and _is_x264_memory_error(err):
            logger.warning(
                "libx264 failed to allocate memory; retrying with low-memory settings."
            )
            cmd = _build_cmd(gpu=False, low_memory=True)
            ok, low_mem_err = _run_ffmpeg(cmd, duration, progress_cb)
            if ok:
                err = (
                    "[CPU low-memory fallback] Xuất thành công bằng libx264 "
                    "với 1 thread/ultrafast vì lần đầu thiếu RAM."
                )
            else:
                err = low_mem_err
        return ok, err
    finally:
        if tmp_srt and Path(tmp_srt).exists():
            try:
                Path(tmp_srt).unlink()
            except Exception:
                pass


def generate_preview(
    project: Project,
    clip: Clip,
    progress_cb: Optional[Callable[[float, float], None]] = None,
) -> tuple[bool, str]:
    """Generate a low-quality preview clip (no overlays, fast)."""
    ff = _ff()
    source_video = getattr(clip, "source_video", "") or project.source_video
    previews_dir = Path(project.output_dir) / "previews"
    ensure_dir(previews_dir)
    out_path = str(previews_dir / f"preview_part{clip.index:02d}.mp4")

    cmd = [
        ff, "-y",
        "-ss", str(clip.start_time),
        "-to", str(clip.end_time),
        "-i", source_video,
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "28",
        "-profile:v", "high",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "96k",
        "-vf", "scale=480:-2",
        out_path,
    ]
    ok, err = _run_ffmpeg(cmd, clip.duration, progress_cb)
    if ok:
        clip.preview_path = out_path
    return ok, err


def generate_caption(project: Project, clip: Clip, platform: str) -> dict:
    """Generate basic caption and hashtags for a clip."""
    name = project.name
    idx = clip.index
    caption = f"🎬 {name} - PART {idx}\n\nTheo dõi để xem phần tiếp theo! 🔥"
    base_tags = [
        f"#{name.replace(' ', '')}",
        f"#part{idx}",
        "#shorts",
        "#phim",
        "#review",
    ]
    platform_tags = {
        "TikTok": ["#tiktok", "#filmtok", "#phimhay"],
        "YouTube Shorts": ["#youtubeshorts", "#shorts"],
        "Facebook Reels": ["#facebookreels", "#reels"],
        "Instagram Reels": ["#instagramreels", "#reels", "#instagram"],
    }
    extra = platform_tags.get(platform, [])
    hashtags = " ".join(base_tags + extra)
    return {
        "index": idx,
        "platform": platform,
        "caption": caption,
        "hashtags": hashtags,
    }


# ─── Concatenate exported clips into one video ──────────────────────────────

def concatenate_clips(
    clip_paths: list[str],
    output_path: str,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> tuple[bool, str]:
    """Concatenate multiple exported clip files into a single video.

    Uses FFmpeg concat demuxer (re-mux, no re-encode when codecs match).
    Falls back to filter_complex concat if re-mux fails.

    Returns (success, output_path_or_error).
    """
    ff = _ff()
    if not clip_paths:
        return False, "Không có clip nào để ghép."

    # Verify all files exist
    missing = [p for p in clip_paths if not Path(p).exists()]
    if missing:
        names = ", ".join(Path(p).name for p in missing)
        return False, f"Thiếu file: {names}"

    if len(clip_paths) == 1:
        # Single clip — just copy
        import shutil
        ensure_dir(Path(output_path).parent)
        shutil.copy2(clip_paths[0], output_path)
        if progress_cb:
            progress_cb(f"✅ Chỉ có 1 clip — đã copy: {Path(output_path).name}")
        return True, output_path

    ensure_dir(Path(output_path).parent)

    if progress_cb:
        progress_cb(f"🔗 Đang ghép {len(clip_paths)} clips thành 1 video...")

    # ── Method 1: concat demuxer (fast, no re-encode) ──
    import tempfile
    concat_list = tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8",
    )
    try:
        for p in clip_paths:
            # FFmpeg concat list format: file 'path'
            safe_path = str(Path(p).resolve()).replace("'", "'\\''")
            concat_list.write(f"file '{safe_path}'\n")
        concat_list.close()

        cmd = [
            ff, "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_list.name,
            "-c", "copy",
            "-movflags", "+faststart",
            output_path,
        ]

        result = subprocess.run(cmd, capture_output=True, timeout=600)

        if result.returncode == 0 and Path(output_path).exists():
            size_mb = Path(output_path).stat().st_size / (1024 * 1024)
            if progress_cb:
                progress_cb(
                    f"✅ Ghép xong: {Path(output_path).name} "
                    f"({size_mb:.1f} MB)"
                )
            return True, output_path

        # ── Method 2: filter_complex concat (re-encode, handles mismatched codecs) ──
        if progress_cb:
            progress_cb("⚠️ Concat demuxer thất bại. Đang thử re-encode...")

        inputs = []
        filter_parts = []
        for i, p in enumerate(clip_paths):
            inputs.extend(["-i", p])
            filter_parts.append(f"[{i}:v:0][{i}:a:0]")

        filter_str = "".join(filter_parts) + f"concat=n={len(clip_paths)}:v=1:a=1[outv][outa]"

        cmd2 = [
            ff, "-y",
            *inputs,
            "-filter_complex", filter_str,
            "-map", "[outv]",
            "-map", "[outa]",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "20",
            "-c:a", "aac",
            "-b:a", "128k",
            "-movflags", "+faststart",
            output_path,
        ]

        # Estimate total duration for timeout
        total_dur = sum(_get_duration(p) for p in clip_paths)
        result2 = subprocess.run(cmd2, capture_output=True, timeout=max(600, int(total_dur * 3)))

        if result2.returncode == 0 and Path(output_path).exists():
            size_mb = Path(output_path).stat().st_size / (1024 * 1024)
            if progress_cb:
                progress_cb(
                    f"✅ Ghép xong (re-encoded): {Path(output_path).name} "
                    f"({size_mb:.1f} MB)"
                )
            return True, output_path

        err = result2.stderr.decode(errors="replace")[-500:]
        logger.error(f"Concat failed: {err}")
        return False, f"Ghép video thất bại:\n{err}"

    finally:
        try:
            Path(concat_list.name).unlink()
        except OSError:
            pass


def _get_duration(video_path: str) -> float:
    """Get video duration in seconds using ffprobe."""
    ff = _ff()
    ffprobe = ff.replace("ffmpeg", "ffprobe") if "ffmpeg" in ff else "ffprobe"
    try:
        result = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", video_path],
            capture_output=True, timeout=30,
        )
        return float(result.stdout.decode().strip() or 0)
    except Exception:
        return 60.0  # fallback estimate
