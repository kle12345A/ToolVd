"""Download videos and social profiles using yt-dlp/gallery-dl."""

import json
import html
import os
import re
import shutil
import socket
import sqlite3
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import parse_qs, urlsplit
from urllib.request import urlopen

from src.core.dependency_manager import (
    download_gallerydl, ffmpeg_path, ffprobe_path, gallerydl_path, ytdlp_path,
)
from src.utils.file_utils import ensure_dir
from src.utils.logger import logger


_COMPATIBLE_FORMAT = (
    "bestvideo[vcodec^=avc1][ext=mp4]+bestaudio[acodec^=mp4a][ext=m4a]/"
    "best[ext=mp4][vcodec^=avc1]/"
    "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best"
)
_VIDEO_EXTENSIONS = {".mp4", ".webm", ".mov", ".m4v", ".mkv"}


def _probe_downloaded_file(path: str) -> tuple[bool, dict | str]:
    """Return container and primary codecs for a downloaded video."""
    probe = ffprobe_path()
    if not probe:
        return False, "Khong tim thay ffprobe. Hay tai FFmpeg trong Cai dat."
    try:
        result = subprocess.run(
            [probe, "-v", "error", "-print_format", "json",
             "-show_entries", "format=format_name:stream=codec_type,codec_name", path],
            capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"Khong the kiem tra file vua tai bang ffprobe: {exc}"
    if result.returncode != 0:
        detail = (result.stderr or "ffprobe khong doc duoc file").strip()
        return False, f"Khong the kiem tra file vua tai: {detail}"
    try:
        data = json.loads(result.stdout or "{}")
    except json.JSONDecodeError as exc:
        return False, f"ffprobe tra ve metadata khong hop le: {exc}"
    streams = data.get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if not video:
        return False, "File vua tai khong co video stream hop le."
    formats = set((data.get("format", {}).get("format_name") or "").split(","))
    return True, {
        "is_mp4": Path(path).suffix.lower() == ".mp4" and "mp4" in formats,
        "video_codec": (video.get("codec_name") or "").lower(),
        "audio_codec": (audio.get("codec_name") or "").lower() if audio else None,
    }


def _compatible_output_path(source: Path) -> Path:
    if source.suffix.lower() == ".mp4":
        return source.with_name(f"{source.stem}.amsr-compatible.mp4")
    candidate = source.with_suffix(".mp4")
    return candidate if not candidate.exists() else source.with_name(
        f"{source.stem}.amsr-compatible.mp4"
    )


def _normalize_downloaded_file(
    path: str,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> tuple[bool, str]:
    """Ensure a downloaded video is MP4/H.264/AAC without needless encoding."""
    source = Path(path)
    if not source.is_file():
        return False, f"Khong tim thay file vua tai: {source}"
    ok, metadata_or_error = _probe_downloaded_file(str(source))
    if not ok:
        return False, str(metadata_or_error)
    metadata = metadata_or_error
    video_ok = metadata["video_codec"] == "h264"
    audio_ok = metadata["audio_codec"] in (None, "aac")
    if metadata["is_mp4"] and video_ok and audio_ok:
        if progress_cb:
            progress_cb("Da tai file tuong thich san")
        return True, str(source)

    ffmpeg = ffmpeg_path()
    if not ffmpeg:
        return False, "Khong tim thay ffmpeg. Hay tai FFmpeg trong Cai dat."
    if progress_cb:
        progress_cb("Dang chuan hoa de mo duoc ngay")
    target = _compatible_output_path(source)
    cmd = [ffmpeg, "-y", "-i", str(source), "-map", "0:v:0", "-map", "0:a?"]
    if video_ok and audio_ok:
        cmd.extend(["-c", "copy"])
    else:
        cmd.extend(["-c:v", "libx264", "-preset", "medium", "-crf", "20",
                    "-pix_fmt", "yuv420p"])
        if metadata["audio_codec"] is None:
            cmd.append("-an")
        else:
            cmd.extend(["-c:a", "aac", "-b:a", "192k"])
    cmd.extend(["-movflags", "+faststart", str(target)])
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8", errors="replace"
        )
    except OSError as exc:
        return False, f"Khong the chay ffmpeg de chuan hoa video: {exc}"
    if result.returncode != 0 or not target.is_file():
        target.unlink(missing_ok=True)
        detail = (result.stderr or "ffmpeg khong tao duoc file dau ra").strip()
        return False, f"Chuan hoa video that bai: {detail[-1000:]}"
    if source.suffix.lower() == ".mp4":
        source.unlink()
        target.replace(source)
        final_path = source
    else:
        source.unlink()
        final_path = target
    logger.info(f"Normalized downloaded video: {final_path}")
    return True, str(final_path)


def _video_files(folder: Path) -> set[Path]:
    return {
        path.resolve() for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in _VIDEO_EXTENSIONS
    }


def _js_runtime_args() -> list[str]:
    """Return yt-dlp arguments for the first installed JavaScript runtime."""
    for runtime in ("deno", "node", "bun", "qjs"):
        executable = shutil.which(runtime)
        if executable:
            return ["--js-runtimes", f"{runtime}:{executable}"]
    return []


def normalize_tiktok_account(account: str) -> tuple[bool, str, str]:
    """Return ``(valid, username, profile_url_or_error)`` for a TikTok account."""
    value = (account or "").strip()
    if not value:
        return False, "", "Vui lòng nhập tên kênh TikTok."
    if re.search(r"tiktok\.com/@[^/]+/video/", value, re.IGNORECASE):
        return False, "", "Đây là link video. Hãy nhập @username của kênh TikTok."

    # Accept @username, username, or a pasted TikTok profile URL.  Video URLs are
    # rejected here so the user does not accidentally start the wrong mode.
    match = re.search(r"(?:tiktok\.com/)?@([A-Za-z0-9._-]+)", value, re.IGNORECASE)
    if match:
        username = match.group(1)
    elif re.fullmatch(r"[A-Za-z0-9._-]+", value):
        username = value
    else:
        return False, "", "Tên kênh TikTok không hợp lệ. Ví dụ: @username"

    if not (2 <= len(username) <= 50):
        return False, "", "Tên kênh TikTok phải có từ 2 đến 50 ký tự."
    return True, username, f"https://www.tiktok.com/@{username}"


def normalize_youtube_account(account: str) -> tuple[bool, str, str]:
    """Normalize a YouTube handle/channel URL to the channel root.

    The root lets yt-dlp select all available tabs and also supports channels
    that publish only Shorts and therefore have no ``/videos`` tab.
    """
    value = (account or "").strip()
    if not value:
        return False, "", "Vui lòng nhập tên kênh YouTube."
    if re.search(r"(?:youtu\.be/|youtube\.com/(?:watch|shorts)/)", value, re.I):
        return False, "", "Đây là link video. Hãy nhập @handle hoặc link kênh YouTube."

    match = re.search(
        r"youtube\.com/(?:@|channel/|c/|user/)([A-Za-z0-9._-]+)", value, re.I
    )
    if match:
        username = match.group(1)
        base_url = value.split("?", 1)[0].rstrip("/")
        base_url = re.sub(r"/(?:videos|shorts|streams)$", "", base_url, flags=re.I)
    else:
        username = value.lstrip("@")
        if not re.fullmatch(r"[A-Za-z0-9._-]{2,100}", username):
            return False, "", "Kênh YouTube không hợp lệ. Ví dụ: @handle"
        base_url = f"https://www.youtube.com/@{username}"
    return True, username, base_url


def normalize_instagram_account(account: str) -> tuple[bool, str, str]:
    """Normalize an Instagram username/profile URL."""
    value = (account or "").strip()
    if not value:
        return False, "", "Vui lòng nhập tài khoản Instagram."
    if re.search(r"instagram\.com/(?:p|reel|reels|stories)/", value, re.I):
        return False, "", "Đây là link bài đăng. Hãy nhập @username hoặc link trang cá nhân Instagram."
    match = re.search(r"instagram\.com/([A-Za-z0-9._]+)", value, re.I)
    username = match.group(1) if match else value.lstrip("@")
    if not re.fullmatch(r"[A-Za-z0-9._]{1,30}", username):
        return False, "", "Tài khoản Instagram không hợp lệ. Ví dụ: @username"
    return True, username, f"https://www.instagram.com/{username}/"


def normalize_facebook_account(account: str) -> tuple[bool, str, str]:
    """Normalize a Facebook Page/profile URL to its Reels tab."""
    value = (account or "").strip()
    if not value:
        return False, "", "Vui lòng nhập tên hoặc link trang Facebook."
    if re.search(
        r"(?:facebook\.com/(?:reel/|watch|share/(?:v|r)/)|fb\.watch/)",
        value,
        re.I,
    ) or re.search(r"facebook\.com/[^/?#]+/videos/[^/?#]+", value, re.I):
        return False, "", (
            "Đây là link một video Facebook. Hãy nhập link trang/profile "
            "để tải nhiều Reels."
        )

    candidate = (
        value
        if "://" in value
        else f"https://www.facebook.com/{value.lstrip('@')}"
    )
    try:
        parsed = urlsplit(candidate)
    except ValueError:
        return False, "", "Link trang Facebook không hợp lệ."
    hostname = (parsed.hostname or "").lower()
    if hostname not in {
        "facebook.com", "www.facebook.com", "m.facebook.com", "web.facebook.com"
    }:
        return False, "", "Link phải thuộc facebook.com."

    if parsed.path.rstrip("/").lower() == "/profile.php":
        profile_ids = parse_qs(parsed.query).get("id") or []
        profile_id = next((item for item in profile_ids if item.isdigit()), "")
        if not profile_id:
            return False, "", "Link profile Facebook chưa có ID hợp lệ."
        return (
            True,
            profile_id,
            f"https://www.facebook.com/profile.php?id={profile_id}&sk=reels",
        )

    parts = [part for part in parsed.path.split("/") if part]
    if parts and parts[-1].lower() in {"reels", "videos"}:
        parts.pop()
    if not parts:
        return False, "", "Link trang Facebook chưa có tên trang/profile."
    username = parts[0]
    if username.lower() in {
        "groups", "watch", "reel", "reels", "share", "marketplace", "gaming"
    }:
        return False, "", (
            "Hãy nhập link trực tiếp của trang/profile Facebook, "
            "ví dụ https://www.facebook.com/TenTrang"
        )
    if not re.fullmatch(r"[A-Za-z0-9._-]{2,100}", username):
        return False, "", "Tên trang Facebook không hợp lệ."
    return True, username, f"https://www.facebook.com/{username}/reels/"


def normalize_douyin_account(account: str) -> tuple[bool, str, str]:
    """Normalize a Douyin user ID/profile URL for profile downloads."""
    value = (account or "").strip()
    if not value:
        return False, "", "Vui lòng nhập link trang cá nhân hoặc user ID Douyin."

    if re.search(r"douyin\.com/(?:video|note)/", value, re.I):
        return False, "", (
            "Đây là link video Douyin. Hãy dán link vào ô tải video từ URL phía trên."
        )
    if re.search(
        r"(?:v\.douyin\.com/|iesdouyin\.com/share/(?:video|note)/)",
        value,
        re.I,
    ):
        return False, "", (
            "Link chia sẻ ngắn dùng cho một video. Hãy dán link vào ô URL phía trên."
        )

    match = re.search(
        r"(?:douyin\.com/user/|iesdouyin\.com/share/user/)"
        r"([A-Za-z0-9._-]+)",
        value,
        re.I,
    )
    if match:
        user_id = match.group(1)
        profile_url = f"https://www.douyin.com/user/{user_id}"
    else:
        user_id = value.lstrip("@")
        if not re.fullmatch(r"[A-Za-z0-9._-]{2,200}", user_id):
            return False, "", (
                "Tài khoản Douyin không hợp lệ. Hãy dán link dạng "
                "https://www.douyin.com/user/..."
            )
        profile_url = f"https://www.douyin.com/user/{user_id}"

    return True, user_id, profile_url


def download_tiktok_channel(
    account: str,
    output_dir: str,
    cookies_browser: str = "",
    cookies_file: str = "",
    max_videos: int = 0,
    progress_cb: Optional[Callable[[str], None]] = None,
    _platform: str = "tiktok",
) -> tuple[bool, str, int]:
    """Download every available video from a TikTok profile.

    A per-channel download archive prevents already downloaded posts from being
    fetched again on subsequent runs. ``max_videos`` limits newly downloaded
    files, not the number of playlist entries scanned, so later runs continue
    with the next batch. Returns ``(success, folder_or_error, count)``.
    """
    ytdlp = ytdlp_path()
    if not ytdlp:
        return False, "Không tìm thấy yt-dlp. Vui lòng tải yt-dlp trước.", 0

    platform_config = {
        "youtube": (
            normalize_youtube_account,
            "YouTube",
            lambda user: f"YouTube_@{user}",
        ),
        "douyin": (
            normalize_douyin_account,
            "Douyin",
            lambda user: f"Douyin_@{user}",
        ),
        "tiktok": (
            normalize_tiktok_account,
            "TikTok",
            lambda user: f"@{user}",
        ),
    }
    normalizer, platform_label, folder_builder = platform_config.get(
        _platform, platform_config["tiktok"]
    )
    valid, username, profile_or_error = normalizer(account)
    channel_folder_name = folder_builder(username)
    if not valid:
        return False, profile_or_error, 0

    base_dir = Path(output_dir)
    # Accept either the parent download folder or the channel folder itself.
    # This prevents @username/@username nesting when the user selects the same
    # channel folder again on a later run.
    channel_dir = (
        base_dir
        if base_dir.name.casefold() == channel_folder_name.casefold()
        else base_dir / channel_folder_name
    )
    ensure_dir(str(channel_dir))
    archive_path = channel_dir / ".downloaded.txt"
    output_template = str(
        channel_dir / "%(upload_date>%Y-%m-%d)s_%(id)s_%(title).80s.%(ext)s"
    )
    file_marker = "__AMSR_DOWNLOADED_FILE__:"
    cmd = [
        ytdlp,
        "--no-update",
        *_js_runtime_args(),
        "--yes-playlist",
        "--lazy-playlist",
        "--ignore-errors",
        "--socket-timeout", "20",
        "--extractor-retries", "3",
        "--retries", "3",
        "--fragment-retries", "3",
        "--download-archive", str(archive_path),
        "-f", _COMPATIBLE_FORMAT,
        "--merge-output-format", "mp4",
        "--restrict-filenames",
        "--windows-filenames",
        "-o", output_template,
        "--newline",
        "--print", f"after_move:{file_marker}%(filepath)s",
    ]
    if max_videos > 0:
        # Do not use --playlist-end here: on a later run that would scan the
        # same first N archived posts and never reach the next batch.
        cmd.extend(["--max-downloads", str(max_videos)])
    if cookies_file:
        cmd.extend(["--cookies", cookies_file])
    elif cookies_browser:
        cmd.extend(["--cookies-from-browser", cookies_browser])
    cmd.append(profile_or_error)

    logger.info(f"Downloading {platform_label} channel: @{username}")
    downloaded_files: list[str] = []
    output_lines: list[str] = []
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        for raw_line in proc.stdout:
            line = raw_line.strip()
            output_lines.append(line)
            output_lines = output_lines[-120:]
            logger.debug(f"yt-dlp channel: {line}")
            if line.startswith(file_marker):
                path = line[len(file_marker):].strip()
                if path and path not in downloaded_files:
                    downloaded_files.append(path)
            if progress_cb:
                progress_cb(line)
        proc.wait()

        joined = "\n".join(output_lines)
        # yt-dlp exits with 101 when --max-downloads is reached. That is the
        # expected successful end of a limited batch.
        batch_limit_reached = (
            proc.returncode == 101
            and max_videos > 0
            and len(downloaded_files) >= max_videos
        )
        if proc.returncode != 0 and not batch_limit_reached:
            if "Could not copy Chrome cookie database" in joined:
                return False, (
                    "Không đọc được cookies Chrome. Hãy đóng hoàn toàn Chrome, "
                    "hoặc chọn Edge/Firefox hay file cookies.txt."
                ), 0
            return False, f"Không thể tải kênh {platform_label}. Kiểm tra log và cookies.", 0
        if not downloaded_files and (
            "does not exist" in joined.lower()
            or "unable to extract" in joined.lower()
            or "no videos found" in joined.lower()
        ):
            return False, (
                "Không lấy được video của kênh. Kiểm tra tên kênh, cookies, "
                "hoặc cập nhật yt-dlp."
            ), 0

        for downloaded_file in downloaded_files:
            normalized, result = _normalize_downloaded_file(downloaded_file, progress_cb)
            if not normalized:
                return False, result, 0

        logger.info(
            f"{platform_label} channel @{username}: downloaded {len(downloaded_files)} new videos"
        )
        return True, str(channel_dir), len(downloaded_files)
    except Exception as e:
        logger.error(f"{platform_label} channel download error: {e}")
        return False, str(e), 0


def download_youtube_channel(account: str, output_dir: str, **kwargs):
    return download_tiktok_channel(
        account, output_dir, _platform="youtube", **kwargs
    )


def download_douyin_profile(account: str, output_dir: str, **kwargs):
    return download_tiktok_channel(
        account, output_dir, _platform="douyin", **kwargs
    )


def _facebook_reel_id(url: str) -> str:
    match = re.search(
        r"facebook\.com/(?:reel/|[^/?#]+/videos/)([A-Za-z0-9._-]+)",
        url,
        re.I,
    )
    return match.group(1) if match else ""


def _free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _collect_facebook_reel_urls(
    profile_url: str,
    already_downloaded: set[str],
    max_videos: int,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> tuple[bool, list[str], str, str]:
    """Collect Reel URLs by scrolling Facebook in a dedicated Chrome profile.

    Returns ``(success, urls_or_empty, browser_cookie_spec, error_or_profile)``.
    The browser profile is persistent, so Facebook login is only needed once.
    """
    try:
        import websocket
    except ImportError:
        return False, [], "", (
            "Thiếu thư viện websocket-client. Hãy chạy: "
            "pip install websocket-client"
        )

    browser, browser_name = _find_chromium_browser()
    if not browser:
        return False, [], "", "Không tìm thấy Chrome hoặc Edge."

    profile_dir = (Path("data") / "facebook_browser_profile").resolve()
    ensure_dir(str(profile_dir))
    port = _free_local_port()
    command = [
        browser,
        f"--remote-debugging-port={port}",
        "--remote-allow-origins=*",
        f"--user-data-dir={profile_dir}",
        "--no-first-run",
        "--no-default-browser-check",
        "--new-window",
        profile_url,
    ]
    if progress_cb:
        progress_cb(
            "Đang mở cửa sổ Facebook riêng. Nếu được yêu cầu, hãy đăng nhập "
            "trong cửa sổ này; tool sẽ tự quét sau khi trang Reels hiện ra."
        )

    creation_flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    process = None
    ws = None
    try:
        process = subprocess.Popen(command, creationflags=creation_flags)
        debugger_url = ""
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline and not debugger_url:
            try:
                with urlopen(
                    f"http://127.0.0.1:{port}/json", timeout=2
                ) as response:
                    targets = json.loads(response.read().decode("utf-8"))
                page = next(
                    (
                        target for target in targets
                        if target.get("type") == "page"
                        and "facebook.com" in target.get("url", "")
                    ),
                    None,
                )
                if page:
                    debugger_url = page.get("webSocketDebuggerUrl", "")
            except (OSError, ValueError, json.JSONDecodeError):
                time.sleep(1)
        if not debugger_url:
            return False, [], "", (
                "Không kết nối được với cửa sổ Chrome/Edge dùng để quét Facebook."
            )

        ws = websocket.create_connection(
            debugger_url, timeout=15, origin="http://localhost"
        )
        command_id = 0

        def evaluate(expression: str):
            nonlocal command_id
            command_id += 1
            wanted_id = command_id
            ws.send(json.dumps({
                "id": wanted_id,
                "method": "Runtime.evaluate",
                "params": {
                    "expression": expression,
                    "returnByValue": True,
                    "awaitPromise": True,
                },
            }))
            while True:
                message = json.loads(ws.recv())
                if message.get("id") == wanted_id:
                    if "error" in message:
                        raise RuntimeError(message["error"].get("message", "CDP error"))
                    return (
                        message.get("result", {})
                        .get("result", {})
                        .get("value")
                    )

        found: dict[str, str] = {}
        stagnant_rounds = 0
        scan_deadline = time.monotonic() + 300
        last_status = 0.0
        script = r"""
(() => {
  const links = Array.from(document.querySelectorAll('a[href]'))
    .map(a => a.href)
    .filter(h => /facebook\.com\/(?:reel\/|[^/?#]+\/videos\/)/i.test(h));
  window.scrollBy(0, Math.max(window.innerHeight * 1.8, 1200));
  return {links, url: location.href, title: document.title};
})()
"""
        while time.monotonic() < scan_deadline:
            result = evaluate(script) or {}
            page_url = str(result.get("url") or "")
            links = result.get("links") or []
            before = len(found)
            for raw_url in links:
                reel_id = _facebook_reel_id(str(raw_url))
                if reel_id and reel_id not in already_downloaded:
                    found.setdefault(
                        reel_id, f"https://www.facebook.com/reel/{reel_id}"
                    )

            if len(found) > before:
                stagnant_rounds = 0
                if progress_cb:
                    progress_cb(
                        f"Đã tìm thấy {len(found)} Reel chưa tải..."
                    )
            else:
                stagnant_rounds += 1

            if max_videos > 0 and len(found) >= max_videos:
                break
            if found and stagnant_rounds >= 10:
                break
            if not found and time.monotonic() - last_status >= 15:
                last_status = time.monotonic()
                if progress_cb:
                    if "login" in page_url.lower():
                        progress_cb(
                            "Đang chờ anh đăng nhập Facebook trong cửa sổ trình duyệt..."
                        )
                    else:
                        progress_cb(
                            "Đang cuộn trang và tìm Reels; hãy giữ cửa sổ "
                            "Facebook mở..."
                        )
            time.sleep(1.2)

        urls = list(found.values())
        if max_videos > 0:
            urls = urls[:max_videos]
        if not urls:
            return False, [], "", (
                "Không tìm thấy Reel chưa tải. Hãy kiểm tra đúng link trang, "
                "đăng nhập Facebook trong cửa sổ vừa mở và mở được tab Reels."
            )
        cookie_spec = f"{browser_name}:{profile_dir}"
        return True, urls, cookie_spec, str(profile_dir)
    except Exception as exc:
        logger.error(f"Facebook browser collector error: {exc}")
        return False, [], "", f"Lỗi khi quét Facebook bằng trình duyệt: {exc}"
    finally:
        if ws is not None:
            try:
                ws.send(json.dumps({
                    "id": 999999,
                    "method": "Browser.close",
                }))
            except Exception:
                pass
            try:
                ws.close()
            except Exception:
                pass
        if process is not None and process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=10)
            except Exception:
                pass


def download_facebook_profile(
    account: str,
    output_dir: str,
    cookies_browser: str = "",
    cookies_file: str = "",
    max_videos: int = 0,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> tuple[bool, str, int]:
    """Download Facebook Reels in resumable batches using a Chrome collector."""
    valid, username, profile_or_error = normalize_facebook_account(account)
    if not valid:
        return False, profile_or_error, 0

    base_dir = Path(output_dir)
    folder_name = f"Facebook_@{username}"
    profile_dir = (
        base_dir
        if base_dir.name.casefold() == folder_name.casefold()
        else base_dir / folder_name
    )
    ensure_dir(str(profile_dir))
    history_path = profile_dir / ".facebook_downloaded.txt"
    already_downloaded = set()
    if history_path.is_file():
        already_downloaded = {
            line.strip() for line in history_path.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines() if line.strip()
        }

    ok, urls, browser_cookie_spec, result = _collect_facebook_reel_urls(
        profile_or_error, already_downloaded, max_videos, progress_cb
    )
    if not ok:
        return False, result, 0

    downloaded = 0
    errors: list[str] = []
    for index, reel_url in enumerate(urls, start=1):
        reel_id = _facebook_reel_id(reel_url)
        if progress_cb:
            progress_cb(f"[{index}/{len(urls)}] Đang tải Facebook Reel {reel_id}...")
        success, path_or_error = download_video(
            reel_url,
            str(profile_dir),
            cookies_browser="" if cookies_file else browser_cookie_spec,
            cookies_file=cookies_file,
            progress_cb=progress_cb,
        )
        if not success:
            errors.append(f"{reel_id}: {path_or_error}")
            continue
        with history_path.open("a", encoding="utf-8") as history:
            history.write(reel_id + "\n")
        downloaded += 1

    if downloaded:
        if errors and progress_cb:
            progress_cb(
                f"Đã tải {downloaded} Reel; bỏ qua {len(errors)} Reel lỗi."
            )
        return True, str(profile_dir), downloaded
    detail = errors[-1] if errors else "Không có Reel nào tải thành công."
    return False, detail, 0


def _archive_count(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        with sqlite3.connect(str(path)) as db:
            row = db.execute("SELECT COUNT(*) FROM archive").fetchone()
            return int(row[0] if row else 0)
    except sqlite3.Error:
        return 0


def download_instagram_profile(
    account: str,
    output_dir: str,
    cookies_browser: str = "",
    cookies_file: str = "",
    max_videos: int = 0,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> tuple[bool, str, int]:
    """Download video files from an Instagram profile with continuation."""
    valid, username, profile_or_error = normalize_instagram_account(account)
    if not valid:
        return False, profile_or_error, 0

    gallerydl = gallerydl_path()
    if not gallerydl:
        ok = download_gallerydl(status_cb=progress_cb)
        gallerydl = gallerydl_path()
        if not ok or not gallerydl:
            return False, "Không tải được bộ hỗ trợ Instagram gallery-dl.", 0

    base_dir = Path(output_dir)
    folder_name = f"Instagram_@{username}"
    profile_dir = base_dir if base_dir.name.casefold() == folder_name.casefold() else base_dir / folder_name
    ensure_dir(str(profile_dir))
    archive_path = profile_dir / ".gallery-dl.sqlite3"
    before = _archive_count(archive_path)
    files_before = _video_files(profile_dir)

    cmd = [
        gallerydl, "--config-ignore", "--no-colors",
        "--directory", str(profile_dir),
        "--download-archive", str(archive_path),
        "--filter", "extension in ('mp4', 'webm', 'mov', 'm4v')",
        "-o", "extractor.instagram.include=posts",
        "-o", "extractor.instagram.order-posts=desc",
        "-o", "extractor.instagram.sleep-request=1.0-2.0",
    ]
    if max_videos > 0:
        cmd.extend(["--range", f"{before + 1}-{before + max_videos}"])
    if cookies_file:
        cmd.extend(["--cookies", cookies_file])
    elif cookies_browser:
        cmd.extend(["--cookies-from-browser", cookies_browser])
    cmd.append(profile_or_error)

    lines = []
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
        )
        for raw_line in proc.stdout:
            line = raw_line.strip()
            lines.append(line)
            lines = lines[-120:]
            logger.debug(f"gallery-dl Instagram: {line}")
            if progress_cb:
                progress_cb(line)
        proc.wait()
        after = _archive_count(archive_path)
        count = max(0, after - before)
        if proc.returncode != 0 and count == 0:
            joined = "\n".join(lines).lower()
            if "cookies" in joined or "login" in joined or "401" in joined:
                return False, "Instagram yêu cầu đăng nhập. Hãy bật cookies trình duyệt đã đăng nhập Instagram.", 0
            return False, "Không thể tải trang Instagram. Hãy bật cookies và kiểm tra lại tài khoản.", 0
        files_after = _video_files(profile_dir)
        for downloaded_file in sorted(files_after - files_before):
            normalized, result = _normalize_downloaded_file(str(downloaded_file), progress_cb)
            if not normalized:
                return False, result, 0
        return True, str(profile_dir), count
    except Exception as e:
        logger.error(f"Instagram profile download error: {e}")
        return False, str(e), 0


def normalize_social_account(platform: str, account: str):
    return {
        "tiktok": normalize_tiktok_account,
        "youtube": normalize_youtube_account,
        "instagram": normalize_instagram_account,
        "douyin": normalize_douyin_account,
        "facebook": normalize_facebook_account,
    }[platform](account)


def download_social_profile(platform: str, account: str, output_dir: str, **kwargs):
    if platform == "youtube":
        return download_youtube_channel(account, output_dir, **kwargs)
    if platform == "instagram":
        return download_instagram_profile(account, output_dir, **kwargs)
    if platform == "douyin":
        return download_douyin_profile(account, output_dir, **kwargs)
    if platform == "facebook":
        return download_facebook_profile(account, output_dir, **kwargs)
    return download_tiktok_channel(account, output_dir, **kwargs)


def normalize_download_url(url: str) -> tuple[bool, str, str]:
    """Convert share/search URLs into URLs understood by the site extractor."""
    value = html.unescape((url or "").strip())
    if not value:
        return False, "", "Vui lòng dán URL video."

    # Chat apps and browsers often copy a link as Markdown:
    # [visible URL](actual URL), with query separators escaped as \& or \_.
    # Prefer the link target, then remove Markdown escaping before parsing.
    markdown_link = re.fullmatch(r"\s*\[[^\]]*\]\((.+)\)\s*", value, re.S)
    if markdown_link:
        value = markdown_link.group(1).strip()
    value = re.sub(r"\\([&_=?:/#.%~-])", r"\1", value).strip()

    try:
        parsed = urlsplit(value)
    except ValueError:
        return False, "", "URL video không hợp lệ."

    hostname = (parsed.hostname or "").lower()
    is_facebook = (
        hostname in {"facebook.com", "fb.watch"}
        or hostname.endswith(".facebook.com")
    )
    if is_facebook:
        facebook_video_path = bool(
            hostname == "fb.watch"
            or re.search(
                r"/(?:watch|reel|reels|videos|video\.php|share/(?:v|r))(?:/|$)",
                parsed.path,
                re.I,
            )
            or (
                parsed.path.rstrip("/").lower() == "/watch"
                and parse_qs(parsed.query).get("v")
            )
        )
        if not facebook_video_path:
            return False, "", (
                "Đây chưa phải link video Facebook. Hãy mở video/Reel, "
                "chọn Chia sẻ → Sao chép liên kết rồi dán lại."
            )
        return True, value, "Đã nhận diện link video Facebook."

    is_douyin = hostname == "douyin.com" or hostname.endswith(".douyin.com")
    if is_douyin and re.search(r"/(?:jingxuan/)?search/", parsed.path, re.I):
        modal_ids = parse_qs(parsed.query).get("modal_id") or []
        video_id = next(
            (item for item in modal_ids if re.fullmatch(r"\d{10,30}", item)),
            "",
        )
        if not video_id:
            return False, "", (
                "Link tìm kiếm Douyin chưa chứa video cụ thể. "
                "Hãy mở video rồi sao chép link có modal_id."
            )
        canonical = f"https://www.douyin.com/video/{video_id}"
        return True, canonical, (
            f"Đã nhận diện video Douyin {video_id} từ link tìm kiếm."
        )

    return True, value, ""


def _find_chromium_browser() -> tuple[str, str]:
    """Return ``(executable, yt-dlp browser name)`` for Chrome or Edge."""
    for command, browser_name in (("chrome", "chrome"), ("msedge", "edge")):
        executable = shutil.which(command)
        if executable:
            return executable, browser_name

    candidates = (
        (
            Path(os.environ.get("PROGRAMFILES", ""))
            / "Google/Chrome/Application/chrome.exe",
            "chrome",
        ),
        (
            Path(os.environ.get("PROGRAMFILES(X86)", ""))
            / "Google/Chrome/Application/chrome.exe",
            "chrome",
        ),
        (
            Path(os.environ.get("LOCALAPPDATA", ""))
            / "Google/Chrome/Application/chrome.exe",
            "chrome",
        ),
        (
            Path(os.environ.get("PROGRAMFILES(X86)", ""))
            / "Microsoft/Edge/Application/msedge.exe",
            "edge",
        ),
        (
            Path(os.environ.get("PROGRAMFILES", ""))
            / "Microsoft/Edge/Application/msedge.exe",
            "edge",
        ),
    )
    for executable, browser_name in candidates:
        if executable.is_file():
            return str(executable), browser_name
    return "", ""


def _create_anonymous_douyin_profile(
    video_url: str,
    progress_cb: Optional[Callable[[str], None]] = None,
):
    """Create disposable anonymous Douyin cookies without reading user profiles."""
    match = re.search(r"douyin\.com/video/(\d+)", video_url, re.I)
    if not match:
        return None, "", "Không tìm thấy ID video Douyin."

    browser, browser_name = _find_chromium_browser()
    if not browser:
        return None, "", (
            "Không tìm thấy Chrome hoặc Edge để tạo phiên Douyin ẩn danh."
        )

    session = tempfile.TemporaryDirectory(prefix="amsr-douyin-")
    profile_dir = session.name
    share_url = f"https://www.iesdouyin.com/share/video/{match.group(1)}/"
    if progress_cb:
        progress_cb(
            "Douyin cần phiên chống bot; đang tạo phiên ẩn danh tạm "
            "(không đọc cookies cá nhân)..."
        )

    command = [
        browser,
        "--headless=new",
        "--disable-gpu",
        "--no-first-run",
        "--no-default-browser-check",
        f"--user-data-dir={profile_dir}",
        "--virtual-time-budget=12000",
        "--dump-dom",
        share_url,
    ]
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
            creationflags=creation_flags,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        session.cleanup()
        return None, "", f"Không tạo được phiên Douyin ẩn danh: {exc}"

    cookie_db = Path(profile_dir) / "Default" / "Network" / "Cookies"
    if result.returncode != 0 or not cookie_db.is_file():
        detail = (result.stderr or "").strip()
        session.cleanup()
        return None, "", (
            "Douyin không tạo được phiên ẩn danh tự động."
            + (f" {detail[-300:]}" if detail else "")
        )
    return session, browser_name, ""


def download_video(
    url: str,
    output_dir: str,
    cookies_browser: str = "",
    cookies_file: str = "",
    progress_cb: Optional[Callable[[str], None]] = None,
) -> tuple[bool, str]:
    """Download video using yt-dlp. Returns (success, output_path_or_error)."""
    valid_url, normalized_url, url_message = normalize_download_url(url)
    if not valid_url:
        return False, url_message
    if url_message and progress_cb:
        progress_cb(url_message)

    ytdlp = ytdlp_path()
    if not ytdlp:
        return False, "yt-dlp không tìm thấy. Vui lòng tải yt-dlp trước."

    anonymous_session = None
    anonymous_browser = ""
    normalized_host = (urlsplit(normalized_url).hostname or "").lower()
    is_facebook_video = (
        normalized_host in {"facebook.com", "fb.watch"}
        or normalized_host.endswith(".facebook.com")
    )
    is_douyin_video = bool(
        re.search(r"(?:^|\.)douyin\.com/video/\d+", normalized_url, re.I)
    )
    if is_douyin_video and not cookies_file and not cookies_browser:
        anonymous_session, anonymous_browser, session_error = (
            _create_anonymous_douyin_profile(normalized_url, progress_cb)
        )
        if not anonymous_session and progress_cb:
            progress_cb(session_error)

    ensure_dir(output_dir)
    output_template = str(Path(output_dir) / "%(id)s_%(title).80s.%(ext)s")
    file_marker = "__AMSR_DOWNLOADED_FILE__:"

    cmd = [
        ytdlp,
        "--no-update",
        *_js_runtime_args(),
        "--no-playlist",
        "-f", _COMPATIBLE_FORMAT,
        "--merge-output-format", "mp4",
        "--restrict-filenames",
        "--windows-filenames",
        "-o", output_template,
        "--newline",
        "--print", f"after_move:{file_marker}%(filepath)s",
    ]
    if cookies_file:
        cmd.extend(["--cookies", cookies_file])
    elif cookies_browser:
        cmd.extend(["--cookies-from-browser", cookies_browser])
    elif anonymous_session:
        cmd.extend([
            "--cookies-from-browser",
            f"{anonymous_browser}:{anonymous_session.name}",
        ])
    cmd.append(normalized_url)

    logger.info(f"Downloading: {normalized_url}")
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        last_file = None
        output_lines = []
        for line in proc.stdout:
            line = line.strip()
            output_lines.append(line)
            output_lines = output_lines[-80:]
            logger.debug(f"yt-dlp: {line}")
            if progress_cb:
                progress_cb(line)
            if line.startswith(file_marker):
                last_file = line[len(file_marker):].strip()
            elif "[download] Destination:" in line:
                last_file = line.split("Destination:")[-1].strip()
            elif line.startswith("[download] ") and " has already been downloaded" in line:
                last_file = line[len("[download] "):].split(" has already been downloaded", 1)[0].strip()
            elif "Merging formats into" in line:
                last_file = line.split('"')[1] if '"' in line else last_file
            elif "[Merger] Merging formats into" in line:
                last_file = line.split('"')[1] if '"' in line else last_file

        proc.wait()
        if proc.returncode != 0:
            joined = "\n".join(output_lines)
            if (
                "Fresh cookies" in joined
                or ("[Douyin]" in joined and "cookies" in joined.lower())
            ):
                return False, (
                    "Douyin yêu cầu cookies mới để tải video này.\n\n"
                    "Cách xử lý:\n"
                    "1. Mở Douyin trong Chrome/Edge và mở thử video cần tải.\n"
                    "2. Quay lại Tab 1, bật 'Dùng cookies trình duyệt'.\n"
                    "3. Chọn đúng trình duyệt vừa mở Douyin rồi tải lại.\n\n"
                    "Nếu trình duyệt không đọc được cookies, hãy export cookies "
                    "Douyin thành file cookies.txt và chọn file đó."
                )
            if "Unable to extract universal data for rehydration" in joined:
                return False, (
                    "TikTok khong tra du lieu video cho yt-dlp lan nay. "
                    "Tool se thu lai trong batch; neu van loi, hay cap nhat yt-dlp trong Cai dat."
                )
            if "Could not copy Chrome cookie database" in joined:
                return False, (
                    "Không đọc được cookies từ Chrome.\n\n"
                    "Cách xử lý nhanh:\n"
                    "1. Đóng toàn bộ Chrome, kể cả cửa sổ nền.\n"
                    "2. Mở lại app rồi tải lại với cookies Chrome.\n\n"
                    "Nếu vẫn lỗi, dùng Edge/Firefox hoặc export cookies ra file cookies.txt "
                    "rồi chọn file đó trong Tab 1."
                )
            if "Sign in to confirm your age" in joined or "age-restricted" in joined:
                return False, (
                    "Video này bị giới hạn tuổi/đòi đăng nhập YouTube.\n\n"
                    "Cách xử lý:\n"
                    "1. Mở YouTube trong Chrome/Edge/Firefox và đăng nhập tài khoản đủ tuổi.\n"
                    "2. Quay lại Tab 1, bật 'Dùng cookies trình duyệt'.\n"
                    "3. Chọn đúng trình duyệt đang đăng nhập rồi tải lại."
                )
            if is_facebook_video:
                return False, (
                    "Không thể tải video Facebook này. Video có thể không công khai "
                    "hoặc Facebook yêu cầu đăng nhập.\n\n"
                    "Cách xử lý:\n"
                    "1. Mở video trong Chrome/Edge/Firefox và đăng nhập Facebook.\n"
                    "2. Đảm bảo tài khoản của bạn xem được video.\n"
                    "3. Quay lại Tab 1, bật 'Dùng cookies trình duyệt', chọn đúng "
                    "trình duyệt rồi tải lại.\n\n"
                    "Chỉ tải video bạn có quyền sử dụng."
                )
            return False, "yt-dlp thoát với lỗi. Kiểm tra logs/app.log."

        # Find the downloaded file if path tracking failed
        if not last_file or not Path(last_file).exists():
            videos = [p for p in Path(output_dir).glob("*")
                      if p.is_file() and p.suffix.lower() in _VIDEO_EXTENSIONS]
            if videos:
                last_file = str(sorted(videos, key=lambda p: p.stat().st_mtime)[-1])

        if last_file and Path(last_file).exists():
            normalized, result = _normalize_downloaded_file(last_file, progress_cb)
            if not normalized:
                return False, result
            logger.info(f"Downloaded: {result}")
            return True, result

        return False, "Không tìm thấy file sau khi tải xong."
    except Exception as e:
        logger.error(f"Download error: {e}")
        return False, str(e)
    finally:
        if anonymous_session:
            anonymous_session.cleanup()
