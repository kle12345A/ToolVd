"""QThread workers for background tasks."""

from PyQt6.QtCore import QThread, pyqtSignal
import json
import time
from pathlib import Path
from datetime import datetime


class DownloadDepsWorker(QThread):
    progress = pyqtSignal(int, int)       # downloaded, total
    status = pyqtSignal(str)
    finished = pyqtSignal(bool, str)      # success, message

    def __init__(self, download_ffmpeg: bool = True, download_ytdlp: bool = True):
        super().__init__()
        self.download_ffmpeg = download_ffmpeg
        self.download_ytdlp = download_ytdlp

    def run(self):
        from src.core.dependency_manager import download_ffmpeg, download_ytdlp
        ok = True
        if self.download_ffmpeg:
            result = download_ffmpeg(
                progress_cb=lambda d, t: self.progress.emit(d, t),
                status_cb=lambda s: self.status.emit(s),
            )
            if not result:
                ok = False
        if self.download_ytdlp:
            result = download_ytdlp(
                progress_cb=lambda d, t: self.progress.emit(d, t),
                status_cb=lambda s: self.status.emit(s),
            )
            if not result:
                ok = False
        msg = "Tải xong tất cả dependencies." if ok else "Một số dependencies tải thất bại."
        self.finished.emit(ok, msg)


class DownloadVideoWorker(QThread):
    log_line = pyqtSignal(str)
    finished = pyqtSignal(bool, str)   # success, path_or_error

    def __init__(
        self,
        url: str,
        output_dir: str,
        cookies_browser: str = "",
        cookies_file: str = "",
    ):
        super().__init__()
        self.url = url
        self.output_dir = output_dir
        self.cookies_browser = cookies_browser
        self.cookies_file = cookies_file

    def run(self):
        from src.core.downloader import download_video
        ok, result = download_video(
            self.url,
            self.output_dir,
            cookies_browser=self.cookies_browser,
            cookies_file=self.cookies_file,
            progress_cb=lambda line: self.log_line.emit(line),
        )
        self.finished.emit(ok, result)


class DownloadSocialProfileWorker(QThread):
    """Download videos from a supported social profile/page."""

    log_line = pyqtSignal(str)
    finished = pyqtSignal(bool, str, int)  # success, folder_or_error, new_count

    def __init__(
        self,
        platform: str,
        account: str,
        output_dir: str,
        cookies_browser: str = "",
        cookies_file: str = "",
        max_videos: int = 0,
    ):
        super().__init__()
        self.platform = platform
        self.account = account
        self.output_dir = output_dir
        self.cookies_browser = cookies_browser
        self.cookies_file = cookies_file
        self.max_videos = max_videos

    def run(self):
        from src.core.downloader import download_social_profile

        ok, result, count = download_social_profile(
            self.platform,
            self.account,
            self.output_dir,
            cookies_browser=self.cookies_browser,
            cookies_file=self.cookies_file,
            max_videos=self.max_videos,
            progress_cb=lambda line: self.log_line.emit(line),
        )
        self.finished.emit(ok, result, count)


# Backward-compatible name for older imports.
DownloadTikTokChannelWorker = DownloadSocialProfileWorker


class BatchDownloadMergeWorker(QThread):
    log_line = pyqtSignal(str)
    progress = pyqtSignal(float, float)   # elapsed, total
    finished = pyqtSignal(bool, str)      # success, merged_path_or_error

    def __init__(
        self,
        urls: list[str],
        output_dir: str,
        cookies_browser: str = "",
        cookies_file: str = "",
        transition: str = "none",
        trans_dur: float = 0.7,
        merge_mode: str = "merge",
    ):
        super().__init__()
        self.urls = urls
        self.output_dir = output_dir
        self.cookies_browser = cookies_browser
        self.cookies_file = cookies_file
        self.transition = transition
        self.trans_dur = trans_dur
        self.merge_mode = merge_mode

    def run(self):
        from src.core.downloader import download_video
        from src.core.video_merger import (
            concatenate_videos, merge_videos, probe_video,
        )

        downloaded = []
        downloaded_meta = []
        failed = []
        for idx, url in enumerate(self.urls, start=1):
            self.log_line.emit(f"[{idx}/{len(self.urls)}] Dang tai: {url}")
            ok = False
            result = ""
            max_attempts = 3
            for attempt in range(1, max_attempts + 1):
                if attempt > 1:
                    self.log_line.emit(
                        f"[{idx}/{len(self.urls)}] Thu lai lan {attempt}/{max_attempts}..."
                    )
                ok, result = download_video(
                    url,
                    self.output_dir,
                    cookies_browser=self.cookies_browser,
                    cookies_file=self.cookies_file,
                    progress_cb=lambda line, i=idx: self.log_line.emit(f"[{i}] {line}"),
                )
                if ok:
                    break
                if attempt < max_attempts:
                    time.sleep(1.5 * attempt)
            if not ok:
                failed.append({
                    "index": idx,
                    "url": url,
                    "error": result,
                })
                if self.merge_mode == "separate":
                    self.log_line.emit(
                        f"[{idx}/{len(self.urls)}] Bo qua URL loi sau {max_attempts} lan: {result}"
                    )
                    continue
                self.finished.emit(False, f"Tai URL #{idx} that bai: {result}")
                return
            downloaded.append(result)
            info = probe_video(result)
            downloaded_meta.append({
                "index": idx,
                "url": url,
                "path": result,
                "filename": Path(result).name,
                "duration": float(info.get("duration", 0) or 0),
            })
            self.log_line.emit(f"[{idx}/{len(self.urls)}] Da tai xong: {Path(result).name}")

        if len(downloaded) == 1 and self.merge_mode != "separate":
            self.finished.emit(True, downloaded[0])
            return

        if not downloaded:
            if failed:
                details = "; ".join(
                    f"#{item['index']}: {item['error']}" for item in failed
                )
                self.finished.emit(False, f"Khong tai duoc video nao. {details}")
                return
            self.finished.emit(False, "Khong co video nao duoc tai.")
            return

        if self.merge_mode == "separate":
            sidecar = str(Path(downloaded[0]).with_suffix(".parts.json"))
            parts = []
            for part_idx, item in enumerate(downloaded_meta, start=1):
                duration = max(0.1, float(item.get("duration", 0) or 0))
                parts.append({
                    **item,
                    "source_index": item["index"],
                    "source_video": item["path"],
                    "start_time": 0.0,
                    "end_time": round(duration, 3),
                    "part_text": f"PART {part_idx}",
                    "separate_source": True,
                })
            with open(sidecar, "w", encoding="utf-8") as f:
                json.dump({
                    "merged_video": "",
                    "mode": "separate",
                    "parts": parts,
                    "failed": failed,
                }, f, ensure_ascii=False, indent=2)
            self.log_line.emit(
                f"Da luu {len(parts)} video rieng thanh Parts: {Path(sidecar).name}"
            )
            if failed:
                self.log_line.emit(
                    f"Bo qua {len(failed)} URL bi loi. Xem logs/app.log neu can chi tiet."
                )
            self.finished.emit(True, downloaded[0])
            return

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = str(Path(self.output_dir) / f"merged_downloads_{stamp}.mp4")
        trans_used = 0.0
        if self.transition and self.transition != "none":
            self.log_line.emit(
                f"Dang ghep {len(downloaded)} video voi hieu ung {self.transition}..."
            )
            durations = [max(0.1, item["duration"]) for item in downloaded_meta]
            max_trans = max(0.1, min(durations) - 0.1)
            trans_used = max(0.1, min(float(self.trans_dur), max_trans))
            ok, result = merge_videos(
                downloaded,
                self.transition,
                self.trans_dur,
                out_path,
                progress_cb=lambda e, t: self.progress.emit(e, t),
            )
        else:
            self.log_line.emit(f"Dang ghep {len(downloaded)} video thanh 1 file...")
            ok, result = concatenate_videos(
                downloaded,
                out_path,
                progress_cb=lambda e, t: self.progress.emit(e, t),
            )
        if ok:
            start = 0.0
            parts = []
            for item in downloaded_meta:
                duration = max(0.1, float(item.get("duration", 0) or 0))
                end = start + duration
                parts.append({
                    **item,
                    "source_index": item["index"],
                    "start_time": round(start, 3),
                    "end_time": round(end, 3),
                    "part_text": f"PART {item['index']}",
                    "transition_overlap": round(trans_used, 3),
                })
                start = end - trans_used
            sidecar = str(Path(out_path).with_suffix(".parts.json"))
            with open(sidecar, "w", encoding="utf-8") as f:
                json.dump({
                    "merged_video": out_path,
                    "transition": self.transition or "none",
                    "transition_duration": trans_used,
                    "parts": parts,
                }, f, ensure_ascii=False, indent=2)
            self.log_line.emit(f"Da luu thong tin chia Part: {Path(sidecar).name}")
        self.finished.emit(ok, out_path if ok else result)


class MetadataWorker(QThread):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, path: str):
        super().__init__()
        self.path = path

    def run(self):
        from src.core.video_manager import get_video_metadata
        meta = get_video_metadata(self.path)
        if meta:
            self.finished.emit(meta)
        else:
            self.error.emit("Không đọc được metadata video.")


class ExportWorker(QThread):
    clip_started = pyqtSignal(int, str)       # index, part_text
    clip_progress = pyqtSignal(int, float, float)   # index, elapsed, total
    clip_done = pyqtSignal(int, bool, str)    # index, success, path_or_error
    all_done = pyqtSignal(list)               # list of exported paths

    def __init__(self, project, clips, cfg, out_dir):
        super().__init__()
        self.project = project
        self.clips = clips
        self.cfg = cfg
        self.out_dir = out_dir

    def run(self):
        from src.core.video_processor import export_clip
        from src.utils.file_utils import sanitize_filename
        from pathlib import Path

        exported = []
        for clip in self.clips:
            if not clip.enabled:
                continue
            clip.export_path = ""
            if getattr(clip, "is_review_master", False):
                filename = (
                    f"{sanitize_filename(self.project.name)}_final_review.mp4"
                )
            else:
                safe_label = sanitize_filename(clip.part_text) or f"part{clip.index:02d}"
                filename = f"part{clip.index:02d}_{safe_label}.mp4"
            out_path = str(Path(self.out_dir) / filename)
            self.clip_started.emit(clip.index, clip.part_text)
            ok, err = export_clip(
                self.project,
                clip,
                self.cfg,
                out_path,
                progress_cb=lambda e, t, idx=clip.index: self.clip_progress.emit(idx, e, t),
            )
            if ok:
                clip.export_path = out_path
                exported.append(out_path)
            # Pass path on success; if there's a fallback warning, append it
            if ok and err:
                # GPU→CPU fallback: success but with warning message
                self.clip_done.emit(clip.index, ok, f"{out_path}\n⚠️ {err}")
            else:
                self.clip_done.emit(clip.index, ok, out_path if ok else err)
        self.all_done.emit(exported)


class ThumbnailWorker(QThread):
    """Extract a single video frame for live preview in export tab."""
    finished = pyqtSignal(str)   # path to thumbnail (empty = failed)

    def __init__(self, video_path: str, out_path: str, time_offset: float = 30.0):
        super().__init__()
        self.video_path = video_path
        self.out_path = out_path
        self.time_offset = time_offset

    def run(self):
        from src.core.video_processor import extract_thumbnail
        ok = extract_thumbnail(self.video_path, self.out_path, self.time_offset)
        self.finished.emit(self.out_path if ok else "")


class WideThumbnailWorker(QThread):
    """Extract a wide (original-aspect-ratio) thumbnail for the crop-adjustment widget."""
    finished = pyqtSignal(str)   # path to thumbnail (empty = failed)

    def __init__(self, video_path: str, out_path: str, time_offset: float = 30.0):
        super().__init__()
        self.video_path = video_path
        self.out_path = out_path
        self.time_offset = time_offset

    def run(self):
        from src.core.video_processor import extract_wide_thumbnail
        ok = extract_wide_thumbnail(self.video_path, self.out_path, self.time_offset)
        self.finished.emit(self.out_path if ok else "")


class PreviewWorker(QThread):
    finished = pyqtSignal(bool, str, object)   # success, path, clip

    def __init__(self, project, clip):
        super().__init__()
        self.project = project
        self.clip = clip

    def run(self):
        from src.core.video_processor import generate_preview
        ok, err = generate_preview(self.project, self.clip)
        path = self.clip.preview_path if ok else err
        self.finished.emit(ok, path, self.clip)


class YouTubeResearchWorker(QThread):
    status = pyqtSignal(str)
    finished = pyqtSignal(bool, str)      # success, report_path_or_error

    def __init__(self, keywords, api_key, out_dir,
                 run_ai=False, ai_provider="gemini", ai_model="",
                 max_per_keyword=20):
        super().__init__()
        self.keywords = keywords
        self.api_key = api_key
        self.out_dir = out_dir
        self.run_ai = run_ai
        self.ai_provider = ai_provider
        self.ai_model = ai_model
        self.max_per_keyword = max_per_keyword
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        from src.core.youtube_research import run_pipeline, StopRequested
        try:
            report = run_pipeline(
                keywords=self.keywords,
                api_key=self.api_key,
                out_dir=self.out_dir,
                run_ai=self.run_ai,
                ai_provider=self.ai_provider,
                ai_model=self.ai_model,
                max_per_keyword=self.max_per_keyword,
                status_cb=lambda m: self.status.emit(m),
                stop_flag=lambda: self._stop,
            )
            self.finished.emit(True, report)
        except StopRequested:
            self.finished.emit(False, "⏹ Đã dừng theo yêu cầu.")
        except Exception as e:
            self.finished.emit(False, str(e))


class MergeVideosWorker(QThread):
    progress = pyqtSignal(float, float)   # elapsed, total
    finished = pyqtSignal(bool, str)      # success, output_path_or_error

    def __init__(self, paths, transition, trans_dur, out_path,
                 target_w=1080, target_h=1920, fps=30,
                 clip_max_durations=None):
        super().__init__()
        self.paths = paths
        self.transition = transition
        self.trans_dur = trans_dur
        self.out_path = out_path
        self.target_w = target_w
        self.target_h = target_h
        self.fps = fps
        self.clip_max_durations = clip_max_durations

    def run(self):
        from src.core.video_merger import concatenate_videos, merge_videos
        try:
            if self.transition == "none":
                ok, msg = concatenate_videos(
                    self.paths,
                    self.out_path,
                    self.target_w,
                    self.target_h,
                    self.fps,
                    progress_cb=lambda e, t: self.progress.emit(e, t),
                    clip_max_durations=self.clip_max_durations,
                )
            else:
                ok, msg = merge_videos(
                    self.paths, self.transition, self.trans_dur, self.out_path,
                    self.target_w, self.target_h, self.fps,
                    progress_cb=lambda e, t: self.progress.emit(e, t),
                    clip_max_durations=self.clip_max_durations,
                )
            if ok:
                # msg may carry a non-fatal warning; pass path + warning
                self.finished.emit(True, self.out_path + (("\n⚠️ " + msg) if msg else ""))
            else:
                self.finished.emit(False, msg)
        except Exception as e:
            self.finished.emit(False, str(e))
