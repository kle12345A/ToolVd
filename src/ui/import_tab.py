"""Tab 1: Import video."""

import json
import os
from pathlib import Path

from PyQt6.QtCore import Qt, QSettings, pyqtSignal
from PyQt6.QtGui import QDragEnterEvent, QDropEvent
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QFileDialog, QProgressBar, QTextEdit, QGroupBox,
    QSizePolicy, QFrame, QCheckBox, QComboBox, QDoubleSpinBox,
)

from src.core.video_merger import TRANSITIONS
from src.ui.workers import (
    DownloadVideoWorker, DownloadSocialProfileWorker,
    BatchDownloadMergeWorker, MetadataWorker,
)
from src.utils.file_utils import format_duration, file_size_str


class DropArea(QFrame):
    files_dropped = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setMinimumHeight(120)
        self.setStyleSheet("""
            QFrame {
                border: 2px dashed #555;
                border-radius: 8px;
                background: #1e1e2e;
            }
            QFrame:hover { border-color: #7aa2f7; }
        """)
        layout = QVBoxLayout(self)
        lbl = QLabel("🎬  Kéo & thả video vào đây\n(mp4, mkv, mov, avi, webm)")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet("color: #888; font-size: 14px; border: none;")
        layout.addWidget(lbl)

    def dragEnterEvent(self, e: QDragEnterEvent):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()
            self.setStyleSheet(self.styleSheet().replace("#555", "#7aa2f7"))

    def dragLeaveEvent(self, e):
        self.setStyleSheet(self.styleSheet().replace("#7aa2f7", "#555"))

    def dropEvent(self, e: QDropEvent):
        self.setStyleSheet(self.styleSheet().replace("#7aa2f7", "#555"))
        paths = [u.toLocalFile() for u in e.mimeData().urls()]
        valid_exts = {".mp4", ".mkv", ".mov", ".avi", ".webm"}
        valid = [p for p in paths if Path(p).suffix.lower() in valid_exts]
        if valid:
            self.files_dropped.emit(valid)


class ImportTab(QWidget):
    video_imported = pyqtSignal(str, dict)   # path, metadata

    def __init__(self, parent=None):
        super().__init__(parent)
        self._dl_worker = None
        self._channel_worker = None
        self._batch_worker = None
        self._meta_worker = None
        self._pending_batch_parts = {}
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # Drop area
        self.drop_area = DropArea()
        self.drop_area.files_dropped.connect(self._on_files_dropped)
        layout.addWidget(self.drop_area)

        # File picker
        pick_row = QHBoxLayout()
        self.btn_browse = QPushButton("📁 Chọn File Video...")
        self.btn_browse.setFixedHeight(36)
        self.btn_browse.clicked.connect(self._browse_file)
        self.lbl_selected = QLabel("Chưa chọn file")
        self.lbl_selected.setStyleSheet("color: #888;")
        pick_row.addWidget(self.btn_browse)
        pick_row.addWidget(self.lbl_selected, 1)
        layout.addLayout(pick_row)

        # URL download
        url_group = QGroupBox(
            "Tải video từ URL (YouTube, TikTok, Facebook, Douyin, ...)"
        )
        url_layout = QVBoxLayout(url_group)
        url_row = QHBoxLayout()
        self.txt_url = QLineEdit()
        self.txt_url.setPlaceholderText(
            "Dán URL video YouTube, TikTok, Facebook/Reels... vào đây"
        )
        self.txt_url.setToolTip(
            "Facebook: mở video hoặc Reel → Chia sẻ → Sao chép liên kết. "
            "Video riêng tư có thể cần bật cookies trình duyệt."
        )
        self.btn_download = QPushButton("⬇ Tải xuống")
        self.btn_download.setFixedWidth(110)
        self.btn_download.clicked.connect(self._start_download)
        url_row.addWidget(self.txt_url)
        url_row.addWidget(self.btn_download)
        url_layout.addLayout(url_row)

        channel_row = QHBoxLayout()
        channel_row.addWidget(QLabel("Toàn bộ video:"))
        self.cmb_social_platform = QComboBox()
        self.cmb_social_platform.addItem("TikTok", "tiktok")
        self.cmb_social_platform.addItem("YouTube", "youtube")
        self.cmb_social_platform.addItem("Instagram", "instagram")
        self.cmb_social_platform.addItem("Facebook Reels", "facebook")
        self.cmb_social_platform.addItem("Douyin", "douyin")
        saved_platform = QSettings("AI Movie Studio", "Downloader").value(
            "social_platform", "tiktok", type=str
        )
        platform_index = self.cmb_social_platform.findData(saved_platform)
        self.cmb_social_platform.setCurrentIndex(platform_index if platform_index >= 0 else 0)
        channel_row.addWidget(self.cmb_social_platform)
        self.txt_tiktok_account = QLineEdit()
        self.txt_tiktok_account.setPlaceholderText("Nhập @username hoặc link trang/kênh")
        self.txt_tiktok_account.setText(
            QSettings("AI Movie Studio", "Downloader").value(
                f"{saved_platform}_account", "", type=str
            )
        )
        self.txt_tiktok_account.setToolTip(
            "Nhập tên tài khoản hoặc link trang cá nhân; Douyin nên dùng link profile. "
            "Những lần sau tool chỉ tải các video mới."
        )
        self.cmb_channel_limit = QComboBox()
        for amount in (10, 20, 50, 100, 200):
            self.cmb_channel_limit.addItem(f"{amount} video", amount)
        self.cmb_channel_limit.addItem("Toàn bộ", 0)
        saved_limit = QSettings("AI Movie Studio", "Downloader").value(
            "tiktok_limit", 20, type=int
        )
        saved_index = self.cmb_channel_limit.findData(saved_limit)
        self.cmb_channel_limit.setCurrentIndex(saved_index if saved_index >= 0 else 1)
        self.cmb_channel_limit.setToolTip(
            "Số video mới cần tải trong mỗi lần. "
            "Lần sau tool sẽ tiếp tục với nhóm video kế tiếp."
        )
        self.btn_download_channel = QPushButton("Tải trang/kênh")
        self.btn_download_channel.setFixedWidth(110)
        self.btn_download_channel.clicked.connect(self._start_channel_download)
        channel_row.addWidget(self.txt_tiktok_account, 1)
        channel_row.addWidget(self.cmb_channel_limit)
        channel_row.addWidget(self.btn_download_channel)
        url_layout.addLayout(channel_row)
        self.cmb_social_platform.currentIndexChanged.connect(self._on_social_platform_changed)
        self._on_social_platform_changed()

        channel_hint = QLabel(
            "Hỗ trợ TikTok, YouTube, Instagram, Facebook Reels và Douyin. "
            "Facebook sẽ mở một cửa sổ Chrome/Edge riêng để đăng nhập và quét Reels. "
            "Tải lại sẽ bỏ qua video đã có và tải tiếp nhóm kế tiếp."
        )
        channel_hint.setStyleSheet("color: #888; font-size: 11px;")
        channel_hint.setWordWrap(True)
        url_layout.addWidget(channel_hint)

        self.chk_batch_download = QCheckBox(
            "Tai nhieu URL roi ghep thanh 1 video truoc khi tao Project"
        )
        self.chk_batch_download.toggled.connect(self._toggle_batch_mode)
        url_layout.addWidget(self.chk_batch_download)

        self.url_rows = []
        self.url_rows_widget = QWidget()
        self.url_rows_layout = QVBoxLayout(self.url_rows_widget)
        self.url_rows_layout.setContentsMargins(0, 0, 0, 0)
        self.url_rows_layout.setSpacing(6)
        self.url_rows_widget.setVisible(False)
        url_layout.addWidget(self.url_rows_widget)
        self._add_url_row()

        self.btn_add_url = QPushButton("+ Them link")
        self.btn_add_url.setFixedWidth(120)
        self.btn_add_url.setVisible(False)
        self.btn_add_url.clicked.connect(self._add_url_row)
        url_layout.addWidget(self.btn_add_url, alignment=Qt.AlignmentFlag.AlignLeft)

        self.batch_mode_row = QHBoxLayout()
        self.batch_mode_row.addWidget(QLabel("Che do:"))
        self.cmb_batch_mode = QComboBox()
        self.cmb_batch_mode.addItem("Ghep thanh 1 video", "merge")
        self.cmb_batch_mode.addItem("Giu rieng: moi video = 1 Part", "separate")
        self.cmb_batch_mode.currentIndexChanged.connect(self._on_batch_merge_mode_changed)
        self.batch_mode_row.addWidget(self.cmb_batch_mode, 1)
        self.batch_mode_widget = QWidget()
        self.batch_mode_widget.setLayout(self.batch_mode_row)
        self.batch_mode_widget.setVisible(False)
        url_layout.addWidget(self.batch_mode_widget)

        self.batch_fx_row = QHBoxLayout()
        self.lbl_batch_transition = QLabel("Hieu ung:")
        self.cmb_batch_transition = QComboBox()
        self.cmb_batch_transition.addItem("Khong co", "none")
        for key, label in TRANSITIONS.items():
            self.cmb_batch_transition.addItem(label, key)
        self.cmb_batch_transition.setCurrentIndex(0)
        self.lbl_batch_trans_dur = QLabel("Thoi luong:")
        self.spn_batch_trans_dur = QDoubleSpinBox()
        self.spn_batch_trans_dur.setRange(0.1, 3.0)
        self.spn_batch_trans_dur.setSingleStep(0.1)
        self.spn_batch_trans_dur.setValue(0.7)
        self.spn_batch_trans_dur.setSuffix(" s")
        self.batch_fx_row.addWidget(self.lbl_batch_transition)
        self.batch_fx_row.addWidget(self.cmb_batch_transition, 1)
        self.batch_fx_row.addWidget(self.lbl_batch_trans_dur)
        self.batch_fx_row.addWidget(self.spn_batch_trans_dur)
        self.batch_fx_widget = QWidget()
        self.batch_fx_widget.setLayout(self.batch_fx_row)
        self.batch_fx_widget.setVisible(False)
        url_layout.addWidget(self.batch_fx_widget)

        cookie_row = QHBoxLayout()
        self.chk_use_cookies = QCheckBox("Dùng cookies trình duyệt")
        self.chk_use_cookies.setToolTip(
            "Bật khi YouTube/Facebook hoặc nền tảng khác yêu cầu đăng nhập."
        )
        self.cmb_cookie_browser = QComboBox()
        self.cmb_cookie_browser.addItem("Chrome", "chrome")
        self.cmb_cookie_browser.addItem("Edge", "edge")
        self.cmb_cookie_browser.addItem("Firefox", "firefox")
        self.cmb_cookie_browser.addItem("Brave", "brave")
        self.cmb_cookie_browser.setEnabled(False)
        self.chk_use_cookies.toggled.connect(self.cmb_cookie_browser.setEnabled)
        self.txt_cookies_file = QLineEdit()
        self.txt_cookies_file.setPlaceholderText("Hoặc chọn file cookies.txt")
        self.txt_cookies_file.setReadOnly(True)
        self.txt_cookies_file.setEnabled(False)
        self.btn_cookies_file = QPushButton("cookies.txt")
        self.btn_cookies_file.setEnabled(False)
        self.btn_cookies_file.clicked.connect(self._browse_cookies_file)
        self.chk_use_cookies.toggled.connect(self.txt_cookies_file.setEnabled)
        self.chk_use_cookies.toggled.connect(self.btn_cookies_file.setEnabled)
        cookie_row.addWidget(self.chk_use_cookies)
        cookie_row.addWidget(self.cmb_cookie_browser)
        cookie_row.addWidget(self.txt_cookies_file, 1)
        cookie_row.addWidget(self.btn_cookies_file)
        cookie_row.addStretch()
        url_layout.addLayout(cookie_row)

        # Download destination folder
        self._download_dir = str(Path("output") / "_downloads")
        dl_dir_row = QHBoxLayout()
        dl_dir_row.addWidget(QLabel("Lưu vào:"))
        self.lbl_dl_dir = QLabel(self._download_dir)
        self.lbl_dl_dir.setStyleSheet("color: #a6adc8; font-size: 11px;")
        self.btn_dl_dir = QPushButton("📁 Chọn thư mục...")
        self.btn_dl_dir.clicked.connect(self._choose_download_dir)
        dl_dir_row.addWidget(self.lbl_dl_dir, 1)
        dl_dir_row.addWidget(self.btn_dl_dir)
        url_layout.addLayout(dl_dir_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        url_layout.addWidget(self.progress_bar)

        self.txt_dl_log = QTextEdit()
        self.txt_dl_log.setReadOnly(True)
        self.txt_dl_log.setMaximumHeight(100)
        self.txt_dl_log.setStyleSheet("background: #111; color: #aaa; font-size: 11px;")
        self.txt_dl_log.setVisible(False)
        url_layout.addWidget(self.txt_dl_log)
        layout.addWidget(url_group)

        # Metadata display
        meta_group = QGroupBox("Thông tin video")
        meta_layout = QVBoxLayout(meta_group)
        self.lbl_meta = QLabel("Chưa có video nào được chọn.")
        self.lbl_meta.setWordWrap(True)
        self.lbl_meta.setStyleSheet("color: #cdd6f4; font-family: monospace;")
        meta_layout.addWidget(self.lbl_meta)

        self.btn_use_video = QPushButton("✅ Dùng video này → Tạo Project")
        self.btn_use_video.setFixedHeight(40)
        self.btn_use_video.setEnabled(False)
        self.btn_use_video.setStyleSheet(
            "background: #313244; color: #a6e3a1; font-weight: bold;"
        )
        self.btn_use_video.clicked.connect(self._use_video)
        meta_layout.addWidget(self.btn_use_video)
        layout.addWidget(meta_group)

        layout.addStretch()

        self._current_path = ""
        self._current_meta = {}

    def _browse_cookies_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Chọn file cookies.txt",
            "",
            "Cookies (*.txt);;All Files (*)",
        )
        if path:
            self.txt_cookies_file.setText(path)

    def _browse_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Chọn file video",
            "",
            "Video Files (*.mp4 *.mkv *.mov *.avi *.webm);;All Files (*)",
        )
        if path:
            self._load_video(path)

    def _on_files_dropped(self, paths: list):
        if paths:
            self._load_video(paths[0])

    def _load_video(self, path: str):
        self._current_path = path
        self.lbl_selected.setText(Path(path).name)
        self.lbl_meta.setText("Đang đọc metadata...")
        self.btn_use_video.setEnabled(False)
        self._meta_worker = MetadataWorker(path)
        self._meta_worker.finished.connect(self._on_meta_loaded)
        self._meta_worker.error.connect(self._on_meta_error)
        self._meta_worker.start()

    def _on_meta_loaded(self, meta: dict):
        batch_parts = self._pending_batch_parts.get(self._current_path)
        if batch_parts:
            meta["batch_parts"] = batch_parts
        self._current_meta = meta
        txt = (
            f"📄 File: {meta.get('filename', '')}\n"
            f"⏱ Thời lượng: {format_duration(meta.get('duration', 0))}\n"
            f"📐 Độ phân giải: {meta.get('width')}x{meta.get('height')}\n"
            f"🎞 FPS: {meta.get('fps')}\n"
            f"🎬 Codec video: {meta.get('video_codec')}\n"
            f"🔊 Codec audio: {meta.get('audio_codec')}\n"
            f"💾 Dung lượng: {meta.get('size_mb')} MB\n"
            f"📦 Định dạng: {meta.get('format', '')}"
        )
        self.lbl_meta.setText(txt)
        self.btn_use_video.setEnabled(True)

    def _on_meta_error(self, err: str):
        self.lbl_meta.setText(f"❌ Lỗi: {err}")
        self.btn_use_video.setEnabled(False)

    def _choose_download_dir(self):
        d = QFileDialog.getExistingDirectory(
            self, "Chọn thư mục lưu video tải về", self._download_dir
        )
        if d:
            self._download_dir = d
            self.lbl_dl_dir.setText(d)

    def _on_dl_log(self, line: str):
        self.txt_dl_log.append(line)
        sb = self.txt_dl_log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _on_dl_finished(self, ok: bool, result: str):
        self.btn_download.setEnabled(True)
        self.btn_download_channel.setEnabled(True)
        self.progress_bar.setVisible(False)
        if ok:
            self.txt_dl_log.append(f"✅ Tải xong: {result}")
            sidecar = Path(result).with_suffix(".parts.json")
            if sidecar.exists():
                try:
                    with open(sidecar, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    parts = data.get("parts") or []
                    if parts:
                        self._pending_batch_parts[result] = parts
                        self.txt_dl_log.append(
                            f"Da san sang tao {len(parts)} Parts tu cac video da tai."
                        )
                except Exception as e:
                    self.txt_dl_log.append(f"Khong doc duoc thong tin chia Part: {e}")
            self._load_video(result)
        else:
            self.txt_dl_log.append(f"❌ Lỗi: {result}")

    def _on_channel_dl_finished(self, ok: bool, result: str, count: int):
        self.btn_download.setEnabled(True)
        self.btn_download_channel.setEnabled(True)
        self.progress_bar.setVisible(False)
        if ok:
            if count:
                self.txt_dl_log.append(
                    f"Đã tải xong {count} video mới. Thư mục: {result}"
                )
            else:
                self.txt_dl_log.append(
                    f"Kênh không có video mới; các video cũ đã được bỏ qua. Thư mục: {result}"
                )
        else:
            platform_name = self.cmb_social_platform.currentText()
            self.txt_dl_log.append(f"Lỗi tải {platform_name}: {result}")

    def _on_social_platform_changed(self):
        platform = self.cmb_social_platform.currentData()
        settings = QSettings("AI Movie Studio", "Downloader")
        settings.setValue("social_platform", platform)
        self.txt_tiktok_account.setToolTip(
            "Nhập tên tài khoản hoặc link trang cá nhân. "
            "Những lần sau tool chỉ tải các video mới."
        )
        self.txt_tiktok_account.setText(
            settings.value(f"{platform}_account", "", type=str)
        )
        if platform == "youtube":
            self.txt_tiktok_account.setPlaceholderText("Nhập @handle hoặc link kênh YouTube")
        elif platform == "instagram":
            self.txt_tiktok_account.setPlaceholderText("Nhập @username hoặc link profile Instagram")
        elif platform == "facebook":
            self.txt_tiktok_account.setPlaceholderText(
                "Dán link trang Facebook, ví dụ facebook.com/TenTrang"
            )
            self.txt_tiktok_account.setToolTip(
                "Tool mở trình duyệt Facebook riêng, tự cuộn trang Reels và "
                "ghi nhớ video đã tải. Lần đầu có thể cần đăng nhập."
            )
        elif platform == "douyin":
            self.txt_tiktok_account.setPlaceholderText(
                "Dán link trang cá nhân hoặc user ID Douyin"
            )
        else:
            self.txt_tiktok_account.setPlaceholderText("Nhập @username hoặc link profile TikTok")

    def _start_channel_download(self):
        account = self.txt_tiktok_account.text().strip()
        platform = self.cmb_social_platform.currentData()
        platform_name = self.cmb_social_platform.currentText()
        from src.core.downloader import normalize_social_account
        valid, username, error_or_url = normalize_social_account(platform, account)
        self.txt_dl_log.setVisible(True)
        if not valid:
            self.txt_dl_log.append(error_or_url)
            return

        from src.core.dependency_manager import is_ytdlp_available
        if not is_ytdlp_available():
            self.txt_dl_log.append(
                "yt-dlp chưa được cài. Vào Cài đặt để cài dependencies."
            )
            return

        settings = QSettings("AI Movie Studio", "Downloader")
        settings.setValue("social_platform", platform)
        settings.setValue(f"{platform}_account", f"@{username}")
        max_videos = int(self.cmb_channel_limit.currentData() or 0)
        QSettings("AI Movie Studio", "Downloader").setValue(
            "tiktok_limit", max_videos
        )
        self.txt_tiktok_account.setText(f"@{username}")
        Path(self._download_dir).mkdir(parents=True, exist_ok=True)
        cookies_browser = (
            self.cmb_cookie_browser.currentData()
            if self.chk_use_cookies.isChecked()
            else ""
        )
        cookies_file = (
            self.txt_cookies_file.text().strip()
            if self.chk_use_cookies.isChecked()
            else ""
        )
        if cookies_file:
            cookies_browser = ""

        self.btn_download.setEnabled(False)
        self.btn_download_channel.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.txt_dl_log.clear()
        limit_label = f"{max_videos} video chưa tải tiếp theo" if max_videos else "toàn bộ kênh"
        self.txt_dl_log.append(f"Đang quét {limit_label} của {platform_name} @{username}...")
        if platform == "facebook":
            self.txt_dl_log.append(
                "Cửa sổ Facebook riêng sẽ mở. Nếu là lần đầu, hãy đăng nhập "
                "và giữ cửa sổ mở để tool tự cuộn trang Reels."
            )
        self._channel_worker = DownloadSocialProfileWorker(
            platform,
            username,
            self._download_dir,
            cookies_browser,
            cookies_file,
            max_videos,
        )
        self._channel_worker.log_line.connect(self._on_dl_log)
        self._channel_worker.finished.connect(self._on_channel_dl_finished)
        self._channel_worker.start()

    def _start_download(self):
        urls = self._download_urls()
        if not urls:
            return
        batch_mode = self.chk_batch_download.isChecked() and len(urls) > 1

        from src.core.dependency_manager import is_ytdlp_available, is_ffmpeg_available
        if not is_ytdlp_available():
            self.txt_dl_log.setVisible(True)
            self.txt_dl_log.append("yt-dlp chua duoc cai. Vao Cai dat de cai dependencies.")
            return
        if batch_mode and not is_ffmpeg_available():
            self.txt_dl_log.setVisible(True)
            self.txt_dl_log.append("FFmpeg chua duoc cai. Can FFmpeg de ghep nhieu video.")
            return

        self.btn_download.setEnabled(False)
        self.btn_download_channel.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.txt_dl_log.setVisible(True)
        self.txt_dl_log.clear()

        downloads_dir = self._download_dir
        Path(downloads_dir).mkdir(parents=True, exist_ok=True)
        cookies_browser = (
            self.cmb_cookie_browser.currentData()
            if self.chk_use_cookies.isChecked()
            else ""
        )
        cookies_file = (
            self.txt_cookies_file.text().strip()
            if self.chk_use_cookies.isChecked()
            else ""
        )
        if cookies_file:
            self.txt_dl_log.append(f"Dung cookies file: {Path(cookies_file).name}")
            cookies_browser = ""
        elif cookies_browser:
            self.txt_dl_log.append(
                f"Dung cookies tu {self.cmb_cookie_browser.currentText()}..."
            )

        if batch_mode:
            mode_label = (
                "giu rieng moi video thanh 1 Part"
                if (self.cmb_batch_mode.currentData() or "merge") == "separate"
                else "ghep thanh 1 file"
            )
            self.txt_dl_log.append(f"Batch: tai {len(urls)} video, {mode_label}.")
            self._batch_worker = BatchDownloadMergeWorker(
                urls,
                downloads_dir,
                cookies_browser,
                cookies_file,
                transition=self.cmb_batch_transition.currentData() or "none",
                trans_dur=self.spn_batch_trans_dur.value(),
                merge_mode=self.cmb_batch_mode.currentData() or "merge",
            )
            self._batch_worker.log_line.connect(self._on_dl_log)
            self._batch_worker.progress.connect(self._on_batch_merge_progress)
            self._batch_worker.finished.connect(self._on_dl_finished)
            self._batch_worker.start()
        else:
            self._dl_worker = DownloadVideoWorker(
                urls[0], downloads_dir, cookies_browser, cookies_file
            )
            self._dl_worker.log_line.connect(self._on_dl_log)
            self._dl_worker.finished.connect(self._on_dl_finished)
            self._dl_worker.start()

    def _toggle_batch_mode(self, checked: bool):
        self.url_rows_widget.setVisible(checked)
        self.btn_add_url.setVisible(checked)
        self.batch_mode_widget.setVisible(checked)
        self._on_batch_merge_mode_changed()
        self.txt_url.setEnabled(not checked)

    def _on_batch_merge_mode_changed(self):
        show_fx = (
            self.chk_batch_download.isChecked()
            and (self.cmb_batch_mode.currentData() or "merge") == "merge"
        )
        self.batch_fx_widget.setVisible(show_fx)

    def _download_urls(self) -> list[str]:
        if self.chk_batch_download.isChecked():
            urls = []
            for edit, _row, _btn in self.url_rows:
                text = edit.text().strip()
                if text:
                    urls.append(text)
            return urls
        url = self.txt_url.text().strip()
        return [url] if url else []

    def _add_url_row(self):
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(6)

        edit = QLineEdit()
        edit.setPlaceholderText(f"URL video #{len(self.url_rows) + 1}")
        edit.textChanged.connect(lambda _text, e=edit: self._split_pasted_urls(e))

        btn_remove = QPushButton("X")
        btn_remove.setFixedWidth(34)
        btn_remove.clicked.connect(lambda _checked=False, r=row: self._remove_url_row(r))

        row_layout.addWidget(edit, 1)
        row_layout.addWidget(btn_remove)
        self.url_rows_layout.addWidget(row)
        self.url_rows.append((edit, row, btn_remove))
        self._refresh_url_rows()
        edit.setFocus()

    def _remove_url_row(self, row: QWidget):
        if len(self.url_rows) <= 1:
            self.url_rows[0][0].clear()
            return
        for item in list(self.url_rows):
            edit, row_widget, _btn = item
            if row_widget is row:
                self.url_rows.remove(item)
                self.url_rows_layout.removeWidget(row_widget)
                row_widget.deleteLater()
                break
        self._refresh_url_rows()

    def _refresh_url_rows(self):
        for idx, (edit, _row, btn_remove) in enumerate(self.url_rows, start=1):
            edit.setPlaceholderText(f"URL video #{idx}")
            btn_remove.setEnabled(len(self.url_rows) > 1)

    def _split_pasted_urls(self, edit: QLineEdit):
        text = edit.text()
        if "\n" not in text and "\r" not in text:
            return
        urls = [line.strip() for line in text.splitlines() if line.strip()]
        if not urls:
            edit.clear()
            return
        edit.blockSignals(True)
        edit.setText(urls[0])
        edit.blockSignals(False)
        for url in urls[1:]:
            self._add_url_row()
            self.url_rows[-1][0].setText(url)

    def _on_batch_merge_progress(self, elapsed: float, total: float):
        if total > 0:
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(min(100, int(elapsed / total * 100)))

    def _use_video(self):
        if self._current_path and self._current_meta:
            self.video_imported.emit(self._current_path, self._current_meta)
