"""Main application window."""

import os
import uuid
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QInputDialog, QMessageBox,
    QTabWidget, QStatusBar, QToolBar, QFileDialog,
    QDialog, QListWidget, QListWidgetItem, QDialogButtonBox,
)

from src.core.project_manager import (
    create_project, load_project, list_projects, save_project, delete_project,
)
from src.core.dependency_manager import (
    check_and_report, ffmpeg_path, is_ffmpeg_available,
)
from src.core.tts_manager import probe_media_duration
from src.core.video_manager import get_video_metadata
from src.core.workflow import next_step, resume_step, step_access
from src.models.project import Project
from src.models.clip import Clip
from src.ui.import_tab import ImportTab
from src.ui.transcribe_tab import TranscribeTab
from src.ui.export_tab import ExportTab
from src.ui.voiceover_tab import VoiceoverTab
from src.ui.settings_dialog import SettingsDialog
from src.utils.logger import logger

APP_TITLE = "AI Movie Short & Review Studio"
APP_VERSION = "2.0.0"


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self._project: Project | None = None
        self._all_features_unlocked = False
        self.setWindowTitle(f"{APP_TITLE} v{APP_VERSION}")
        self.setMinimumSize(1050, 680)
        self._apply_dark_theme()
        self._resize_to_screen()
        self._setup_ui()
        self._check_deps_on_startup()

    def _resize_to_screen(self):
        """Scale window to 88 % of available screen area, then center."""
        screen = QApplication.primaryScreen()
        if not screen:
            self.resize(1280, 820)
            return
        avail = screen.availableGeometry()
        w = max(1050, min(1600, int(avail.width()  * 0.88)))
        h = max(680,  min(1020, int(avail.height() * 0.88)))
        self.resize(w, h)
        self.move(
            avail.x() + (avail.width()  - w) // 2,
            avail.y() + (avail.height() - h) // 2,
        )

    def _apply_dark_theme(self):
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #1e1e2e;
                color: #cdd6f4;
                font-family: 'Segoe UI', sans-serif;
                font-size: 13px;
            }
            QTabWidget::pane { border: 1px solid #313244; }
            QTabBar::tab {
                background: #313244; color: #cdd6f4;
                padding: 8px 16px; border-radius: 4px 4px 0 0;
            }
            QTabBar::tab:selected { background: #45475a; color: #cba6f7; }
            QGroupBox {
                border: 1px solid #313244; border-radius: 6px;
                margin-top: 8px; padding-top: 6px;
                font-weight: bold;
            }
            QGroupBox::title { color: #89b4fa; subcontrol-origin: margin; left: 8px; }
            QPushButton {
                background: #313244; color: #cdd6f4;
                border: 1px solid #45475a; border-radius: 4px; padding: 5px 12px;
            }
            QPushButton:hover { background: #45475a; }
            QPushButton:disabled { color: #6c7086; }
            QLineEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                background: #181825; color: #cdd6f4;
                border: 1px solid #45475a; border-radius: 4px; padding: 4px;
            }
            QTableWidget {
                background: #181825; gridline-color: #313244;
                selection-background-color: #313244;
            }
            QHeaderView::section {
                background: #313244; color: #89b4fa;
                border: none; padding: 4px;
            }
            QProgressBar {
                background: #181825; border: 1px solid #45475a; border-radius: 3px;
            }
            QProgressBar::chunk { background: #7aa2f7; border-radius: 3px; }
            QScrollBar:vertical { background: #181825; width: 8px; }
            QScrollBar::handle:vertical { background: #45475a; border-radius: 4px; }
            QStatusBar { background: #181825; color: #888; }
            QToolBar { background: #181825; border: none; spacing: 4px; }
            QSplitter::handle { background: #313244; }
            QSlider::groove:horizontal { background: #313244; height: 4px; border-radius: 2px; }
            QSlider::handle:horizontal { background: #7aa2f7; width: 14px; height: 14px;
                margin: -5px 0; border-radius: 7px; }
        """)

    def _setup_ui(self):
        # Toolbar
        toolbar = QToolBar("Main")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        action_new = QAction("📂 Project mới", self)
        action_new.triggered.connect(self._new_project_from_toolbar)
        toolbar.addAction(action_new)

        action_open = QAction("📁 Mở Project", self)
        action_open.triggered.connect(self._open_project)
        toolbar.addAction(action_open)

        toolbar.addSeparator()

        # Output folder button (Phase 2 request)
        action_outdir = QAction("💾 Thư mục output", self)
        action_outdir.setToolTip("Chọn thư mục lưu tất cả video xuất")
        action_outdir.triggered.connect(self._set_global_output_dir)
        toolbar.addAction(action_outdir)

        toolbar.addSeparator()

        action_settings = QAction("⚙ Cài đặt", self)
        action_settings.triggered.connect(self._open_settings)
        toolbar.addAction(action_settings)

        toolbar.addSeparator()
        self.lbl_project = QLabel("  Chưa có project  ")
        self.lbl_project.setStyleSheet("color: #888; font-style: italic;")
        toolbar.addWidget(self.lbl_project)

        toolbar.addSeparator()
        self.lbl_outdir = QLabel("  📁 Output: mặc định  ")
        self.lbl_outdir.setStyleSheet("color: #888; font-size: 11px;")
        toolbar.addWidget(self.lbl_outdir)

        # Central widget with tabs
        central = QWidget()
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 4)

        flow_row = QHBoxLayout()
        self.lbl_flow = QLabel(
            "Quy trình: Video & Project → Lời thoại & Ghép giọng → Phụ đề → Xuất bản"
        )
        self.lbl_flow.setWordWrap(True)
        self.lbl_flow.setStyleSheet(
            "color:#89b4fa; background:#181825; border:1px solid #313244;"
            "border-radius:5px; padding:7px 10px;"
        )
        self.btn_back = QPushButton("← Bước trước")
        self.btn_back.clicked.connect(self._go_previous_step)
        self.btn_next = QPushButton("Bước tiếp theo →")
        self.btn_next.setStyleSheet(
            "background:#45475a; color:#a6e3a1; font-weight:bold;"
        )
        self.btn_next.clicked.connect(self._go_next_step)
        flow_row.addWidget(self.lbl_flow, 1)
        flow_row.addWidget(self.btn_back)
        flow_row.addWidget(self.btn_next)
        main_layout.addLayout(flow_row)

        self.tabs = QTabWidget()

        # Step 1: select/download a video and create a project.
        self.import_tab = ImportTab()
        self.import_tab.video_imported.connect(self._on_video_imported)
        self.tabs.addTab(self.import_tab, "1  Video & Project")

        # Step 2: paste narration, generate audio, and merge it with the video.
        self.voiceover_tab = VoiceoverTab()
        self.voiceover_tab.merged_video_ready.connect(self._on_voiceover_video_ready)
        self.tabs.addTab(self.voiceover_tab, "2  Lời thoại & Ghép giọng")

        # Step 3: transcribe the newly voiced video and create subtitles.
        self.subtitle_tab = TranscribeTab(mode="narration")
        self.subtitle_tab.transcript_ready.connect(
            self._on_narration_transcript_ready
        )
        self.subtitle_tab.subtitles_generated.connect(self._on_subtitles_generated)
        self.tabs.addTab(self.subtitle_tab, "3  Phụ đề")

        # Step 4: final visual adjustments and export.
        self.export_tab = ExportTab()
        self.export_tab.export_finished.connect(self._on_export_finished)
        self.tabs.addTab(self.export_tab, "4  Xuất bản")
        self._utility_tab_indices = ()

        self._primary_step_ids = [
            "source", "voice", "narration_subtitle", "export",
        ]
        self._step_widgets = {
            "source": self.import_tab,
            "voice": self.voiceover_tab,
            "narration_subtitle": self.subtitle_tab,
            "export": self.export_tab,
        }
        self._widget_step_ids = {
            widget: step_id for step_id, widget in self._step_widgets.items()
        }
        self.tabs.currentChanged.connect(self._update_flow_header)

        main_layout.addWidget(self.tabs)
        self.setCentralWidget(central)
        self._update_step_access()
        self._update_flow_header()

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Sẵn sàng.")

    # ─── Dependency check ─────────────────────────────────────────

    def _check_deps_on_startup(self):
        if not is_ffmpeg_available():
            result = QMessageBox.question(
                self,
                "Thiếu FFmpeg",
                "FFmpeg chưa được tải. Bạn muốn tải ngay bây giờ không?\n"
                "(Cần kết nối internet, khoảng ~80MB)",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if result == QMessageBox.StandardButton.Yes:
                self._open_settings()

    # ─── Project management ───────────────────────────────────────

    def _new_project_from_toolbar(self):
        self.tabs.setCurrentWidget(self.import_tab)
        self.status_bar.showMessage("Chọn video để tạo project mới.")

    def _open_project(self):
        projects = list_projects()
        if not projects:
            QMessageBox.information(self, "Không có project", "Chưa có project nào. Hãy import video trước.")
            return

        dlg = _ProjectListDialog(projects, self)
        if dlg.exec() == QDialog.DialogCode.Accepted and dlg.selected_dir:
            project = load_project(dlg.selected_dir)
            if project:
                self._load_project(project)
            else:
                QMessageBox.warning(self, "Lỗi", "Không tải được project.")

    def _on_video_imported(self, path: str, metadata: dict):
        name, ok = QInputDialog.getText(
            self, "Tên Project", "Nhập tên project:", text=Path(path).stem
        )
        if not ok or not name.strip():
            return
        project = create_project(name.strip(), path, metadata)
        batch_parts = metadata.get("batch_parts") or []
        if batch_parts:
            clips = []
            for idx, item in enumerate(batch_parts, start=1):
                start = float(item.get("start_time", 0) or 0)
                end = float(item.get("end_time", start) or start)
                if end <= start:
                    continue
                clips.append(Clip(
                    id=str(uuid.uuid4())[:8],
                    index=idx,
                    start_time=round(start, 3),
                    end_time=round(end, 3),
                    part_text=f"CẢNH {idx:02d}",
                    source_video=item.get("source_video", "") or "",
                ))
            if clips:
                project.clips = clips
                project.scene_revision = 1
                save_project(project)
        self._load_project(project)
        self.tabs.setCurrentWidget(self.voiceover_tab)
        source_note = (
            f"Đã nhận {len(project.clips)} đoạn video từ batch. "
            if len(project.clips) > 1 else ""
        )
        self.status_bar.showMessage(
            f"Project '{project.name}' đã được tạo. {source_note}"
            "Bước tiếp theo: dán lời thoại và tạo giọng đọc."
        )

    def _load_project(self, project: Project):
        self._project = project
        if not project.clips:
            duration = float(project.video_metadata.get("duration", 0) or 0)
            if duration > 0:
                project.clips = [Clip(
                    id=str(uuid.uuid4())[:8],
                    index=1,
                    start_time=0.0,
                    end_time=round(duration, 3),
                    part_text="",
                    source_video=project.source_video,
                )]
                project.scene_revision = max(1, project.scene_revision)
                save_project(project)
        self.lbl_project.setText(f"  📽 {project.name}  ")
        self.lbl_project.setStyleSheet("color: #a6e3a1; font-weight: bold;")
        self.export_tab.load_project(project)
        self.subtitle_tab.load_project(project)
        self.voiceover_tab.load_project(project)
        self._scene_signature = self._scene_state_signature(project)
        self._update_step_access()
        resume_id = resume_step(project)
        self.tabs.setCurrentWidget(
            self._step_widgets.get(resume_id, self.voiceover_tab)
        )
        # Show global output dir if set
        if project.global_output_dir:
            self._update_outdir_label(project.global_output_dir)
        self.status_bar.showMessage(f"Đã mở project: {project.name}")
        logger.info(f"Loaded project: {project.name}")

    def _on_project_changed(self):
        if self._project:
            new_signature = self._scene_state_signature(self._project)
            if new_signature != getattr(self, "_scene_signature", None):
                self._project.scene_revision += 1
                self._project.workflow_stage = "scenes"
                # Preserve scripts for review, but invalidate every rendered
                # artifact that was derived from the old scene timeline.
                self._project.voiceover_audio = ""
                self._project.review_base_video = ""
                self._project.review_voice_video = ""
                self._project.narration_transcript_file = ""
                self._project.narration_subtitle_file = ""
                self._project.final_video = ""
                self._project.review_video_duration = 0.0
                self._project.review_voice_duration = 0.0
                self._project.review_sync_error = 0.0
                self._project.export_config.global_subtitle_file = ""
                self._project.export_config.subtitle_enabled = False
                for clip in self._project.clips:
                    clip.preview_path = ""
                    clip.export_path = ""
                    clip.subtitle_file = ""
                save_project(self._project)
                self._scene_signature = new_signature
            # Use refresh_clips for export tab so user's export settings
            # (mode, colors, position, etc.) are NOT wiped when clips are edited.
            self.export_tab.refresh_clips(self._project)
            self.subtitle_tab.load_project(self._project)
            self.voiceover_tab.load_project(self._project)
            self._update_step_access()

    def _on_source_transcript_ready(self, transcript: dict):
        """Retained for compatibility with integrations from older builds."""
        if self._project:
            self._project.workflow_stage = "source_transcript"
            save_project(self._project)
        self._update_step_access()
        self.status_bar.showMessage(
            f"Đã lưu phiên âm nguồn: {len(transcript.get('segments', []))} đoạn."
        )

    # Compatibility with code/tests that still call the old callback name.
    def _on_transcript_ready(self, transcript: dict):
        self._on_source_transcript_ready(transcript)

    def _on_narration_transcript_ready(self, transcript: dict):
        if self._project:
            self._project.workflow_stage = "narration_transcript"
            save_project(self._project)
        self._update_step_access()
        self.status_bar.showMessage(
            f"Đã nhận {len(transcript.get('segments', []))} đoạn từ giọng đọc. "
            "Chọn style và bấm Tạo phụ đề toàn video."
        )

    def _on_subtitles_generated(self, style: str):
        """Refresh final export after the narration-wide subtitle is ready."""
        if not self._project:
            return
        self._project.workflow_stage = "narration_subtitle"
        self._project.subtitle_voice_revision = self._project.voice_scene_revision
        save_project(self._project)
        # Refresh clip list and subtitle-file status badge in export tab
        self.export_tab.refresh_clips(self._project)
        self._update_step_access()
        # Auto-select the style that was just generated in the export dropdown
        for i in range(self.export_tab.cmb_sub_style.count()):
            if self.export_tab.cmb_sub_style.itemData(i) == style:
                self.export_tab.cmb_sub_style.setCurrentIndex(i)
                break
        style_labels = {"plain": "Plain (chữ trắng)", "karaoke": "Karaoke (highlight từng từ)"}
        style_label = style_labels.get(style, style)
        self.status_bar.showMessage(
            f"✅ Subtitle đã tạo xong ({style_label}). "
            "Đang chuyển sang bước Xuất bản."
        )
        self.tabs.setCurrentWidget(self.export_tab)

    def _on_export_finished(self, exported: list):
        if self._project and exported:
            self._project.final_video = exported[0]
            self._project.workflow_stage = "export"
            save_project(self._project)
        self._update_step_access()
        self.status_bar.showMessage(f"Xuất xong {len(exported)} video.")

    def _on_scenes_applied(self):
        """Refresh simplified screens if an older integration edits clips."""
        if self._project:
            self._on_project_changed()
            self.tabs.setCurrentWidget(self.voiceover_tab)

    def _on_ai_scripts_saved(self):
        """Copy the confirmed script to Voice and continue forward."""
        if self._project and self._project.voiceover_script:
            self._project.workflow_stage = "script"
            self._project.script_scene_revision = self._project.scene_revision
            save_project(self._project)
            self.voiceover_tab.txt_script.setPlainText(
                self._project.voiceover_script
            )
        self._update_step_access()
        self.tabs.setCurrentWidget(self.voiceover_tab)
        self.status_bar.showMessage(
            "Lời thoại đã được lưu. Bước tiếp theo: tạo giọng đọc."
        )

    def _on_voiceover_video_ready(self, video_path: str):
        """Adopt a narrated review artifact without mutating the raw source."""
        if not self._project:
            return
        merged = Path(video_path)
        if not merged.exists():
            return

        if not getattr(self._project, "original_source_video", ""):
            self._project.original_source_video = (
                self._project.video_metadata.get("path", "")
                or self._project.source_video
            )

        # Source video, source transcript and source clip coordinates remain
        # immutable. Downstream narration artifacts have their own fields.
        self._project.review_voice_video = str(merged)
        base_candidate = merged.with_name(
            f"{self._project.name}_review_no_sub_base.mp4"
        )
        self._project.review_base_video = (
            str(base_candidate) if base_candidate.exists() else ""
        )
        self._project.narration_transcript_file = ""
        self._project.narration_subtitle_file = ""
        self._project.final_video = ""
        self._project.export_config.global_subtitle_file = ""
        self._project.export_config.subtitle_enabled = False
        self._project.export_config.subtitle_style = "none"
        self._project.export_config.part_text_enabled = False
        self._project.export_config.watermark_text = ""

        ff = ffmpeg_path()
        video_duration = (
            probe_media_duration(str(merged), ff) if ff else 0.0
        )
        voice_duration = (
            probe_media_duration(self._project.voiceover_audio, ff)
            if ff
            and self._project.voiceover_audio
            and Path(self._project.voiceover_audio).exists()
            else 0.0
        )
        self._project.review_video_duration = video_duration
        # Timed dubbing explicitly allows overflow: the merge trims audio at
        # the video endpoint. Downstream QA describes the merged timeline,
        # not the longer standalone TTS file.
        if self._project.workflow_mode == "dubbing_timed_manual" and video_duration > 0:
            voice_duration = min(voice_duration, video_duration)
        self._project.review_voice_duration = voice_duration
        self._project.review_sync_error = (
            abs(video_duration - voice_duration)
            if video_duration > 0 and voice_duration > 0 else 0.0
        )
        self._project.workflow_stage = "voice"
        self._project.voice_scene_revision = self._project.scene_revision

        save_project(self._project)
        self.subtitle_tab.load_project(self._project)
        self.export_tab.refresh_clips(self._project)
        self._update_step_access()
        tolerance = max(0.15, video_duration * 0.005) if video_duration > 0 else 0.15
        sync_ok = (
            video_duration <= 0
            or voice_duration <= 0
            or self._project.review_sync_error <= tolerance
        )
        self.tabs.setCurrentWidget(
            self.subtitle_tab if sync_ok else self.voiceover_tab
        )
        qa = ""
        if video_duration > 0 and voice_duration > 0:
            qa = (
                f" • video {video_duration:.2f}s • voice {voice_duration:.2f}s "
                f"• lệch {self._project.review_sync_error:.2f}s"
            )
        if sync_ok:
            self.status_bar.showMessage(
                f"✅ Đã dựng master có giọng, sạch PART/subtitle{qa}. "
                "Bước tiếp theo: tạo phụ đề từ giọng đọc."
            )
        else:
            self.status_bar.showMessage(
                f"❌ Master chưa đạt kiểm tra đồng bộ{qa}; ngưỡng {tolerance:.2f}s. "
                "Hãy chỉnh kịch bản/tốc độ và tạo lại giọng."
            )

    # ─── Output folder (toolbar) ──────────────────────────────────

    def _has_transcript(self) -> bool:
        return bool(
            self._project
            and (
                self._project.source_transcript_file
                or self._project.transcript_file
            )
            and Path(
                self._project.source_transcript_file
                or self._project.transcript_file
            ).exists()
        )

    def _has_clip_subtitles(self) -> bool:
        return bool(
            self._project
            and self._project.narration_subtitle_file
            and Path(self._project.narration_subtitle_file).exists()
        )

    def _update_step_access(self):
        """Apply workflow guidance without hiding features in unlocked mode."""
        if not hasattr(self, "tabs"):
            return

        access = step_access(self._project)
        for step_id, widget in self._step_widgets.items():
            idx = self.tabs.indexOf(widget)
            info = access.get(step_id, {})
            prerequisites_ok = bool(info.get("enabled", False))
            ok = self._all_features_unlocked or prerequisites_ok
            reason = info.get("reason", "")
            self.tabs.setTabEnabled(idx, ok)
            if prerequisites_ok:
                tooltip = ""
            elif self._all_features_unlocked:
                tooltip = (
                    "🔓 Được mở thủ công. "
                    + (reason or "Bước này chưa đủ dữ liệu đầu vào.")
                )
            else:
                tooltip = f"🔒 {reason or 'Chưa đủ dữ liệu đầu vào.'}"
            self.tabs.setTabToolTip(idx, tooltip)

        for idx in getattr(self, "_utility_tab_indices", ()):
            self.tabs.setTabVisible(idx, self._all_features_unlocked)

        current_widget = self.tabs.currentWidget()
        current_id = self._widget_step_ids.get(current_widget)
        if (
            not self._all_features_unlocked
            and current_id
            and not access.get(current_id, {}).get("enabled", False)
        ):
            target_id = resume_step(self._project)
            self.tabs.setCurrentWidget(
                self._step_widgets.get(target_id, self.import_tab)
            )
        self._update_flow_header()

    def _set_unlock_all(self, enabled: bool):
        """Show every feature while retaining runtime input validation."""
        self._all_features_unlocked = bool(enabled)
        self._update_step_access()
        if self._all_features_unlocked:
            self.status_bar.showMessage(
                "🔓 Đã mở tất cả chức năng. Các bước thiếu dữ liệu sẽ cảnh báo "
                "khi bạn bấm thực thi."
            )
        else:
            self.status_bar.showMessage(
                "🔒 Đã bật lại quy trình tuần tự theo điều kiện đầu vào."
            )

    @staticmethod
    def _scene_state_signature(project: Project) -> tuple:
        """Stable source-coordinate signature used to invalidate descendants."""
        return tuple(
            (
                clip.id,
                clip.index,
                round(float(clip.start_time), 3),
                round(float(clip.end_time), 3),
                bool(clip.enabled),
                getattr(clip, "source_video", "") or project.source_video,
                clip.part_text,
            )
            for clip in project.clips
        )

    def _go_to_step(self, step_id: str) -> bool:
        widget = self._step_widgets.get(step_id)
        if not widget:
            return False
        access = step_access(self._project).get(step_id, {})
        prerequisites_ok = bool(access.get("enabled", False))
        if not prerequisites_ok and not self._all_features_unlocked:
            self.status_bar.showMessage(
                access.get("reason", "Bước này chưa đủ dữ liệu đầu vào.")
            )
            return False
        self.tabs.setCurrentWidget(widget)
        if not prerequisites_ok:
            self.status_bar.showMessage(
                "🔓 Đã mở bước này thủ công. "
                + access.get("reason", "Hãy bổ sung dữ liệu trước khi chạy.")
            )
        return True

    def _go_previous_step(self):
        current_id = self._widget_step_ids.get(self.tabs.currentWidget())
        if current_id not in self._primary_step_ids:
            self._go_to_step(resume_step(self._project))
            return
        pos = self._primary_step_ids.index(current_id)
        if self._all_features_unlocked and pos > 0:
            self._go_to_step(self._primary_step_ids[pos - 1])
            return
        for previous in reversed(self._primary_step_ids[:pos]):
            if self._go_to_step(previous):
                return

    def _go_next_step(self):
        current_id = self._widget_step_ids.get(self.tabs.currentWidget())
        if current_id not in self._primary_step_ids:
            self._go_to_step(resume_step(self._project))
            return
        pos = self._primary_step_ids.index(current_id)
        if self._all_features_unlocked:
            if pos + 1 < len(self._primary_step_ids):
                self._go_to_step(self._primary_step_ids[pos + 1])
            return
        target_id = next_step(current_id, self._project)
        if target_id and self._go_to_step(target_id):
            return
        access = step_access(self._project)
        if pos + 1 < len(self._primary_step_ids):
            info = access.get(self._primary_step_ids[pos + 1], {})
            self.status_bar.showMessage(
                info.get("reason", "Hãy hoàn tất bước hiện tại trước.")
            )

    def _update_flow_header(self, *_):
        if not hasattr(self, "_widget_step_ids"):
            return
        current_id = self._widget_step_ids.get(self.tabs.currentWidget())
        if current_id not in self._primary_step_ids:
            self.lbl_flow.setText(
                "🔧 Công cụ độc lập — không làm thay đổi tiến độ project. "
                "Bấm “Bước trước/tiếp theo” để quay lại quy trình review."
            )
            self.btn_back.setEnabled(True)
            self.btn_next.setEnabled(True)
            return

        access = step_access(self._project)
        pos = self._primary_step_ids.index(current_id)
        current_name = self.tabs.tabText(
            self.tabs.indexOf(self._step_widgets[current_id])
        ).split("  ", 1)[-1]
        markers = []
        for step_id in self._primary_step_ids:
            info = access.get(step_id, {})
            if info.get("complete"):
                marker = "✓"
            elif step_id == current_id:
                marker = "●"
            elif info.get("enabled"):
                marker = "○"
            elif self._all_features_unlocked:
                marker = "◇"
            else:
                marker = "🔒"
            markers.append(marker)
        qa = ""
        if self._project and self._project.review_voice_video:
            qa = (
                f" • A/V lệch {self._project.review_sync_error:.2f}s"
                if self._project.review_sync_error >= 0 else ""
            )
        self.lbl_flow.setText(
            (
                "🔓 Tất cả chức năng • "
                if self._all_features_unlocked else ""
            )
            + f"Bước {pos + 1}/{len(self._primary_step_ids)} • {current_name}{qa}\n"
            + "  ".join(
                f"{markers[i]} {i + 1}" for i in range(len(markers))
            )
        )
        self.btn_back.setEnabled(pos > 0)
        if pos + 1 < len(self._primary_step_ids):
            next_info = access.get(self._primary_step_ids[pos + 1], {})
            next_enabled = (
                self._all_features_unlocked
                or bool(next_info.get("enabled", False))
            )
            self.btn_next.setEnabled(next_enabled)
            if next_info.get("enabled"):
                tooltip = ""
            elif self._all_features_unlocked:
                tooltip = (
                    "🔓 Có thể mở trực tiếp; "
                    + next_info.get("reason", "chưa đủ dữ liệu đầu vào.")
                )
            else:
                tooltip = next_info.get("reason", "Chưa đủ dữ liệu.")
            self.btn_next.setToolTip(tooltip)
            self.btn_next.setText("Bước tiếp theo →")
        else:
            self.btn_next.setEnabled(False)
            self.btn_next.setText("Đã ở bước cuối")

    def _set_global_output_dir(self):
        d = QFileDialog.getExistingDirectory(
            self, "Chọn thư mục lưu tất cả video xuất",
            self._project.global_output_dir if self._project else "",
        )
        if d:
            self._update_outdir_label(d)
            if self._project:
                self._project.global_output_dir = d
                self.export_tab.txt_out_dir.setText(d)
                save_project(self._project)
            self.status_bar.showMessage(f"Output folder: {d}")

    def _update_outdir_label(self, path: str):
        short = Path(path).name or path
        self.lbl_outdir.setText(f"  📁 {short}  ")
        self.lbl_outdir.setStyleSheet("color: #f9e2af; font-size: 11px;")
        self.lbl_outdir.setToolTip(path)

    # ─── Settings ─────────────────────────────────────────────────

    def _open_settings(self):
        dlg = SettingsDialog(self)
        dlg.exec()


class _ProjectListDialog(QDialog):
    def __init__(self, projects: list[dict], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Chọn Project")
        self.setMinimumSize(520, 360)
        self.selected_dir = ""

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Chọn project để mở:"))

        self.list_widget = QListWidget()
        self._populate(projects)
        self.list_widget.itemDoubleClicked.connect(self._accept)
        layout.addWidget(self.list_widget)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Open | QDialogButtonBox.StandardButton.Cancel
        )
        # Add a Delete button on the left side of the button box
        self.btn_delete = QPushButton("🗑 Xóa project")
        self.btn_delete.setStyleSheet("color: #f38ba8;")
        self.btn_delete.clicked.connect(self._delete_selected)
        btns.addButton(self.btn_delete, QDialogButtonBox.ButtonRole.DestructiveRole)
        btns.accepted.connect(self._accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _populate(self, projects: list[dict]):
        self.list_widget.clear()
        for p in projects:
            item = QListWidgetItem(
                f"📽 {p['name']}  —  {p['updated_at']}  ({p['clips_count']} parts)"
            )
            item.setData(Qt.ItemDataRole.UserRole, p["dir"])
            self.list_widget.addItem(item)

    def _delete_selected(self):
        item = self.list_widget.currentItem()
        if not item:
            QMessageBox.information(self, "Chưa chọn", "Hãy chọn project muốn xóa.")
            return
        name = item.text()
        proj_dir = item.data(Qt.ItemDataRole.UserRole)
        reply = QMessageBox.question(
            self, "Xác nhận xóa",
            f"Xóa vĩnh viễn project này?\n\n{name}\n\n"
            "Toàn bộ clip, phụ đề, video đã xuất trong project sẽ bị xóa "
            "và KHÔNG thể khôi phục.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        if delete_project(proj_dir):
            self._populate(list_projects())
            if self.list_widget.count() == 0:
                QMessageBox.information(
                    self, "Đã xóa", "Đã xóa. Không còn project nào."
                )
                self.reject()
        else:
            QMessageBox.warning(self, "Lỗi", "Không xóa được project.")

    def _accept(self):
        item = self.list_widget.currentItem()
        if item:
            self.selected_dir = item.data(Qt.ItemDataRole.UserRole)
            self.accept()
