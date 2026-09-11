"""Tab 4: Edit Parts - clip list, split/merge/reorder, transcript view."""

import math
import re
import uuid
from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QSplitter,
    QLabel, QPushButton, QLineEdit, QSpinBox, QDoubleSpinBox,
    QGroupBox, QTableWidget, QTableWidgetItem, QHeaderView,
    QFileDialog, QMessageBox, QAbstractItemView, QTextEdit,
    QCheckBox,
)

from src.models.project import Project
from src.models.clip import Clip
from src.utils.file_utils import format_duration, hms_to_seconds, seconds_to_hms
from src.core.project_manager import generate_clips, save_project
from src.core.ai_client import _load_transcript_segments, _extract_clip_transcript
from src.core.video_manager import get_video_metadata
from src.ui.export_tab import _CropAdjustWidget
from src.ui.workers import WideThumbnailWorker


class EditorTab(QWidget):
    project_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._project: Project | None = None
        self._segments: list[dict] = []
        self._wide_thumb_worker: WideThumbnailWorker | None = None
        self._pre_crop_w: float = 1.0
        self._pre_crop_h: float = 1.0
        self._source_duration_cache: dict[str, float] = {}
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # Top controls
        top = QHBoxLayout()
        top.addWidget(QLabel("Thời lượng mỗi Part (giây):"))
        self.spn_duration = QDoubleSpinBox()
        self.spn_duration.setRange(5, 3600)
        self.spn_duration.setValue(60)
        self.spn_duration.setSuffix(" s")
        self.spn_duration.setFixedWidth(100)
        top.addWidget(self.spn_duration)

        top.addWidget(QLabel("Số part tối đa (0 = tất cả):"))
        self.spn_max_parts = QSpinBox()
        self.spn_max_parts.setRange(0, 9999)
        self.spn_max_parts.setValue(0)
        self.spn_max_parts.setFixedWidth(80)
        top.addWidget(self.spn_max_parts)

        self.btn_generate = QPushButton("⚡ Tự động chia Parts")
        self.btn_generate.setFixedHeight(34)
        self.btn_generate.clicked.connect(self._generate_clips)
        top.addWidget(self.btn_generate)
        top.addStretch()

        self.lbl_summary = QLabel("Chưa có project.")
        self.lbl_summary.setStyleSheet("color: #888;")
        top.addWidget(self.lbl_summary)
        layout.addLayout(top)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # ── Left: clip table + subtitle ──
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self.table = QTableWidget()
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels(
            ["On", "#", "Video", "Bat dau", "Ket thuc", "Thoi luong", "Subtitle"]
        )
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(0, 30)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(1, 50)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked)
        self.table.cellChanged.connect(self._on_cell_changed)
        self.table.itemSelectionChanged.connect(self._on_selection_changed)
        left_layout.addWidget(self.table)

        tbl_btns = QHBoxLayout()
        self.btn_check_all = QPushButton("Chọn tất cả")
        self.btn_check_all.clicked.connect(lambda: self._check_all(True))
        self.btn_uncheck_all = QPushButton("Bỏ chọn tất cả")
        self.btn_uncheck_all.clicked.connect(lambda: self._check_all(False))
        self.btn_del_clip = QPushButton("🗑 Xóa Part")
        self.btn_del_clip.clicked.connect(self._delete_selected_clip)
        tbl_btns.addWidget(self.btn_check_all)
        tbl_btns.addWidget(self.btn_uncheck_all)
        tbl_btns.addStretch()
        tbl_btns.addWidget(self.btn_del_clip)
        left_layout.addLayout(tbl_btns)

        sub_group = QGroupBox("Subtitle toàn bộ (.srt)")
        sub_layout = QHBoxLayout(sub_group)
        self.txt_subtitle = QLineEdit()
        self.txt_subtitle.setPlaceholderText("Chưa chọn file .srt")
        self.txt_subtitle.setReadOnly(True)
        btn_srt = QPushButton("📂 Chọn .srt")
        btn_srt.clicked.connect(self._browse_srt)
        btn_clear_srt = QPushButton("✕")
        btn_clear_srt.setFixedWidth(30)
        btn_clear_srt.clicked.connect(self._clear_srt)
        sub_layout.addWidget(self.txt_subtitle)
        sub_layout.addWidget(btn_srt)
        sub_layout.addWidget(btn_clear_srt)
        left_layout.addWidget(sub_group)

        splitter.addWidget(left_widget)

        # ── Right: editing tools ──
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(4, 0, 0, 0)
        right_layout.setSpacing(8)

        # Edit times
        edit_group = QGroupBox("✏ Chỉnh thời gian Part đang chọn")
        eg = QHBoxLayout(edit_group)
        eg.addWidget(QLabel("Bắt đầu:"))
        self.txt_start = QLineEdit()
        self.txt_start.setPlaceholderText("00:01:30.000")
        self.txt_start.setFixedWidth(110)
        eg.addWidget(self.txt_start)
        eg.addWidget(QLabel("Kết thúc:"))
        self.txt_end = QLineEdit()
        self.txt_end.setPlaceholderText("00:02:00.000")
        self.txt_end.setFixedWidth(110)
        eg.addWidget(self.txt_end)
        btn_apply_time = QPushButton("✔ Áp dụng")
        btn_apply_time.clicked.connect(self._apply_time_edit)
        eg.addWidget(btn_apply_time)
        eg.addStretch()
        right_layout.addWidget(edit_group)

        # Part operations
        ops_group = QGroupBox("🔧 Thao tác Part")
        og = QGridLayout(ops_group)
        self.btn_add = QPushButton("➕ Thêm Part")
        self.btn_add.clicked.connect(self._add_part)
        self.btn_dup = QPushButton("⎘ Nhân đôi")
        self.btn_dup.clicked.connect(self._duplicate_part)
        self.btn_up = QPushButton("⬆ Lên")
        self.btn_up.clicked.connect(lambda: self._move_part(-1))
        self.btn_down = QPushButton("⬇ Xuống")
        self.btn_down.clicked.connect(lambda: self._move_part(1))
        self.btn_merge = QPushButton("⤵ Gộp với Part sau")
        self.btn_merge.clicked.connect(self._merge_with_next)
        og.addWidget(self.btn_add, 0, 0)
        og.addWidget(self.btn_dup, 0, 1)
        og.addWidget(self.btn_up, 1, 0)
        og.addWidget(self.btn_down, 1, 1)
        og.addWidget(self.btn_merge, 2, 0, 1, 2)

        split_row = QHBoxLayout()
        split_row.addWidget(QLabel("✂ Tách tại:"))
        self.txt_split = QLineEdit()
        self.txt_split.setPlaceholderText("mm:ss (trống = giữa Part)")
        self.txt_split.setFixedWidth(150)
        split_row.addWidget(self.txt_split)
        self.btn_split = QPushButton("Tách Part")
        self.btn_split.clicked.connect(self._split_part)
        split_row.addWidget(self.btn_split)
        split_row.addStretch()
        og.addLayout(split_row, 3, 0, 1, 2)
        right_layout.addWidget(ops_group)

        pre_group = QGroupBox("Cat/co khung hinh truoc khi xuat")
        pre_layout = QVBoxLayout(pre_group)
        self.chk_pre_crop = QCheckBox("Bat cat/co khung hinh nguon")
        self.chk_pre_crop.toggled.connect(self._on_pre_crop_toggled)
        pre_layout.addWidget(self.chk_pre_crop)
        self._pre_crop_widget = _CropAdjustWidget(
            lock_aspect=False,
            label="Bat cat/co khung hinh de chon vung",
        )
        self._pre_crop_widget.setVisible(False)
        self._pre_crop_widget.crop_changed.connect(self._on_pre_crop_drag)
        pre_layout.addWidget(self._pre_crop_widget)
        self.lbl_pre_crop = QLabel("X: 50% / Y: 50% / W: 100% / H: 100%")
        self.lbl_pre_crop.setStyleSheet("color:#888; font-size:11px;")
        pre_layout.addWidget(self.lbl_pre_crop)
        pre_hint = QLabel("Buoc nay ap dung truoc Tab Xuat Video. No crop/zoom video nguon truoc khi them blur_bg, title, subtitle.")
        pre_hint.setStyleSheet("color:#6c7086; font-size:10px;")
        pre_hint.setWordWrap(True)
        pre_layout.addWidget(pre_hint)
        right_layout.addWidget(pre_group)

        # Transcript view for selected part
        tr_group = QGroupBox("💬 Lời thoại trong Part đang chọn")
        tg = QVBoxLayout(tr_group)
        self.txt_transcript = QTextEdit()
        self.txt_transcript.setReadOnly(True)
        self.txt_transcript.setPlaceholderText(
            "Chọn một Part để xem lời thoại trong khoảng thời gian đó.\n"
            "(Cần đã phiên âm ở Tab 2)"
        )
        self.txt_transcript.setStyleSheet(
            "background:#181825; color:#cdd6f4; font-size:13px;"
        )
        tg.addWidget(self.txt_transcript)
        right_layout.addWidget(tr_group, 1)

        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 6)
        splitter.setStretchFactor(1, 4)
        splitter.setSizes([720, 480])
        layout.addWidget(splitter, 1)

    # ─── Public API ───────────────────────────────────────────────

    def load_project(self, project: Project):
        self._project = project
        self._source_duration_cache.clear()
        self.spn_duration.setValue(project.clip_duration)
        self._segments = _load_transcript_segments(project) or []
        sf = project.export_config.global_subtitle_file
        self.txt_subtitle.setText(Path(sf).name if sf else "")
        cfg = project.export_config
        self.chk_pre_crop.blockSignals(True)
        self.chk_pre_crop.setChecked(getattr(cfg, "pre_crop_enabled", False))
        self.chk_pre_crop.blockSignals(False)
        self._pre_crop_w = getattr(cfg, "pre_crop_w", 1.0)
        self._pre_crop_h = getattr(cfg, "pre_crop_h", 1.0)
        self._pre_crop_widget.set_crop(
            getattr(cfg, "pre_crop_x", 0.5),
            getattr(cfg, "pre_crop_y", 0.5),
            self._pre_crop_w,
            self._pre_crop_h,
        )
        self._update_pre_crop_label(
            getattr(cfg, "pre_crop_x", 0.5),
            getattr(cfg, "pre_crop_y", 0.5),
            self._pre_crop_w,
            self._pre_crop_h,
        )
        self._on_pre_crop_toggled(self.chk_pre_crop.isChecked())
        self._refresh_table()

    # ─── Helpers ──────────────────────────────────────────────────

    def _selected_row(self) -> int:
        rows = self.table.selectionModel().selectedRows()
        if not rows or not self._project:
            return -1
        r = rows[0].row()
        return r if 0 <= r < len(self._project.clips) else -1

    def _video_duration(self) -> float:
        if self._project and self._project.video_metadata:
            try:
                duration = float(
                    self._project.video_metadata.get("duration", 0) or 0
                )
            except (TypeError, ValueError):
                return 0.0
            return duration if math.isfinite(duration) and duration > 0 else 0.0
        return 0.0

    def _effective_source(self, clip: Clip) -> str:
        if not self._project:
            return ""
        return str(
            getattr(clip, "source_video", "")
            or self._project.source_video
            or ""
        )

    @staticmethod
    def _source_key(source_video: str) -> str:
        if not source_video:
            return ""
        try:
            return str(Path(source_video).resolve(strict=False)).casefold()
        except (OSError, RuntimeError):
            return str(source_video).casefold()

    def _source_duration(self, clip: Clip) -> float:
        """Return the duration of the exact source used by ``clip``."""
        if not self._project:
            return 0.0

        source_video = self._effective_source(clip)
        source_key = self._source_key(source_video)
        if not source_key:
            return 0.0
        if source_key in self._source_duration_cache:
            return self._source_duration_cache[source_key]

        project_source_key = self._source_key(self._project.source_video)
        duration = 0.0
        if source_key == project_source_key:
            duration = self._video_duration()
        else:
            metadata_path = str(
                self._project.video_metadata.get("path", "") or ""
            )
            if source_key == self._source_key(metadata_path):
                duration = self._video_duration()

        if duration <= 0 and source_video and Path(source_video).exists():
            metadata = get_video_metadata(source_video) or {}
            try:
                duration = float(metadata.get("duration", 0) or 0)
            except (TypeError, ValueError):
                duration = 0.0

        if not math.isfinite(duration) or duration <= 0:
            duration = 0.0
        self._source_duration_cache[source_key] = duration
        return duration

    def _validate_time_range(
        self,
        clip: Clip,
        start_time,
        end_time,
    ) -> tuple[bool, str]:
        source_video = self._effective_source(clip)
        if not source_video:
            return False, "Cảnh chưa có video nguồn."

        try:
            start = float(start_time)
            end = float(end_time)
        except (TypeError, ValueError):
            return False, "Thời gian bắt đầu/kết thúc không đúng định dạng."

        if not math.isfinite(start) or not math.isfinite(end):
            return False, "Thời gian phải là một số hữu hạn."

        duration = self._source_duration(clip)
        if duration <= 0:
            return (
                False,
                f"Không đọc được thời lượng video nguồn: {Path(source_video).name}",
            )
        if start < 0:
            return False, "Thời gian bắt đầu không được nhỏ hơn 0."
        if start >= end:
            return False, "Thời gian bắt đầu phải nhỏ hơn thời gian kết thúc."
        if end > duration:
            return (
                False,
                "Thời gian kết thúc vượt quá video nguồn "
                f"({seconds_to_hms(duration)}).",
            )
        return True, ""

    def _same_source(self, first: Clip, second: Clip) -> bool:
        return bool(
            self._source_key(self._effective_source(first))
            and self._source_key(self._effective_source(first))
            == self._source_key(self._effective_source(second))
        )

    def _restore_time_widgets(self, row: int, clip: Clip):
        """Restore table/editor values without recursively firing edits."""
        self.table.blockSignals(True)
        start_item = self.table.item(row, 3)
        end_item = self.table.item(row, 4)
        duration_item = self.table.item(row, 5)
        if start_item:
            start_item.setText(seconds_to_hms(clip.start_time))
        if end_item:
            end_item.setText(seconds_to_hms(clip.end_time))
        if duration_item:
            duration_item.setText(format_duration(clip.duration))
        self.table.blockSignals(False)

        if self._selected_row() == row:
            self.txt_start.setText(seconds_to_hms(clip.start_time))
            self.txt_end.setText(seconds_to_hms(clip.end_time))

    def _warn_invalid_time(self, message: str):
        QMessageBox.warning(self, "Thời gian không hợp lệ", message)
        window = self.window()
        status_bar_getter = getattr(window, "statusBar", None)
        if callable(status_bar_getter):
            status_bar = status_bar_getter()
            if status_bar is not None:
                status_bar.showMessage(message, 6000)

    @staticmethod
    def _clear_clip_derivatives(clip: Clip):
        """Clear artifacts/content that no longer belongs to a derived range."""
        clip.subtitle_file = ""
        clip.preview_path = ""
        clip.export_path = ""
        clip.voiceover_script = ""
        clip.custom_subtitle = ""
        clip.custom_header = ""

    def _renumber(self):
        """Keep indices sequential; update default scene titles."""
        for i, c in enumerate(self._project.clips):
            is_default = bool(
                re.fullmatch(
                    r"(?:PART|CẢNH)\s+\d+",
                    (c.part_text or "").strip(),
                    flags=re.IGNORECASE,
                )
            )
            c.index = i + 1
            if is_default:
                c.part_text = f"CẢNH {i + 1:02d}"

    def _commit(self, select_row: int | None = None):
        """Save, refresh table, re-select a row, notify others."""
        self._renumber()
        save_project(self._project)
        self._refresh_table()
        if select_row is not None and 0 <= select_row < len(self._project.clips):
            self.table.selectRow(select_row)
        self.project_changed.emit()

    # ─── Generate / table ─────────────────────────────────────────

    def _generate_clips(self):
        if not self._project:
            return
        self._project.clip_duration = self.spn_duration.value()
        clips = generate_clips(self._project)
        max_p = self.spn_max_parts.value()
        if max_p > 0:
            clips = clips[:max_p]
        self._project.clips = clips
        save_project(self._project)
        self._refresh_table()
        self.project_changed.emit()

    def _refresh_table(self):
        if not self._project:
            return
        clips = self._project.clips
        self.table.blockSignals(True)
        self.table.setRowCount(len(clips))
        for row, clip in enumerate(clips):
            chk = QTableWidgetItem()
            chk.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
            chk.setCheckState(
                Qt.CheckState.Checked if clip.enabled else Qt.CheckState.Unchecked
            )
            self.table.setItem(row, 0, chk)
            idx_item = QTableWidgetItem(f"#{clip.index}")
            idx_item.setFlags(idx_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, 1, idx_item)
            video_name = Path(getattr(clip, "source_video", "") or self._project.source_video).name
            video_item = QTableWidgetItem(video_name)
            video_item.setFlags(video_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, 2, video_item)
            self.table.setItem(row, 3, QTableWidgetItem(seconds_to_hms(clip.start_time)))
            self.table.setItem(row, 4, QTableWidgetItem(seconds_to_hms(clip.end_time)))
            dur_item = QTableWidgetItem(format_duration(clip.duration))
            dur_item.setFlags(dur_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, 5, dur_item)
            srt_name = Path(clip.subtitle_file).name if clip.subtitle_file else ""
            self.table.setItem(row, 6, QTableWidgetItem(srt_name))
        self.table.blockSignals(False)
        total = len(clips)
        enabled = sum(1 for c in clips if c.enabled)
        total_dur = sum(c.duration for c in clips)
        self.lbl_summary.setText(
            f"{total} parts | {enabled} được chọn | Tổng: {format_duration(total_dur)}"
        )

    def _on_cell_changed(self, row: int, col: int):
        if not self._project or row >= len(self._project.clips):
            return
        clip = self._project.clips[row]
        item = self.table.item(row, col)
        if item is None:
            return
        if col == 0:
            clip.enabled = item.checkState() == Qt.CheckState.Checked
        elif col in (3, 4):
            val = hms_to_seconds(item.text())
            candidate_start = val if col == 3 else clip.start_time
            candidate_end = val if col == 4 else clip.end_time
            valid, error = self._validate_time_range(
                clip,
                candidate_start,
                candidate_end,
            )
            if not valid:
                self._restore_time_widgets(row, clip)
                self._warn_invalid_time(error)
                return
            clip.start_time = float(candidate_start)
            clip.end_time = float(candidate_end)
            self._restore_time_widgets(row, clip)
        save_project(self._project)
        self.project_changed.emit()

    def _on_selection_changed(self):
        row = self._selected_row()
        if row < 0:
            return
        clip = self._project.clips[row]
        self.txt_start.setText(seconds_to_hms(clip.start_time))
        self.txt_end.setText(seconds_to_hms(clip.end_time))
        self._update_transcript_view(clip)
        if self.chk_pre_crop.isChecked():
            self._load_pre_crop_thumbnail(clip)

    def _update_pre_crop_label(self, cx: float, cy: float, cw: float, ch: float):
        self.lbl_pre_crop.setText(
            f"X: {int(cx*100)}% / Y: {int(cy*100)}% / W: {int(cw*100)}% / H: {int(ch*100)}%"
        )

    def _on_pre_crop_toggled(self, checked: bool):
        self._pre_crop_widget.setVisible(checked)
        if self._project:
            cfg = self._project.export_config
            cfg.pre_crop_enabled = checked
            save_project(self._project)
            row = self._selected_row()
            if checked and row >= 0:
                self._load_pre_crop_thumbnail(self._project.clips[row])
            self.project_changed.emit()

    def _on_pre_crop_drag(self, cx: float, cy: float, cw: float, ch: float):
        if not self._project:
            return
        self._pre_crop_w = cw
        self._pre_crop_h = ch
        cfg = self._project.export_config
        cfg.pre_crop_enabled = self.chk_pre_crop.isChecked()
        cfg.pre_crop_x = max(0.0, min(1.0, cx))
        cfg.pre_crop_y = max(0.0, min(1.0, cy))
        cfg.pre_crop_w = max(0.05, min(1.0, cw))
        cfg.pre_crop_h = max(0.05, min(1.0, ch))
        self._update_pre_crop_label(cfg.pre_crop_x, cfg.pre_crop_y, cfg.pre_crop_w, cfg.pre_crop_h)
        save_project(self._project)
        self.project_changed.emit()

    def _load_pre_crop_thumbnail(self, clip: Clip):
        if not self._project:
            return
        source_video = getattr(clip, "source_video", "") or self._project.source_video
        if not source_video or not Path(source_video).exists():
            return
        previews = Path(self._project.output_dir) / "previews"
        previews.mkdir(parents=True, exist_ok=True)
        path = str(previews / f"pre_crop_thumb_part{clip.index:02d}.jpg")
        if Path(path).exists():
            self._on_pre_crop_thumb_ready(path)
            return
        offset = min(30.0, max(0.0, clip.start_time + clip.duration * 0.1))
        self._wide_thumb_worker = WideThumbnailWorker(source_video, path, offset)
        self._wide_thumb_worker.finished.connect(self._on_pre_crop_thumb_ready)
        self._wide_thumb_worker.start()

    def _on_pre_crop_thumb_ready(self, path: str):
        if path and Path(path).exists():
            self._pre_crop_widget.set_thumbnail(QPixmap(path))

    def _update_transcript_view(self, clip: Clip):
        if not self._segments:
            self.txt_transcript.setPlainText(
                "(Chưa có transcript — hãy phiên âm ở Tab 2 để xem lời thoại)"
            )
            return
        text = _extract_clip_transcript(
            self._segments, clip.start_time, clip.end_time, max_chars=4000
        )
        self.txt_transcript.setPlainText(text or "(Không có lời thoại trong đoạn này)")

    # ─── Time edit ────────────────────────────────────────────────

    def _apply_time_edit(self):
        row = self._selected_row()
        if row < 0:
            return
        clip = self._project.clips[row]
        start_time = hms_to_seconds(self.txt_start.text())
        end_time = hms_to_seconds(self.txt_end.text())
        valid, error = self._validate_time_range(
            clip,
            start_time,
            end_time,
        )
        if not valid:
            self._restore_time_widgets(row, clip)
            self._warn_invalid_time(error)
            return
        clip.start_time = float(start_time)
        clip.end_time = float(end_time)
        self._commit(select_row=row)

    def _check_all(self, state: bool):
        if not self._project:
            return
        for clip in self._project.clips:
            clip.enabled = state
        save_project(self._project)
        self._refresh_table()
        self.project_changed.emit()

    def _delete_selected_clip(self):
        row = self._selected_row()
        if row < 0:
            return
        if QMessageBox.question(
            self, "Xóa Part", f"Xóa Part #{row + 1}?"
        ) == QMessageBox.StandardButton.Yes:
            del self._project.clips[row]
            self._commit(select_row=min(row, len(self._project.clips) - 1))

    # ─── Part operations ──────────────────────────────────────────

    def _add_part(self):
        if not self._project:
            return
        row = self._selected_row()
        clip_dur = self.spn_duration.value()

        anchor = None
        if row >= 0:
            anchor = self._project.clips[row]
        elif self._project.clips:
            anchor = self._project.clips[-1]

        start = anchor.end_time if anchor else 0.0
        source_video = getattr(anchor, "source_video", "") if anchor else ""
        new = Clip(
            id=str(uuid.uuid4())[:8],
            index=0,
            start_time=round(start, 3),
            end_time=round(start + clip_dur, 3),
            part_text="CẢNH 00",
            source_video=source_video,
        )

        vdur = self._source_duration(new)
        if vdur <= 0:
            self._warn_invalid_time(
                "Không đọc được thời lượng video nguồn nên chưa thể thêm cảnh."
            )
            return
        end = start + clip_dur
        end = min(end, vdur)
        if end <= start:
            self._warn_invalid_time(
                "Không còn khoảng thời gian hợp lệ ở cuối video nguồn."
            )
            return
        new.end_time = round(end, 3)
        valid, error = self._validate_time_range(
            new,
            new.start_time,
            new.end_time,
        )
        if not valid:
            self._warn_invalid_time(error)
            return
        insert_at = row + 1 if row >= 0 else len(self._project.clips)
        self._project.clips.insert(insert_at, new)
        self._commit(select_row=insert_at)

    def _duplicate_part(self):
        row = self._selected_row()
        if row < 0:
            return
        src = self._project.clips[row]
        valid, error = self._validate_time_range(
            src,
            src.start_time,
            src.end_time,
        )
        if not valid:
            self._restore_time_widgets(row, src)
            self._warn_invalid_time(error)
            return
        dup = Clip.from_dict(src.to_dict())
        dup.id = str(uuid.uuid4())[:8]
        self._clear_clip_derivatives(dup)
        self._project.clips.insert(row + 1, dup)
        self._commit(select_row=row + 1)

    def _move_part(self, delta: int):
        row = self._selected_row()
        if row < 0:
            return
        new = row + delta
        if not (0 <= new < len(self._project.clips)):
            return
        clips = self._project.clips
        clips[row], clips[new] = clips[new], clips[row]
        self._commit(select_row=new)

    def _merge_with_next(self):
        row = self._selected_row()
        if row < 0 or row + 1 >= len(self._project.clips):
            QMessageBox.information(self, "Gộp Part", "Không có Part kế tiếp để gộp.")
            return
        cur = self._project.clips[row]
        nxt = self._project.clips[row + 1]
        if not self._same_source(cur, nxt):
            self._warn_invalid_time(
                "Chỉ có thể gộp hai cảnh dùng cùng một video nguồn."
            )
            return

        for clip in (cur, nxt):
            valid, error = self._validate_time_range(
                clip,
                clip.start_time,
                clip.end_time,
            )
            if not valid:
                self._restore_time_widgets(row, cur)
                self._warn_invalid_time(error)
                return

        merged_start = min(cur.start_time, nxt.start_time)
        merged_end = max(cur.end_time, nxt.end_time)
        valid, error = self._validate_time_range(
            cur,
            merged_start,
            merged_end,
        )
        if not valid:
            self._restore_time_widgets(row, cur)
            self._warn_invalid_time(error)
            return

        cur.start_time = merged_start
        cur.end_time = merged_end
        self._clear_clip_derivatives(cur)
        del self._project.clips[row + 1]
        self._commit(select_row=row)

    def _split_part(self):
        row = self._selected_row()
        if row < 0:
            return
        clip = self._project.clips[row]
        valid, error = self._validate_time_range(
            clip,
            clip.start_time,
            clip.end_time,
        )
        if not valid:
            self._restore_time_widgets(row, clip)
            self._warn_invalid_time(error)
            return
        txt = self.txt_split.text().strip()
        if txt:
            split_t = hms_to_seconds(txt)
        else:
            split_t = (clip.start_time + clip.end_time) / 2.0
        if (
            split_t is None
            or not math.isfinite(split_t)
            or not (clip.start_time < split_t < clip.end_time)
        ):
            QMessageBox.warning(
                self, "Tách Part",
                f"Mốc tách phải nằm trong khoảng "
                f"{seconds_to_hms(clip.start_time)} – {seconds_to_hms(clip.end_time)}.",
            )
            return
        second = Clip.from_dict(clip.to_dict())
        second.id = str(uuid.uuid4())[:8]
        second.start_time = round(split_t, 3)
        second.end_time = clip.end_time
        clip.end_time = round(split_t, 3)
        self._clear_clip_derivatives(clip)
        self._clear_clip_derivatives(second)
        self._project.clips.insert(row + 1, second)
        self.txt_split.clear()
        self._commit(select_row=row)

    # ─── Subtitle ─────────────────────────────────────────────────

    def _browse_srt(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Chọn file subtitle", "", "Subtitle (*.srt);;All Files (*)"
        )
        if path and self._project:
            self._project.export_config.global_subtitle_file = path
            self._project.export_config.subtitle_enabled = True
            self.txt_subtitle.setText(Path(path).name)
            save_project(self._project)

    def _clear_srt(self):
        if self._project:
            self._project.export_config.global_subtitle_file = ""
            self._project.export_config.subtitle_enabled = False
            self.txt_subtitle.clear()
            save_project(self._project)
