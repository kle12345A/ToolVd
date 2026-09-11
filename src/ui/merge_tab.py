"""Tab 9: Ghép video — merge multiple clips into one with transitions."""

import os
import random
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QPushButton, QComboBox, QDoubleSpinBox, QSpinBox,
    QListWidget, QListWidgetItem, QGroupBox, QProgressBar, QTextEdit,
    QFileDialog, QMessageBox, QAbstractItemView, QCheckBox,
)

from src.core.video_merger import TRANSITIONS, probe_video
from src.ui.workers import MergeVideosWorker

_RESOLUTIONS = [
    ("Dọc 9:16 (1080×1920)", 1080, 1920),
    ("Ngang 16:9 (1920×1080)", 1920, 1080),
    ("Vuông 1:1 (1080×1080)", 1080, 1080),
]
_VIDEO_FILTER = "Video (*.mp4 *.mov *.mkv *.avi *.webm *.m4v)"


class MergeTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker: MergeVideosWorker | None = None
        self._project = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        hint = QLabel(
            "🎬 Ghép nhiều video thành 1 (như CapCut) với hiệu ứng chuyển cảnh. "
            "Có thể chèn thêm video nhân vật/cảm xúc giữa các đoạn như reaction cutaway."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#94e2d5; padding:4px;")
        layout.addWidget(hint)

        # ── Main video list + reorder controls ──
        list_group = QGroupBox("Video chính (theo thứ tự câu chuyện)")
        lg = QVBoxLayout(list_group)
        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        lg.addWidget(self.list_widget)

        row = QHBoxLayout()
        btn_add = QPushButton("➕ Thêm video...")
        btn_add.clicked.connect(self._add_videos)
        btn_add_parts = QPushButton("📥 Thêm Parts đã xuất")
        btn_add_parts.clicked.connect(self._add_exported_parts)
        btn_up = QPushButton("⬆ Lên")
        btn_up.clicked.connect(lambda: self._move(-1))
        btn_down = QPushButton("⬇ Xuống")
        btn_down.clicked.connect(lambda: self._move(1))
        btn_remove = QPushButton("🗑 Xóa")
        btn_remove.clicked.connect(self._remove_selected)
        for b in (btn_add, btn_add_parts, btn_up, btn_down, btn_remove):
            row.addWidget(b)
        lg.addLayout(row)
        layout.addWidget(list_group, 1)

        # ── Insert / reaction videos ──
        insert_group = QGroupBox("Video nhân vật / cảm xúc chèn giữa các đoạn")
        ig = QVBoxLayout(insert_group)

        self.chk_insert_enabled = QCheckBox(
            "Bật chèn video nhân vật sau mỗi video chính"
        )
        self.chk_insert_enabled.toggled.connect(self._update_insert_enabled)
        ig.addWidget(self.chk_insert_enabled)

        self.insert_list_widget = QListWidget()
        self.insert_list_widget.setMaximumHeight(105)
        self.insert_list_widget.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        ig.addWidget(self.insert_list_widget)

        insert_row = QHBoxLayout()
        btn_add_insert = QPushButton("➕ Thêm video nhân vật...")
        btn_add_insert.clicked.connect(self._add_insert_videos)
        btn_insert_up = QPushButton("⬆ Lên")
        btn_insert_up.clicked.connect(lambda: self._move_insert(-1))
        btn_insert_down = QPushButton("⬇ Xuống")
        btn_insert_down.clicked.connect(lambda: self._move_insert(1))
        btn_remove_insert = QPushButton("🗑 Xóa")
        btn_remove_insert.clicked.connect(self._remove_selected_insert)
        for b in (btn_add_insert, btn_insert_up, btn_insert_down, btn_remove_insert):
            insert_row.addWidget(b)
        ig.addLayout(insert_row)

        insert_form = QFormLayout()
        self.cmb_insert_mode = QComboBox()
        self.cmb_insert_mode.addItem("Xoay vòng theo thứ tự", "round_robin")
        self.cmb_insert_mode.addItem("Ngẫu nhiên mỗi lần chèn", "random")
        insert_form.addRow("Cách chọn video:", self.cmb_insert_mode)

        self.spn_insert_max_dur = QDoubleSpinBox()
        self.spn_insert_max_dur.setRange(0.0, 10.0)
        self.spn_insert_max_dur.setSingleStep(0.1)
        self.spn_insert_max_dur.setValue(1.5)
        self.spn_insert_max_dur.setSuffix(" giây")
        self.spn_insert_max_dur.setSpecialValueText("Dùng đủ clip")
        insert_form.addRow("Cắt video nhân vật tối đa:", self.spn_insert_max_dur)

        self.chk_insert_after_last = QCheckBox("Chèn cả sau video cuối")
        self.chk_insert_after_last.setChecked(False)
        insert_form.addRow(self.chk_insert_after_last)
        ig.addLayout(insert_form)

        layout.addWidget(insert_group)
        self._insert_controls = [
            self.insert_list_widget,
            btn_add_insert,
            btn_insert_up,
            btn_insert_down,
            btn_remove_insert,
            self.cmb_insert_mode,
            self.spn_insert_max_dur,
            self.chk_insert_after_last,
        ]
        self._update_insert_enabled(False)

        # ── Transition + output settings ──
        opt_group = QGroupBox("Hiệu ứng & khung hình")
        form = QFormLayout(opt_group)

        self.cmb_transition = QComboBox()
        self.cmb_transition.addItem("Cắt thẳng (không hiệu ứng)", "none")
        for key, label in TRANSITIONS.items():
            self.cmb_transition.addItem(label, key)
        self.cmb_transition.currentIndexChanged.connect(self._update_transition_enabled)
        form.addRow("Hiệu ứng chuyển cảnh:", self.cmb_transition)

        self.spn_trans_dur = QDoubleSpinBox()
        self.spn_trans_dur.setRange(0.1, 3.0)
        self.spn_trans_dur.setSingleStep(0.1)
        self.spn_trans_dur.setValue(0.7)
        self.spn_trans_dur.setSuffix(" giây")
        form.addRow("Thời lượng chuyển cảnh:", self.spn_trans_dur)
        self._update_transition_enabled()

        self.cmb_res = QComboBox()
        for label, w, h in _RESOLUTIONS:
            self.cmb_res.addItem(label, (w, h))
        form.addRow("Khung hình:", self.cmb_res)

        self.spn_fps = QSpinBox()
        self.spn_fps.setRange(24, 60)
        self.spn_fps.setValue(30)
        form.addRow("FPS:", self.spn_fps)
        layout.addWidget(opt_group)

        # ── Actions ──
        act_row = QHBoxLayout()
        self.btn_merge = QPushButton("🎬 Ghép video")
        self.btn_merge.setMinimumHeight(40)
        self.btn_merge.setStyleSheet(
            "background:#7aa2f7; color:#1e1e2e; font-weight:bold; font-size:14px;")
        self.btn_merge.clicked.connect(self._start_merge)
        self.btn_open = QPushButton("📂 Mở video")
        self.btn_open.setEnabled(False)
        self.btn_open.clicked.connect(self._open_output)
        act_row.addWidget(self.btn_merge, 2)
        act_row.addWidget(self.btn_open, 1)
        layout.addLayout(act_row)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setMaximumHeight(90)
        self.txt_log.setStyleSheet("background:#111; color:#cdd6f4; font-size:11px;")
        layout.addWidget(self.txt_log)

        self._output_path = ""

    # ── Project integration ──
    def load_project(self, project):
        self._project = project

    # ── List management ──
    def _add_item(self, path: str):
        info = probe_video(path)
        dur = info["duration"]
        item = QListWidgetItem(f"{Path(path).name}   ({dur:.1f}s)")
        item.setData(Qt.ItemDataRole.UserRole, path)
        self.list_widget.addItem(item)

    def _add_insert_item(self, path: str):
        info = probe_video(path)
        dur = info["duration"]
        item = QListWidgetItem(f"{Path(path).name}   ({dur:.1f}s)")
        item.setData(Qt.ItemDataRole.UserRole, path)
        self.insert_list_widget.addItem(item)

    def _add_videos(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Chọn video để ghép", "", _VIDEO_FILTER)
        for p in paths:
            self._add_item(p)

    def _add_insert_videos(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Chọn video nhân vật/cảm xúc để chèn", "", _VIDEO_FILTER)
        for p in paths:
            self._add_insert_item(p)
        if paths:
            self.chk_insert_enabled.setChecked(True)

    def _add_exported_parts(self):
        if not self._project or not getattr(self._project, "clips", None):
            QMessageBox.information(
                self, "Chưa có", "Chưa có project hoặc chưa xuất Part nào.")
            return
        added = 0
        for c in self._project.clips:
            ep = getattr(c, "export_path", "")
            if ep and Path(ep).exists():
                self._add_item(ep)
                added += 1
        if added == 0:
            QMessageBox.information(
                self, "Chưa có", "Chưa tìm thấy Part nào đã xuất. Hãy xuất ở Tab 5 trước.")

    def _move(self, delta: int):
        row = self.list_widget.currentRow()
        if row < 0:
            return
        new = row + delta
        if not (0 <= new < self.list_widget.count()):
            return
        item = self.list_widget.takeItem(row)
        self.list_widget.insertItem(new, item)
        self.list_widget.setCurrentRow(new)

    def _remove_selected(self):
        for item in self.list_widget.selectedItems():
            self.list_widget.takeItem(self.list_widget.row(item))

    def _move_insert(self, delta: int):
        row = self.insert_list_widget.currentRow()
        if row < 0:
            return
        new = row + delta
        if not (0 <= new < self.insert_list_widget.count()):
            return
        item = self.insert_list_widget.takeItem(row)
        self.insert_list_widget.insertItem(new, item)
        self.insert_list_widget.setCurrentRow(new)

    def _remove_selected_insert(self):
        for item in self.insert_list_widget.selectedItems():
            self.insert_list_widget.takeItem(self.insert_list_widget.row(item))

    def _paths(self) -> list[str]:
        return [
            self.list_widget.item(i).data(Qt.ItemDataRole.UserRole)
            for i in range(self.list_widget.count())
        ]

    def _insert_paths(self) -> list[str]:
        return [
            self.insert_list_widget.item(i).data(Qt.ItemDataRole.UserRole)
            for i in range(self.insert_list_widget.count())
        ]

    def _update_insert_enabled(self, enabled: bool):
        for w in getattr(self, "_insert_controls", []):
            w.setEnabled(enabled)

    def _update_transition_enabled(self):
        self.spn_trans_dur.setEnabled(self.cmb_transition.currentData() != "none")

    def _build_merge_sequence(self, main_paths: list[str]) -> tuple[list[str], list[float | None]]:
        if not self.chk_insert_enabled.isChecked():
            return main_paths, [None] * len(main_paths)

        inserts = self._insert_paths()
        if not inserts:
            return main_paths, [None] * len(main_paths)

        sequence: list[str] = []
        caps: list[float | None] = []
        max_insert_dur = float(self.spn_insert_max_dur.value() or 0.0)
        mode = self.cmb_insert_mode.currentData()
        after_last = self.chk_insert_after_last.isChecked()

        insert_index = 0
        for i, path in enumerate(main_paths):
            sequence.append(path)
            caps.append(None)

            should_insert = after_last or i < len(main_paths) - 1
            if not should_insert:
                continue

            if mode == "random":
                insert_path = random.choice(inserts)
            else:
                insert_path = inserts[insert_index % len(inserts)]
                insert_index += 1

            sequence.append(insert_path)
            caps.append(max_insert_dur if max_insert_dur > 0 else None)

        return sequence, caps

    # ── Merge ──
    def _start_merge(self):
        paths = self._paths()
        if not paths:
            QMessageBox.warning(self, "Thiếu video", "Hãy thêm ít nhất 1 video chính.")
            return
        if self.chk_insert_enabled.isChecked() and not self._insert_paths():
            QMessageBox.warning(
                self,
                "Thiếu video nhân vật",
                "Bạn đã bật chế độ chèn nhưng chưa thêm video nhân vật/cảm xúc.",
            )
            return

        merge_paths, clip_caps = self._build_merge_sequence(paths)
        if len(merge_paths) < 2:
            QMessageBox.warning(self, "Thiếu video", "Cần ít nhất 2 đoạn để ghép.")
            return
        default_dir = ""
        if self._project and getattr(self._project, "output_dir", ""):
            default_dir = str(Path(self._project.output_dir) / "merged.mp4")
        else:
            default_dir = "merged.mp4"
        out, _ = QFileDialog.getSaveFileName(
            self, "Lưu video ghép", default_dir, "Video (*.mp4)")
        if not out:
            return
        if not out.lower().endswith(".mp4"):
            out += ".mp4"
        self._output_path = out

        w, h = self.cmb_res.currentData()
        self.txt_log.clear()
        if self.chk_insert_enabled.isChecked():
            inserted = len(merge_paths) - len(paths)
            self.txt_log.append(
                f"⏳ Đang ghép {len(paths)} video chính + {inserted} video nhân vật..."
            )
        else:
            self.txt_log.append(f"⏳ Đang ghép {len(paths)} video...")
        self.btn_merge.setEnabled(False)
        self.btn_open.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)

        self._worker = MergeVideosWorker(
            merge_paths, self.cmb_transition.currentData(),
            self.spn_trans_dur.value(), out, w, h, self.spn_fps.value(),
            clip_max_durations=clip_caps,
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    def _on_progress(self, elapsed: float, total: float):
        if total > 0:
            self.progress.setValue(min(100, int(elapsed / total * 100)))

    def _on_finished(self, ok: bool, result: str):
        self.btn_merge.setEnabled(True)
        self.progress.setVisible(False)
        if ok:
            parts = result.split("\n", 1)
            self.txt_log.append(f"✅ Ghép xong: {Path(parts[0]).name}")
            if len(parts) > 1:
                self.txt_log.append(parts[1])
            self.btn_open.setEnabled(True)
        else:
            self.txt_log.append(f"❌ Lỗi: {result}")

    def _open_output(self):
        if self._output_path and os.path.exists(self._output_path):
            try:
                os.startfile(self._output_path)
            except Exception as e:
                QMessageBox.warning(self, "Lỗi", f"Không mở được: {e}")
