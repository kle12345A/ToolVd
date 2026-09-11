"""Tab 8: YouTube channel research & reup-scoring tool (standalone, no project)."""

import os
from pathlib import Path

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QPushButton, QComboBox, QCheckBox, QSpinBox,
    QPlainTextEdit, QTextEdit, QGroupBox, QMessageBox, QFileDialog,
)

from src.core.ai_client import load_api_key
from src.core.youtube_research import (
    DEFAULT_KEYWORDS, SUB_MIN, SUB_MAX,
)
from src.ui.workers import YouTubeResearchWorker

_AI_PROVIDERS = [
    ("Gemini (Google)", "gemini"),
    ("Groq", "groq"),
    ("OpenRouter", "openrouter"),
    ("Ollama (local)", "ollama"),
]
_DEFAULT_OUT = str(Path("data") / "youtube_research")


class YouTubeResearchTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker: YouTubeResearchWorker | None = None
        self._report_path = ""
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        hint = QLabel(
            "🔎 Tìm & chấm điểm kênh YouTube để reup TikTok US. "
            "Cần API key YouTube Data v3 (Cài đặt → API Keys)."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#94e2d5; padding:4px;")
        layout.addWidget(hint)

        # ── Input group ──
        ig = QGroupBox("Tiêu chí tìm kiếm")
        form = QFormLayout(ig)

        self.txt_keywords = QPlainTextEdit()
        self.txt_keywords.setPlaceholderText("Mỗi dòng 1 keyword...")
        self.txt_keywords.setPlainText("\n".join(DEFAULT_KEYWORDS))
        self.txt_keywords.setFixedHeight(110)
        form.addRow("Keywords:", self.txt_keywords)

        self.spn_per_kw = QSpinBox()
        self.spn_per_kw.setRange(1, 50)
        self.spn_per_kw.setValue(20)
        form.addRow("Số kênh / keyword:", self.spn_per_kw)

        sub_row = QHBoxLayout()
        self.spn_sub_min = QSpinBox()
        self.spn_sub_min.setRange(0, 100_000_000)
        self.spn_sub_min.setValue(SUB_MIN)
        self.spn_sub_min.setGroupSeparatorShown(True)
        self.spn_sub_max = QSpinBox()
        self.spn_sub_max.setRange(0, 100_000_000)
        self.spn_sub_max.setValue(SUB_MAX)
        self.spn_sub_max.setGroupSeparatorShown(True)
        sub_row.addWidget(QLabel("Min:"))
        sub_row.addWidget(self.spn_sub_min)
        sub_row.addWidget(QLabel("Max:"))
        sub_row.addWidget(self.spn_sub_max)
        sub_w = QWidget()
        sub_w.setLayout(sub_row)
        form.addRow("Subscribers:", sub_w)

        layout.addWidget(ig)

        # ── AI group ──
        ag = QGroupBox("Chấm điểm AI (tùy chọn — 1 request/kênh, tiết kiệm token)")
        af = QFormLayout(ag)
        self.chk_ai = QCheckBox("Bật AI chấm KÊNH + đưa ra lý do nên/không nên reup")
        self.chk_ai.setChecked(False)
        self.chk_ai.toggled.connect(self._on_ai_toggled)
        af.addRow(self.chk_ai)
        self.cmb_provider = QComboBox()
        for label, key in _AI_PROVIDERS:
            self.cmb_provider.addItem(label, key)
        self.cmb_provider.setEnabled(False)
        af.addRow("Provider:", self.cmb_provider)
        layout.addWidget(ag)

        # ── Output dir ──
        out_row = QHBoxLayout()
        self.lbl_out = QLabel(_DEFAULT_OUT)
        self.lbl_out.setStyleSheet("color:#a6adc8;")
        btn_out = QPushButton("📁 Chọn thư mục")
        btn_out.clicked.connect(self._choose_out)
        out_row.addWidget(QLabel("Lưu vào:"))
        out_row.addWidget(self.lbl_out, 1)
        out_row.addWidget(btn_out)
        layout.addLayout(out_row)

        # ── Action buttons ──
        btn_row = QHBoxLayout()
        self.btn_run = QPushButton("▶ Bắt đầu nghiên cứu")
        self.btn_run.setFixedHeight(40)
        self.btn_run.clicked.connect(self._on_run)
        self.btn_stop = QPushButton("⏹ Dừng")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._on_stop)
        self.btn_open = QPushButton("📂 Mở báo cáo")
        self.btn_open.setEnabled(False)
        self.btn_open.clicked.connect(self._open_report)
        btn_row.addWidget(self.btn_run, 2)
        btn_row.addWidget(self.btn_stop, 1)
        btn_row.addWidget(self.btn_open, 1)
        layout.addLayout(btn_row)

        # ── Log ──
        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setStyleSheet("background:#111; color:#cdd6f4; font-size:12px;")
        layout.addWidget(self.txt_log, 1)

    # ── Slots ──
    def _on_ai_toggled(self, checked: bool):
        self.cmb_provider.setEnabled(checked)

    def _choose_out(self):
        d = QFileDialog.getExistingDirectory(self, "Chọn thư mục output", _DEFAULT_OUT)
        if d:
            self.lbl_out.setText(d)

    def _on_run(self):
        api_key = load_api_key("youtube_data")
        if not api_key:
            QMessageBox.warning(
                self, "Thiếu API key",
                "Chưa có YouTube Data API v3 key.\n\n"
                "Vào ⚙ Cài đặt → API Keys → nhập 'YouTube Data API v3' rồi Lưu.",
            )
            return

        keywords = [
            ln.strip() for ln in self.txt_keywords.toPlainText().splitlines()
            if ln.strip()
        ]
        if not keywords:
            QMessageBox.warning(self, "Thiếu keyword", "Nhập ít nhất 1 keyword.")
            return

        run_ai = self.chk_ai.isChecked()
        provider = self.cmb_provider.currentData() or "gemini"

        self.txt_log.clear()
        self._set_running(True)

        self._worker = YouTubeResearchWorker(
            keywords=keywords,
            api_key=api_key,
            out_dir=self.lbl_out.text(),
            run_ai=run_ai,
            ai_provider=provider,
            ai_model="",
            max_per_keyword=self.spn_per_kw.value(),
        )
        self._worker.status.connect(self._on_status)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    def _on_stop(self):
        if self._worker:
            self._worker.stop()
            self._on_status("⏹ Đang dừng...")
            self.btn_stop.setEnabled(False)

    def _on_status(self, msg: str):
        self.txt_log.append(msg)
        sb = self.txt_log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _on_finished(self, ok: bool, result: str):
        self._set_running(False)
        if ok:
            self._report_path = result
            self.btn_open.setEnabled(True)
            self._on_status(f"🎉 Hoàn tất! Báo cáo: {result}")
        else:
            self._on_status(f"❌ {result}")

    def _open_report(self):
        if self._report_path and os.path.exists(self._report_path):
            try:
                os.startfile(self._report_path)  # Windows
            except Exception as e:
                QMessageBox.warning(self, "Lỗi", f"Không mở được: {e}")

    def _set_running(self, running: bool):
        self.btn_run.setEnabled(not running)
        self.btn_stop.setEnabled(running)

    # No-op to stay compatible with main_window's load_project loop
    def load_project(self, project):
        pass
