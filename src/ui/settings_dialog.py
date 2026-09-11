"""Settings dialog: API keys, dependencies."""

import json
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QPushButton, QLineEdit, QTabWidget,
    QWidget, QGroupBox, QProgressBar, QTextEdit,
    QMessageBox,
)

from src.ui.workers import DownloadDepsWorker
from src.core.dependency_manager import check_and_report, ffmpeg_path, ytdlp_path

DATA_DIR = Path("data")
API_KEYS_FILE = DATA_DIR / "api_keys.json"

_PROVIDERS = [
    ("OpenAI", "openai"),
    ("Gemini (Google) — free 15 RPM", "gemini"),
    ("Groq — free Llama/Mixtral", "groq"),
    ("OpenRouter — free models", "openrouter"),
    ("Claude (Anthropic)", "claude"),
    ("YouTube Data API v3 — tìm kênh", "youtube_data"),
    ("Ollama (local, no key)", "ollama"),
    ("LM Studio (local)", "lmstudio"),
    ("ElevenLabs TTS", "elevenlabs"),
    ("OpenAI TTS", "openai_tts"),
    ("Azure TTS", "azure_tts"),
    ("Google TTS", "google_tts"),
]


def _load_api_keys() -> dict:
    DATA_DIR.mkdir(exist_ok=True)
    if API_KEYS_FILE.exists():
        try:
            with open(API_KEYS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_api_keys(keys: dict) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    with open(API_KEYS_FILE, "w", encoding="utf-8") as f:
        json.dump(keys, f, ensure_ascii=False, indent=2)


class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Cài đặt")
        self.setMinimumSize(560, 480)
        self._dep_worker: DownloadDepsWorker | None = None
        self._key_fields: dict[str, QLineEdit] = {}
        self._setup_ui()
        self._load_keys_to_ui()
        self._refresh_dep_status()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        tabs = QTabWidget()

        # ── Tab 1: Dependencies ──────────────────────────────────
        dep_tab = QWidget()
        dep_layout = QVBoxLayout(dep_tab)

        status_group = QGroupBox("Trạng thái")
        status_form = QFormLayout(status_group)
        self.lbl_ffmpeg = QLabel("...")
        self.lbl_ytdlp = QLabel("...")
        status_form.addRow("FFmpeg:", self.lbl_ffmpeg)
        status_form.addRow("yt-dlp:", self.lbl_ytdlp)
        dep_layout.addWidget(status_group)

        btn_row = QHBoxLayout()
        self.btn_dl_ffmpeg = QPushButton("⬇ Tải FFmpeg")
        self.btn_dl_ffmpeg.clicked.connect(lambda: self._download_deps(ffmpeg=True, ytdlp=False))
        self.btn_dl_ytdlp = QPushButton("⬇ Tải yt-dlp")
        self.btn_dl_ytdlp.clicked.connect(lambda: self._download_deps(ffmpeg=False, ytdlp=True))
        self.btn_dl_all = QPushButton("⬇ Tải tất cả")
        self.btn_dl_all.clicked.connect(lambda: self._download_deps(ffmpeg=True, ytdlp=True))
        btn_row.addWidget(self.btn_dl_ffmpeg)
        btn_row.addWidget(self.btn_dl_ytdlp)
        btn_row.addWidget(self.btn_dl_all)
        dep_layout.addLayout(btn_row)

        self.dep_progress = QProgressBar()
        self.dep_progress.setVisible(False)
        dep_layout.addWidget(self.dep_progress)

        self.dep_log = QTextEdit()
        self.dep_log.setReadOnly(True)
        self.dep_log.setMaximumHeight(120)
        self.dep_log.setStyleSheet("background: #111; color: #aaa; font-size: 11px;")
        dep_layout.addWidget(self.dep_log)
        dep_layout.addStretch()

        tabs.addTab(dep_tab, "🔧 Dependencies")

        # ── Tab 2: API Keys ──────────────────────────────────────
        api_tab = QWidget()
        api_layout = QVBoxLayout(api_tab)

        api_group = QGroupBox("API Keys (lưu local, không upload)")
        api_form = QFormLayout(api_group)

        for label, key_id in _PROVIDERS:
            field = QLineEdit()
            field.setEchoMode(QLineEdit.EchoMode.Password)
            field.setPlaceholderText(f"Nhập API key {label}...")
            self._key_fields[key_id] = field
            api_form.addRow(f"{label}:", field)

        api_layout.addWidget(api_group)

        btn_save_keys = QPushButton("💾 Lưu API Keys")
        btn_save_keys.setFixedHeight(36)
        btn_save_keys.clicked.connect(self._save_keys)
        api_layout.addWidget(btn_save_keys)
        api_layout.addStretch()
        tabs.addTab(api_tab, "🔑 API Keys")

        layout.addWidget(tabs)

        btn_close = QPushButton("Đóng")
        btn_close.clicked.connect(self.accept)
        layout.addWidget(btn_close)

    def _refresh_dep_status(self):
        status = check_and_report()
        ok_style = "color: #a6e3a1;"
        err_style = "color: #f38ba8;"
        ff = ffmpeg_path()
        yl = ytdlp_path()
        self.lbl_ffmpeg.setText(
            f"✅ Sẵn sàng  ({ff})" if status["ffmpeg"] else "❌ Chưa cài"
        )
        self.lbl_ffmpeg.setStyleSheet(ok_style if status["ffmpeg"] else err_style)
        self.lbl_ytdlp.setText(
            f"✅ Sẵn sàng  ({yl})" if status["ytdlp"] else "❌ Chưa cài"
        )
        self.lbl_ytdlp.setStyleSheet(ok_style if status["ytdlp"] else err_style)

    def _download_deps(self, ffmpeg: bool, ytdlp: bool):
        self._set_dl_buttons(False)
        self.dep_progress.setVisible(True)
        self.dep_progress.setRange(0, 0)
        self.dep_log.clear()

        self._dep_worker = DownloadDepsWorker(ffmpeg, ytdlp)
        self._dep_worker.progress.connect(
            lambda d, t: (
                self.dep_progress.setRange(0, t),
                self.dep_progress.setValue(d),
            )
        )
        self._dep_worker.status.connect(lambda s: self.dep_log.append(s))
        self._dep_worker.finished.connect(self._on_dep_done)
        self._dep_worker.start()

    def _on_dep_done(self, ok: bool, msg: str):
        self._set_dl_buttons(True)
        self.dep_progress.setVisible(False)
        self.dep_log.append(msg)
        self._refresh_dep_status()

    def _set_dl_buttons(self, enabled: bool):
        for b in (self.btn_dl_ffmpeg, self.btn_dl_ytdlp, self.btn_dl_all):
            b.setEnabled(enabled)

    def _load_keys_to_ui(self):
        keys = _load_api_keys()
        for key_id, field in self._key_fields.items():
            val = keys.get(key_id, "")
            if val:
                field.setText(val)

    def _save_keys(self):
        keys = {}
        for key_id, field in self._key_fields.items():
            val = field.text().strip()
            if val:
                keys[key_id] = val
        _save_api_keys(keys)
        QMessageBox.information(self, "Đã lưu", "API keys đã được lưu vào data/api_keys.json")
