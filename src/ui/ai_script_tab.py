"""Tab 6: AI Kịch bản — AI-generated voiceover review scripts per Part."""

from __future__ import annotations

import json

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QPushButton, QComboBox, QCheckBox, QSpinBox,
    QGroupBox, QTextEdit, QSplitter, QScrollArea,
    QMessageBox, QProgressBar, QSizePolicy, QFrame,
)

from src.core.project_manager import save_project
from src.models.project import Project
from src.utils.logger import logger

_SCRIPT_TARGET_WPS = 4.6

from src.core.ai_client import (
    REVIEW_STYLES,
    GEMINI_FREE_MODELS,
    GROQ_FREE_MODELS,
    OPENROUTER_FREE_MODELS,
    OLLAMA_SUGGESTED,
    generate_review_scripts,
    load_transcript_text,
)


# ─── Background worker ─────────────────────────────────────────────────────────

class _AIWorker(QThread):
    status  = pyqtSignal(str)
    finished = pyqtSignal(bool, object)   # success, result_dict_or_error_str

    def __init__(self, project, provider, model, style_key,
                 output_language, duration_per_part_sec, num_parts,
                 clip_durations=None, extra_prompt=""):
        super().__init__()
        self.project = project
        self.provider = provider
        self.model = model
        self.style_key = style_key
        self.output_language = output_language
        self.duration_per_part_sec = duration_per_part_sec
        self.num_parts = num_parts
        self.clip_durations = clip_durations
        self.extra_prompt = extra_prompt

    def run(self):
        try:
            result = generate_review_scripts(
                self.project,
                provider=self.provider,
                model=self.model,
                style_key=self.style_key,
                output_language=self.output_language,
                duration_per_part_sec=self.duration_per_part_sec,
                num_parts=self.num_parts,
                clip_durations=self.clip_durations,
                extra_prompt=self.extra_prompt,
                status_cb=lambda s: self.status.emit(s),
            )
            self.finished.emit(True, result)
        except Exception as e:
            logger.error(f"AI script generation failed: {e}")
            self.finished.emit(False, str(e))


# ─── Per-part script card ──────────────────────────────────────────────────────

class _PartScriptCard(QFrame):
    """Editable script card for one Part."""

    def __init__(self, index: int, title: str, script: str,
                 estimated_seconds: int = 60, parent=None):
        super().__init__(parent)
        self.index = index
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet(
            "QFrame { background: #1e1e2e; border: 1px solid #313244;"
            " border-radius: 6px; }"
        )

        layout = QVBoxLayout(self)
        layout.setSpacing(4)

        # Header row
        hdr = QHBoxLayout()
        self.lbl_title = QLabel(title)
        self.lbl_title.setStyleSheet(
            "color: #cba6f7; font-weight: bold; font-size: 13px;"
        )
        hdr.addWidget(self.lbl_title, 1)

        self.lbl_stats = QLabel(f"~{estimated_seconds}s")
        self.lbl_stats.setStyleSheet("color: #888; font-size: 11px;")
        hdr.addWidget(self.lbl_stats)
        layout.addLayout(hdr)

        self.txt = QTextEdit()
        self.txt.setPlainText(script)
        self.txt.setMinimumHeight(110)
        self.txt.setMaximumHeight(220)
        self.txt.setStyleSheet(
            "background: #181825; color: #cdd6f4; font-size: 13px;"
            " border: 1px solid #313244; border-radius: 4px;"
        )
        self.txt.textChanged.connect(self._update_stats)
        layout.addWidget(self.txt)

        self._update_stats()

    def _update_stats(self):
        text = self.txt.toPlainText()
        words = len(text.split())
        secs = max(1, int(words / _SCRIPT_TARGET_WPS))
        self.lbl_stats.setText(f"{words} từ · ~{secs}s đọc")

    def get_script(self) -> str:
        return self.txt.toPlainText().strip()

    def set_title(self, title: str):
        self.lbl_title.setText(title)
        self.lbl_title.setStyleSheet(
            "color: #cba6f7; font-weight: bold; font-size: 13px;"
        )


# ─── Main tab ──────────────────────────────────────────────────────────────────

class AIScriptTab(QWidget):
    scripts_saved = pyqtSignal()   # emitted after saving to clips

    def __init__(self, parent=None):
        super().__init__(parent)
        self._project: Project | None = None
        self._worker: _AIWorker | None = None
        self._cards: list[_PartScriptCard] = []
        self._setup_ui()

    # ─── UI ────────────────────────────────────────────────────────────────────

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(6)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # ── Left: settings ─────────────────────────────────────────────────
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left_scroll.setStyleSheet("QScrollArea{border:none;}")
        # Options panel takes ~40% of the width (4:6 split) so dropdowns/labels
        # are never clipped; the splitter lets the user fine-tune it.
        left_scroll.setMinimumWidth(340)

        lw = QWidget()
        ll = QVBoxLayout(lw)
        ll.setSpacing(8)
        ll.setContentsMargins(4, 4, 8, 4)

        # Project status
        status_g = QGroupBox("Trạng thái Project")
        status_fl = QFormLayout(status_g)
        self.lbl_project_name = QLabel("Chưa có project")
        self.lbl_project_name.setStyleSheet("color: #888;")
        status_fl.addRow("Project:", self.lbl_project_name)
        self.lbl_transcript_status = QLabel("❌ Chưa có transcript")
        self.lbl_transcript_status.setStyleSheet("color: #f38ba8;")
        status_fl.addRow("Transcript:", self.lbl_transcript_status)
        self.lbl_clips_count = QLabel("0 Parts")
        status_fl.addRow("Số Parts:", self.lbl_clips_count)
        ll.addWidget(status_g)

        # AI Provider
        ai_g = QGroupBox("🤖 AI Provider")
        ai_fl = QFormLayout(ai_g)

        self.cmb_provider = QComboBox()
        self.cmb_provider.addItem("Google Gemini (free 15 RPM)", "gemini")
        self.cmb_provider.addItem("Groq (free Llama/Mixtral)", "groq")
        self.cmb_provider.addItem("OpenRouter (free models)", "openrouter")
        self.cmb_provider.addItem("Ollama (local, miễn phí)", "ollama")
        self.cmb_provider.currentIndexChanged.connect(self._on_provider_changed)
        ai_fl.addRow("Provider:", self.cmb_provider)

        self.cmb_model = QComboBox()
        ai_fl.addRow("Model:", self.cmb_model)

        # Free tier hint
        self.lbl_free_hint = QLabel()
        self.lbl_free_hint.setStyleSheet(
            "color: #a6e3a1; font-size: 10px; padding: 3px;"
        )
        self.lbl_free_hint.setWordWrap(True)
        ai_fl.addRow(self.lbl_free_hint)

        self.lbl_key_status = QLabel()
        self.lbl_key_status.setStyleSheet("color: #888; font-size: 10px;")
        ai_fl.addRow("API Key:", self.lbl_key_status)

        ll.addWidget(ai_g)

        # Generation settings
        gen_g = QGroupBox("⚙ Cài đặt kịch bản")
        gen_fl = QFormLayout(gen_g)

        self.cmb_style = QComboBox()
        for key, label in REVIEW_STYLES.items():
            self.cmb_style.addItem(label, key)
        gen_fl.addRow("Phong cách:", self.cmb_style)

        self.cmb_language = QComboBox()
        self.cmb_language.addItem("🇻🇳 Tiếng Việt", "vi")
        self.cmb_language.addItem("🇺🇸 English", "en")
        gen_fl.addRow("Ngôn ngữ kịch bản:", self.cmb_language)

        self.chk_auto_duration = QCheckBox("Tự động theo thời lượng từng Part")
        self.chk_auto_duration.setChecked(True)
        self.chk_auto_duration.setToolTip(
            "Khi bật: kịch bản mỗi Part khớp với thời lượng thực của clip.\n"
            "Khi tắt: bạn chọn thời lượng cố định cho tất cả Parts."
        )
        self.chk_auto_duration.toggled.connect(self._on_auto_duration_toggled)
        gen_fl.addRow(self.chk_auto_duration)

        self.spn_duration = QSpinBox()
        self.spn_duration.setRange(5, 900)
        self.spn_duration.setValue(60)
        self.spn_duration.setSuffix(" giây")
        self.spn_duration.setEnabled(False)
        self.spn_duration.valueChanged.connect(self._update_calc)
        gen_fl.addRow("Thời lượng mỗi Part:", self.spn_duration)

        self.chk_auto_parts = QCheckBox("Tự động theo số Parts đã chia")
        self.chk_auto_parts.setChecked(True)
        self.chk_auto_parts.toggled.connect(self._on_auto_parts_toggled)
        gen_fl.addRow(self.chk_auto_parts)

        self.spn_num_parts = QSpinBox()
        self.spn_num_parts.setRange(1, 100)
        self.spn_num_parts.setValue(3)
        self.spn_num_parts.setEnabled(False)
        self.spn_num_parts.valueChanged.connect(self._update_calc)
        gen_fl.addRow("Số Parts:", self.spn_num_parts)

        self.lbl_calc = QLabel()
        self.lbl_calc.setStyleSheet(
            "color: #89b4fa; font-size: 11px; background: #313244;"
            " border-radius: 4px; padding: 4px;"
        )
        self.lbl_calc.setWordWrap(True)
        gen_fl.addRow(self.lbl_calc)

        self.txt_extra_prompt = QTextEdit()
        self.txt_extra_prompt.setMaximumHeight(95)
        self.txt_extra_prompt.setPlaceholderText(
            "Ví dụ: Viết giọng chuyên gia factory, dùng câu ngắn, "
            "không nói quá kỹ thuật, nhấn vào quy trình đang thấy trên video..."
        )
        self.txt_extra_prompt.setStyleSheet(
            "background: #181825; color: #cdd6f4; font-size: 12px;"
        )
        gen_fl.addRow("Prompt/Yêu cầu thêm:", self.txt_extra_prompt)

        ll.addWidget(gen_g)

        # Generate button
        self.btn_generate = QPushButton("✨ Tạo kịch bản AI")
        self.btn_generate.setMinimumHeight(40)
        self.btn_generate.setStyleSheet(
            "background: #7aa2f7; color: #1e1e2e;"
            " font-size: 14px; font-weight: bold;"
        )
        self.btn_generate.clicked.connect(self._generate)
        ll.addWidget(self.btn_generate)

        # Save buttons — stacked vertically so labels never get clipped
        # in the narrow side panel (responsive to small screens).
        self.btn_save_all = QPushButton("💾 Lưu tất cả vào Parts")
        self.btn_save_all.clicked.connect(self._save_all_to_clips)
        self.btn_save_all.setEnabled(False)
        ll.addWidget(self.btn_save_all)

        self.btn_to_voiceover = QPushButton("🎤 → Tab Lồng tiếng")
        self.btn_to_voiceover.clicked.connect(self._send_to_voiceover)
        self.btn_to_voiceover.setEnabled(False)
        ll.addWidget(self.btn_to_voiceover)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)   # indeterminate
        self.progress_bar.setVisible(False)
        ll.addWidget(self.progress_bar)

        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setMinimumHeight(60)
        self.txt_log.setMaximumHeight(110)
        self.txt_log.setStyleSheet("background: #111; color: #aaa; font-size: 11px;")
        ll.addWidget(self.txt_log)

        ll.addStretch()
        left_scroll.setWidget(lw)
        splitter.addWidget(left_scroll)

        # ── Right: overview + per-part script cards ────────────────────────
        right_w = QWidget()
        right_l = QVBoxLayout(right_w)
        right_l.setContentsMargins(4, 4, 4, 4)
        right_l.setSpacing(6)

        right_l.addWidget(QLabel("📋 Tổng quan bộ phim:"))
        self.txt_overview = QTextEdit()
        self.txt_overview.setMaximumHeight(80)
        self.txt_overview.setPlaceholderText(
            "Sau khi tạo kịch bản, tổng quan sẽ xuất hiện ở đây..."
        )
        self.txt_overview.setStyleSheet(
            "background: #181825; color: #cdd6f4; font-size: 13px;"
        )
        right_l.addWidget(self.txt_overview)

        right_l.addWidget(QLabel("🎬 Kịch bản từng Part (có thể chỉnh sửa trực tiếp):"))

        # Scrollable cards area
        cards_scroll = QScrollArea()
        cards_scroll.setWidgetResizable(True)
        cards_scroll.setStyleSheet("QScrollArea{border:none;}")
        self._cards_widget = QWidget()
        self._cards_layout = QVBoxLayout(self._cards_widget)
        self._cards_layout.setSpacing(8)
        self._cards_layout.setContentsMargins(4, 4, 4, 4)
        self._cards_layout.addStretch()
        cards_scroll.setWidget(self._cards_widget)
        right_l.addWidget(cards_scroll, 1)

        splitter.addWidget(right_w)
        # 4:6 ratio — options panel : overview/scripts
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 6)
        splitter.setSizes([460, 690])
        root.addWidget(splitter, 1)

        # Initialise dynamic controls
        self._on_provider_changed()
        self._update_calc()

    # ─── Public API ────────────────────────────────────────────────────────────

    def load_project(self, project: Project):
        self._project = project
        self.lbl_project_name.setText(project.name)
        self.lbl_project_name.setStyleSheet("color: #a6e3a1; font-weight: bold;")
        self.txt_extra_prompt.setPlainText(
            getattr(project, "ai_script_extra_prompt", "") or ""
        )

        # Transcript status
        transcript = load_transcript_text(project)
        if transcript:
            chars = len(transcript)
            self.lbl_transcript_status.setText(f"✅ {chars:,} ký tự")
            self.lbl_transcript_status.setStyleSheet("color: #a6e3a1;")
        else:
            self.lbl_transcript_status.setText("Chưa có - sẽ dùng mô tả hình ảnh/visual")
            self.lbl_transcript_status.setStyleSheet("color: #f9e2af;")

        enabled_clips = self._enabled_clips()
        n = len(enabled_clips)
        self.spn_num_parts.setMaximum(max(1, n))
        if n > 0:
            total_dur = sum(c.duration for c in enabled_clips)
            avg_dur = total_dur / n
            self.lbl_clips_count.setText(
                f"{n} Parts đã chọn | Tổng: {total_dur/60:.1f} phút | "
                f"TB: {avg_dur:.0f}s/Part"
            )
            self.spn_num_parts.setValue(n)
            # Set manual duration to average clip length
            self.spn_duration.setValue(max(5, int(avg_dur)))
        else:
            self.spn_num_parts.setValue(1)
            self.lbl_clips_count.setText("0 Parts đã chọn")

        # Load saved scripts
        self._reload_cards_from_clips()
        self._update_calc()

    # ─── Internal helpers ──────────────────────────────────────────────────────

    def _enabled_clips(self) -> list:
        if not self._project:
            return []
        return [
            clip for clip in (self._project.clips or [])
            if getattr(clip, "enabled", True)
        ]

    def _on_provider_changed(self, _=None):
        provider = self.cmb_provider.currentData() or "gemini"
        self.cmb_model.clear()

        if provider == "gemini":
            for mid, label in GEMINI_FREE_MODELS:
                self.cmb_model.addItem(label, mid)
            self.lbl_free_hint.setText(
                "✅ Free: 15 req/min · 1M tokens/min\n"
                "Lấy key tại: aistudio.google.com"
            )
        elif provider == "groq":
            for mid, label in GROQ_FREE_MODELS:
                self.cmb_model.addItem(label, mid)
            self.lbl_free_hint.setText(
                "✅ Free: 14,400 req/day · rất nhanh\n"
                "Lấy key tại: console.groq.com"
            )
        elif provider == "openrouter":
            for mid, label in OPENROUTER_FREE_MODELS:
                self.cmb_model.addItem(label, mid)
            self.lbl_free_hint.setText(
                "✅ Free models (đánh dấu :free)\n"
                "Lấy key tại: openrouter.ai/keys"
            )
        elif provider == "ollama":
            for mid, label in OLLAMA_SUGGESTED:
                self.cmb_model.addItem(label, mid)
            self.lbl_free_hint.setText(
                "✅ Hoàn toàn miễn phí, chạy local\n"
                "Cài tại: ollama.com  |  Không cần API key"
            )

        # Check API key
        self._refresh_key_status()

    def _refresh_key_status(self):
        provider = self.cmb_provider.currentData() or "gemini"
        if provider == "ollama":
            self.lbl_key_status.setText("✅ Không cần API key")
            self.lbl_key_status.setStyleSheet("color: #a6e3a1; font-size: 10px;")
            return
        from src.core.ai_client import load_api_key
        key = load_api_key(provider)
        if key:
            masked = key[:6] + "..." + key[-4:] if len(key) > 10 else "***"
            self.lbl_key_status.setText(f"✅ Đã có ({masked})")
            self.lbl_key_status.setStyleSheet("color: #a6e3a1; font-size: 10px;")
        else:
            self.lbl_key_status.setText("❌ Chưa có — vào ⚙ Cài đặt → API Keys")
            self.lbl_key_status.setStyleSheet("color: #f38ba8; font-size: 10px;")

    def _on_auto_duration_toggled(self, checked: bool):
        self.spn_duration.setEnabled(not checked)
        self._update_calc()

    def _on_auto_parts_toggled(self, checked: bool):
        self.spn_num_parts.setEnabled(not checked)
        self._update_calc()

    def _update_calc(self):
        num = self._effective_num_parts()
        enabled_clips = self._enabled_clips()
        if not enabled_clips:
            self.lbl_calc.setText("Chưa có Part nào được chọn để tạo kịch bản")
            return
        if self.chk_auto_duration.isChecked():
            clips = enabled_clips[:num]
            total = sum(c.duration for c in clips)
            avg = total / max(1, len(clips))
            words = int(avg * _SCRIPT_TARGET_WPS)
            self.lbl_calc.setText(
                f"{num} Parts | Tổng video: {total/60:.1f} phút\n"
                f"TB: {avg:.0f}s/Part · ~{words} từ/Part\n"
                f"Kịch bản sẽ khớp thời lượng từng clip"
            )
        else:
            dur_sec = self.spn_duration.value()
            words = int(dur_sec * _SCRIPT_TARGET_WPS)
            self.lbl_calc.setText(
                f"{num} Parts × {dur_sec}s\n"
                f"≈ {words} từ/Part"
            )

    def _effective_num_parts(self) -> int:
        enabled_count = len(self._enabled_clips())
        if enabled_count <= 0:
            return 0
        if self.chk_auto_parts.isChecked():
            return enabled_count
        return min(self.spn_num_parts.value(), enabled_count)

    # ─── Cards ─────────────────────────────────────────────────────────────────

    def _clear_cards(self):
        self._cards = []
        self.btn_save_all.setEnabled(False)
        self.btn_to_voiceover.setEnabled(False)
        # Remove all but the trailing stretch
        while self._cards_layout.count() > 1:
            item = self._cards_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _add_card(self, index: int, title: str, script: str,
                  estimated_seconds: int = 60):
        card = _PartScriptCard(index, title, script, estimated_seconds)
        # Insert before the trailing stretch
        pos = max(0, self._cards_layout.count() - 1)
        self._cards_layout.insertWidget(pos, card)
        self._cards.append(card)

    def _reload_cards_from_clips(self):
        """Populate cards from clips' saved voiceover_script (if any)."""
        self._clear_cards()
        enabled_clips = self._enabled_clips()
        if not enabled_clips:
            return
        has_scripts = any(
            c.voiceover_script for c in enabled_clips
        )
        if not has_scripts:
            return
        for clip in enabled_clips:
            if clip.voiceover_script:
                self._add_card(
                    clip.index,
                    clip.part_text,
                    clip.voiceover_script,
                )
        if self._cards:
            self.btn_save_all.setEnabled(True)
            self.btn_to_voiceover.setEnabled(True)

    # ─── Generate ──────────────────────────────────────────────────────────────

    def _generate(self):
        if not self._project:
            QMessageBox.warning(self, "Chưa có project",
                                "Vui lòng mở project trước.")
            return

        enabled_clips = self._enabled_clips()
        if not enabled_clips:
            QMessageBox.warning(
                self,
                "Chưa chọn Part",
                "Hãy bật ít nhất một Part ở Tab Chỉnh sửa trước khi tạo kịch bản.",
            )
            return

        transcript = load_transcript_text(self._project)

        provider = self.cmb_provider.currentData() or "gemini"
        if provider != "ollama":
            from src.core.ai_client import load_api_key
            if not load_api_key(provider):
                QMessageBox.warning(
                    self, "Thiếu API Key",
                    f"Chưa có API key cho {provider}.\n\n"
                    "Vào ⚙ Cài đặt → tab API Keys để nhập.",
                )
                return

        num_parts = self._effective_num_parts()
        extra_prompt = self.txt_extra_prompt.toPlainText().strip()
        self._project.ai_script_extra_prompt = extra_prompt
        save_project(self._project)

        # Per-clip durations or fixed duration
        clip_durations = None
        selected_clips = enabled_clips[:num_parts]
        if self.chk_auto_duration.isChecked():
            clip_durations = [
                c.duration for c in selected_clips
            ]
            avg_dur = sum(clip_durations) / max(1, len(clip_durations))
            duration_sec = int(avg_dur)
        else:
            duration_sec = self.spn_duration.value()

        self.btn_generate.setEnabled(False)
        self.btn_save_all.setEnabled(False)
        self.btn_to_voiceover.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.txt_log.clear()
        self.txt_log.append("🚀 Đang tạo kịch bản...")

        self._worker = _AIWorker(
            project=self._project,
            provider=provider,
            model=self.cmb_model.currentData() or "",
            style_key=self.cmb_style.currentData() or "summary",
            output_language=self.cmb_language.currentData() or "vi",
            duration_per_part_sec=duration_sec,
            num_parts=num_parts,
            clip_durations=clip_durations,
            extra_prompt=extra_prompt,
        )
        self._worker.status.connect(self._on_log)
        self._worker.finished.connect(self._on_generate_done)
        self._worker.start()

    def _on_log(self, msg: str):
        self.txt_log.append(msg)
        self.txt_log.verticalScrollBar().setValue(
            self.txt_log.verticalScrollBar().maximum()
        )

    def _on_generate_done(self, ok: bool, result):
        self.btn_generate.setEnabled(True)
        self.progress_bar.setVisible(False)

        if not ok:
            QMessageBox.critical(self, "Lỗi AI", str(result))
            self._on_log(f"❌ {result}")
            return

        # result is a dict: {overview, parts: [{index, title, script, estimated_seconds}]}
        self.txt_overview.setPlainText(result.get("overview", ""))
        self._clear_cards()

        parts = result.get("parts", [])
        dur_sec = self.spn_duration.value()
        for part in parts:
            self._add_card(
                part.get("index", 1),
                part.get("title", f"PART {part.get('index', 1)}"),
                part.get("script", ""),
                part.get("estimated_seconds", dur_sec),
            )

        if self._cards:
            self.btn_save_all.setEnabled(True)
            self.btn_to_voiceover.setEnabled(True)

        self._on_log(f"✅ Đã tạo {len(parts)} kịch bản Part.")

    # ─── Save ──────────────────────────────────────────────────────────────────

    def _save_all_to_clips(self):
        if not self._project:
            return
        if not self._cards:
            return

        # Match cards only to enabled clips. Clear their previous scripts first
        # so a partial AI response cannot silently retain stale narration.
        enabled_clips = self._enabled_clips()
        clip_map = {c.index: c for c in enabled_clips}
        for clip in enabled_clips:
            clip.voiceover_script = ""
        saved = 0

        # Results loaded from the project carry the real clip indexes. Fresh
        # AI responses may instead number Parts sequentially (1..N), even when
        # enabled clip indexes have gaps. Only trust index mapping when every
        # card maps unambiguously; otherwise preserve enabled-clip order.
        card_indexes = [card.index for card in self._cards]
        use_index_mapping = (
            len(set(card_indexes)) == len(card_indexes)
            and all(index in clip_map for index in card_indexes)
        )

        for pos, card in enumerate(self._cards):
            script = card.get_script()
            if use_index_mapping:
                clip_map[card.index].voiceover_script = script
                saved += 1
            elif pos < len(enabled_clips):
                enabled_clips[pos].voiceover_script = script
                saved += 1

        # Save only the narration Parts to the voiceover script.
        # The overview is for editing context, not something to read over the video.
        combined = []
        for card in self._cards:
            combined.append(f"=== {card.lbl_title.text()} ===\n{card.get_script()}")
        self._project.voiceover_script = "\n\n".join(combined)
        save_project(self._project)

        self._on_log(f"💾 Đã lưu {saved} kịch bản vào Parts.")
        QMessageBox.information(
            self, "Đã lưu",
            f"✅ Đã lưu kịch bản cho {saved} Parts.\n\n"
            "Bạn có thể vào Tab 7 (Lồng tiếng) để tạo giọng đọc từ kịch bản này.",
        )
        self.scripts_saved.emit()

    def _send_to_voiceover(self):
        """Combine all part scripts and signal the main window to switch tab."""
        if not self._cards:
            return
        combined = []
        for card in self._cards:
            combined.append(f"=== {card.lbl_title.text()} ===\n{card.get_script()}")
        full_script = "\n\n".join(combined)

        # Save to project first
        if self._project:
            self._project.voiceover_script = full_script
            save_project(self._project)

        # Emit — main_window listens and copies script to voiceover tab
        self.scripts_saved.emit()
        QMessageBox.information(
            self, "Đã chuyển sang Tab Lồng tiếng",
            "Kịch bản đã được lưu.\nVào Tab 7 (Lồng tiếng) để tạo giọng đọc.",
        )
