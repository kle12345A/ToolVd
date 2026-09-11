"""Tab 3: AI Cắt cảnh chính — AI-powered key scene extraction from long movies."""

from __future__ import annotations

from copy import copy
import math
import uuid

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QPushButton, QComboBox, QDoubleSpinBox, QSpinBox,
    QGroupBox, QTextEdit, QScrollArea, QSplitter,
    QMessageBox, QProgressBar, QFrame, QCheckBox,
)

from src.core.project_manager import save_project
from src.models.project import Project
from src.models.clip import Clip
from src.utils.file_utils import format_duration
from src.utils.logger import logger

from src.core.ai_client import (
    GEMINI_FREE_MODELS,
    GROQ_FREE_MODELS,
    OPENROUTER_FREE_MODELS,
    OLLAMA_SUGGESTED,
    load_api_key,
    preferred_transcript_path,
)


# ── Worker ────────────────────────────────────────────────────────────────────

class _SceneWorker(QThread):
    status = pyqtSignal(str)
    finished = pyqtSignal(bool, object)   # ok, result_dict | error_str

    def __init__(self, project, provider, model, target_minutes,
                 max_scenes, output_language, use_visual_timeline=True,
                 scene_threshold=0.32, max_detected_scenes=5000):
        super().__init__()
        self.project = project
        self.provider = provider
        self.model = model
        self.target_minutes = target_minutes
        self.max_scenes = max_scenes
        self.output_language = output_language
        self.use_visual_timeline = use_visual_timeline
        self.scene_threshold = scene_threshold
        self.max_detected_scenes = max_detected_scenes

    def run(self):
        try:
            from src.core.ai_scene_analyzer import analyze_key_scenes
            analysis_project = copy(self.project)
            source_transcript = preferred_transcript_path(self.project)
            if source_transcript:
                analysis_project.transcript_file = source_transcript
            result = analyze_key_scenes(
                analysis_project,
                provider=self.provider,
                model=self.model,
                target_minutes=self.target_minutes,
                max_scenes=self.max_scenes,
                output_language=self.output_language,
                use_visual_timeline=self.use_visual_timeline,
                scene_threshold=self.scene_threshold,
                max_detected_scenes=self.max_detected_scenes,
                status_cb=lambda s: self.status.emit(s),
            )
            self.finished.emit(True, result)
        except Exception as e:
            logger.error(f"AI scene analysis failed: {e}")
            self.finished.emit(False, str(e))


# ── Scene card ────────────────────────────────────────────────────────────────

class _SceneCard(QFrame):
    """One scene row with checkbox, time, description."""
    toggled = pyqtSignal()

    def __init__(self, scene: dict, parent=None):
        super().__init__(parent)
        self.scene = scene
        importance = scene.get("importance", "medium")
        border = "#f9e2af" if importance == "high" else "#45475a"
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet(
            f"QFrame {{ background: #1e1e2e; border: 1px solid {border};"
            f" border-radius: 5px; }}"
        )

        rl = QHBoxLayout(self)
        rl.setContentsMargins(8, 6, 8, 6)

        self.chk = QCheckBox()
        self.chk.setChecked(scene.get("enabled", True))
        self.chk.toggled.connect(self.toggled.emit)
        rl.addWidget(self.chk)

        idx = scene.get("index", 0)
        start = scene.get("start_time", 0)
        end = scene.get("end_time", 0)
        dur = scene.get("duration", end - start)

        lbl_idx = QLabel(f"<b style='color:#cba6f7;'>#{idx}</b>")
        lbl_idx.setFixedWidth(30)
        rl.addWidget(lbl_idx)

        lbl_time = QLabel(
            f"{format_duration(start)} → {format_duration(end)}"
        )
        lbl_time.setFixedWidth(160)
        lbl_time.setStyleSheet("color: #89b4fa;")
        rl.addWidget(lbl_time)

        lbl_dur = QLabel(f"{dur:.0f}s")
        lbl_dur.setFixedWidth(40)
        lbl_dur.setStyleSheet("color: #a6e3a1; font-weight: bold;")
        rl.addWidget(lbl_dur)

        desc = scene.get("description", "")
        lbl_desc = QLabel(desc)
        lbl_desc.setWordWrap(True)
        lbl_desc.setStyleSheet("color: #bac2de;")
        rl.addWidget(lbl_desc, 1)

        if importance == "high":
            star = QLabel("⭐")
            star.setToolTip("Cảnh quan trọng")
            rl.addWidget(star)

    def is_checked(self) -> bool:
        return self.chk.isChecked()


# ── Main Tab ──────────────────────────────────────────────────────────────────

class AISceneTab(QWidget):
    scenes_applied = pyqtSignal()   # emitted after clips are created

    def __init__(self, parent=None):
        super().__init__(parent)
        self._project: Project | None = None
        self._worker: _SceneWorker | None = None
        self._cards: list[_SceneCard] = []
        self._result: dict | None = None
        self._setup_ui()

    # ── UI ─────────────────────────────────────────────────────────────────────

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(6)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # ── Left: settings ────────────────────────────────────────────────
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        left_scroll.setStyleSheet("QScrollArea{border:none;}")
        left_scroll.setMinimumWidth(280)
        left_scroll.setMaximumWidth(380)

        lw = QWidget()
        ll = QVBoxLayout(lw)
        ll.setSpacing(8)
        ll.setContentsMargins(4, 4, 8, 4)

        # Project status
        sg = QGroupBox("📊 Trạng thái")
        sfl = QFormLayout(sg)
        self.lbl_project = QLabel("Chưa có project")
        self.lbl_project.setStyleSheet("color: #888;")
        sfl.addRow("Project:", self.lbl_project)
        self.lbl_transcript = QLabel("❌ Chưa có")
        self.lbl_transcript.setStyleSheet("color: #f38ba8;")
        sfl.addRow("Transcript:", self.lbl_transcript)
        self.lbl_video_dur = QLabel("-")
        sfl.addRow("Thời lượng phim:", self.lbl_video_dur)
        ll.addWidget(sg)

        # Target settings
        tg = QGroupBox("🎯 Mục tiêu")
        tfl = QFormLayout(tg)

        self.spn_target = QDoubleSpinBox()
        self.spn_target.setRange(0.1, 60.0)
        self.spn_target.setValue(1.0)      # placeholder — sẽ tính lại khi load project
        self.spn_target.setSuffix(" phút")
        self.spn_target.setDecimals(1)
        self.spn_target.setSingleStep(0.1)
        tfl.addRow("Thời lượng tóm tắt:", self.spn_target)

        self.spn_max_scenes = QSpinBox()
        self.spn_max_scenes.setRange(3, 30)
        self.spn_max_scenes.setValue(12)
        self.spn_max_scenes.setSuffix(" cảnh")
        tfl.addRow("Số cảnh tối đa:", self.spn_max_scenes)

        self.chk_visual_timeline = QCheckBox("Phat hien hang loat phan canh + map loi thoai")
        self.chk_visual_timeline.setChecked(True)
        self.chk_visual_timeline.setToolTip(
            "Dung FFmpeg scene score de cat phim thanh nhieu shot, "
            "sau do map transcript vao dung canh truoc khi gui AI."
        )
        tfl.addRow(self.chk_visual_timeline)

        self.spn_scene_threshold = QDoubleSpinBox()
        self.spn_scene_threshold.setRange(0.05, 0.80)
        self.spn_scene_threshold.setSingleStep(0.05)
        self.spn_scene_threshold.setDecimals(2)
        self.spn_scene_threshold.setValue(0.32)
        self.spn_scene_threshold.setToolTip(
            "Thap hon = phat hien nhieu canh hon; cao hon = chi cat khi thay doi ro."
        )
        tfl.addRow("Do nhay cat canh:", self.spn_scene_threshold)

        self.spn_detect_limit = QSpinBox()
        self.spn_detect_limit.setRange(100, 10000)
        self.spn_detect_limit.setValue(5000)
        self.spn_detect_limit.setSingleStep(100)
        self.spn_detect_limit.setSuffix(" canh")
        tfl.addRow("Gioi han canh tho:", self.spn_detect_limit)

        self.lbl_avg = QLabel()
        self.lbl_avg.setStyleSheet(
            "color: #89b4fa; font-size: 11px; background: #313244;"
            " border-radius: 4px; padding: 4px;"
        )
        self.lbl_avg.setWordWrap(True)
        tfl.addRow(self.lbl_avg)
        self.spn_target.valueChanged.connect(self._update_calc)
        self.spn_max_scenes.valueChanged.connect(self._update_calc)

        ll.addWidget(tg)

        # AI provider
        ag = QGroupBox("🤖 AI Provider")
        afl = QFormLayout(ag)

        self.cmb_provider = QComboBox()
        self.cmb_provider.addItem("Google Gemini (free)", "gemini")
        self.cmb_provider.addItem("Groq (free, nhanh)", "groq")
        self.cmb_provider.addItem("OpenRouter (free)", "openrouter")
        self.cmb_provider.addItem("Ollama (local)", "ollama")
        self.cmb_provider.currentIndexChanged.connect(self._on_provider_changed)
        afl.addRow("Provider:", self.cmb_provider)

        self.cmb_model = QComboBox()
        afl.addRow("Model:", self.cmb_model)

        self.lbl_key = QLabel()
        self.lbl_key.setStyleSheet("color: #888; font-size: 10px;")
        afl.addRow("API Key:", self.lbl_key)

        self.cmb_lang = QComboBox()
        self.cmb_lang.addItem("🇻🇳 Tiếng Việt", "vi")
        self.cmb_lang.addItem("🇺🇸 English", "en")
        afl.addRow("Ngôn ngữ:", self.cmb_lang)

        ll.addWidget(ag)

        # Run button
        self.btn_run = QPushButton("🎬 Phân tích cảnh chính")
        self.btn_run.setFixedHeight(44)
        self.btn_run.setStyleSheet(
            "background: #cba6f7; color: #1e1e2e;"
            " font-size: 15px; font-weight: bold;"
        )
        self.btn_run.clicked.connect(self._run_analysis)
        ll.addWidget(self.btn_run)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        ll.addWidget(self.progress)

        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setMaximumHeight(130)
        self.txt_log.setStyleSheet(
            "background: #111; color: #aaa; font-size: 11px;"
        )
        ll.addWidget(self.txt_log)

        # Apply button
        self.btn_apply = QPushButton("✅ Tạo Parts từ cảnh đã chọn")
        self.btn_apply.setFixedHeight(38)
        self.btn_apply.setStyleSheet(
            "background: #a6e3a1; color: #1e1e2e; font-weight: bold;"
        )
        self.btn_apply.setEnabled(False)
        self.btn_apply.clicked.connect(self._apply_scenes)
        ll.addWidget(self.btn_apply)

        ll.addStretch()
        left_scroll.setWidget(lw)
        splitter.addWidget(left_scroll)

        # ── Right: results ────────────────────────────────────────────────
        right = QWidget()
        right_l = QVBoxLayout(right)
        right_l.setContentsMargins(4, 4, 4, 4)
        right_l.setSpacing(6)

        # Summary
        self.lbl_summary = QLabel(
            "💡 Nhấn \"Phân tích cảnh chính\" để AI tìm những cảnh quan trọng nhất "
            "trong phim dài, tự động cắt thành video tóm tắt."
        )
        self.lbl_summary.setWordWrap(True)
        self.lbl_summary.setStyleSheet(
            "color: #bac2de; background: #313244; padding: 10px;"
            " border-radius: 6px; font-size: 13px;"
        )
        right_l.addWidget(self.lbl_summary)

        # Stats bar
        stats_row = QHBoxLayout()
        self.lbl_stats = QLabel()
        self.lbl_stats.setStyleSheet(
            "color: #a6e3a1; font-weight: bold; font-size: 13px;"
        )
        stats_row.addWidget(self.lbl_stats)
        stats_row.addStretch()

        btn_all = QPushButton("Chọn tất cả")
        btn_all.clicked.connect(lambda: self._check_all(True))
        btn_none = QPushButton("Bỏ chọn tất cả")
        btn_none.clicked.connect(lambda: self._check_all(False))
        stats_row.addWidget(btn_all)
        stats_row.addWidget(btn_none)

        self.lbl_selected = QLabel()
        self.lbl_selected.setStyleSheet("color: #f9e2af; font-weight: bold;")
        stats_row.addWidget(self.lbl_selected)
        right_l.addLayout(stats_row)

        # Scene cards (scrollable)
        cards_scroll = QScrollArea()
        cards_scroll.setWidgetResizable(True)
        cards_scroll.setStyleSheet("QScrollArea{border:none;}")
        self._cards_widget = QWidget()
        self._cards_layout = QVBoxLayout(self._cards_widget)
        self._cards_layout.setSpacing(4)
        self._cards_layout.setContentsMargins(2, 2, 2, 2)
        self._cards_layout.addStretch()
        cards_scroll.setWidget(self._cards_widget)
        right_l.addWidget(cards_scroll, 1)

        splitter.addWidget(right)
        splitter.setSizes([320, 700])
        root.addWidget(splitter, 1)

        # Init
        self._on_provider_changed()
        self._update_calc()

    # ── Public ─────────────────────────────────────────────────────────────────

    def load_project(self, project: Project):
        self._project = project
        self.lbl_project.setText(project.name)
        self.lbl_project.setStyleSheet("color: #a6e3a1; font-weight: bold;")

        # Video duration + auto-suggest target
        dur = float(project.video_metadata.get("duration", 0) or 0)
        if dur:
            self.lbl_video_dur.setText(f"{dur/60:.0f} phút ({dur:.0f}s)")
            video_minutes = dur / 60.0
            max_target = max(0.1, min(60.0, video_minutes))
            self.spn_target.setRange(0.1, max_target)
            # Suggest about 10% of the source, never longer than the source.
            suggested = max(0.1, min(15.0, round(video_minutes * 0.1, 1)))
            suggested = min(suggested, max_target)
            self.spn_target.setValue(suggested)
        else:
            self.lbl_video_dur.setText("-")
            self.spn_target.setRange(0.1, 60.0)

        # Transcript
        tf = preferred_transcript_path(project)
        if tf:
            from src.core.ai_client import load_transcript_text
            txt = load_transcript_text(project)
            self.lbl_transcript.setText(f"✅ {len(txt):,} ký tự")
            self.lbl_transcript.setStyleSheet("color: #a6e3a1;")
        else:
            self.lbl_transcript.setText("❌ Chưa có — chạy Tab 2 (Phiên âm) trước")
            self.lbl_transcript.setStyleSheet("color: #f38ba8;")

        self._update_calc()

    # ── Internal ───────────────────────────────────────────────────────────────

    def _on_provider_changed(self, _=None):
        prov = self.cmb_provider.currentData() or "gemini"
        self.cmb_model.clear()
        models = {
            "gemini": GEMINI_FREE_MODELS,
            "groq": GROQ_FREE_MODELS,
            "openrouter": OPENROUTER_FREE_MODELS,
            "ollama": OLLAMA_SUGGESTED,
        }.get(prov, [])
        for mid, label in models:
            self.cmb_model.addItem(label, mid)
        if prov == "ollama":
            self.lbl_key.setText("✅ Không cần API key")
            self.lbl_key.setStyleSheet("color: #a6e3a1; font-size: 10px;")
        else:
            key = load_api_key(prov)
            if key:
                self.lbl_key.setText("✅ Đã có")
                self.lbl_key.setStyleSheet("color: #a6e3a1; font-size: 10px;")
            else:
                self.lbl_key.setText("❌ Chưa có — vào ⚙ Cài đặt")
                self.lbl_key.setStyleSheet("color: #f38ba8; font-size: 10px;")

    def _update_calc(self):
        target = self._effective_target_minutes()
        mx = self.spn_max_scenes.value()
        avg = target * 60 / mx
        vid_dur = 0
        if self._project:
            vid_dur = self._project.video_metadata.get("duration", 0)
        ratio = f" ({target/vid_dur*60*100:.0f}% phim)" if vid_dur > 0 else ""
        self.lbl_avg.setText(
            f"~{mx} cảnh × {avg:.0f}s trung bình = {target:.1f} phút{ratio}\n"
            f"AI sẽ chọn cảnh quan trọng nhất, bỏ qua phần phụ"
        )

    def _effective_target_minutes(self) -> float:
        target = max(0.1, float(self.spn_target.value()))
        if not self._project:
            return target
        duration = float(self._project.video_metadata.get("duration", 0) or 0)
        if duration > 0:
            target = min(target, duration / 60.0)
        return max(0.0, target)

    def _log(self, msg: str):
        self.txt_log.append(msg)
        sb = self.txt_log.verticalScrollBar()
        sb.setValue(sb.maximum())

    # ── Run ────────────────────────────────────────────────────────────────────

    def _run_analysis(self):
        if not self._project:
            QMessageBox.warning(self, "Chưa có project",
                                "Vui lòng mở project trước.")
            return

        tf = preferred_transcript_path(self._project)
        if not tf:
            QMessageBox.warning(
                self, "Chưa có transcript",
                "Bạn cần phiên âm video trước (Tab 2).\n\n"
                "AI phân tích nội dung phiên âm để tìm cảnh chính.",
            )
            return

        prov = self.cmb_provider.currentData() or "gemini"
        if prov != "ollama" and not load_api_key(prov):
            QMessageBox.warning(
                self, "Thiếu API Key",
                f"Chưa có API key cho {prov}.\n"
                "Vào ⚙ Cài đặt → API Keys.",
            )
            return

        self.btn_run.setEnabled(False)
        self.btn_apply.setEnabled(False)
        self.progress.setVisible(True)
        self.txt_log.clear()
        self._log("🚀 Bắt đầu phân tích cảnh chính...")

        self._worker = _SceneWorker(
            project=self._project,
            provider=prov,
            model=self.cmb_model.currentData() or "",
            target_minutes=self._effective_target_minutes(),
            max_scenes=self.spn_max_scenes.value(),
            output_language=self.cmb_lang.currentData() or "vi",
            use_visual_timeline=self.chk_visual_timeline.isChecked(),
            scene_threshold=self.spn_scene_threshold.value(),
            max_detected_scenes=self.spn_detect_limit.value(),
        )
        self._worker.status.connect(self._log)
        self._worker.finished.connect(self._on_done)
        self._worker.start()

    def _on_done(self, ok: bool, result):
        self.btn_run.setEnabled(True)
        self.progress.setVisible(False)

        if not ok:
            QMessageBox.critical(self, "Lỗi AI", str(result))
            self._log(f"❌ {result}")
            return

        self._result = result
        self._show_results(result)
        self.btn_apply.setEnabled(True)
        self._log(
            f"✅ Tìm thấy {len(result.get('scenes', []))} cảnh. "
            "Chọn/bỏ chọn rồi nhấn \"Tạo Parts\"."
        )

    # ── Results display ────────────────────────────────────────────────────────

    def _clear_cards(self):
        self._cards = []
        while self._cards_layout.count() > 1:
            item = self._cards_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _show_results(self, result: dict):
        self._clear_cards()

        summary = result.get("recap_timeline") or result.get("summary", "")
        if summary:
            self.lbl_summary.setText(f"📋 {summary}")

        scenes = result.get("scenes", [])
        total = sum(s.get("duration", 0) for s in scenes)
        self.lbl_stats.setText(
            f"🎬 {len(scenes)} cảnh | ⏱ Tổng: {total/60:.1f} phút ({total:.0f}s)"
        )

        for scene in scenes:
            card = _SceneCard(scene)
            card.toggled.connect(self._update_selected)
            pos = max(0, self._cards_layout.count() - 1)
            self._cards_layout.insertWidget(pos, card)
            self._cards.append(card)

        self._update_selected()

    def _check_all(self, state: bool):
        for card in self._cards:
            card.chk.setChecked(state)

    def _update_selected(self):
        scenes = self._result.get("scenes", []) if self._result else []
        sel = []
        for i, card in enumerate(self._cards):
            if card.is_checked() and i < len(scenes):
                sel.append(scenes[i])
        total = sum(s.get("duration", 0) for s in sel)
        self.lbl_selected.setText(
            f"Đã chọn: {len(sel)} cảnh | {total/60:.1f} phút"
        )

    # ── Apply scenes → clips ──────────────────────────────────────────────────

    def _apply_scenes(self):
        if not self._project or not self._result:
            return

        scenes = self._result.get("scenes", [])
        selected = []
        rejected = 0
        for i, card in enumerate(self._cards):
            if card.is_checked() and i < len(scenes):
                scene = dict(scenes[i])
                try:
                    start = float(scene.get("start_time", 0))
                    end = float(scene.get("end_time", 0))
                except (TypeError, ValueError):
                    rejected += 1
                    continue
                if (
                    not math.isfinite(start)
                    or not math.isfinite(end)
                    or start < 0
                    or end <= start
                ):
                    rejected += 1
                    continue
                scene["start_time"] = start
                scene["end_time"] = end
                scene["duration"] = end - start
                selected.append(scene)

        selected.sort(key=lambda item: (
            item["start_time"],
            item["end_time"],
        ))

        if not selected:
            QMessageBox.warning(self, "Chưa chọn cảnh",
                                "Hãy chọn ít nhất 1 cảnh có khoảng thời gian hợp lệ.")
            return
        if rejected:
            self._log(
                f"⚠️ Đã bỏ qua {rejected} cảnh có khoảng thời gian rỗng/không hợp lệ."
            )

        # Ask about existing clips
        if self._project.clips:
            resp = QMessageBox.question(
                self,
                "Clips hiện tại",
                f"Project đang có {len(self._project.clips)} cảnh.\n\n"
                "• Yes — Thay thế toàn bộ bằng cảnh AI\n"
                "• No — Thêm cảnh AI vào cuối",
                QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.No
                | QMessageBox.StandardButton.Cancel,
            )
            if resp == QMessageBox.StandardButton.Cancel:
                return
            if resp == QMessageBox.StandardButton.Yes:
                self._project.clips = []

        start_idx = len(self._project.clips) + 1
        new_clips = []
        for i, scene in enumerate(selected):
            clip = Clip(
                id=str(uuid.uuid4())[:8],
                index=start_idx + i,
                start_time=scene["start_time"],
                end_time=scene["end_time"],
                enabled=True,
                part_text=f"CẢNH {start_idx + i:02d}",
            )
            new_clips.append(clip)

        self._project.clips.extend(new_clips)

        summary = self._result.get("recap_timeline") or self._result.get("summary", "")
        if summary:
            self._project.voiceover_script = summary

        save_project(self._project)
        self.scenes_applied.emit()

        total_dur = sum(c.duration for c in new_clips)
        QMessageBox.information(
            self,
            "✅ Đã tạo phân cảnh",
            f"Đã tạo {len(new_clips)} cảnh chính.\n"
            f"Tổng: {total_dur/60:.1f} phút ({total_dur:.0f}s)\n\n"
            "Chuyển sang Tab 4 để chỉnh sửa,\n"
            "hoặc Tab 5 để xuất video.",
        )
