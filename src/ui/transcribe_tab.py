"""Phiên âm nguồn hoặc narration bằng faster-whisper."""

import json
import threading
from pathlib import Path

from PyQt6.QtCore import Qt, QThread, QTimer, QUrl, pyqtSignal
from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QPushButton, QComboBox, QCheckBox, QSlider,
    QGroupBox, QTextEdit, QProgressBar, QSplitter,
    QMessageBox, QFileDialog, QSizePolicy,
)

from src.core.transcriber import (
    WHISPER_MODELS, is_faster_whisper_available,
    is_cuda_available, transcribe_video, load_transcript,
)
from src.core.subtitle_builder import (
    build_subtitle_for_clip,
    build_subtitle_for_full_video,
)
from src.core.dependency_manager import ffmpeg_path
from src.core.project_manager import save_project
from src.core.video_manager import get_video_metadata
from src.models.project import Project
from src.utils.logger import logger


_LANGUAGES = [
    ("auto", "Tự động phát hiện"),
    ("vi",   "Tiếng Việt"),
    ("en",   "English"),
    ("zh",   "中文"),
    ("ja",   "日本語"),
    ("ko",   "한국어"),
    ("es",   "Español"),
    ("fr",   "Français"),
    ("de",   "Deutsch"),
    ("th",   "ภาษาไทย"),
]

_SPEED_OPTIONS = [
    (1, "🚀 Nhanh nhất (beam=1, greedy)"),
    (3, "⚖️ Cân bằng (beam=3)"),
    (5, "🎯 Chính xác nhất (beam=5, mặc định)"),
]


# ─── Workers ──────────────────────────────────────────────────────────────────

class _TranscribeWorker(QThread):
    log = pyqtSignal(str)
    finished = pyqtSignal(object)   # transcript dict or None

    def __init__(self, video_path, output_dir, model, language, use_gpu, beam_size):
        super().__init__()
        self.video_path = video_path
        self.output_dir = output_dir
        self.model = model
        self.language = language
        self.use_gpu = use_gpu
        self.beam_size = beam_size
        self._cancel = threading.Event()

    def cancel(self):
        self._cancel.set()

    def run(self):
        result = transcribe_video(
            self.video_path,
            self.output_dir,
            model_size=self.model,
            language=self.language or None,
            use_gpu=self.use_gpu,
            ffmpeg_bin=ffmpeg_path(),
            progress_cb=lambda s: self.log.emit(s),
            cancel_event=self._cancel,
            beam_size=self.beam_size,
        )
        self.finished.emit(result)


class _MultiPartTranscribeWorker(QThread):
    """Transcribe only the selected clips, one by one.

    For each clip:
      1. Extract its audio segment from the source video (FFmpeg -ss/-to)
      2. Transcribe that segment with faster-whisper
      3. Shift timestamps back to absolute positions in the source video
      4. Merge into a combined transcript
    """
    log = pyqtSignal(str)
    finished = pyqtSignal(object)   # merged transcript dict or None

    def __init__(self, source_video, clips, output_dir,
                 model, language, use_gpu, beam_size):
        super().__init__()
        self.source_video = source_video
        self.clips = clips          # list[Clip] — only enabled clips
        self.output_dir = output_dir
        self.model = model
        self.language = language
        self.use_gpu = use_gpu
        self.beam_size = beam_size
        self._cancel = threading.Event()

    def cancel(self):
        self._cancel.set()

    def run(self):
        import subprocess
        ff = ffmpeg_path()
        if not ff:
            self.log.emit("❌ FFmpeg chưa được tải.")
            self.finished.emit(None)
            return

        all_segments = []
        detected_lang = ""
        seg_dir = Path(self.output_dir)
        seg_dir.mkdir(parents=True, exist_ok=True)

        for i, clip in enumerate(self.clips):
            if self._cancel.is_set():
                self.log.emit("⛔ Đã hủy.")
                self.finished.emit(None)
                return

            dur = clip.end_time - clip.start_time
            self.log.emit(
                f"🎬 [{i+1}/{len(self.clips)}] Đang trích xuất "
                f"{clip.part_text} ({dur:.0f}s)..."
            )

            # A batch Part may belong to its own source file.  Falling back to
            # the project source preserves the regular single-video workflow.
            clip_source = (
                getattr(clip, "source_video", "") or self.source_video
            )
            if not clip_source or not Path(clip_source).exists():
                self.log.emit(
                    f"⚠️ Không tìm thấy video nguồn của {clip.part_text}: "
                    f"{clip_source or '(trống)'}"
                )
                continue

            # 1. Extract segment
            seg_path = str(seg_dir / f"_seg_part{clip.index:02d}.mp4")
            try:
                Path(seg_path).unlink(missing_ok=True)
            except OSError:
                pass
            cmd = [
                ff, "-y",
                "-ss", str(clip.start_time),
                "-to", str(clip.end_time),
                "-i", clip_source,
                "-c", "copy",
                seg_path,
            ]
            try:
                proc = subprocess.run(cmd, capture_output=True, timeout=120)
            except Exception as e:
                self.log.emit(f"⚠️ FFmpeg lỗi: {e}")
                continue

            if proc.returncode != 0 or not Path(seg_path).exists():
                error = proc.stderr.decode(errors="replace")[-300:]
                self.log.emit(
                    f"⚠️ Không trích xuất được {clip.part_text}: {error}"
                )
                continue

            # 2. Transcribe segment
            self.log.emit(
                f"🎙 [{i+1}/{len(self.clips)}] Phiên âm {clip.part_text}..."
            )
            result = transcribe_video(
                seg_path,
                str(seg_dir),
                model_size=self.model,
                language=self.language or None,
                use_gpu=self.use_gpu,
                ffmpeg_bin=ff,
                progress_cb=lambda s: self.log.emit(f"  {s}"),
                cancel_event=self._cancel,
                beam_size=self.beam_size,
            )
            if not result:
                self.log.emit(f"⚠️ Phiên âm {clip.part_text} thất bại.")
                continue

            if not detected_lang:
                detected_lang = result.get("language", "")

            # 3. Shift timestamps to absolute positions
            offset = clip.start_time
            for seg in result.get("segments", []):
                seg["start"] += offset
                seg["end"] += offset
                seg["source_clip_id"] = getattr(clip, "id", "")
                for w in seg.get("words", []):
                    w["start"] += offset
                    w["end"] += offset
                all_segments.append(seg)

            n_segs = len(result.get("segments", []))
            self.log.emit(
                f"✅ {clip.part_text}: {n_segs} đoạn phiên âm."
            )

            # Cleanup temp segment
            try:
                Path(seg_path).unlink(missing_ok=True)
            except Exception:
                pass

        if not all_segments:
            self.log.emit("❌ Không có đoạn nào được phiên âm.")
            self.finished.emit(None)
            return

        # Sort segments by time
        all_segments.sort(key=lambda s: s["start"])

        # Build merged transcript
        merged = {
            "language": detected_lang,
            "segments": all_segments,
            "text": " ".join(s.get("text", "") for s in all_segments),
        }

        # Save merged transcript
        out_path = seg_dir / "transcript.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)

        self.log.emit(
            f"✅ Phiên âm xong {len(self.clips)} Part "
            f"→ {len(all_segments)} đoạn tổng cộng."
        )
        self.finished.emit(merged)


class _SubtitleGenWorker(QThread):
    log = pyqtSignal(str)
    finished = pyqtSignal(int, int)   # generated, skipped

    def __init__(self, project, transcript, style, cfg, mode="source"):
        super().__init__()
        self.project = project
        self.transcript = transcript
        self.style = style
        self.cfg = cfg
        self.mode = mode

    def run(self):
        sub_dir = str(Path(self.project.output_dir) / "subtitles")
        gen = skip = 0

        if self.mode == "narration":
            review_video = getattr(self.project, "review_voice_video", "") or ""
            duration = None
            if review_video and Path(review_video).exists():
                metadata = get_video_metadata(review_video)
                if metadata:
                    duration = float(metadata.get("duration", 0) or 0) or None
            if duration is None:
                try:
                    duration = (
                        float(self.transcript.get("duration", 0) or 0) or None
                    )
                except (TypeError, ValueError):
                    duration = None

            path = build_subtitle_for_full_video(
                self.transcript,
                output_dir=sub_dir,
                style=self.style,
                width=self.cfg.width,
                height=self.cfg.height,
                fontsize=self.cfg.subtitle_fontsize,
                text_color=self.cfg.subtitle_color,
                highlight_color=self.cfg.subtitle_highlight_color,
                position=self.cfg.subtitle_position,
                duration=duration,
                filename_stem="review_subtitle",
            )
            if not path:
                self.log.emit(
                    "⚠️ Transcript narration không có câu hợp lệ để tạo subtitle."
                )
                self.finished.emit(0, 1)
                return

            self.project.narration_subtitle_file = path
            self.cfg.global_subtitle_file = path
            self.cfg.subtitle_enabled = True
            self.cfg.subtitle_style = self.style
            save_project(self.project)
            self.log.emit(f"✅ Subtitle toàn video: {Path(path).name}")
            self.finished.emit(1, 0)
            return

        # Only generate subtitles for ENABLED clips (checked in Tab 4)
        enabled_clips = [c for c in self.project.clips if c.enabled]
        if not enabled_clips:
            self.log.emit("⚠️ Không có Part nào được chọn (✓) ở Tab 4.")
            self.finished.emit(0, 0)
            return

        for clip in enabled_clips:
            path = build_subtitle_for_clip(
                self.transcript,
                clip_start=clip.start_time,
                clip_end=clip.end_time,
                output_dir=sub_dir,
                clip_index=clip.index,
                style=self.style,
                width=self.cfg.width,
                height=self.cfg.height,
                fontsize=self.cfg.subtitle_fontsize,
                text_color=self.cfg.subtitle_color,
                highlight_color=self.cfg.subtitle_highlight_color,
                position=self.cfg.subtitle_position,
            )
            if path:
                clip.subtitle_file = path
                gen += 1
                self.log.emit(f"✅ PART {clip.index}: {Path(path).name}")
            else:
                skip += 1
                self.log.emit(f"⚠️ PART {clip.index}: không có lời thoại")
        save_project(self.project)
        self.finished.emit(gen, skip)


# ─── Main tab ─────────────────────────────────────────────────────────────────

class TranscribeTab(QWidget):
    """Transcription UI for either the immutable source or final narration."""

    transcript_ready = pyqtSignal(dict)
    source_transcript_ready = pyqtSignal(dict)
    narration_transcript_ready = pyqtSignal(dict)
    narration_subtitle_ready = pyqtSignal(str)
    # Compatibility signal. Carries "plain", "karaoke", or "word".
    subtitles_generated = pyqtSignal(str)

    def __init__(self, parent=None, mode: str = "source"):
        # Also accept TranscribeTab("narration") while preserving the original
        # positional-parent constructor used by PyQt callers.
        if isinstance(parent, str) and parent in {"source", "narration"}:
            mode, parent = parent, None
        super().__init__(parent)
        if mode not in {"source", "narration"}:
            raise ValueError("TranscribeTab mode must be 'source' or 'narration'")
        self.mode = mode
        self._project: Project | None = None
        self._transcript: dict | None = None
        self._worker: _TranscribeWorker | None = None
        self._sub_worker: _SubtitleGenWorker | None = None
        self._setup_ui()
        self._setup_player()

    # ─── UI setup ─────────────────────────────────────────────────────────────

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # Workflow hint
        workflow_text = (
            "💡 Phiên âm video nguồn để AI hiểu nội dung và chọn phân cảnh."
            if self.mode == "source"
            else
            "💡 Phiên âm video đã ghép giọng, sau đó tạo một subtitle "
            "cho toàn bộ timeline review."
        )
        self.lbl_workflow = QLabel(workflow_text)
        self.lbl_workflow.setStyleSheet(
            "color: #89b4fa; font-size: 11px; padding: 4px 8px;"
            "background: #1e1e2e; border-radius: 4px;"
        )
        self.lbl_workflow.setWordWrap(True)
        layout.addWidget(self.lbl_workflow)

        # Dependency status
        self.lbl_dep = QLabel()
        self._update_dep_label()
        layout.addWidget(self.lbl_dep)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # ── Left: Settings + Controls ──────────────────────────────────────
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left.setMinimumWidth(300)
        left.setMaximumWidth(420)

        cfg_group = QGroupBox("Cấu hình phiên âm")
        cfg_form = QFormLayout(cfg_group)

        self.cmb_model = QComboBox()
        for m in WHISPER_MODELS:
            self.cmb_model.addItem(m)
        self.cmb_model.setCurrentText("base")
        self.cmb_model.currentTextChanged.connect(self._update_model_hint)
        cfg_form.addRow("Model Whisper:", self.cmb_model)

        self.cmb_lang = QComboBox()
        for code, label in _LANGUAGES:
            self.cmb_lang.addItem(label, code)
        cfg_form.addRow("Ngôn ngữ:", self.cmb_lang)

        self.cmb_speed = QComboBox()
        for beam, label in _SPEED_OPTIONS:
            self.cmb_speed.addItem(label, beam)
        self.cmb_speed.setCurrentIndex(2)   # beam=5 default
        cfg_form.addRow("Tốc độ:", self.cmb_speed)

        self.chk_gpu = QCheckBox("Dùng GPU (CUDA) nếu có")
        self.chk_gpu.setChecked(is_cuda_available())
        cfg_form.addRow(self.chk_gpu)

        # Transcription scope: whole video or selected part(s)
        self.cmb_scope = QComboBox()
        scope_label = (
            "🎤 Toàn bộ video đã ghép giọng"
            if self.mode == "narration"
            else "📹 Toàn bộ video nguồn"
        )
        self.cmb_scope.addItem(scope_label, "full")
        self.cmb_scope.setEnabled(self.mode == "source")
        # Parts will be populated when project loads
        cfg_form.addRow("Phạm vi:", self.cmb_scope)

        left_layout.addWidget(cfg_group)

        # Model hint
        self.lbl_model_hint = QLabel()
        self.lbl_model_hint.setWordWrap(True)
        self.lbl_model_hint.setStyleSheet("color: #888; font-size: 11px;")
        self._update_model_hint(self.cmb_model.currentText())
        left_layout.addWidget(self.lbl_model_hint)

        # Transcribe / Cancel buttons
        btn_row = QHBoxLayout()
        transcribe_label = (
            "🎙 Phiên âm giọng đọc"
            if self.mode == "narration"
            else "🎙 Phiên âm video nguồn"
        )
        self.btn_transcribe = QPushButton(transcribe_label)
        self.btn_transcribe.setFixedHeight(40)
        self.btn_transcribe.setStyleSheet(
            "background: #313244; color: #cba6f7; font-weight: bold;"
        )
        self.btn_transcribe.clicked.connect(self._start_transcribe)
        btn_row.addWidget(self.btn_transcribe)

        self.btn_cancel = QPushButton("⛔ Hủy")
        self.btn_cancel.setFixedHeight(40)
        self.btn_cancel.setFixedWidth(90)
        self.btn_cancel.setStyleSheet("background: #45475a; color: #f38ba8;")
        self.btn_cancel.setVisible(False)
        self.btn_cancel.clicked.connect(self._cancel_transcribe)
        btn_row.addWidget(self.btn_cancel)
        left_layout.addLayout(btn_row)

        # Load existing
        self.btn_load = QPushButton("📂 Tải transcript có sẵn (.json)")
        self.btn_load.clicked.connect(self._load_existing)
        left_layout.addWidget(self.btn_load)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(False)
        left_layout.addWidget(self.progress_bar)

        # Subtitle generation
        self.subtitle_group = QGroupBox("Tạo subtitle toàn video review")
        sub_layout = QVBoxLayout(self.subtitle_group)

        style_row = QHBoxLayout()
        style_row.addWidget(QLabel("Style:"))
        self.cmb_sub_style = QComboBox()
        self.cmb_sub_style.addItem("Plain (chữ trắng viền đen)", "plain")
        self.cmb_sub_style.addItem("Karaoke (highlight từng từ)", "karaoke")
        self.cmb_sub_style.addItem("Word-by-word (hiện từng chữ)", "word")
        style_row.addWidget(self.cmb_sub_style, 1)
        sub_layout.addLayout(style_row)

        self.btn_gen_subs = QPushButton("⚡ Tạo 1 subtitle cho toàn bộ video")
        self.btn_gen_subs.setFixedHeight(36)
        self.btn_gen_subs.setEnabled(False)
        self.btn_gen_subs.clicked.connect(self._generate_subtitles)
        sub_layout.addWidget(self.btn_gen_subs)

        self.lbl_sub_status = QLabel("")
        self.lbl_sub_status.setStyleSheet("color: #a6e3a1;")
        sub_layout.addWidget(self.lbl_sub_status)
        self.subtitle_group.setVisible(self.mode == "narration")
        left_layout.addWidget(self.subtitle_group)
        left_layout.addStretch()

        splitter.addWidget(left)

        # ── Right: Preview player + Log + Transcript ───────────────────────
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)

        # ── Embedded video player ──────────────────────────────────────────
        player_title = (
            "🎬 Video đã ghép giọng và subtitle đồng bộ"
            if self.mode == "narration"
            else "🎬 Xem trước transcript nguồn"
        )
        self.player_group = QGroupBox(player_title)
        player_vbox = QVBoxLayout(self.player_group)
        player_vbox.setSpacing(4)

        self._video_widget = QVideoWidget()
        self._video_widget.setMinimumHeight(180)
        self._video_widget.setMaximumHeight(240)
        self._video_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        player_vbox.addWidget(self._video_widget)

        # Playback controls
        ctrl_row = QHBoxLayout()
        self.btn_play = QPushButton("▶")
        self.btn_play.setFixedSize(34, 28)
        self.btn_play.setEnabled(False)
        self.btn_play.clicked.connect(self._toggle_play)

        self.sld_pos = QSlider(Qt.Orientation.Horizontal)
        self.sld_pos.setEnabled(False)
        self.sld_pos.sliderMoved.connect(self._seek)

        self.lbl_time = QLabel("0:00 / 0:00")
        self.lbl_time.setFixedWidth(88)
        self.lbl_time.setStyleSheet("font-size: 11px; color: #888;")

        ctrl_row.addWidget(self.btn_play)
        ctrl_row.addWidget(self.sld_pos, 1)
        ctrl_row.addWidget(self.lbl_time)
        player_vbox.addLayout(ctrl_row)

        # Current subtitle display (live sync)
        self.lbl_cur_sub = QLabel("")
        self.lbl_cur_sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_cur_sub.setWordWrap(True)
        self.lbl_cur_sub.setStyleSheet(
            "background:#111; color:white; font-size:14px; font-weight:bold;"
            "padding:6px 10px; border-radius:4px; min-height:38px;"
        )
        player_vbox.addWidget(self.lbl_cur_sub)

        right_layout.addWidget(self.player_group)

        # Log
        right_layout.addWidget(QLabel("Log:"))
        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setStyleSheet("background: #111; color: #a6e3a1; font-size: 11px;")
        self.txt_log.setMaximumHeight(140)
        right_layout.addWidget(self.txt_log)

        # Transcript preview
        right_layout.addWidget(QLabel("Nội dung phiên âm (chỉ xem):"))
        self.txt_transcript = QTextEdit()
        self.txt_transcript.setReadOnly(True)
        self.txt_transcript.setPlaceholderText(
            "Transcript sẽ hiển thị ở đây sau khi phiên âm xong..."
        )
        self.txt_transcript.setStyleSheet(
            "background: #181825; color: #cdd6f4; font-size: 12px;"
        )
        right_layout.addWidget(self.txt_transcript, 1)

        splitter.addWidget(right)
        splitter.setSizes([340, 660])
        layout.addWidget(splitter, 1)

    def _setup_player(self):
        """Initialize QMediaPlayer for in-app subtitle preview."""
        self._media_player = QMediaPlayer()
        self._audio_output = QAudioOutput()
        self._audio_output.setVolume(0.9)
        self._media_player.setAudioOutput(self._audio_output)
        self._media_player.setVideoOutput(self._video_widget)
        self._media_player.durationChanged.connect(self._on_duration_changed)
        self._media_player.positionChanged.connect(self._on_position_changed)
        self._media_player.playbackStateChanged.connect(self._on_playback_state)

    # ─── Public API ────────────────────────────────────────────────────────────

    def load_project(self, project: Project):
        self._project = project
        self._update_dep_label()
        self._transcript = None
        self.txt_log.clear()
        self.txt_transcript.clear()
        self.lbl_cur_sub.setText("")
        self.btn_gen_subs.setEnabled(False)
        self.lbl_sub_status.setText("")
        self.btn_transcribe.setEnabled(True)
        self.btn_play.setEnabled(False)
        self.sld_pos.setEnabled(False)
        self._media_player.stop()
        self._media_player.setSource(QUrl())

        # Populate transcription scope combo
        self.cmb_scope.blockSignals(True)
        self.cmb_scope.clear()
        if self.mode == "narration":
            self.cmb_scope.addItem("🎤 Toàn bộ video đã ghép giọng", "full")
            self.cmb_scope.setEnabled(False)
        else:
            self.cmb_scope.setEnabled(True)
            enabled = [c for c in project.clips if c.enabled] if project.clips else []
            if enabled:
                total_dur = sum(c.end_time - c.start_time for c in enabled)
                self.cmb_scope.addItem(
                    f"✅ Chỉ {len(enabled)} Part đã chọn ({total_dur:.0f}s)",
                    "selected",
                )
            self.cmb_scope.addItem("📹 Toàn bộ video nguồn", "full")
            if project.clips:
                for clip in project.clips:
                    dur = clip.end_time - clip.start_time
                    mark = "✓" if clip.enabled else "✗"
                    self.cmb_scope.addItem(
                        f"{mark} {clip.part_text} ({dur:.0f}s)",
                        f"part_{clip.index}",
                    )
        self.cmb_scope.blockSignals(False)

        video_to_load = self._preview_video_path()
        if video_to_load and Path(video_to_load).exists():
            self._media_player.setSource(QUrl.fromLocalFile(video_to_load))
            self.btn_play.setEnabled(True)
            self.sld_pos.setEnabled(True)
            if self.mode == "narration":
                self.txt_log.append(
                    f"🎤 Video narration: {Path(video_to_load).name}"
                )
            else:
                self.txt_log.append(
                    f"📹 Video nguồn: {Path(video_to_load).name}"
                )
        elif self.mode == "source":
            missing_label = (
                "video nguồn"
            )
            self.txt_log.append(f"❌ Chưa có {missing_label} hợp lệ.")
        else:
            self.txt_log.append(
                "⚠️ Chưa có review_voice_video để xem trước."
            )

        transcription_input = self._transcription_input_path()
        if not transcription_input or not Path(transcription_input).exists():
            self.btn_transcribe.setEnabled(False)
            if self.mode == "narration":
                self.txt_log.append(
                    "❌ Chưa có voiceover_audio hoặc review_voice_video "
                    "để phiên âm narration."
                )

        # Auto-load only the transcript artifact for this mode.
        transcript_path = self._stored_transcript_path()
        if transcript_path and Path(transcript_path).exists():
            t = load_transcript(transcript_path)
            if t:
                self._set_transcript(t)
                self.txt_log.append(f"📋 Đã tải transcript: {transcript_path}")

        if self.mode == "narration":
            subtitle_path = (
                getattr(project, "narration_subtitle_file", "") or ""
            )
            if subtitle_path and Path(subtitle_path).exists():
                self.lbl_sub_status.setText(
                    f"✅ Subtitle toàn video: {Path(subtitle_path).name}"
                )

    def _preview_video_path(self) -> str:
        if not self._project:
            return ""
        if self.mode == "narration":
            return getattr(self._project, "review_voice_video", "") or ""
        return getattr(self._project, "source_video", "") or ""

    def _transcription_input_path(self) -> str:
        if not self._project:
            return ""
        if self.mode == "narration":
            narration_audio = (
                getattr(self._project, "voiceover_audio", "") or ""
            )
            if narration_audio and Path(narration_audio).exists():
                return narration_audio
            return getattr(self._project, "review_voice_video", "") or ""
        return getattr(self._project, "source_video", "") or ""

    def _transcript_output_dir(self) -> Path:
        if not self._project:
            return Path()
        branch = "narration" if self.mode == "narration" else "source"
        return Path(self._project.output_dir) / "audio" / branch

    def _canonical_transcript_path(self) -> str:
        return str(self._transcript_output_dir() / "transcript.json")

    def _stored_transcript_path(self) -> str:
        if not self._project:
            return ""
        if self.mode == "narration":
            return (
                getattr(self._project, "narration_transcript_file", "") or ""
            )
        source_path = (
            getattr(self._project, "source_transcript_file", "") or ""
        )
        if source_path:
            return source_path
        # ``transcript_file`` is a legacy source alias.  Once a narration
        # transcript exists, never guess that the legacy path is still source.
        if getattr(self._project, "narration_transcript_file", ""):
            return ""
        return getattr(self._project, "transcript_file", "") or ""

    def _persist_transcript(self, transcript: dict) -> str:
        """Save this mode's transcript without overwriting the other mode."""
        if not self._project:
            return ""
        output_dir = self._transcript_output_dir()
        output_dir.mkdir(parents=True, exist_ok=True)
        transcript_path = str(output_dir / "transcript.json")
        with open(transcript_path, "w", encoding="utf-8") as handle:
            json.dump(transcript, handle, ensure_ascii=False, indent=2)

        if self.mode == "narration":
            self._project.narration_transcript_file = transcript_path
        else:
            self._project.source_transcript_file = transcript_path
            # Compatibility alias consumed by the existing AI modules.
            self._project.transcript_file = transcript_path
            self._project.transcript_language = transcript.get("language", "")
        save_project(self._project)
        return transcript_path

    # ─── Internal ──────────────────────────────────────────────────────────────

    def _update_dep_label(self):
        if is_faster_whisper_available():
            self.lbl_dep.setText("✅ faster-whisper sẵn sàng")
            self.lbl_dep.setStyleSheet("color: #a6e3a1;")
        else:
            self.lbl_dep.setText(
                "❌ faster-whisper chưa cài. Chạy: pip install faster-whisper"
            )
            self.lbl_dep.setStyleSheet("color: #f38ba8;")

    def _update_model_hint(self, model: str):
        hints = {
            "tiny":     "~75MB  | Nhanh nhất, kém chính xác",
            "base":     "~145MB | Cân bằng tốt cho CPU",
            "small":    "~465MB | Khuyến nghị cho tiếng Việt",
            "medium":   "~1.5GB | Chính xác cao, cần RAM nhiều",
            "large-v2": "~3GB   | Rất chính xác, nên dùng GPU",
            "large-v3": "~3GB   | Tốt nhất, cần GPU mạnh",
        }
        self.lbl_model_hint.setText(hints.get(model, ""))

    # ── Transcription ─────────────────────────────────────────────────────────

    def _start_transcribe(self):
        if not self._project:
            QMessageBox.warning(self, "Chưa có project", "Vui lòng tạo project trước.")
            return
        video_path = self._transcription_input_path()
        if not video_path or not Path(video_path).exists():
            label = (
                "voiceover_audio hoặc review_voice_video"
                if self.mode == "narration"
                else "video nguồn"
            )
            QMessageBox.warning(
                self,
                "Thiếu video",
                f"Không tìm thấy {label} để phiên âm.",
            )
            return
        if not is_faster_whisper_available():
            QMessageBox.warning(
                self, "Thiếu thư viện",
                "faster-whisper chưa được cài.\nChạy: pip install faster-whisper"
            )
            return

        lang_code = self.cmb_lang.currentData()
        if lang_code == "auto":
            lang_code = ""
        beam_size = self.cmb_speed.currentData()
        model = self.cmb_model.currentText()
        use_gpu = self.chk_gpu.isChecked()

        scope = (
            "full"
            if self.mode == "narration"
            else (self.cmb_scope.currentData() or "full")
        )
        self._transcribe_scope = scope
        audio_dir = str(self._transcript_output_dir())

        self.btn_transcribe.setEnabled(False)
        self.btn_cancel.setVisible(True)
        self.progress_bar.setVisible(True)
        self.txt_log.clear()

        if scope == "selected":
            # ── Multi-part: chỉ phiên âm các Part đã chọn (✓) ─────────
            enabled_clips = [c for c in self._project.clips if c.enabled]
            if not enabled_clips:
                QMessageBox.warning(
                    self, "Không có Part",
                    "Chưa có Part nào được chọn (✓) ở Tab 4."
                )
                self.btn_transcribe.setEnabled(True)
                self.btn_cancel.setVisible(False)
                self.progress_bar.setVisible(False)
                return

            total_dur = sum(c.duration for c in enabled_clips)
            self.txt_log.append(
                f"🚀 Phiên âm {len(enabled_clips)} Part đã chọn "
                f"(tổng ~{total_dur:.0f}s)..."
            )
            self._worker = _MultiPartTranscribeWorker(
                video_path,
                enabled_clips,
                audio_dir,
                model, lang_code, use_gpu, beam_size,
            )
            self._worker.log.connect(self._on_log)
            self._worker.finished.connect(self._on_transcribe_done)
            self._worker.start()

        elif scope.startswith("part_"):
            # ── Single part ────────────────────────────────────────────
            try:
                part_idx = int(scope.split("_")[1])
            except (ValueError, IndexError):
                QMessageBox.warning(self, "Lỗi", "Part không hợp lệ.")
                self.btn_transcribe.setEnabled(True)
                self.btn_cancel.setVisible(False)
                self.progress_bar.setVisible(False)
                return
            clip = next(
                (c for c in self._project.clips if c.index == part_idx), None
            )
            if not clip:
                QMessageBox.warning(self, "Lỗi", f"Không tìm thấy Part {part_idx}.")
                self.btn_transcribe.setEnabled(True)
                self.btn_cancel.setVisible(False)
                self.progress_bar.setVisible(False)
                return

            self.txt_log.append(
                f"🎬 Phiên âm {clip.part_text} "
                f"({clip.start_time:.0f}s → {clip.end_time:.0f}s)..."
            )
            # Use multi-part worker with just one clip for consistency
            self._worker = _MultiPartTranscribeWorker(
                video_path,
                [clip],
                audio_dir,
                model, lang_code, use_gpu, beam_size,
            )
            self._worker.log.connect(self._on_log)
            self._worker.finished.connect(self._on_transcribe_done)
            self._worker.start()

        else:
            # ── Full video ─────────────────────────────────────────────
            target_label = (
                (
                    "track giọng đọc sạch"
                    if Path(video_path).suffix.lower()
                    in {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac"}
                    else "video đã ghép giọng"
                )
                if self.mode == "narration"
                else "video nguồn"
            )
            self.txt_log.append(f"🚀 Bắt đầu phiên âm toàn bộ {target_label}...")
            self._worker = _TranscribeWorker(
                video_path,
                audio_dir,
                model, lang_code, use_gpu, beam_size,
            )
            self._worker.log.connect(self._on_log)
            self._worker.finished.connect(self._on_transcribe_done)
            self._worker.start()

    def _cancel_transcribe(self):
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self.txt_log.append("⛔ Đang dừng...")
            self.btn_cancel.setEnabled(False)

    def _on_log(self, line: str):
        self.txt_log.append(line)
        sb = self.txt_log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _on_transcribe_done(self, transcript):
        self.btn_transcribe.setEnabled(True)
        self.btn_cancel.setVisible(False)
        self.btn_cancel.setEnabled(True)
        self.progress_bar.setVisible(False)
        if transcript:
            # NOTE: _MultiPartTranscribeWorker already shifts timestamps
            # internally for "selected" and "part_N" scopes, so we do NOT
            # shift again here.  Only merge into existing transcript when
            # transcribing a single part on top of an existing transcript.
            scope = getattr(self, "_transcribe_scope", "full")
            if scope.startswith("part_") and self._transcript:
                try:
                    part_idx = int(scope.split("_")[1])
                    clip = next(
                        (c for c in self._project.clips if c.index == part_idx),
                        None,
                    )
                    if clip:
                        self._merge_part_transcript(
                            self._transcript, transcript, clip
                        )
                        transcript = self._transcript
                        self.txt_log.append(
                            "🔗 Đã ghép transcript Part vào transcript chung."
                        )
                except Exception as e:
                    self.txt_log.append(f"⚠️ Lỗi merge transcript: {e}")

            transcript_path = self._persist_transcript(transcript)
            self._set_transcript(transcript)
            self.txt_log.append(f"💾 Đã lưu riêng: {transcript_path}")
            self.transcript_ready.emit(transcript)
            if self.mode == "narration":
                self.narration_transcript_ready.emit(transcript)
            else:
                self.source_transcript_ready.emit(transcript)
        else:
            self.txt_log.append("❌ Phiên âm thất bại hoặc đã bị hủy.")

    def _merge_part_transcript(self, base: dict, part: dict, clip):
        """Merge a per-part transcript into the base transcript.

        Replaces any existing segments that overlap with the clip's time range.
        """
        base_segs = base.get("segments", [])
        part_segs = part.get("segments", [])

        # Remove existing segments that overlap with this clip's range
        filtered = [
            s for s in base_segs
            if s["end"] <= clip.start_time or s["start"] >= clip.end_time
        ]
        # Add the new part's segments
        filtered.extend(part_segs)
        # Sort by start time
        filtered.sort(key=lambda s: s["start"])
        base["segments"] = filtered
        base["text"] = " ".join(
            (segment.get("text") or "").strip()
            for segment in filtered
            if (segment.get("text") or "").strip()
        )

    def _set_transcript(self, transcript: dict):
        self._transcript = transcript
        self.btn_gen_subs.setEnabled(self.mode == "narration")
        lines = []
        for seg in transcript.get("segments", []):
            lines.append(f"[{seg['start']:.1f}s] {seg['text']}")
        self.txt_transcript.setPlainText("\n".join(lines))

    def _load_existing(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Chọn file transcript", "", "JSON (*.json);;All Files (*)"
        )
        if path:
            t = load_transcript(path)
            if t:
                self._set_transcript(t)
                if self._project:
                    saved_path = self._persist_transcript(t)
                    self.transcript_ready.emit(t)
                    if self.mode == "narration":
                        self.narration_transcript_ready.emit(t)
                    else:
                        self.source_transcript_ready.emit(t)
                    self.txt_log.append(
                        f"✅ Đã nhập và lưu riêng: {saved_path}"
                    )
                else:
                    self.txt_log.append(f"✅ Đã tải: {path}")
            else:
                QMessageBox.warning(self, "Lỗi", "Không đọc được file transcript.")

    # ── Subtitle generation ───────────────────────────────────────────────────

    def _generate_subtitles(self):
        if not self._transcript or not self._project:
            return
        if self.mode != "narration":
            QMessageBox.information(
                self,
                "Subtitle narration",
                "Subtitle cuối chỉ được tạo ở chế độ phiên âm narration.",
            )
            return

        style = self.cmb_sub_style.currentData()
        self.btn_gen_subs.setEnabled(False)
        self.lbl_sub_status.setText("Đang tạo subtitle...")

        self._sub_worker = _SubtitleGenWorker(
            self._project,
            self._transcript,
            style,
            self._project.export_config,
            mode=self.mode,
        )
        self._sub_worker.log.connect(self._on_log)
        self._sub_worker.finished.connect(self._on_subs_done)
        self._sub_worker.start()

    def _on_subs_done(self, gen: int, skip: int):
        self.btn_gen_subs.setEnabled(True)
        style = self.cmb_sub_style.currentData()
        if gen <= 0:
            self.lbl_sub_status.setText(
                "❌ Không tạo được subtitle từ transcript narration."
            )
            return

        subtitle_path = (
            getattr(self._project, "narration_subtitle_file", "") or ""
        )
        self.lbl_sub_status.setText(
            f"✅ Subtitle toàn video: {Path(subtitle_path).name}"
        )
        self._project.export_config.subtitle_enabled = True
        self._project.export_config.subtitle_style = style
        self._project.export_config.global_subtitle_file = subtitle_path
        save_project(self._project)
        self.narration_subtitle_ready.emit(subtitle_path)
        self.subtitles_generated.emit(style)

    # ── In-app media player ───────────────────────────────────────────────────

    def _toggle_play(self):
        state = self._media_player.playbackState()
        if state == QMediaPlayer.PlaybackState.PlayingState:
            self._media_player.pause()
        else:
            self._media_player.play()

    def _seek(self, pos_ms: int):
        self._media_player.setPosition(pos_ms)

    def _on_playback_state(self, state):
        playing = state == QMediaPlayer.PlaybackState.PlayingState
        self.btn_play.setText("⏸" if playing else "▶")

    def _on_duration_changed(self, duration_ms: int):
        self.sld_pos.setRange(0, max(duration_ms, 1))
        self._update_time_label(0, duration_ms)

    def _on_position_changed(self, pos_ms: int):
        self.sld_pos.blockSignals(True)
        self.sld_pos.setValue(pos_ms)
        self.sld_pos.blockSignals(False)
        dur = self._media_player.duration()
        self._update_time_label(pos_ms, dur)

        # Sync subtitle display
        if self._transcript:
            secs = pos_ms / 1000.0
            text = ""
            for seg in self._transcript.get("segments", []):
                if seg["start"] <= secs <= seg["end"]:
                    text = seg["text"]
                    break
            self.lbl_cur_sub.setText(text)

    @staticmethod
    def _fmt_ms(ms: int) -> str:
        s = ms // 1000
        return f"{s // 60}:{s % 60:02d}"

    def _update_time_label(self, pos_ms: int, dur_ms: int):
        self.lbl_time.setText(
            f"{self._fmt_ms(pos_ms)} / {self._fmt_ms(dur_ms)}"
        )
