"""Tab 5: Lồng tiếng – edge-tts, VoxCPM2 và OmniVoice Studio."""

import json
import re
import shutil
import tempfile
from pathlib import Path

from PyQt6.QtCore import Qt, QThread, QUrl, pyqtSignal
from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QPushButton, QComboBox, QSlider, QDoubleSpinBox, QSpinBox,
    QGroupBox, QTextEdit, QSplitter,
    QMessageBox, QFileDialog, QSizePolicy,
    QProgressBar, QCheckBox, QLineEdit, QScrollArea,
)

from src.core.tts_manager import (
    is_edge_tts_available, get_edge_tts_voices,
    generate_edge_tts, merge_voice_with_video,
    is_nghitts_available, generate_nghitts, NGHITTS_VOICES,
    is_voxcpm_available, get_voxcpm_runtime_status,
    generate_voxcpm, VOXCPM_DEFAULT_MODEL,
    align_interpreter_segments, align_voice_segments,
    build_clean_review_video, split_voiceover_sections,
    strip_voiceover_part_labels, parse_timed_dubbing_script,
    is_omnivoice_running, preprocess_narration_text,
    probe_media_duration,
    VI_VOICES, EN_VOICES,
)
from src.core.dependency_manager import ffmpeg_path
from src.core.dubbing_budget import word_budget, timeline_overflow, group_dubbing_segments
from src.core.tts_manager import trim_dubbing_edges
from src.core.transcriber import transcribe_video, load_transcript
from src.core.ai_client import translate_timed_segments
from src.core.project_manager import save_project
from src.models.project import Project
from src.utils.file_utils import ensure_dir
from src.utils.logger import logger


# Vietnamese is counted by whitespace-separated syllables. Use a conservative
# fallback until the selected provider/voice has been measured in this project.
_VI_EST_WORDS_PER_SEC = 3.3
_EN_EST_WORDS_PER_SEC = 2.5
_VOICE_RATE_MIN = 1.10
_VOICE_RATE_MAX = 1.15
# TTS duration varies slightly between runs. Keep the selectable range capped
# at 1.15x, but permit a tiny alignment correction instead of rejecting 1.16x.
_VOICE_RATE_HARD_MAX = 1.17
_DUBBING_GRACE_WORDS = 3
_DUBBING_GRACE_SECONDS = 1.5


def _fmt_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds or 0)))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def _count_words(text: str) -> int:
    return len([w for w in (text or "").split() if w.strip()])


def _scene_budget_violations(
    scenes: list[dict],
    words_per_second: float,
    max_rate: float = _VOICE_RATE_HARD_MAX,
) -> list[dict]:
    """Return scenes whose script cannot fit without exceeding max_rate."""
    violations: list[dict] = []
    pace = max(0.1, float(words_per_second))
    rate = max(0.1, float(max_rate))
    for scene in scenes:
        duration = max(0.1, float(scene.get("duration", 0.0) or 0.0))
        words = _count_words(scene.get("text", ""))
        max_words = max(1, int(duration * pace * rate))
        estimated_seconds = words / (pace * rate) if words else 0.0
        if words > max_words:
            violations.append({
                "index": int(scene.get("index", len(violations) + 1)),
                "words": words,
                "max_words": max_words,
                "remove": max(1, words - max_words),
                "duration": duration,
                "estimated_seconds": estimated_seconds,
                "overflow_seconds": max(0.0, estimated_seconds - duration),
            })
    return violations


def _small_dubbing_overflow_allowed(
    overflow_seconds: float,
    excess_words: int,
    video_duration: float,
) -> bool:
    """Accept minor estimation drift while still rejecting real overflow."""
    seconds_limit = min(
        _DUBBING_GRACE_SECONDS,
        max(0.75, max(0.0, float(video_duration or 0.0)) * 0.04),
    )
    return (
        int(excess_words or 0) <= _DUBBING_GRACE_WORDS
        and float(overflow_seconds or 0.0) <= seconds_limit
    )


class _VoiceGenWorker(QThread):
    log = pyqtSignal(str)
    progress = pyqtSignal(int)           # 0-100 percentage
    finished = pyqtSignal(bool, str)     # success, audio_path_or_error

    def __init__(
        self,
        scenes,
        voice,
        output_path,
        rate,
        volume,
        ffmpeg_bin,
        rate_factor,
        provider="edge_tts",
        voxcpm_options=None,
    ):
        super().__init__()
        self.scenes = scenes
        self.voice = voice
        self.output_path = output_path
        self.rate = rate
        self.volume = volume
        self.ffmpeg_bin = ffmpeg_bin
        self.rate_factor = rate_factor
        self.provider = provider
        self.voxcpm_options = dict(voxcpm_options or {})
        self.adjustments: list[dict] = []

    def run(self):
        work_dir = Path(tempfile.mkdtemp(
            prefix="_voice_parts_",
            dir=str(Path(self.output_path).parent),
        ))
        raw_segments: list[tuple[str, float, int]] = []
        scene_count = max(1, len(self.scenes))
        try:
            for pos, scene in enumerate(self.scenes):
                part_index = int(scene.get("index", pos + 1))
                suffix = ".wav" if self.provider in {"voxcpm", "nghitts"} else ".mp3"
                raw_path = str(work_dir / f"part_{part_index:03d}{suffix}")
                self.log.emit(
                    f"🎤 Tạo voice Part {part_index}/{scene_count} "
                    f"cho cảnh {float(scene['duration']):.2f}s..."
                )
                if self.provider == "voxcpm":
                    ok, result = generate_voxcpm(
                        scene["text"],
                        raw_path,
                        progress_cb=lambda s: self.log.emit(s),
                        **self.voxcpm_options,
                    )
                    self.progress.emit(
                        min(90, int(((pos + 1) / scene_count) * 90))
                    )
                elif self.provider == "nghitts":
                    ok, result = generate_nghitts(
                        scene["text"],
                        raw_path,
                        voice_name=self.voice,
                        rate_factor=self.rate_factor,
                        volume_percent=self.voxcpm_options.get("volume_percent", 0),
                        progress_cb=lambda s: self.log.emit(s),
                    )
                    self.progress.emit(
                        min(90, int(((pos + 1) / scene_count) * 90))
                    )
                else:
                    ok, result = generate_edge_tts(
                        scene["text"],
                        self.voice,
                        raw_path,
                        rate=self.rate,
                        volume=self.volume,
                        progress_cb=lambda s: self.log.emit(s),
                        progress_pct_cb=lambda p, i=pos: self.progress.emit(
                            min(90, int(((i + p / 100.0) / scene_count) * 90))
                        ),
                    )
                if not ok:
                    self.finished.emit(False, result)
                    return
                raw_segments.append((
                    result,
                    float(scene["duration"]),
                    part_index,
                ))

            self.progress.emit(92)
            ok, result, adjustments = align_voice_segments(
                raw_segments,
                self.output_path,
                self.ffmpeg_bin,
                progress_cb=lambda s: self.log.emit(s),
                allow_slowdown=False,
                max_tempo=(
                    _VOICE_RATE_HARD_MAX
                    / max(_VOICE_RATE_MIN, self.rate_factor)
                ),
            )
            self.adjustments = adjustments
            self.progress.emit(100 if ok else 0)
            self.finished.emit(ok, result)
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)


class _QuickPreviewWorker(QThread):
    finished = pyqtSignal(bool, str)

    def __init__(self, provider, voice, output_path, rate_factor, volume_percent):
        super().__init__()
        self.provider = provider
        self.voice = voice
        self.output_path = output_path
        self.rate_factor = rate_factor
        self.volume_percent = volume_percent

    def run(self):
        sample = "Xin chào, đây là giọng đọc thử. Chúc bạn một ngày vui vẻ."
        if self.provider == "nghitts":
            result = generate_nghitts(
                sample,
                self.output_path,
                voice_name=self.voice,
                rate_factor=self.rate_factor,
                volume_percent=self.volume_percent,
            )
        else:
            rate_pct = int(round((self.rate_factor - 1.0) * 100))
            rate = f"+{rate_pct}%" if rate_pct >= 0 else f"{rate_pct}%"
            volume = int(self.volume_percent)
            result = generate_edge_tts(
                sample,
                self.voice,
                self.output_path,
                rate=rate,
                volume=f"+{volume}%" if volume >= 0 else f"{volume}%",
            )
        self.finished.emit(*result)


class _ChineseScanWorker(QThread):
    log = pyqtSignal(str)
    finished = pyqtSignal(bool, object)

    def __init__(self, source_video, output_dir, model, ffmpeg_bin):
        super().__init__()
        self.source_video = source_video
        self.output_dir = output_dir
        self.model = model
        self.ffmpeg_bin = ffmpeg_bin

    def run(self):
        result = transcribe_video(
            self.source_video,
            str(Path(self.output_dir) / "dubbing_transcript"),
            model_size=self.model,
            language="zh",
            use_gpu=False,
            ffmpeg_bin=self.ffmpeg_bin,
            progress_cb=lambda line: self.log.emit(line),
            beam_size=5,
        )
        if result and result.get("segments"):
            self.finished.emit(True, result)
        else:
            self.finished.emit(False, "Không nhận diện được lời thoại tiếng Trung.")


class _DubbingWorker(QThread):
    log = pyqtSignal(str)
    progress = pyqtSignal(int)
    finished = pyqtSignal(bool, object)

    def __init__(
        self, source_video, output_dir, transcript_path, whisper_model,
        ai_provider, ai_model, voice_provider, voice, rate_factor,
        volume_percent, ffmpeg_bin, video_duration, original_audio_volume,
        voxcpm_options=None, manual_segments=None,
    ):
        super().__init__()
        self.source_video = source_video
        self.output_dir = Path(output_dir)
        self.transcript_path = transcript_path
        self.whisper_model = whisper_model
        self.ai_provider = ai_provider
        self.ai_model = ai_model
        self.voice_provider = voice_provider
        self.voice = voice
        self.rate_factor = rate_factor
        self.volume_percent = volume_percent
        self.ffmpeg_bin = ffmpeg_bin
        self.video_duration = video_duration
        self.original_audio_volume = original_audio_volume
        self.voxcpm_options = dict(voxcpm_options or {})
        self.manual_segments = list(manual_segments or [])

    def run(self):
        work_dir = Path(tempfile.mkdtemp(prefix="_dub_", dir=str(self.output_dir)))
        try:
            if self.manual_segments:
                translated = self.manual_segments
                self.log.emit(
                    f"📝 Đã nhận {len(translated)} đoạn tiếng Việt theo timestamp; bỏ qua AI dịch."
                )
            else:
                transcript = (
                    load_transcript(self.transcript_path)
                    if self.transcript_path and Path(self.transcript_path).is_file()
                    else None
                )
                if transcript and not str(transcript.get("language", "")).lower().startswith("zh"):
                    transcript = None
                if not transcript or not transcript.get("segments"):
                    self.log.emit("🎧 Chưa có phiên âm nguồn — nhận diện tiếng Trung trước…")
                    transcript = transcribe_video(
                        self.source_video,
                        str(self.output_dir / "dubbing_transcript"),
                        model_size=self.whisper_model,
                        language="zh",
                        use_gpu=False,
                        ffmpeg_bin=self.ffmpeg_bin,
                        progress_cb=lambda line: self.log.emit(line),
                        beam_size=5,
                    )
                if not transcript or not transcript.get("segments"):
                    raise RuntimeError("Không nhận diện được lời thoại tiếng Trung trong video.")
                self.progress.emit(15)
                translated = translate_timed_segments(
                    transcript["segments"], self.ai_provider, self.ai_model,
                    status_cb=lambda line: self.log.emit(line),
                )
            self.progress.emit(30)
            original_segments = translated
            translated = group_dubbing_segments(original_segments)
            self.log.emit(f"📝 Gom {len(original_segments)} mốc thành {len(translated)} cụm đọc liền mạch; giữ nguyên lời.")
            measurements = []
            timed_sequence: list[tuple[str, float, float, int]] = []
            selected_max_rate = max(
                _VOICE_RATE_MIN,
                min(_VOICE_RATE_MAX, self.rate_factor),
            )
            self.log.emit(
                f"🎚 Lồng tiếng: tự chọn nhịp chung {_VOICE_RATE_MIN:.2f}x–"
                f"{selected_max_rate:.2f}x theo thời lượng thực tế."
            )
            for pos, segment in enumerate(translated, start=1):
                start = max(0.0, float(segment["start"]))
                end = max(start + 0.1, float(segment["end"]))
                suffix = ".wav" if self.voice_provider in {"nghitts", "voxcpm"} else ".mp3"
                raw_path = str(work_dir / f"voice_{pos:04d}{suffix}")
                self.log.emit(
                    f"🎙 Câu {pos}/{len(translated)} [{start:.2f}–{end:.2f}s]: "
                    f"{segment['text_vi'][:70]}"
                )
                if self.voice_provider == "nghitts":
                    ok, result = generate_nghitts(
                        segment["text_vi"], raw_path, voice_name=self.voice,
                        rate_factor=1.0,
                        volume_percent=self.volume_percent,
                    )
                elif self.voice_provider == "voxcpm":
                    ok, result = generate_voxcpm(
                        segment["text_vi"], raw_path, **self.voxcpm_options
                    )
                else:
                    rate_pct = 0  # Apply the bounded speed once, after measuring audio.
                    volume = int(self.volume_percent)
                    ok, result = generate_edge_tts(
                        segment["text_vi"], self.voice, raw_path,
                        rate=f"+{rate_pct}%" if rate_pct >= 0 else f"{rate_pct}%",
                        volume=f"+{volume}%" if volume >= 0 else f"{volume}%",
                        progress_cb=lambda line: self.log.emit(line),
                    )
                if not ok:
                    raise RuntimeError(result)
                self.log.emit(
                    f"✅ Đã tạo audio câu {pos}/{len(translated)}; "
                    "chưa ghép video."
                )
                result = trim_dubbing_edges(
                    result, str(work_dir / f"trimmed_{pos:04d}.wav"), self.ffmpeg_bin
                )
                timed_sequence.append((result, start, end, pos))
                measurements.append({"text": segment["text_vi"],
                                     "words": len(segment["text_vi"].split()),
                                     "seconds": probe_media_duration(result, self.ffmpeg_bin)})
                self.progress.emit(30 + int(45 * pos / max(1, len(translated))))
            total_duration = max(
                self.video_duration,
                max((item[2] for item in timed_sequence), default=0.0),
            )
            timeline = str(self.output_dir / "audio" / "dubbing_vi.mp3")
            (self.output_dir / "dubbing_measurements.json").write_text(
                json.dumps({"provider": self.voice_provider, "voice": self.voice,
                            "version": 2, "rows": measurements}, ensure_ascii=False), encoding="utf-8"
            )
            ensure_dir(Path(timeline).parent)
            ok, result, adjustments = align_interpreter_segments(
                timed_sequence,
                timeline,
                self.ffmpeg_bin,
                total_duration,
                progress_cb=lambda line: self.log.emit(line),
                min_tempo=_VOICE_RATE_MIN,
                max_tempo=selected_max_rate,
                allow_overflow=True,
            )
            if not ok:
                raise RuntimeError(result)
            self.log.emit("✅ Đã căn xong các mốc; đang ghép audio vào video…")
            for segment, adjustment in zip(translated, adjustments):
                voice_start = float(adjustment["placed_start"])
                voice_duration = (
                    float(adjustment["source_duration"])
                    / max(0.01, float(adjustment["tempo_factor"]))
                )
                segment["voice_start"] = voice_start
                segment["voice_end"] = voice_start + voice_duration
                segment["tempo_factor"] = float(adjustment["tempo_factor"])
            self.progress.emit(85)
            output_video = str(self.output_dir / "dubbed_vi.mp4")
            ok, result = merge_voice_with_video(
                self.source_video, timeline, output_video, self.ffmpeg_bin,
                self.original_audio_volume,
                progress_cb=lambda line: self.log.emit(line),
            )
            if not ok:
                raise RuntimeError(result)
            translation_path = self.output_dir / "dubbing_translation.json"
            translation_path.write_text(
                json.dumps({"segments": original_segments, "voice_groups": translated}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            self.progress.emit(100)
            self.finished.emit(True, {
                "video": result, "audio": timeline,
                "translation": str(translation_path),
                "segments": original_segments,
            })
        except Exception as exc:
            logger.exception("Automatic dubbing failed")
            self.finished.emit(False, str(exc))
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)


class _MergeWorker(QThread):
    log = pyqtSignal(str)
    finished = pyqtSignal(bool, str)

    def __init__(
        self,
        project,
        clean_video_path,
        audio_path,
        output_path,
        ffmpeg_bin,
        orig_vol,
    ):
        super().__init__()
        self.project = project
        self.clean_video_path = clean_video_path
        self.audio_path = audio_path
        self.output_path = output_path
        self.ffmpeg_bin = ffmpeg_bin
        self.orig_vol = orig_vol

    def run(self):
        ok, result = build_clean_review_video(
            self.project,
            self.clean_video_path,
            progress_cb=lambda s: self.log.emit(s),
        )
        if not ok:
            self.finished.emit(False, result)
            return

        self.log.emit(
            f"🎤 Ghép giọng đọc vào video review sạch: "
            f"{Path(self.clean_video_path).name}"
        )
        ok, result = merge_voice_with_video(
            self.clean_video_path, self.audio_path, self.output_path,
            self.ffmpeg_bin, self.orig_vol,
            progress_cb=lambda s: self.log.emit(s),
        )

        self.finished.emit(ok, result)


class VoiceoverTab(QWidget):
    merged_video_ready = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._project: Project | None = None
        self._voices: list[dict] = []
        self._voice_worker = None
        self._quick_preview_worker = None
        self._merge_worker = None
        self._generated_audio = ""
        self._quick_preview_audio = ""
        self._video_duration = 0.0
        self._scene_mapping_error = ""
        self._setup_ui()
        self._setup_audio_player()
        self._on_voice_mode_changed()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Dep status
        self.lbl_dep = QLabel()
        self._refresh_dep()
        layout.addWidget(self.lbl_dep)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # ── Left: voice + settings ───────────────────────────────
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # TTS provider
        prov_group = QGroupBox("Nguồn giọng đọc")
        prov_layout = QFormLayout(prov_group)

        self.cmb_provider = QComboBox()
        self.cmb_provider.addItem("edge-tts (miễn phí)", "edge_tts")
        self.cmb_provider.addItem("Ngọc Huyền (mới) — local", "nghitts")
        self.cmb_provider.addItem("VoxCPM2 (local / clone giọng)", "voxcpm")
        self.cmb_provider.currentIndexChanged.connect(self._on_provider_changed)
        prov_layout.addRow("Provider:", self.cmb_provider)

        self.cmb_voice_mode = QComboBox()
        self.cmb_voice_mode.addItem("Theo từng đoạn", "parts")
        self.cmb_voice_mode.addItem("Một lời thoại cho toàn video", "story")
        self.cmb_voice_mode.addItem("Lồng tiếng theo mốc [giây] — tự dịch", "dub_timed_manual")
        self.cmb_voice_mode.setCurrentIndex(1)
        self.cmb_voice_mode.currentIndexChanged.connect(self._on_voice_mode_changed)
        prov_layout.addRow("Cách đọc:", self.cmb_voice_mode)

        self.cmb_dub_ai = QComboBox()
        self.cmb_dub_ai.addItem("Google Gemini", "gemini")
        self.cmb_dub_ai.addItem("Groq", "groq")
        self.cmb_dub_ai.addItem("OpenRouter", "openrouter")
        self.cmb_dub_ai.addItem("Ollama local", "ollama")
        self.cmb_dub_ai.currentIndexChanged.connect(self._on_dub_ai_changed)
        prov_layout.addRow("AI dịch Trung → Việt:", self.cmb_dub_ai)
        self.txt_dub_model = QLineEdit("gemini-2.5-flash-lite")
        prov_layout.addRow("Model dịch:", self.txt_dub_model)
        self.cmb_dub_whisper = QComboBox()
        self.cmb_dub_whisper.addItems(["tiny", "base", "small", "medium", "large-v3"])
        self.cmb_dub_whisper.setCurrentText("base")
        prov_layout.addRow("Whisper tiếng Trung:", self.cmb_dub_whisper)

        # Voice selector
        self.cmb_voice = QComboBox()
        self._populate_quick_voices()
        prov_layout.addRow("Giọng đọc:", self.cmb_voice)

        self.btn_refresh_voices = QPushButton("🔄 Tải danh sách giọng đầy đủ")
        self.btn_refresh_voices.clicked.connect(self._load_all_voices)
        prov_layout.addRow(self.btn_refresh_voices)

        self.btn_quick_preview = QPushButton("🔊 Nghe thử nhanh")
        self.btn_quick_preview.setToolTip(
            "Chỉ tạo một câu ngắn; file mẫu được lưu lại để lần sau phát ngay."
        )
        self.btn_quick_preview.clicked.connect(self._quick_preview_voice)
        prov_layout.addRow(self.btn_quick_preview)

        # Rate and volume
        self.spn_rate = QDoubleSpinBox()
        self.spn_rate.setRange(0.50, 2.00)
        self.spn_rate.setDecimals(2)
        self.spn_rate.setSingleStep(0.05)
        self.spn_rate.setValue(1.00)
        self.spn_rate.setSuffix("x")
        self.spn_rate.setToolTip(
            "1.00x = tốc độ gốc. App sẽ đo audio thật và căn nhẹ từng cảnh "
            "để khớp timeline."
        )
        self.spn_rate.valueChanged.connect(self._update_story_stats)
        prov_layout.addRow("Tốc độ nói (CapCut):", self.spn_rate)

        self.lbl_effective_rate = QLabel("Nhịp khớp video: chờ tạo voice")
        self.lbl_effective_rate.setStyleSheet("color:#89b4fa; font-size:11px;")
        self.lbl_effective_rate.setWordWrap(True)
        prov_layout.addRow(self.lbl_effective_rate)

        self.spn_vol = QDoubleSpinBox()
        self.spn_vol.setRange(-100, 100)
        self.spn_vol.setValue(0)
        self.spn_vol.setSuffix("%")
        prov_layout.addRow("Âm lượng:", self.spn_vol)

        self.chk_narration_mode = QCheckBox("Chế độ tường thuật (tự nhiên hơn)")
        self.chk_narration_mode.setChecked(True)
        self.chk_narration_mode.setToolTip(
            "Tự động xử lý text trước khi đọc:\n"
            "- Loại bỏ tiêu đề PART\n"
            "- Thêm ngắt nghỉ tự nhiên\n"
            "- Chuẩn hóa dấu câu"
        )
        prov_layout.addRow(self.chk_narration_mode)
        left_layout.addWidget(prov_group)

        # VoxCPM is optional because its PyTorch stack and model weights are
        # several GB. All imports remain lazy so the regular app stays light.
        self.voxcpm_group = QGroupBox("VoxCPM2 — thiết kế / clone giọng")
        vox_layout = QFormLayout(self.voxcpm_group)

        self.cmb_voxcpm_mode = QComboBox()
        self.cmb_voxcpm_mode.addItem("Voice Design (không cần audio)", "design")
        self.cmb_voxcpm_mode.addItem("Clone giọng (audio 5–30 giây)", "clone")
        self.cmb_voxcpm_mode.addItem("Hi-Fi clone (audio + transcript)", "hifi")
        self.cmb_voxcpm_mode.currentIndexChanged.connect(
            self._on_voxcpm_mode_changed
        )
        vox_layout.addRow("Chế độ:", self.cmb_voxcpm_mode)

        self.txt_voxcpm_style = QLineEdit()
        self.txt_voxcpm_style.setPlaceholderText(
            "Ví dụ: giọng nữ trẻ, ấm áp, kể chuyện tự nhiên"
        )
        self.txt_voxcpm_style.setText("Giọng kể chuyện tự nhiên, ấm áp, rõ ràng")
        vox_layout.addRow("Mô tả giọng:", self.txt_voxcpm_style)

        self.txt_voxcpm_reference = QLineEdit()
        self.txt_voxcpm_reference.setPlaceholderText(
            "Audio sạch 5–30 giây (.wav/.flac/.mp3)"
        )
        btn_ref = QPushButton("📂")
        btn_ref.setFixedWidth(38)
        btn_ref.clicked.connect(self._browse_voxcpm_reference)
        ref_row = QHBoxLayout()
        ref_row.addWidget(self.txt_voxcpm_reference, 1)
        ref_row.addWidget(btn_ref)
        vox_layout.addRow("Audio mẫu:", ref_row)
        self.btn_voxcpm_reference = btn_ref

        self.txt_voxcpm_prompt = QLineEdit()
        self.txt_voxcpm_prompt.setPlaceholderText(
            "Transcript chính xác từng chữ trong audio mẫu"
        )
        vox_layout.addRow("Transcript mẫu:", self.txt_voxcpm_prompt)

        self.txt_voxcpm_model = QLineEdit(VOXCPM_DEFAULT_MODEL)
        self.txt_voxcpm_model.setToolTip(
            "Hugging Face model ID hoặc đường dẫn thư mục model đã tải."
        )
        vox_layout.addRow("Model:", self.txt_voxcpm_model)

        self.cmb_voxcpm_device = QComboBox()
        self.cmb_voxcpm_device.addItem("Tự động (an toàn VRAM)", "auto")
        self.cmb_voxcpm_device.addItem("CUDA / NVIDIA", "cuda")
        self.cmb_voxcpm_device.addItem("CPU", "cpu")
        vox_layout.addRow("Thiết bị:", self.cmb_voxcpm_device)

        tuning_row = QHBoxLayout()
        self.spn_voxcpm_cfg = QDoubleSpinBox()
        self.spn_voxcpm_cfg.setRange(1.0, 3.0)
        self.spn_voxcpm_cfg.setSingleStep(0.1)
        self.spn_voxcpm_cfg.setValue(2.0)
        self.spn_voxcpm_steps = QSpinBox()
        self.spn_voxcpm_steps.setRange(4, 30)
        self.spn_voxcpm_steps.setValue(10)
        tuning_row.addWidget(QLabel("CFG"))
        tuning_row.addWidget(self.spn_voxcpm_cfg)
        tuning_row.addWidget(QLabel("Bước"))
        tuning_row.addWidget(self.spn_voxcpm_steps)
        vox_layout.addRow("Chất lượng:", tuning_row)

        vox_flags = QHBoxLayout()
        self.chk_voxcpm_normalize = QCheckBox("Chuẩn hóa số/ngày")
        self.chk_voxcpm_normalize.setChecked(True)
        self.chk_voxcpm_denoise = QCheckBox("Khử nhiễu audio mẫu")
        vox_flags.addWidget(self.chk_voxcpm_normalize)
        vox_flags.addWidget(self.chk_voxcpm_denoise)
        vox_layout.addRow(vox_flags)

        self.lbl_voxcpm_note = QLabel(
            "Lần đầu chạy sẽ tải model vài GB. VoxCPM2 cần khoảng 8 GB VRAM; "
            "GPU nhỏ hơn sẽ được chuyển sang CPU ở chế độ Tự động."
        )
        self.lbl_voxcpm_note.setWordWrap(True)
        self.lbl_voxcpm_note.setStyleSheet(
            "color:#f9e2af; font-size:11px;"
        )
        vox_layout.addRow(self.lbl_voxcpm_note)
        self.voxcpm_group.setVisible(False)
        left_layout.addWidget(self.voxcpm_group)

        # Merge settings
        merge_group = QGroupBox("Ghép âm thanh")
        merge_layout = QFormLayout(merge_group)

        self.lbl_orig_vol = QLabel("15%")
        self.sld_orig_vol = QSlider(Qt.Orientation.Horizontal)
        self.sld_orig_vol.setRange(0, 100)
        self.sld_orig_vol.setValue(15)
        self.sld_orig_vol.valueChanged.connect(
            lambda v: self.lbl_orig_vol.setText(f"{v}%")
        )
        vol_row = QHBoxLayout()
        vol_row.addWidget(self.sld_orig_vol)
        vol_row.addWidget(self.lbl_orig_vol)
        merge_layout.addRow("Âm gốc còn lại:", vol_row)

        self._orig_vol_before_mute = 15
        self.chk_mute_original = QCheckBox("Tắt hoàn toàn âm thanh gốc của video")
        self.chk_mute_original.toggled.connect(self._toggle_original_audio)
        merge_layout.addRow(self.chk_mute_original)

        merge_layout.addWidget(QLabel(
            "0% = tắt âm gốc | 15% = nhỏ dưới nền | 100% = giữ nguyên"
        ))
        clean_note = QLabel(
            "Video được dựng lại trực tiếp từ nguồn thô với PART/subtitle tắt hoàn toàn. "
            "Sau khi ghép xong, tool tự chuyển sang màn hình tạo phụ đề."
        )
        clean_note.setWordWrap(True)
        clean_note.setStyleSheet("color:#a6e3a1; font-size:11px;")
        merge_layout.addRow(clean_note)
        left_layout.addWidget(merge_group)

        # Action buttons
        self.btn_gen_voice = QPushButton("🎤 Tạo giọng đọc")
        self.btn_gen_voice.setFixedHeight(36)
        self.btn_gen_voice.clicked.connect(self._generate_voice)
        left_layout.addWidget(self.btn_gen_voice)

        self.btn_scan_chinese = QPushButton("🔎 Quét lời Trung theo mốc thời gian")
        self.btn_scan_chinese.setFixedHeight(36)
        self.btn_scan_chinese.clicked.connect(self._scan_chinese_timeline)
        self.btn_scan_chinese.setVisible(False)
        left_layout.addWidget(self.btn_scan_chinese)

        self.btn_auto_dub = QPushButton("🎬 Tạo voice & ghép theo các mốc")
        self.btn_auto_dub.setFixedHeight(40)
        self.btn_auto_dub.setStyleSheet(
            "background:#45475a; color:#a6e3a1; font-weight:bold;"
        )
        self.btn_auto_dub.clicked.connect(self._start_auto_dubbing)
        self.btn_auto_dub.setVisible(False)
        left_layout.addWidget(self.btn_auto_dub)

        # In-app audio player controls
        audio_row = QHBoxLayout()
        self.btn_preview_voice = QPushButton("▶ Phát")
        self.btn_preview_voice.setEnabled(False)
        self.btn_preview_voice.setFixedWidth(72)
        self.btn_preview_voice.clicked.connect(self._toggle_audio_preview)
        self.btn_stop_voice = QPushButton("⏹")
        self.btn_stop_voice.setEnabled(False)
        self.btn_stop_voice.setFixedWidth(36)
        self.btn_stop_voice.clicked.connect(self._stop_audio_preview)
        self.lbl_audio_time = QLabel("0:00")
        self.lbl_audio_time.setStyleSheet("color: #888; font-size: 11px;")
        audio_row.addWidget(self.btn_preview_voice)
        audio_row.addWidget(self.btn_stop_voice)
        audio_row.addWidget(self.lbl_audio_time)
        audio_row.addStretch()
        left_layout.addLayout(audio_row)

        self.btn_merge = QPushButton("🎞 Ghép voice (không PART, không subtitle)")
        self.btn_merge.setFixedHeight(40)
        self.btn_merge.setStyleSheet(
            "background: #313244; color: #f9e2af; font-weight: bold;"
        )
        self.btn_merge.setEnabled(False)
        self.btn_merge.clicked.connect(self._merge_voice)
        left_layout.addWidget(self.btn_merge)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(False)
        left_layout.addWidget(self.progress_bar)

        left_layout.addStretch()
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        left_scroll.setWidget(left)
        splitter.addWidget(left_scroll)

        # ── Right: Script editor + log ───────────────────────────
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)

        right_layout.addWidget(QLabel("Dán lời thoại cần đọc vào đây:"))

        script_btns = QHBoxLayout()
        btn_clear = QPushButton("🗑 Xóa")
        btn_clear.setFixedWidth(70)
        btn_clear.clicked.connect(lambda: self.txt_script.clear())
        btn_save = QPushButton("💾 Lưu script")
        btn_save.clicked.connect(self._save_script)
        btn_load = QPushButton("📂 Tải file .txt")
        btn_load.clicked.connect(self._load_script_file)
        script_btns.addWidget(btn_clear)
        script_btns.addStretch()
        script_btns.addWidget(btn_load)
        script_btns.addWidget(btn_save)
        right_layout.addLayout(script_btns)

        self.txt_script = QTextEdit()
        self.txt_script.setPlaceholderText(
            "Dán toàn bộ lời thoại vào đây...\n\n"
            "Ví dụ:\nTrong tập này, bác sĩ House đối mặt với ca bệnh bí ẩn...\n"
            "Chuyện gì xảy ra khi không ai biết đáp án?"
        )
        self.txt_script.setStyleSheet("background: #181825; color: #cdd6f4; font-size: 13px;")
        self.txt_script.textChanged.connect(self._update_story_stats)
        right_layout.addWidget(self.txt_script, 1)

        self.lbl_story_stats = QLabel("")
        self.lbl_story_stats.setWordWrap(True)
        self.lbl_story_stats.setStyleSheet(
            "background:#313244; color:#89b4fa; border-radius:4px; padding:6px; font-size:12px;"
        )
        right_layout.addWidget(self.lbl_story_stats)

        right_layout.addWidget(QLabel("Log:"))
        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setMaximumHeight(130)
        self.txt_log.setStyleSheet("background: #111; color: #aaa; font-size: 11px;")
        right_layout.addWidget(self.txt_log)

        splitter.addWidget(right)
        splitter.setSizes([320, 580])
        layout.addWidget(splitter, 1)

    def _toggle_original_audio(self, muted: bool):
        """Make muting the source audio an explicit one-click choice."""
        if muted:
            if self.sld_orig_vol.value() > 0:
                self._orig_vol_before_mute = self.sld_orig_vol.value()
            self.sld_orig_vol.setValue(0)
            self.sld_orig_vol.setEnabled(False)
        else:
            self.sld_orig_vol.setEnabled(True)
            self.sld_orig_vol.setValue(max(1, self._orig_vol_before_mute))

    # ─── Public API ───────────────────────────────────────────────

    def _words_per_second(self) -> float:
        voice_id = (
            self.cmb_voice.currentData()
            if hasattr(self, "cmb_voice") else ""
        ) or ""
        provider = (
            self.cmb_provider.currentData()
            if hasattr(self, "cmb_provider") else "edge_tts"
        )
        if self._project:
            measurements = getattr(
                self._project, "voice_pace_measurements", {}
            ) or {}
            key = f"{provider}|{voice_id}"
            try:
                measured = float(
                    (measurements.get(key) or {}).get("words_per_second", 0.0)
                )
                if measured > 0:
                    return measured
            except (TypeError, ValueError, AttributeError):
                pass
        return (
            _VI_EST_WORDS_PER_SEC
            if provider in {"nghitts", "voxcpm"}
            or str(voice_id).lower().startswith("vi-")
            else _EN_EST_WORDS_PER_SEC
        )

    def _save_voice_pace_measurement(self, adjustments: list[dict]) -> None:
        """Save measured 1.0x pace for the selected provider and voice."""
        if not self._project or not adjustments or not self._voice_worker:
            return
        words = sum(
            _count_words(scene.get("text", ""))
            for scene in self._voice_worker.scenes
        )
        measured_seconds = sum(
            max(0.0, float(item.get("source_duration", 0.0) or 0.0))
            for item in adjustments
        )
        rate = max(0.1, float(self._voice_worker.rate_factor))
        base_seconds = measured_seconds * rate
        if words <= 0 or base_seconds <= 0:
            return
        provider = self.cmb_provider.currentData() or "edge_tts"
        voice = self.cmb_voice.currentData() or ""
        key = f"{provider}|{voice}"
        measurements = dict(
            getattr(self._project, "voice_pace_measurements", {}) or {}
        )
        measurements[key] = {
            "words_per_second": round(words / base_seconds, 4),
            "sample_words": words,
            "sample_seconds": round(measured_seconds, 3),
            "rate_factor": round(rate, 3),
        }
        self._project.voice_pace_measurements = measurements
        save_project(self._project)

    def _estimate_script_seconds(self, text: str) -> float:
        words = _count_words(text)
        rate_factor = max(0.25, self.spn_rate.value())
        return words / (self._words_per_second() * rate_factor) if words else 0.0

    def _max_words_for_video(self) -> int:
        # Capacity is always calculated at the absolute permitted ceiling.
        rate_factor = _VOICE_RATE_HARD_MAX
        return int(
            max(0.0, self._video_duration)
            * self._words_per_second()
            * rate_factor
        )

    def _recommended_rate_factor(self, text: str) -> float:
        words = _count_words(text)
        if not words or self._video_duration <= 0:
            return self.spn_rate.value()
        base_seconds = words / self._words_per_second()
        return max(
            _VOICE_RATE_MIN,
            min(_VOICE_RATE_MAX, base_seconds / self._video_duration),
        )

    def _dubbing_measurements(self):
        if self._project:
            try:
                saved = json.loads((Path(self._project.output_dir) / "dubbing_measurements.json").read_text(encoding="utf-8"))
                if saved.get("version") == 2 and saved["provider"] == self.cmb_provider.currentData() and saved["voice"] == self.cmb_voice.currentData():
                    return saved["rows"]
            except (OSError, ValueError, KeyError):
                pass
        return []

    def _validate_dubbing_budget(self, segments, duration):
        measurements = self._dubbing_measurements()
        # Long content is evaluated at the selected maximum rate. Short
        # content is evaluated at 1.10x, the slowest allowed pace, because that
        # is what the renderer now tries first to minimize silent gaps.
        selected_max_rate = max(
            _VOICE_RATE_MIN,
            min(_VOICE_RATE_MAX, self.spn_rate.value()),
        )
        rows_fast = word_budget(segments, selected_max_rate, measurements)
        rows_slow = word_budget(segments, _VOICE_RATE_MIN, measurements)
        overflow = timeline_overflow(rows_fast, duration)
        excess_words = sum(max(0, row["remove"]) for row in rows_fast)
        # A sub-second / 2% estimate difference is normal between TTS runs.
        # Let synthesis measure it exactly instead of rejecting useful input.
        grace_seconds = max(0.75, float(duration or 0.0) * 0.02)
        minor_overflow = _small_dubbing_overflow_allowed(
            overflow,
            excess_words,
            duration,
        )
        if overflow <= grace_seconds or minor_overflow:
            if overflow > grace_seconds:
                self.txt_log.append(
                    f"⚠️ Chênh lệch nhỏ: dư khoảng {overflow:.2f}s / "
                    f"{excess_words} từ; vẫn cho phép tạo và ghép voice."
                )
            cursor = 0.0
            gaps: list[tuple[float, float]] = []
            for row in rows_slow:
                start = max(0.0, float(row["start"]))
                gap = max(0.0, start - cursor)
                if gap > 0:
                    gaps.append((start, gap))
                cursor = max(cursor, start) + float(row["duration"])
            tail_gap = max(0.0, float(duration or 0.0) - cursor)
            if tail_gap > 0:
                gaps.append((float(duration or 0.0), tail_gap))

            long_gaps = [(start, gap) for start, gap in gaps if gap >= 2.0]
            if not long_gaps:
                return True

            missing = sum(max(0, row["add"]) for row in rows_slow)
            gap_details = "\n".join(
                f"Trước mốc {start:.1f}s: nghỉ khoảng {gap:.2f}s"
                for start, gap in long_gaps
            )
            budget_details = "\n".join(
                f"[{row['start']:.1f}s] {row['words']} từ; nên khoảng "
                f"{row['target']} từ (có thể thêm ~{row['add']} từ)."
                for row in rows_slow
                if row["add"] > 0
            )
            self.txt_log.append(
                "⚠️ Phát hiện khoảng nghỉ dài trước khi tạo voice:\n"
                + gap_details
            )
            answer = QMessageBox.question(
                self,
                "Lời thoại còn quá ngắn",
                f"Ở nhịp chậm nhất {_VOICE_RATE_MIN:.2f}x vẫn có "
                f"{len(long_gaps)} khoảng nghỉ từ 2 giây trở lên.\n\n"
                f"{gap_details}\n\n"
                f"Nên bổ sung tổng cộng khoảng {missing} từ vào đúng các mốc:\n"
                f"{budget_details}\n\n"
                "Bạn vẫn muốn tạo video với các khoảng nghỉ này?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            return answer == QMessageBox.StandardButton.Yes
        rows = rows_fast
        details = "\n".join(
            f"[{row['start']:.1f}s] Đang có {row['words']} từ / mục tiêu ~{row['target']} từ: "
            + (f"thừa ~{row['remove']} từ → bớt ~{row['remove']} từ" if row['remove']
               else f"thiếu ~{row['add']} từ → có thể thêm ~{row['add']} từ" if row['add']
               else "vừa đủ, không cần thêm/bớt")
            for row in rows
        )
        self._update_story_stats()
        self.txt_log.append("Kiểm tra trước khi tạo voice:\n" + details)
        QMessageBox.warning(
            self,
            "Lời đọc dài hơn video",
            f"Ở {self.spn_rate.value():.2f}x, dự tính lời vượt video {overflow:.2f}s.\n"
            f"Các câu dài: thừa ~{sum(row['remove'] for row in rows)} từ → cần bớt tương ứng.\n"
            f"Các câu ngắn: thiếu ~{sum(row['add'] for row in rows)} từ → có thể thêm tương ứng (không bắt buộc).\n\n"
            f"Hãy sửa lời rồi tạo lại. Tool không tạo voice khi dự tính cần "
            f"tốc độ vượt {_VOICE_RATE_HARD_MAX:.2f}x."
        )
        return False

    def _update_story_stats(self):
        if not hasattr(self, "lbl_story_stats"):
            return
        text = self.txt_script.toPlainText() if hasattr(self, "txt_script") else ""
        if self.cmb_voice_mode.currentData() == "dub_timed_manual" and self._video_duration > 0:
            try:
                segments = parse_timed_dubbing_script(text, self._video_duration)
            except ValueError:
                self.lbl_story_stats.setText("Dán lời Việt theo dạng [0.0s] Nội dung để tính số từ.")
                return
            measurements = self._dubbing_measurements()
            rows = word_budget(segments, self.spn_rate.value(), measurements)
            remove = sum(r["remove"] for r in rows)
            add = sum(r["add"] for r in rows)
            words = sum(r["words"] for r in rows)
            details = [f"[{r['start']:.1f}s] Đang có {r['words']} từ / mục tiêu ~{r['target']} từ: "
                       + (f"thừa ~{r['remove']} từ → bớt ~{r['remove']} từ" if r['remove']
                          else f"thiếu ~{r['add']} từ → có thể thêm ~{r['add']} từ (không bắt buộc)" if r['add']
                          else "vừa đủ, không cần thêm/bớt") for r in rows]
            self.lbl_story_stats.setText(
                f"{self.spn_rate.value():.2f}x | {words} từ | "
                f"Câu dài: thừa ~{remove} từ → bớt ~{remove}; câu ngắn: thiếu ~{add} từ → có thể thêm ~{add}. "
                + ("Ước tính từ audio đã đo." if measurements else "Ước tính, chưa đo giọng thực tế.")
                + " Rê chuột vào đây để xem từng mốc."
            )
            self.lbl_story_stats.setToolTip("\n".join(details))
            self._dubbing_budget_details = "\n".join(details)
            return
        words = _count_words(text)
        est_seconds = self._estimate_script_seconds(text)
        video_seconds = max(0.0, self._video_duration)

        base_style = "background:#313244; border-radius:4px; padding:6px; font-size:12px;"
        if not words:
            if video_seconds > 0:
                msg = f"Video: {_fmt_duration(video_seconds)} | Dan truyen/script vao de tinh thoi luong doc."
            else:
                msg = "Dan truyen/script vao day de tinh thoi luong doc."
            self.lbl_story_stats.setText(msg)
            self.lbl_story_stats.setStyleSheet(base_style + " color:#89b4fa;")
            return

        selected_rate = self.spn_rate.value()
        base = (
            f"Video: {_fmt_duration(video_seconds)} | "
            f"Lời đọc ước tính: {_fmt_duration(est_seconds)} ở {selected_rate:.2f}x"
        )
        if video_seconds <= 0:
            self.lbl_story_stats.setText(
                f"Lời đọc ước tính: {_fmt_duration(est_seconds)} | "
                "Chưa có thời lượng video."
            )
            self.lbl_story_stats.setStyleSheet(base_style + " color:#89b4fa;")
            return

        fastest_seconds = words / (
            self._words_per_second() * _VOICE_RATE_HARD_MAX
        )
        overflow_seconds = fastest_seconds - video_seconds
        if overflow_seconds <= 0:
            msg = (
                f"{base} | Có thể ghép trong giới hạn "
                f"{_VOICE_RATE_MIN:.2f}x–{_VOICE_RATE_HARD_MAX:.2f}x."
            )
            color = "#a6e3a1"
        else:
            msg = (
                f"{base} | Ở tốc độ tối đa {_VOICE_RATE_HARD_MAX:.2f}x "
                f"vẫn dư khoảng {overflow_seconds:.2f} giây."
            )
            color = "#f38ba8"
        self.lbl_story_stats.setText(msg)
        self.lbl_story_stats.setStyleSheet(base_style + f" color:{color};")

    def _on_voice_mode_changed(self, _=None):
        if not hasattr(self, "txt_script"):
            return
        mode = self.cmb_voice_mode.currentData() or "parts"
        is_dub = mode == "dub_timed_manual"
        self.spn_rate.setRange(_VOICE_RATE_MIN, _VOICE_RATE_MAX)
        form = self.cmb_dub_ai.parentWidget().layout()
        if isinstance(form, QFormLayout):
            form.setRowVisible(self.cmb_dub_ai, False)
            form.setRowVisible(self.txt_dub_model, False)
            form.setRowVisible(self.cmb_dub_whisper, is_dub)
        self.btn_scan_chinese.setVisible(is_dub)
        self.btn_auto_dub.setVisible(is_dub)
        self.btn_gen_voice.setVisible(not is_dub)
        # Timed dubbing already creates voice and merges the working video in
        # one operation. Hiding the regular merge button avoids presenting a
        # second, disabled button that looks like a blocked required step.
        self.btn_merge.setVisible(not is_dub)
        self.txt_script.setReadOnly(False)
        if is_dub:
            self.txt_script.setPlaceholderText(
                "Bấm Quét lời Trung để lấy dạng [0.0s] câu tiếng Trung.\n"
                "Sau đó tự dịch/sửa thành tiếng Việt nhưng giữ nguyên các mốc [giây]."
            )
        elif mode == "story":
            self.txt_script.setPlaceholderText(
                "Dan truyen/script co san vao day. App se tinh so tu, thoi gian doc va so voi video."
            )
        else:
            self.txt_script.setPlaceholderText("Nhap script review phim tai day...")
        self._update_story_stats()

    def _on_dub_ai_changed(self, _=None):
        defaults = {
            "gemini": "gemini-2.5-flash-lite",
            "groq": "llama-3.3-70b-versatile",
            "openrouter": "google/gemini-2.0-flash-exp:free",
            "ollama": "llama3.1:8b",
        }
        self.txt_dub_model.setText(
            defaults.get(self.cmb_dub_ai.currentData(), "")
        )

    def load_project(self, project: Project):
        self._audio_player.stop()
        self._project = project
        self._generated_audio = ""
        self.txt_script.clear()
        self.btn_preview_voice.setEnabled(False)
        self.btn_stop_voice.setEnabled(False)
        self.btn_merge.setEnabled(False)
        self.lbl_effective_rate.setText("Nhịp khớp video: chờ tạo voice")
        self._video_duration = 0.0
        enabled_clips = [clip for clip in project.clips if clip.enabled]
        if enabled_clips:
            self._video_duration = sum(clip.duration for clip in enabled_clips)
        ff = ffmpeg_path()
        source_video = getattr(project, "source_video", "") or ""
        if self._video_duration <= 0 and ff and source_video and Path(source_video).exists():
            self._video_duration = probe_media_duration(source_video, ff)
        if self._video_duration <= 0:
            self._video_duration = float(project.video_metadata.get("duration", 0) or 0)
        if self._video_duration <= 0 and project.clips:
            self._video_duration = sum(c.duration for c in project.clips)
        if project.voiceover_script:
            self.txt_script.setPlainText(project.voiceover_script)
        audio_is_current = (
            project.voiceover_audio
            and Path(project.voiceover_audio).exists()
            and bool((project.voiceover_script or "").strip())
            and getattr(project, "voice_scene_revision", -1)
            == getattr(project, "scene_revision", 0)
            and bool(getattr(project, "voice_alignment_ok", False))
        )
        if audio_is_current:
            self._generated_audio = project.voiceover_audio
            self.btn_preview_voice.setEnabled(True)
            self.btn_stop_voice.setEnabled(True)
            self.btn_merge.setEnabled(True)
            self._audio_player.setSource(QUrl.fromLocalFile(project.voiceover_audio))
        elif project.voiceover_audio and Path(project.voiceover_audio).exists():
            self.txt_log.append(
                "⚠️ Audio cũ không còn khớp revision cảnh/kịch bản; "
                "hãy tạo lại trước khi ghép."
            )
        self._refresh_dep()
        self._update_story_stats()

    # ─── Internal ─────────────────────────────────────────────────

    def _refresh_dep(self):
        edge_ok = is_edge_tts_available()
        nghi_ok = is_nghitts_available()
        vox_ok, vox_detail = get_voxcpm_runtime_status()
        omni_ok = is_omnivoice_running()
        parts = []
        if edge_ok:
            parts.append("✅ edge-tts")
        else:
            parts.append("❌ edge-tts (pip install edge-tts)")
        parts.append(
            "✅ NGHI-TTS local" if nghi_ok
            else "❌ NGHI-TTS (pip install piper-tts)"
        )
        if vox_ok:
            version = vox_detail.split(" — ", 1)[0]
            parts.append(f"✅ {version}")
        else:
            parts.append("⚪ VoxCPM (chạy install_voxcpm.bat)")
        self.lbl_dep.setToolTip(vox_detail)
        if omni_ok:
            parts.append("✅ OmniVoice Studio")
        else:
            parts.append("⚪ OmniVoice Studio (chưa chạy)")
        self.lbl_dep.setText("  |  ".join(parts))
        self.lbl_dep.setStyleSheet(
            "color: #a6e3a1;"
            if (edge_ok or nghi_ok or vox_ok or omni_ok)
            else "color: #f38ba8;"
        )

    def _populate_quick_voices(self):
        self.cmb_voice.clear()
        for voice_id, label in VI_VOICES:
            self.cmb_voice.addItem(label, voice_id)
        for voice_id, label in EN_VOICES:
            self.cmb_voice.addItem(label, voice_id)

    def _load_all_voices(self):
        if not is_edge_tts_available():
            self.txt_log.append(
                "❌ edge-tts chưa được cài trong venv. Hãy chạy lại run.bat "
                "hoặc cài requirements.txt."
            )
            return
        self.btn_gen_voice.setEnabled(False)
        self.btn_refresh_voices.setEnabled(False)
        self.txt_log.append("⏳ Đang tải danh sách giọng từ edge-tts...")

        class _FetchWorker(QThread):
            done = pyqtSignal(list, str)
            def run(self_inner):
                try:
                    voices = get_edge_tts_voices(raise_errors=True)
                    self_inner.done.emit(voices, "")
                except Exception as exc:
                    self_inner.done.emit([], str(exc))

        self._fetch_worker = _FetchWorker()
        self._fetch_worker.done.connect(self._on_voices_loaded)
        self._fetch_worker.start()

    def _on_voices_loaded(self, voices: list, error: str = ""):
        self.btn_gen_voice.setEnabled(True)
        self.btn_refresh_voices.setEnabled(True)
        if not voices:
            detail = error or "Dịch vụ Edge TTS không trả về dữ liệu."
            self.txt_log.append(f"❌ Không tải được danh sách giọng: {detail}")
            return
        self._voices = voices
        self.cmb_voice.clear()
        for v in voices:
            display = f"{v['ShortName']} — {v['FriendlyName']}"
            self.cmb_voice.addItem(display, v["ShortName"])
        self.txt_log.append(f"✅ Đã tải {len(voices)} giọng.")

    def _on_provider_changed(self, idx: int):
        provider = self.cmb_provider.currentData()
        self._refresh_dep()
        is_edge = provider == "edge_tts"
        is_nghi = provider == "nghitts"
        is_vox = provider == "voxcpm"
        self.cmb_voice.setEnabled(is_edge or is_nghi)
        self.btn_refresh_voices.setVisible(is_edge)
        self.btn_quick_preview.setVisible(is_edge or is_nghi)
        form = self.cmb_voice.parentWidget().layout()
        if isinstance(form, QFormLayout):
            form.setRowVisible(self.cmb_voice, is_edge or is_nghi)
            form.setRowVisible(self.btn_refresh_voices, is_edge)
            form.setRowVisible(self.btn_quick_preview, is_edge or is_nghi)
        if is_nghi:
            self.cmb_voice.clear()
            for name in NGHITTS_VOICES:
                self.cmb_voice.addItem(name, name)
        elif is_edge:
            self._populate_quick_voices()
        self.voxcpm_group.setVisible(is_vox)
        if is_vox:
            self._on_voxcpm_mode_changed()
            if not is_voxcpm_available():
                _, detail = get_voxcpm_runtime_status()
                self.txt_log.append(
                    "⚠️ Runtime này chưa thấy VoxCPM.\n"
                    f"{detail}\n"
                    "Hãy đóng cửa sổ này và chạy voxcpm.bat."
                )
        if provider == "omnivoice":
            if not is_omnivoice_running():
                self.txt_log.append("⚠️ OmniVoice Studio chưa chạy tại localhost:8000")
        self._update_story_stats()

    def _on_voxcpm_mode_changed(self, _=None):
        if not hasattr(self, "cmb_voxcpm_mode"):
            return
        mode = self.cmb_voxcpm_mode.currentData() or "design"
        needs_reference = mode in {"clone", "hifi"}
        needs_transcript = mode == "hifi"
        self.txt_voxcpm_reference.setEnabled(needs_reference)
        self.btn_voxcpm_reference.setEnabled(needs_reference)
        self.txt_voxcpm_prompt.setEnabled(needs_transcript)
        # Upstream ignores style control in Hi-Fi continuation mode.
        self.txt_voxcpm_style.setEnabled(not needs_transcript)

    def _browse_voxcpm_reference(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Chọn audio mẫu cho VoxCPM",
            "",
            "Audio (*.wav *.flac *.mp3 *.m4a *.ogg);;Tất cả file (*)",
        )
        if path:
            self.txt_voxcpm_reference.setText(path)

    def _build_voice_scenes(self, script_text: str) -> list[dict]:
        """Map narration sections to enabled clips in chronological order."""
        sections = split_voiceover_sections(script_text)
        clips = [
            clip for clip in (self._project.clips if self._project else [])
            if clip.enabled
        ]

        def clean(text: str) -> str:
            text = strip_voiceover_part_labels(text)
            return (
                preprocess_narration_text(text)
                if self.chk_narration_mode.isChecked()
                else text
            )

        self._scene_mapping_error = ""
        if not clips:
            duration = max(0.1, self._video_duration)
            return [{"index": 1, "duration": duration, "text": clean(script_text)}]

        # Story mode is intentionally one continuous narration take. It never
        # substitutes stale per-clip AI scripts for the text currently pasted.
        if (self.cmb_voice_mode.currentData() or "parts") == "story":
            return [{
                "index": 1,
                "duration": sum(clip.duration for clip in clips),
                "text": clean(script_text),
            }]

        if clips and len(sections) == len(clips):
            pairs = zip(clips, sections)
            return [
                {
                    "index": clip.index,
                    "duration": clip.duration,
                    "text": clean(section),
                }
                for clip, section in pairs
            ]

        if clips and len(sections) == 1 and all(
            (clip.voiceover_script or "").strip() for clip in clips
        ) and (
            self._project
            and script_text.strip() == (self._project.voiceover_script or "").strip()
        ):
            return [
                {
                    "index": clip.index,
                    "duration": clip.duration,
                    "text": clean(clip.voiceover_script),
                }
                for clip in clips
            ]

        self._scene_mapping_error = (
            f"Kịch bản có {len(sections)} phần nhưng đang chọn {len(clips)} cảnh. "
            "Hãy tạo/lưu đúng một phần kịch bản cho mỗi cảnh; app không tự gộp "
            "toàn bộ lời đọc vì sẽ làm lệch phân cảnh."
        )
        return []

    def _generate_voice(self):
        text = self.txt_script.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "Thiếu script", "Nhập script trước khi tạo giọng.")
            return
        provider = self.cmb_provider.currentData() or "edge_tts"
        if provider == "edge_tts" and not is_edge_tts_available():
            QMessageBox.warning(
                self, "Thiếu thư viện",
                "edge-tts chưa cài.\nChạy: pip install edge-tts"
            )
            return
        if provider == "nghitts" and not is_nghitts_available():
            QMessageBox.warning(
                self, "Thiếu thư viện",
                "Giọng Ngọc Huyền cần piper-tts. Hãy chạy lại run.bat để cài."
            )
            return
        if provider == "voxcpm" and not is_voxcpm_available():
            QMessageBox.warning(
                self,
                "Thiếu VoxCPM",
                "VoxCPM chưa được cài trong môi trường của app.\n\n"
                "Chạy file install_voxcpm.bat, chờ cài xong rồi mở lại app.",
            )
            return

        ff = ffmpeg_path()
        if not ff:
            QMessageBox.warning(
                self, "Thiếu FFmpeg",
                "Cần FFmpeg để đo và căn voice theo từng phân cảnh."
            )
            return

        scenes = self._build_voice_scenes(text)
        if not scenes:
            QMessageBox.warning(
                self,
                "Kịch bản chưa khớp phân cảnh",
                self._scene_mapping_error
                or "Không thể ghép kịch bản với danh sách cảnh đang chọn.",
            )
            return
        spoken_text = " ".join(scene["text"] for scene in scenes)
        estimated_seconds = self._estimate_script_seconds(spoken_text)
        video_seconds = max(0.0, self._video_duration)

        violations = _scene_budget_violations(
            scenes,
            self._words_per_second(),
            _VOICE_RATE_HARD_MAX,
        )
        if violations:
            details = "\n".join(
                f"Cảnh {item['index']}: dự tính {item['estimated_seconds']:.2f}s / "
                f"video {item['duration']:.2f}s — dư khoảng "
                f"{item['overflow_seconds']:.2f}s."
                for item in violations
            )
            total_overflow = sum(item["overflow_seconds"] for item in violations)
            self.txt_log.append(
                "❌ Dừng trước khi tạo voice: kịch bản không thể khớp video "
                f"trong giới hạn tối đa {_VOICE_RATE_HARD_MAX:.2f}x.\n"
                + details
            )
            QMessageBox.warning(
                self,
                "Kịch bản quá dài",
                f"Chưa tạo voice vì cần tốc độ vượt {_VOICE_RATE_HARD_MAX:.2f}x.\n\n"
                f"Lời đọc dự tính dư tổng cộng khoảng {total_overflow:.2f} giây. "
                "Hãy rút ngắn nội dung tương ứng rồi thử lại.\n\n"
                f"{details}",
            )
            self._update_story_stats()
            return

        voice = self.cmb_voice.currentData() or "vi-VN-HoaiMyNeural"
        rate_factor = self.spn_rate.value()
        vol_val = self.spn_vol.value()
        rate_pct = int(round((rate_factor - 1.0) * 100))
        rate_str = f"+{rate_pct}%" if rate_pct >= 0 else f"{rate_pct}%"
        vol_str = f"+{int(vol_val)}%" if vol_val >= 0 else f"{int(vol_val)}%"
        voxcpm_options = {"volume_percent": vol_val}
        if provider == "voxcpm":
            vox_mode = self.cmb_voxcpm_mode.currentData() or "design"
            reference = self.txt_voxcpm_reference.text().strip()
            prompt_text = self.txt_voxcpm_prompt.text().strip()
            if vox_mode in {"clone", "hifi"} and not reference:
                QMessageBox.warning(
                    self,
                    "Thiếu audio mẫu",
                    "Chế độ clone cần một file audio giọng mẫu dài khoảng 5–30 giây.",
                )
                return
            if reference and not Path(reference).is_file():
                QMessageBox.warning(
                    self,
                    "Audio mẫu không tồn tại",
                    f"Không tìm thấy file:\n{reference}",
                )
                return
            if vox_mode == "hifi" and not prompt_text:
                QMessageBox.warning(
                    self,
                    "Thiếu transcript mẫu",
                    "Hi-Fi clone cần transcript chính xác của audio mẫu.",
                )
                return
            voxcpm_options = {
                "model_id": (
                    self.txt_voxcpm_model.text().strip()
                    or VOXCPM_DEFAULT_MODEL
                ),
                "device": self.cmb_voxcpm_device.currentData() or "auto",
                "reference_wav_path": (
                    reference if vox_mode in {"clone", "hifi"} else ""
                ),
                "prompt_text": prompt_text if vox_mode == "hifi" else "",
                "voice_description": (
                    self.txt_voxcpm_style.text().strip()
                    if vox_mode != "hifi" else ""
                ),
                "cfg_value": self.spn_voxcpm_cfg.value(),
                "inference_timesteps": self.spn_voxcpm_steps.value(),
                "normalize": self.chk_voxcpm_normalize.isChecked(),
                "denoise": (
                    self.chk_voxcpm_denoise.isChecked()
                    and vox_mode in {"clone", "hifi"}
                ),
                "optimize": True,
                "volume_percent": vol_val,
            }

        if self._project:
            out_dir = str(Path(self._project.output_dir) / "audio")
        else:
            out_dir = "output/_voice"
        ensure_dir(out_dir)
        out_path = str(Path(out_dir) / "voiceover.mp3")

        self.btn_gen_voice.setEnabled(False)
        self._generated_audio = ""
        self.btn_preview_voice.setEnabled(False)
        self.btn_stop_voice.setEnabled(False)
        self.btn_merge.setEnabled(False)
        if self._project:
            self._project.voiceover_script = text
            self._project.voiceover_audio = ""
            self._project.review_voice_video = ""
            self._project.narration_transcript_file = ""
            self._project.narration_subtitle_file = ""
            self._project.final_video = ""
            self._project.voice_alignment_ok = False
            self._project.voice_rate_min = 0.0
            self._project.voice_rate_max = 0.0
            save_project(self._project)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.progress_bar.setFormat("Đang tạo giọng… %p%")
        self.txt_log.clear()
        words = _count_words(spoken_text)
        self.txt_log.append(
            f"📏 Script: {words:,} từ | Ước tính đọc: {_fmt_duration(estimated_seconds)}"
        )
        self.txt_log.append(
            f"🎚 Tốc độ nền: {rate_factor:.2f}x | "
            f"Căn tự động: {len(scenes)} phân cảnh"
        )
        if provider == "voxcpm":
            self.txt_log.append(
                "🎙 Engine: VoxCPM2 local. Tốc độ chính xác sẽ được FFmpeg "
                "căn riêng sau khi model tạo từng cảnh."
            )
            if vox_mode == "design" and len(scenes) > 1:
                self.txt_log.append(
                    "⚠️ Voice Design có thể đổi nhẹ âm sắc giữa các lần tạo. "
                    "Muốn một giọng tuyệt đối nhất quán qua nhiều cảnh, hãy "
                    "chọn Clone giọng và dùng cùng một audio mẫu."
                )
        elif provider == "nghitts":
            self.txt_log.append(
                "🎙 Engine: NGHI-TTS/Piper chạy local. Model chỉ tải một lần; "
                "các lần tạo sau không cần gọi dịch vụ giọng nói."
            )
        if video_seconds > 0:
            self.txt_log.append(f"🎞 Video: {_fmt_duration(video_seconds)}")
            if estimated_seconds > video_seconds * 1.05:
                max_words = self._max_words_for_video()
                self.txt_log.append(
                    f"⚠️ Script dài hơn video khoảng {_fmt_duration(estimated_seconds - video_seconds)}. "
                    f"Gợi ý {_fmt_duration(video_seconds)}: "
                    f"{self._recommended_rate_factor(spoken_text):.2f}x "
                    f"hoặc rút còn ~{max_words:,} từ."
                )

        self._voice_worker = _VoiceGenWorker(
            scenes,
            voice,
            out_path,
            rate_str,
            vol_str,
            ff,
            rate_factor,
            provider=provider,
            voxcpm_options=voxcpm_options,
        )
        self._voice_worker.log.connect(self._on_log)
        self._voice_worker.progress.connect(self._on_voice_progress)
        self._voice_worker.finished.connect(self._on_voice_done)
        self._voice_worker.start()

    def _on_log(self, line: str):
        self.txt_log.append(line)
        sb = self.txt_log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _on_voice_progress(self, pct: int):
        self.progress_bar.setValue(pct)
        self.progress_bar.setFormat(f"Đang tạo giọng… {pct}%")

    def _on_voice_done(self, ok: bool, result: str):
        self.btn_gen_voice.setEnabled(True)
        self.progress_bar.setValue(100 if ok else 0)
        self.progress_bar.setFormat("✅ Hoàn tất!" if ok else "❌ Lỗi")
        # Hide after 3 seconds
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(3000, lambda: self.progress_bar.setVisible(False))
        adjustments = getattr(self._voice_worker, "adjustments", [])
        if adjustments:
            self._save_voice_pace_measurement(adjustments)
            self._update_story_stats()
        if ok:
            self._generated_audio = result
            self.btn_preview_voice.setEnabled(True)
            self.btn_stop_voice.setEnabled(True)
            self.btn_merge.setEnabled(True)
            # Pre-load into player
            self._audio_player.setSource(QUrl.fromLocalFile(result))
            if self._project:
                self._project.voiceover_audio = result
                self._project.voice_scene_revision = self._project.scene_revision
                self._project.workflow_stage = "voice_audio"
                self._project.voice_alignment_ok = False
                self._project.voice_rate_min = 0.0
                self._project.voice_rate_max = 0.0
                ff = ffmpeg_path()
                if ff:
                    voice_dur = probe_media_duration(result, ff)
                    video_dur = self._video_duration
                    if voice_dur > 0 and video_dur > 0:
                        self.txt_log.append(
                            f"⏱ Voice đã căn: {voice_dur:.2f}s | "
                            f"Video review: {video_dur:.2f}s"
                        )
                if adjustments:
                    raw_total = sum(item["source_duration"] for item in adjustments)
                    target_total = sum(item["target_duration"] for item in adjustments)
                    base_rate = getattr(self._voice_worker, "rate_factor", 1.0)
                    effective = (
                        base_rate * raw_total / target_total
                        if target_total > 0 else base_rate
                    )
                    scene_rates = [
                        base_rate * item["tempo_factor"] for item in adjustments
                    ]
                    self._project.voice_rate_min = min(scene_rates)
                    self._project.voice_rate_max = max(scene_rates)
                    self.lbl_effective_rate.setText(
                        f"Nhịp khớp video: ~{effective:.2f}x tổng thể | "
                        f"từng cảnh {min(scene_rates):.2f}x–{max(scene_rates):.2f}x"
                    )
                    self.txt_log.append(
                        f"✅ Tốc độ thực tế sau căn cảnh: ~{effective:.2f}x "
                        f"(không cắt mất câu cuối)."
                    )
                    unsafe = [
                        rate for rate in scene_rates
                        if rate < _VOICE_RATE_MIN - 0.001
                        or rate > _VOICE_RATE_HARD_MAX + 0.001
                    ]
                    within_grace = [
                        rate for rate in scene_rates
                        if _VOICE_RATE_MAX + 0.001 < rate
                        <= _VOICE_RATE_HARD_MAX + 0.001
                    ]
                    if unsafe:
                        self.btn_merge.setEnabled(False)
                        self.txt_log.append(
                            f"❌ Có cảnh cần tốc độ ngoài {_VOICE_RATE_MIN:.2f}x–"
                            f"{_VOICE_RATE_HARD_MAX:.2f}x. Hãy chỉnh độ dài kịch bản "
                            "rồi tạo lại; app không tăng tốc quá giới hạn."
                        )
                    else:
                        self._project.voice_alignment_ok = True
                        if within_grace:
                            self.txt_log.append(
                                f"⚠️ Sai số căn nhỏ: tốc độ thực tế vượt "
                                f"{_VOICE_RATE_MAX:.2f}x nhưng không quá "
                                f"{_VOICE_RATE_HARD_MAX:.2f}x; vẫn cho phép ghép."
                            )
                if not self._project.voice_alignment_ok:
                    self.btn_merge.setEnabled(False)
                save_project(self._project)
        else:
            QMessageBox.warning(self, "Lỗi TTS", f"Tạo giọng thất bại:\n{result}")

    def _quick_preview_voice(self):
        provider = self.cmb_provider.currentData() or "edge_tts"
        if provider not in {"edge_tts", "nghitts"}:
            return
        if provider == "edge_tts" and not is_edge_tts_available():
            QMessageBox.warning(self, "Thiếu thư viện", "Chưa cài edge-tts.")
            return
        if provider == "nghitts" and not is_nghitts_available():
            QMessageBox.warning(
                self, "Thiếu thư viện",
                "Giọng Ngọc Huyền cần piper-tts. Hãy chạy lại run.bat để cài."
            )
            return

        voice = self.cmb_voice.currentData() or "vi-VN-HoaiMyNeural"
        rate_factor = self.spn_rate.value()
        volume = self.spn_vol.value()
        safe_voice = re.sub(r"[^A-Za-z0-9_-]+", "_", str(voice))
        suffix = ".wav" if provider == "nghitts" else ".mp3"
        cache_dir = Path("data") / "voice_previews"
        ensure_dir(cache_dir)
        cache_path = cache_dir / (
            f"{provider}_{safe_voice}_r{int(rate_factor * 100)}_v{int(volume)}{suffix}"
        )
        if cache_path.is_file() and cache_path.stat().st_size > 1000:
            self._play_audio_file(str(cache_path.resolve()))
            return

        self.btn_quick_preview.setEnabled(False)
        self.btn_quick_preview.setText("⏳ Đang tạo mẫu…")
        if provider == "nghitts":
            self.txt_log.append(
                "⏳ Đang chuẩn bị nghe thử Ngọc Huyền. Lần đầu cần tải model; "
                "những lần sau app phát file đã lưu."
            )
        self._quick_preview_worker = _QuickPreviewWorker(
            provider,
            voice,
            str(cache_path.resolve()),
            rate_factor,
            volume,
        )
        self._quick_preview_worker.finished.connect(self._on_quick_preview_done)
        self._quick_preview_worker.start()

    def _on_quick_preview_done(self, ok: bool, result: str):
        self.btn_quick_preview.setEnabled(True)
        self.btn_quick_preview.setText("🔊 Nghe thử nhanh")
        if not ok:
            self.txt_log.append(f"❌ Không tạo được bản nghe thử: {result}")
            QMessageBox.warning(self, "Lỗi nghe thử", result)
            return
        self._quick_preview_audio = result
        self.txt_log.append("✅ Bản nghe thử đã sẵn sàng và được lưu cho lần sau.")
        self._play_audio_file(result)

    def _scan_chinese_timeline(self):
        if not self._project:
            QMessageBox.warning(self, "Chưa có project", "Hãy tạo project từ video tiếng Trung trước.")
            return
        source = (
            getattr(self._project, "original_source_video", "")
            or getattr(self._project, "source_video", "")
        )
        if not source or not Path(source).is_file():
            QMessageBox.warning(self, "Thiếu video", "Không tìm thấy video nguồn tiếng Trung.")
            return
        ff = ffmpeg_path()
        if not ff:
            QMessageBox.warning(self, "Thiếu FFmpeg", "Cần FFmpeg để quét lời thoại.")
            return
        current = self.txt_script.toPlainText().strip()
        if current:
            backup = Path(self._project.output_dir) / "dubbing_script_backup.txt"
            backup.write_text(current, encoding="utf-8")
            self.txt_log.append(f"💾 Đã sao lưu nội dung đang sửa: {backup}")
        self.btn_scan_chinese.setEnabled(False)
        self.btn_auto_dub.setEnabled(False)
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setFormat("Đang quét lời Trung…")
        self.progress_bar.setVisible(True)
        self._scan_worker = _ChineseScanWorker(
            source, self._project.output_dir,
            self.cmb_dub_whisper.currentText(), ff,
        )
        self._scan_worker.log.connect(self._on_log)
        self._scan_worker.finished.connect(self._on_chinese_scan_done)
        self._scan_worker.start()

    def _on_chinese_scan_done(self, ok: bool, result):
        self.btn_scan_chinese.setEnabled(True)
        self.btn_auto_dub.setEnabled(True)
        self.progress_bar.setVisible(False)
        if not ok:
            QMessageBox.warning(self, "Lỗi quét lời Trung", str(result))
            return
        lines = [
            f"[{float(segment.get('start', 0)):.1f}s] {(segment.get('text') or '').strip()}"
            for segment in result.get("segments", [])
            if (segment.get("text") or "").strip()
        ]
        self.txt_script.setPlainText("\n".join(lines))
        transcript_path = str(
            Path(self._project.output_dir) / "dubbing_transcript" / "transcript.json"
        )
        self._project.source_transcript_file = transcript_path
        self._project.transcript_file = transcript_path
        self._project.transcript_language = "zh"
        self._project.voiceover_script = "\n".join(lines)
        save_project(self._project)
        self.txt_log.append(
            "✅ Đã quét xong. Hãy dịch/sửa từng dòng sang tiếng Việt và giữ nguyên [mốc giây]."
        )

    def _start_auto_dubbing(self):
        if not self._project:
            QMessageBox.warning(self, "Chưa có project", "Hãy tạo project từ video tiếng Trung trước.")
            return
        source = (
            getattr(self._project, "original_source_video", "")
            or getattr(self._project, "source_video", "")
        )
        if not source or not Path(source).is_file():
            QMessageBox.warning(self, "Thiếu video", "Không tìm thấy video nguồn tiếng Trung.")
            return
        ff = ffmpeg_path()
        if not ff:
            QMessageBox.warning(self, "Thiếu FFmpeg", "Cần FFmpeg để lồng tiếng theo timeline.")
            return
        duration = self._video_duration or probe_media_duration(source, ff)
        try:
            manual_segments = parse_timed_dubbing_script(
                self.txt_script.toPlainText(), duration
            )
        except ValueError as exc:
            QMessageBox.warning(
                self, "Sai định dạng mốc thời gian",
                f"{exc}\n\nVí dụ: [0.0s] Nội dung tiếng Việt",
            )
            return
        if not self._validate_dubbing_budget(manual_segments, duration):
            return
        voice_provider = self.cmb_provider.currentData() or "edge_tts"
        voice = self.cmb_voice.currentData() or "vi-VN-HoaiMyNeural"
        if voice_provider == "edge_tts" and not is_edge_tts_available():
            QMessageBox.warning(self, "Thiếu edge-tts", "Hãy chạy lại run.bat để cài edge-tts.")
            return
        if voice_provider == "nghitts" and not is_nghitts_available():
            QMessageBox.warning(self, "Thiếu Piper", "Hãy chạy lại run.bat để cài piper-tts.")
            return
        voxcpm_options = {}
        if voice_provider == "voxcpm":
            if not is_voxcpm_available():
                QMessageBox.warning(self, "Thiếu VoxCPM", "VoxCPM chưa được cài trong app.")
                return
            mode = self.cmb_voxcpm_mode.currentData() or "design"
            reference = self.txt_voxcpm_reference.text().strip()
            prompt = self.txt_voxcpm_prompt.text().strip()
            if mode in {"clone", "hifi"} and not Path(reference).is_file():
                QMessageBox.warning(self, "Thiếu audio mẫu", "Chọn audio mẫu hợp lệ cho VoxCPM.")
                return
            voxcpm_options = {
                "model_id": self.txt_voxcpm_model.text().strip() or VOXCPM_DEFAULT_MODEL,
                "device": self.cmb_voxcpm_device.currentData() or "auto",
                "reference_wav_path": reference if mode in {"clone", "hifi"} else "",
                "prompt_text": prompt if mode == "hifi" else "",
                "voice_description": self.txt_voxcpm_style.text().strip() if mode != "hifi" else "",
                "cfg_value": self.spn_voxcpm_cfg.value(),
                "inference_timesteps": self.spn_voxcpm_steps.value(),
                "normalize": self.chk_voxcpm_normalize.isChecked(),
                "denoise": self.chk_voxcpm_denoise.isChecked(),
                "optimize": True,
                "volume_percent": self.spn_vol.value(),
            }
        transcript_path = (
            getattr(self._project, "source_transcript_file", "")
            or getattr(self._project, "transcript_file", "")
        )
        ensure_dir(self._project.output_dir)
        timed_script = self.txt_script.toPlainText().strip()
        (Path(self._project.output_dir) / "dubbing_vi_timed.txt").write_text(
            timed_script, encoding="utf-8"
        )
        self._project.voiceover_script = timed_script
        save_project(self._project)
        self.btn_auto_dub.setEnabled(False)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Đang lồng tiếng… %p%")
        self.progress_bar.setVisible(True)
        self.txt_log.clear()
        self.txt_log.append(
            f"🎬 Đã nhận {len(manual_segments)} đoạn tự dịch: tạo giọng → "
            "căn đúng timestamp → ghép video. Không gọi model AI."
        )
        self._dubbing_worker = _DubbingWorker(
            source, self._project.output_dir, transcript_path,
            self.cmb_dub_whisper.currentText(),
            self.cmb_dub_ai.currentData() or "gemini",
            self.txt_dub_model.text().strip(),
            voice_provider, voice, self.spn_rate.value(), self.spn_vol.value(),
            ff, duration,
            self.sld_orig_vol.value() / 100.0,
            voxcpm_options, manual_segments,
        )
        self._dubbing_worker.log.connect(self._on_log)
        self._dubbing_worker.progress.connect(self._on_voice_progress)
        self._dubbing_worker.finished.connect(self._on_auto_dubbing_done)
        self._dubbing_worker.start()

    def _on_auto_dubbing_done(self, ok: bool, result):
        self.btn_auto_dub.setEnabled(True)
        self.progress_bar.setVisible(False)
        if not ok:
            self._update_story_stats()
            details = getattr(self, "_dubbing_budget_details", "")
            self.txt_log.append(details)
            error_text = str(result)
            is_tts_service_error = any(
                marker in error_text.lower()
                for marker in (
                    "no audio was received",
                    "không trả về audio",
                    "connect",
                    "timeout",
                    "websocket",
                )
            )
            box = QMessageBox(self)
            box.setWindowTitle(
                "Edge TTS tạm thời không trả audio"
                if is_tts_service_error else "Không thể hoàn tất ghép giọng"
            )
            box.setText(error_text)
            box.setInformativeText(
                "Video chưa được ghép xong nên tool chưa chuyển sang tab Phụ đề. "
                "Hãy bấm Tạo voice & ghép theo các mốc để thử lại."
                if is_tts_service_error else
                "Video chưa được ghép xong. Bấm Show Details để xem từng mốc."
            )
            box.setDetailedText(details)
            box.exec()
            return
        lines = [
            f"[{s['start']:.1f}s] {s['text_vi']}"
            for s in result["segments"]
        ]
        self.txt_script.setPlainText("\n".join(lines))
        self._generated_audio = result["audio"]
        self.btn_merge.setEnabled(True)
        self.btn_preview_voice.setEnabled(True)
        self.btn_stop_voice.setEnabled(True)
        self._audio_player.setSource(QUrl.fromLocalFile(result["audio"]))
        if self._project:
            self._project.voiceover_script = "\n".join(lines)
            self._project.voiceover_audio = result["audio"]
            self._project.review_voice_video = result["video"]
            self._project.workflow_mode = "dubbing_timed_manual"
            self._project.workflow_stage = "voice_video"
            self._project.voice_alignment_ok = True
            self._project.voice_scene_revision = self._project.scene_revision
            save_project(self._project)
        self.txt_log.append(f"✅ Đã ghép giọng vào video làm việc. Chuyển sang Phụ đề: {result['video']}")
        self.merged_video_ready.emit(result["video"])

    def _play_audio_file(self, audio_path: str):
        path = str(Path(audio_path).resolve())
        self._audio_player.stop()
        self._audio_player.setSource(QUrl.fromLocalFile(path))
        self._audio_player.play()

    def _setup_audio_player(self):
        self._audio_player = QMediaPlayer()
        self._audio_out = QAudioOutput()
        self._audio_out.setVolume(0.95)
        self._audio_player.setAudioOutput(self._audio_out)
        self._audio_player.positionChanged.connect(self._on_audio_position)
        self._audio_player.playbackStateChanged.connect(self._on_audio_state)

    def _toggle_audio_preview(self):
        if not self._generated_audio or not Path(self._generated_audio).exists():
            return
        state = self._audio_player.playbackState()
        if state == QMediaPlayer.PlaybackState.PlayingState:
            self._audio_player.pause()
        else:
            # Load if source changed
            if self._audio_player.source().toLocalFile() != self._generated_audio:
                self._audio_player.setSource(QUrl.fromLocalFile(self._generated_audio))
            self._audio_player.play()

    def _stop_audio_preview(self):
        self._audio_player.stop()

    def _on_audio_state(self, state):
        playing = state == QMediaPlayer.PlaybackState.PlayingState
        self.btn_preview_voice.setText("⏸ Dừng" if playing else "▶ Phát")

    def _on_audio_position(self, pos_ms: int):
        s = pos_ms // 1000
        self.lbl_audio_time.setText(f"{s // 60}:{s % 60:02d}")

    def _merge_voice(self):
        if not self._project:
            QMessageBox.warning(self, "Chưa có project", "Vui lòng tạo project trước.")
            return
        current_script = self.txt_script.toPlainText().strip()
        if (
            not current_script
            or current_script != (self._project.voiceover_script or "").strip()
        ):
            QMessageBox.warning(
                self,
                "Kịch bản đã thay đổi",
                "Kịch bản trên màn hình khác bản đã dùng để tạo audio. "
                "Hãy bấm Tạo giọng đọc lại trước khi ghép.",
            )
            return
        if (
            self._project.voice_scene_revision != self._project.scene_revision
            or not getattr(self._project, "voice_alignment_ok", False)
        ):
            QMessageBox.warning(
                self,
                "Audio không còn hợp lệ",
                "Audio không khớp revision cảnh hiện tại hoặc cần tốc độ quá cực đoan. "
                "Hãy tạo lại giọng đọc.",
            )
            return
        if not self._generated_audio or not Path(self._generated_audio).exists():
            QMessageBox.warning(self, "Chưa có giọng", "Tạo giọng trước khi ghép.")
            return

        source_candidates = [
            getattr(self._project, "original_source_video", "") or "",
            getattr(self._project, "source_video", "") or "",
            *[
                getattr(clip, "source_video", "") or ""
                for clip in self._project.clips
            ],
        ]
        if not any(path and Path(path).exists() for path in source_candidates):
            QMessageBox.warning(
                self,
                "Không tìm thấy video nguồn",
                "Project không còn file nguồn để dựng video review sạch subtitle.",
            )
            return

        out_dir = str(Path(self._project.output_dir) / "final")
        ensure_dir(out_dir)
        out_path = str(Path(out_dir) / f"{self._project.name}_review.mp4")
        clean_video_path = str(
            Path(out_dir) / f"{self._project.name}_review_no_sub_base.mp4"
        )

        orig_vol = self.sld_orig_vol.value() / 100.0
        ff = ffmpeg_path()
        if not ff:
            QMessageBox.warning(self, "Thiếu FFmpeg", "Vui lòng tải/cài FFmpeg trước khi ghép video.")
            return
        self.btn_merge.setEnabled(False)
        self.progress_bar.setRange(0, 0)  # indeterminate for merge
        self.progress_bar.setFormat(
            "Đang dựng video review sạch và ghép voice..."
        )
        self.progress_bar.setVisible(True)

        self.txt_log.append(
            "🎬 Dựng lại video không chữ PART/subtitle, sau đó ghép voice đã căn cảnh."
        )

        self._merge_worker = _MergeWorker(
            self._project,
            clean_video_path,
            self._generated_audio,
            out_path,
            ff,
            orig_vol,
        )
        self._merge_worker.log.connect(self._on_log)
        self._merge_worker.finished.connect(self._on_merge_done)
        self._merge_worker.start()

    def _on_merge_done(self, ok: bool, result: str):
        self.btn_merge.setEnabled(True)
        self.progress_bar.setVisible(False)
        if ok:
            self.txt_log.append(f"✅ Video đã ghép giọng: {result}")
            self.merged_video_ready.emit(result)
        else:
            QMessageBox.warning(self, "Lỗi ghép", f"Ghép thất bại:\n{result}")

    def _save_script(self):
        if self._project:
            new_script = self.txt_script.toPlainText().strip()
            if new_script != (self._project.voiceover_script or "").strip():
                self._project.voiceover_audio = ""
                self._project.review_voice_video = ""
                self._project.narration_transcript_file = ""
                self._project.narration_subtitle_file = ""
                self._project.final_video = ""
                self._project.voice_alignment_ok = False
                self._generated_audio = ""
                self.btn_preview_voice.setEnabled(False)
                self.btn_stop_voice.setEnabled(False)
                self.btn_merge.setEnabled(False)
            self._project.voiceover_script = new_script
            self._project.script_scene_revision = self._project.scene_revision
            save_project(self._project)
            self.txt_log.append("💾 Đã lưu script vào project.")

    def _load_script_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Chọn file script", "", "Text (*.txt);;All Files (*)"
        )
        if path:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    self.txt_script.setPlainText(f.read())
            except Exception as e:
                QMessageBox.warning(self, "Lỗi", str(e))
