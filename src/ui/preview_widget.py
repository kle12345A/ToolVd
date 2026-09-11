"""Embedded video player widget using QMediaPlayer."""

import os
from pathlib import Path

from PyQt6.QtCore import Qt, QUrl, QTimer
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QSlider, QLabel, QSizePolicy,
)

from src.utils.file_utils import format_duration


class VideoPreviewWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._duration = 0.0
        self._setup_ui()
        self._setup_player()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.video_widget = QVideoWidget()
        self.video_widget.setMinimumSize(320, 180)
        self.video_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.video_widget.setStyleSheet("background: #1a1a1a;")
        layout.addWidget(self.video_widget)

        # Seek bar
        self.seek_slider = QSlider(Qt.Orientation.Horizontal)
        self.seek_slider.setRange(0, 1000)
        self.seek_slider.sliderMoved.connect(self._on_seek)
        layout.addWidget(self.seek_slider)

        # Controls row
        ctrl = QHBoxLayout()
        self.btn_play = QPushButton("▶ Phát")
        self.btn_play.setFixedWidth(90)
        self.btn_play.clicked.connect(self._toggle_play)

        self.btn_stop = QPushButton("⏹")
        self.btn_stop.setFixedWidth(40)
        self.btn_stop.clicked.connect(self._stop)

        self.lbl_time = QLabel("00:00 / 00:00")
        self.lbl_time.setMinimumWidth(130)

        self.btn_external = QPushButton("📂 Mở ngoài")
        self.btn_external.setFixedWidth(110)
        self.btn_external.clicked.connect(self._open_external)

        ctrl.addWidget(self.btn_play)
        ctrl.addWidget(self.btn_stop)
        ctrl.addWidget(self.lbl_time)
        ctrl.addStretch()
        ctrl.addWidget(self.btn_external)
        layout.addLayout(ctrl)

        self._current_path = ""

    def _setup_player(self):
        self.player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.player.setAudioOutput(self.audio_output)
        self.player.setVideoOutput(self.video_widget)
        self.audio_output.setVolume(1.0)

        self.player.positionChanged.connect(self._on_position_changed)
        self.player.durationChanged.connect(self._on_duration_changed)
        self.player.playbackStateChanged.connect(self._on_state_changed)
        self.player.errorOccurred.connect(self._on_error)

    def load_video(self, path: str):
        self._current_path = path
        self.player.stop()
        self.player.setSource(QUrl.fromLocalFile(path))
        self.seek_slider.setValue(0)
        self.lbl_time.setText("00:00 / 00:00")
        self.btn_play.setText("▶ Phát")

    def load_segment(self, path: str, start_sec: float, end_sec: float):
        """Load video and optionally seek to start."""
        self.load_video(path)
        self._segment_start = start_sec
        self._segment_end = end_sec
        # Seek after source loads
        QTimer.singleShot(300, lambda: self.player.setPosition(int(start_sec * 1000)))

    def _toggle_play(self):
        state = self.player.playbackState()
        if state == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def _stop(self):
        self.player.stop()

    def _on_seek(self, value: int):
        if self._duration > 0:
            pos_ms = int(value / 1000 * self._duration * 1000)
            self.player.setPosition(pos_ms)

    def _on_position_changed(self, pos_ms: int):
        if self._duration > 0:
            ratio = pos_ms / (self._duration * 1000)
            self.seek_slider.blockSignals(True)
            self.seek_slider.setValue(int(ratio * 1000))
            self.seek_slider.blockSignals(False)
        cur = pos_ms / 1000
        self.lbl_time.setText(
            f"{format_duration(cur)} / {format_duration(self._duration)}"
        )

    def _on_duration_changed(self, dur_ms: int):
        self._duration = dur_ms / 1000

    def _on_state_changed(self, state):
        if state == QMediaPlayer.PlaybackState.PlayingState:
            self.btn_play.setText("⏸ Dừng")
        else:
            self.btn_play.setText("▶ Phát")

    def _on_error(self, error, error_str):
        from src.utils.logger import logger
        logger.warning(f"Player error: {error_str}")

    def _open_external(self):
        if self._current_path and Path(self._current_path).exists():
            os.startfile(self._current_path)
