"""Tab 5: Export settings + live preview panel (with interactive crop tool)."""

from pathlib import Path

from PyQt6.QtCore import Qt, QRect, QThread, QTimer, QUrl, pyqtSignal
from PyQt6.QtGui import (
    QColor, QDesktopServices, QFont, QFontMetrics, QPainter, QPen, QPixmap,
)
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QPushButton, QComboBox, QCheckBox, QSpinBox, QDoubleSpinBox,
    QLineEdit, QFileDialog, QProgressBar, QTextEdit, QPlainTextEdit,
    QGroupBox, QMessageBox, QScrollArea, QSplitter, QSizePolicy, QSlider,
    QListWidget, QListWidgetItem,
)

from src.models.project import Project, ExportConfig, PLATFORMS, ASPECT_MODES, PART_TEXT_MODES, SUBTITLE_STYLES
from src.models.clip import Clip
from src.core.project_manager import save_project, save_export_history, save_captions
from src.core.video_manager import get_video_metadata
from src.core.video_processor import (
    _refine_delogo_region,
    generate_caption,
    is_nvenc_available,
)
from src.core.ai_client import (
    GEMINI_FREE_MODELS,
    GROQ_FREE_MODELS,
    OPENROUTER_FREE_MODELS,
    OLLAMA_SUGGESTED,
    generate_clip_titles,
    load_api_key,
)
from src.ui.workers import ExportWorker, ThumbnailWorker, WideThumbnailWorker


_COLORS = ["white", "yellow", "#ff6b6b", "#7aa2f7", "#a6e3a1", "black"]
_POSITIONS = {"top": "Trên cùng", "center": "Giữa", "bottom": "Dưới cùng"}

# Common Windows fonts that support Vietnamese characters
_SUBTITLE_FONTS = [
    ("Arial (mặc định)",          "Arial"),
    ("Times New Roman (serif)",    "Times New Roman"),
    ("Verdana (rộng, dễ đọc)",    "Verdana"),
    ("Tahoma (Windows UI)",        "Tahoma"),
    ("Calibri (hiện đại)",         "Calibri"),
    ("Segoe UI (Win 10/11)",       "Segoe UI"),
    ("Impact (đậm, kịch tính)",   "Impact"),
    ("Georgia (sang trọng)",       "Georgia"),
    ("Trebuchet MS (gọn gàng)",   "Trebuchet MS"),
    ("Comic Sans MS (vui tươi)",  "Comic Sans MS"),
    ("Courier New (mono)",         "Courier New"),
]

# Internal render size — 1/3 of export resolution (1080×1920), 9:16
_PW, _PH = 360, 640


class _AITitleWorker(QThread):
    status = pyqtSignal(str)
    finished = pyqtSignal(bool, object)

    def __init__(self, project, provider, model, use_bottom_title):
        super().__init__()
        self.project = project
        self.provider = provider
        self.model = model
        self.use_bottom_title = use_bottom_title

    def run(self):
        try:
            result = generate_clip_titles(
                self.project,
                self.provider,
                self.model,
                self.use_bottom_title,
                status_cb=lambda s: self.status.emit(s),
            )
            self.finished.emit(True, result)
        except Exception as e:
            self.finished.emit(False, str(e))


def _to_qcolor(name: str) -> QColor:
    _map = {"white": "#ffffff", "black": "#000000", "yellow": "#ffff00"}
    c = QColor(_map.get(name, name))
    return c if c.isValid() else QColor("#ffffff")


# ─── Preview label (interactive — draggable text) ─────────────────────────────

class _PreviewLabel(QLabel):
    """9:16 preview with draggable PART text & subtitle overlays.

    Text bounding boxes are registered by the parent via `set_text_rects()`.
    Dragging emits `text_dragged(name, y_pct)` where y_pct ∈ [0, 1].
    """
    text_dragged = pyqtSignal(str, float)   # "part" or "sub", y_pct 0–1

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background:#111; border-radius:6px;")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._raw: QPixmap | None = None
        # Draggable text tracking  (key → QRect in _PW×_PH coords)
        self._text_rects: dict[str, QRect] = {}
        self._dragging: str | None = None
        self._drag_offset_y: int = 0
        self.setMouseTracking(True)
        self._show_placeholder()

    def update_raw(self, px: QPixmap):
        self._raw = px
        self._rescale()

    def set_text_rects(self, rects: dict[str, QRect]):
        """Register bounding boxes of text labels in _PW×_PH coordinate space."""
        self._text_rects = rects

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._rescale()

    def _rescale(self):
        if self._raw and not self._raw.isNull():
            scaled = self._raw.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            super().setPixmap(scaled)

    def _show_placeholder(self):
        px = QPixmap(_PW, _PH)
        px.fill(QColor("#111"))
        p = QPainter(px)
        p.setPen(QColor("#555"))
        p.drawText(px.rect(), Qt.AlignmentFlag.AlignCenter, "Chưa có\nthumbnail")
        p.end()
        self._raw = px
        super().setPixmap(px)

    # ── coordinate helpers ────────────────────────────────────────────────

    def _widget_to_raw(self, pos):
        """Map widget pixel position → _PW×_PH raw coordinate."""
        pm = self.pixmap()
        if not pm or pm.isNull():
            return None
        # Pixmap is centred in widget
        ox = (self.width() - pm.width()) // 2
        oy = (self.height() - pm.height()) // 2
        rx = (pos.x() - ox) * _PW / max(1, pm.width())
        ry = (pos.y() - oy) * _PH / max(1, pm.height())
        return rx, ry

    def _hit_test(self, rx, ry) -> str | None:
        """Return the key of the text rect under (rx, ry), or None."""
        for key, rect in self._text_rects.items():
            # Expand hit area for easier grabbing (±12 px in raw coords)
            expanded = rect.adjusted(-12, -12, 12, 12)
            if expanded.contains(int(rx), int(ry)):
                return key
        return None

    # ── mouse events ─────────────────────────────────────────────────────

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return
        raw = self._widget_to_raw(event.pos())
        if not raw:
            return
        rx, ry = raw
        hit = self._hit_test(rx, ry)
        if hit:
            self._dragging = hit
            rect = self._text_rects[hit]
            self._drag_offset_y = int(ry - rect.center().y())
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event):
        raw = self._widget_to_raw(event.pos())
        if not raw:
            return
        rx, ry = raw
        if self._dragging:
            # Compute new y_pct (centre of text relative to canvas)
            new_cy = ry - self._drag_offset_y
            y_pct = max(0.0, min(1.0, new_cy / _PH))
            self.text_dragged.emit(self._dragging, y_pct)
        else:
            hit = self._hit_test(rx, ry)
            if hit:
                self.setCursor(Qt.CursorShape.OpenHandCursor)
            else:
                self.setCursor(Qt.CursorShape.ArrowCursor)

    def mouseReleaseEvent(self, event):
        if self._dragging:
            self._dragging = None
            self.setCursor(Qt.CursorShape.ArrowCursor)


# ─── Crop adjustment widget ────────────────────────────────────────────────────

class _CropAdjustWidget(QWidget):
    """Source preview with a movable and resizable selection box."""
    crop_changed = pyqtSignal(float, float, float, float)  # cx, cy, cw, ch

    def __init__(self, parent=None, *, lock_aspect: bool = True, label: str = ""):
        super().__init__(parent)
        self._wide: QPixmap | None = None
        self._cx = 0.5
        self._cy = 0.5
        self._cw = 1.0
        self._ch = 1.0
        self._aspect = 9 / 16
        self._lock_aspect = lock_aspect
        self._empty_label = label or "Chon custom crop de hien thi khung cat"
        self._drag_last = None
        self._drag_mode = ""
        self.setMinimumHeight(170)
        self.setMaximumHeight(260)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setStyleSheet("background:#111; border-radius:4px;")
        self.setToolTip("Keo trong khung de di chuyen. Keo canh/goc trang de thay doi vung cat.")
        self.setMouseTracking(True)

    def set_thumbnail(self, pix: QPixmap):
        self._wide = pix
        self._fit_crop_to_aspect()
        self.update()

    def set_aspect(self, w: int, h: int):
        self._aspect = max(0.05, float(w) / max(1.0, float(h)))
        self._fit_crop_to_aspect()
        self.update()

    def set_crop(self, cx: float, cy: float, cw: float = 1.0, ch: float = 1.0):
        self._cx = max(0.0, min(1.0, cx))
        self._cy = max(0.0, min(1.0, cy))
        self._cw = max(0.05, min(1.0, cw))
        self._ch = max(0.05, min(1.0, ch))
        self._fit_crop_to_aspect()
        self.update()

    def _fit_crop_to_aspect(self):
        if not self._lock_aspect:
            self._cw = max(0.05, min(1.0, self._cw))
            self._ch = max(0.05, min(1.0, self._ch))
            return
        if not self._wide or self._wide.isNull():
            return
        src_aspect = self._wide.width() / max(1, self._wide.height())
        if src_aspect >= self._aspect:
            self._cw = min(self._cw, self._ch * self._aspect / src_aspect)
        else:
            self._ch = min(self._ch, self._cw * src_aspect / self._aspect)
        self._cw = max(0.05, min(1.0, self._cw))
        self._ch = max(0.05, min(1.0, self._ch))

    def _layout(self):
        W, H = self.width(), self.height()
        if not self._wide or self._wide.isNull():
            return None
        sc = self._wide.scaled(W, H, Qt.AspectRatioMode.KeepAspectRatio,
                               Qt.TransformationMode.SmoothTransformation)
        ox = (W - sc.width()) // 2
        oy = (H - sc.height()) // 2
        tw, th = sc.width(), sc.height()
        cw = max(16, int(tw * self._cw))
        ch = max(16, int(th * self._ch))
        x = ox + int(max(0, tw - cw) * self._cx)
        y = oy + int(max(0, th - ch) * self._cy)
        return ox, oy, tw, th, QRect(x, y, cw, ch)

    def _hit_test(self, pos):
        lay = self._layout()
        if lay is None:
            return ""
        _ox, _oy, _tw, _th, r = lay
        m = 10
        l = abs(pos.x() - r.left()) <= m
        rr = abs(pos.x() - r.right()) <= m
        t = abs(pos.y() - r.top()) <= m
        b = abs(pos.y() - r.bottom()) <= m
        if l and t: return "tl"
        if rr and t: return "tr"
        if l and b: return "bl"
        if rr and b: return "br"
        if l: return "l"
        if rr: return "r"
        if t: return "t"
        if b: return "b"
        if r.contains(pos): return "move"
        return ""

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        W, H = self.width(), self.height()
        p.fillRect(0, 0, W, H, QColor("#111"))
        lay = self._layout()
        if lay is None:
            p.setPen(QColor("#555"))
            p.drawText(QRect(0, 0, W, H), Qt.AlignmentFlag.AlignCenter,
                       self._empty_label)
            p.end()
            return
        ox, oy, tw, th, crop = lay
        sc = self._wide.scaled(W, H, Qt.AspectRatioMode.KeepAspectRatio,
                               Qt.TransformationMode.SmoothTransformation)
        p.drawPixmap(ox, oy, sc)
        dim = QColor(0, 0, 0, 155)
        p.fillRect(QRect(ox, oy, crop.left() - ox, th), dim)
        p.fillRect(QRect(crop.right(), oy, ox + tw - crop.right(), th), dim)
        p.fillRect(QRect(crop.left(), oy, crop.width(), crop.top() - oy), dim)
        p.fillRect(QRect(crop.left(), crop.bottom(), crop.width(), oy + th - crop.bottom()), dim)
        p.setPen(QPen(QColor("#ffffff"), 2))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(crop)
        p.setPen(QPen(QColor(255, 255, 255, 90), 1, Qt.PenStyle.DotLine))
        p.drawLine(crop.center().x(), crop.top(), crop.center().x(), crop.bottom())
        p.drawLine(crop.left(), crop.center().y(), crop.right(), crop.center().y())
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor("#ffffff"))
        for x, y in [
            (crop.left(), crop.top()), (crop.center().x(), crop.top()), (crop.right(), crop.top()),
            (crop.left(), crop.center().y()), (crop.right(), crop.center().y()),
            (crop.left(), crop.bottom()), (crop.center().x(), crop.bottom()), (crop.right(), crop.bottom()),
        ]:
            p.drawRoundedRect(QRect(x - 5, y - 5, 10, 10), 4, 4)
        p.end()

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return
        self._drag_mode = self._hit_test(event.pos())
        if not self._drag_mode:
            return
        self._drag_last = event.pos()
        self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event):
        if self._drag_last is None:
            mode = self._hit_test(event.pos())
            if mode in ("l", "r"):
                self.setCursor(Qt.CursorShape.SizeHorCursor)
            elif mode in ("t", "b"):
                self.setCursor(Qt.CursorShape.SizeVerCursor)
            elif mode in ("tl", "tr", "bl", "br"):
                self.setCursor(Qt.CursorShape.SizeFDiagCursor)
            elif mode == "move":
                self.setCursor(Qt.CursorShape.OpenHandCursor)
            else:
                self.setCursor(Qt.CursorShape.ArrowCursor)
            return
        lay = self._layout()
        if lay is None:
            return
        _ox, _oy, tw, th, _crop = lay
        dx = event.pos().x() - self._drag_last.x()
        dy = event.pos().y() - self._drag_last.y()
        if self._drag_mode == "move":
            self._cx = max(0.0, min(1.0, self._cx + dx / max(1, tw * (1 - self._cw))))
            self._cy = max(0.0, min(1.0, self._cy + dy / max(1, th * (1 - self._ch))))
        else:
            if self._lock_aspect:
                sign = -1 if self._drag_mode in ("r", "b", "br") else 1
                delta = sign * max(dx / max(1, tw), dy / max(1, th), key=abs)
                self._cw = max(0.05, min(1.0, self._cw + delta))
                if self._wide and not self._wide.isNull():
                    src_aspect = self._wide.width() / max(1, self._wide.height())
                    self._ch = self._cw * src_aspect / self._aspect
                self._fit_crop_to_aspect()
            else:
                ddx = dx / max(1, tw)
                ddy = dy / max(1, th)
                left = max(0.0, min(1.0 - self._cw, self._cx * (1.0 - self._cw)))
                top = max(0.0, min(1.0 - self._ch, self._cy * (1.0 - self._ch)))
                right = left + self._cw
                bottom = top + self._ch
                if "l" in self._drag_mode:
                    left = max(0.0, min(right - 0.05, left + ddx))
                if "r" in self._drag_mode:
                    right = min(1.0, max(left + 0.05, right + ddx))
                if "t" in self._drag_mode:
                    top = max(0.0, min(bottom - 0.05, top + ddy))
                if "b" in self._drag_mode:
                    bottom = min(1.0, max(top + 0.05, bottom + ddy))
                self._cw = max(0.05, min(1.0, right - left))
                self._ch = max(0.05, min(1.0, bottom - top))
                self._cx = left / max(0.01, 1.0 - self._cw)
                self._cy = top / max(0.01, 1.0 - self._ch)
        self._drag_last = event.pos()
        self.update()
        self.crop_changed.emit(self._cx, self._cy, self._cw, self._ch)

    def mouseReleaseEvent(self, event):
        self._drag_last = None
        self._drag_mode = ""
        self.setCursor(Qt.CursorShape.ArrowCursor)


class ExportTab(QWidget):
    export_finished = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._project: Project | None = None
        self._worker: ExportWorker | None = None
        self._thumb_worker: ThumbnailWorker | None = None
        self._wide_thumb_worker: WideThumbnailWorker | None = None
        self._ai_title_worker: _AITitleWorker | None = None
        self._thumb_pixmap: QPixmap | None = None
        self._wide_pixmap: QPixmap | None = None
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.timeout.connect(self._update_preview)
        # Drag-and-drop y position tracking (-1 = auto from dropdown)
        self._part_text_y_pct: float = -1.0
        self._subtitle_y_pct: float = -1.0
        self._crop_w: float = 1.0
        self._crop_h: float = 1.0
        self._wm_x: float = 0.82
        self._wm_y: float = 0.88
        self._wm_w: float = 0.16
        self._wm_h: float = 0.08
        self._wm_regions: list[dict] = []
        self._wm_region_index: int = 0
        self._bg_image_path: str = ""
        self._bg_music_path: str = ""
        self._setup_ui()

    # ─── UI ────────────────────────────────────────────────────────────────────

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(6)

        top_splitter = QSplitter(Qt.Orientation.Horizontal)

        # ── Left: scrollable form ──────────────────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea{border:none;}")

        fw = QWidget()
        fl = QVBoxLayout(fw)
        fl.setSpacing(8)
        fl.setContentsMargins(4, 4, 8, 4)

        # Platform + Aspect
        pg = QGroupBox("Nền tảng & Định dạng xuất")
        pfl = QFormLayout(pg)
        self.cmb_platform = QComboBox()
        self.cmb_platform.addItems(PLATFORMS)
        pfl.addRow("Nền tảng:", self.cmb_platform)
        self.lbl_resolution = QLabel("1080 × 1920 (9:16)")
        self.lbl_resolution.setStyleSheet("color:#a6e3a1;")
        pfl.addRow("Độ phân giải:", self.lbl_resolution)
        self.cmb_aspect = QComboBox()
        for key, label in ASPECT_MODES.items():
            self.cmb_aspect.addItem(label, key)
        self.cmb_aspect.currentIndexChanged.connect(self._on_aspect_mode_changed)
        pfl.addRow("Xử lý khung hình:", self.cmb_aspect)
        fl.addWidget(pg)

        # ── Crop adjustment group (visible only for custom_crop) ───────────
        self._crop_group = QGroupBox("✂ Điều chỉnh vị trí cắt")
        cg = QFormLayout(self._crop_group)

        self._sld_crop_x = QSlider(Qt.Orientation.Horizontal)
        self._sld_crop_x.setRange(0, 100)
        self._sld_crop_x.setValue(50)
        self._sld_crop_x.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._sld_crop_x.setTickInterval(25)
        self._sld_crop_x.valueChanged.connect(self._on_crop_slider_changed)
        cg.addRow("", self._sld_crop_x)
        self._sld_crop_x.setVisible(False)

        self._sld_crop_y = QSlider(Qt.Orientation.Horizontal)
        self._sld_crop_y.setRange(0, 100)
        self._sld_crop_y.setValue(50)
        self._sld_crop_y.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._sld_crop_y.setTickInterval(25)
        self._sld_crop_y.valueChanged.connect(self._on_crop_slider_changed)
        cg.addRow("", self._sld_crop_y)
        self._sld_crop_y.setVisible(False)

        self._lbl_crop_xy = QLabel("X: 50% / Y: 50%")
        self._lbl_crop_xy.setStyleSheet("color:#888; font-size:11px;")
        cg.addRow(self._lbl_crop_xy)

        lbl_crop_hint = QLabel("Dung khung cat ben phai: keo trong khung de di chuyen, keo canh/goc de phong/thu vung cat.")
        lbl_crop_hint.setStyleSheet("color:#6c7086; font-size:10px;")
        lbl_crop_hint.setWordWrap(True)
        cg.addRow(lbl_crop_hint)

        self._crop_group.setVisible(False)
        fl.addWidget(self._crop_group)

        # ── Background image group (visible only for image_bg) ─────────────
        self._bgimg_group = QGroupBox("🖼 Ảnh nền tùy chỉnh")
        bgl = QFormLayout(self._bgimg_group)
        bg_row = QHBoxLayout()
        self._lbl_bgimg = QLabel("(chưa chọn ảnh)")
        self._lbl_bgimg.setStyleSheet("color:#888; font-size:11px;")
        self._lbl_bgimg.setWordWrap(True)
        btn_pick_bg = QPushButton("Chọn ảnh...")
        btn_pick_bg.clicked.connect(self._on_pick_bg_image)
        btn_clear_bg = QPushButton("Xóa")
        btn_clear_bg.clicked.connect(self._on_clear_bg_image)
        bg_row.addWidget(self._lbl_bgimg, 1)
        bg_row.addWidget(btn_pick_bg)
        bg_row.addWidget(btn_clear_bg)
        bg_w = QWidget(); bg_w.setLayout(bg_row)
        bgl.addRow(bg_w)
        bg_hint = QLabel("Ảnh sẽ phủ kín khung hình. Dùng kèm 'Thu hẹp video + tiêu đề' để có bố cục như mẫu.")
        bg_hint.setStyleSheet("color:#6c7086; font-size:10px;")
        bg_hint.setWordWrap(True)
        bgl.addRow(bg_hint)
        self._bgimg_group.setVisible(False)
        fl.addWidget(self._bgimg_group)

        music_group = QGroupBox("🎵 Nhạc nền")
        music_layout = QFormLayout(music_group)
        self.chk_background_music = QCheckBox("Bật nhạc nền khi xuất")
        self.chk_background_music.toggled.connect(
            self._on_background_music_toggled
        )
        music_layout.addRow(self.chk_background_music)

        music_row = QHBoxLayout()
        self.lbl_background_music = QLabel("(chưa chọn nhạc)")
        self.lbl_background_music.setWordWrap(True)
        self.lbl_background_music.setStyleSheet(
            "color:#888; font-size:11px;"
        )
        self.btn_pick_background_music = QPushButton("Chọn nhạc...")
        self.btn_pick_background_music.clicked.connect(
            self._on_pick_background_music
        )
        self.btn_clear_background_music = QPushButton("Xóa")
        self.btn_clear_background_music.clicked.connect(
            self._on_clear_background_music
        )
        music_row.addWidget(self.lbl_background_music, 1)
        music_row.addWidget(self.btn_pick_background_music)
        music_row.addWidget(self.btn_clear_background_music)
        music_widget = QWidget()
        music_widget.setLayout(music_row)
        music_layout.addRow("File nhạc:", music_widget)

        self.spn_background_music_volume = QSpinBox()
        self.spn_background_music_volume.setRange(0, 100)
        self.spn_background_music_volume.setValue(15)
        self.spn_background_music_volume.setSuffix(" %")
        music_layout.addRow(
            "Âm lượng nhạc:", self.spn_background_music_volume
        )
        music_hint = QLabel(
            "Nhạc tự lặp đến hết video. Nên để 8–20% để không lấn giọng đọc."
        )
        music_hint.setWordWrap(True)
        music_hint.setStyleSheet("color:#6c7086; font-size:10px;")
        music_layout.addRow(music_hint)
        fl.addWidget(music_group)

        self._wm_group = QGroupBox("Xu ly watermark")
        wml = QFormLayout(self._wm_group)
        self.chk_remove_wm = QCheckBox("Bat xu ly watermark")
        self.chk_remove_wm.toggled.connect(self._on_remove_wm_toggled)
        wml.addRow(self.chk_remove_wm)
        region_row = QHBoxLayout()
        self.cmb_wm_region = QComboBox()
        self.cmb_wm_region.currentIndexChanged.connect(self._on_wm_region_changed)
        self.btn_add_wm_region = QPushButton("+ Thêm vùng")
        self.btn_add_wm_region.clicked.connect(self._add_wm_region)
        self.btn_remove_wm_region = QPushButton("− Xóa vùng")
        self.btn_remove_wm_region.clicked.connect(self._remove_wm_region)
        region_row.addWidget(self.cmb_wm_region, 1)
        region_row.addWidget(self.btn_add_wm_region)
        region_row.addWidget(self.btn_remove_wm_region)
        region_widget = QWidget(); region_widget.setLayout(region_row)
        wml.addRow("Vùng watermark:", region_widget)
        self.cmb_remove_wm_mode = QComboBox()
        self.cmb_remove_wm_mode.addItem("Blur vung watermark", "blur")
        self.cmb_remove_wm_mode.addItem(
            "Vá nền mềm (Delogo)", "delogo"
        )
        self.cmb_remove_wm_mode.addItem("Mosaic / pixel hoa", "mosaic")
        self.cmb_remove_wm_mode.addItem("Che bang mau (xoa bang cach phu mau)", "cover")
        self.cmb_remove_wm_mode.currentIndexChanged.connect(self._schedule_preview)
        wml.addRow("Kieu xu ly:", self.cmb_remove_wm_mode)
        self.spn_remove_wm_strength = QSpinBox()
        self.spn_remove_wm_strength.setRange(1, 5)
        self.spn_remove_wm_strength.setValue(3)
        self.spn_remove_wm_strength.setSuffix(" / 5")
        self.spn_remove_wm_strength.valueChanged.connect(self._schedule_preview)
        wml.addRow("Độ hòa nền:", self.spn_remove_wm_strength)
        self.cmb_remove_wm_color = QComboBox()
        self.cmb_remove_wm_color.addItems(_COLORS)
        self.cmb_remove_wm_color.setCurrentText("black")
        self.cmb_remove_wm_color.currentTextChanged.connect(self._schedule_preview)
        wml.addRow("Mau che:", self.cmb_remove_wm_color)
        self.lbl_remove_wm_xy = QLabel("X: 82% / Y: 88% / W: 16% / H: 8%")
        self.lbl_remove_wm_xy.setStyleSheet("color:#888; font-size:11px;")
        wml.addRow(self.lbl_remove_wm_xy)
        wm_hint = QLabel(
            "Chọn một vùng trong danh sách rồi kéo khung Watermark bên phải. "
            "Delogo xử lý đúng toàn bộ vùng đã chọn, kể cả khi vùng phủ hết "
            "chiều ngang video."
        )
        wm_hint.setStyleSheet("color:#6c7086; font-size:10px;")
        wm_hint.setWordWrap(True)
        wml.addRow(wm_hint)
        fl.addWidget(self._wm_group)

        # PART text
        tg = QGroupBox("Chèn PART text")
        self._part_text_group = tg
        tl = QFormLayout(tg)

        self.chk_part_text = QCheckBox("Bật PART text")
        self.chk_part_text.setChecked(True)
        self.chk_part_text.toggled.connect(self._toggle_part_text)
        self.chk_part_text.toggled.connect(self._schedule_preview)
        tl.addRow(self.chk_part_text)

        self.cmb_part_mode = QComboBox()
        for key, label in PART_TEXT_MODES.items():
            self.cmb_part_mode.addItem(label, key)
        self.cmb_part_mode.currentIndexChanged.connect(self._on_part_mode_changed)
        self.cmb_part_mode.currentIndexChanged.connect(self._schedule_preview)
        tl.addRow("Chế độ:", self.cmb_part_mode)

        self.cmb_part_pos = QComboBox()
        for key, label in _POSITIONS.items():
            self.cmb_part_pos.addItem(label, key)
        self.cmb_part_pos.currentIndexChanged.connect(self._on_part_pos_changed)
        tl.addRow("Vị trí:", self.cmb_part_pos)

        self.spn_fontsize = QSpinBox()
        self.spn_fontsize.setRange(20, 200)
        self.spn_fontsize.setValue(72)
        self.spn_fontsize.valueChanged.connect(self._schedule_preview)
        tl.addRow("Font size (px):", self.spn_fontsize)

        self.cmb_part_font = QComboBox()
        for display, family in _SUBTITLE_FONTS:
            self.cmb_part_font.addItem(display, family)
        self.cmb_part_font.currentIndexChanged.connect(self._schedule_preview)
        tl.addRow("Phong chu:", self.cmb_part_font)

        style_row_part = QHBoxLayout()
        self.chk_part_bold = QCheckBox("Bold")
        self.chk_part_bold.setChecked(True)
        self.chk_part_bold.toggled.connect(self._schedule_preview)
        self.chk_part_italic = QCheckBox("Italic")
        self.chk_part_italic.toggled.connect(self._schedule_preview)
        self.chk_part_underline = QCheckBox("Gach chan")
        self.chk_part_underline.toggled.connect(self._schedule_preview)
        style_row_part.addWidget(self.chk_part_bold)
        style_row_part.addWidget(self.chk_part_italic)
        style_row_part.addWidget(self.chk_part_underline)
        style_row_part.addStretch()
        style_part_w = QWidget(); style_part_w.setLayout(style_row_part)
        tl.addRow("Kieu chu:", style_part_w)

        self.cmb_text_color = QComboBox()
        self.cmb_text_color.addItems(_COLORS)
        self.cmb_text_color.currentTextChanged.connect(self._schedule_preview)
        tl.addRow("Màu chữ:", self.cmb_text_color)

        self.cmb_bg_color = QComboBox()
        self.cmb_bg_color.addItems(_COLORS)
        self.cmb_bg_color.setCurrentText("black")
        self.cmb_bg_color.currentTextChanged.connect(self._schedule_preview)
        tl.addRow("Màu nền:", self.cmb_bg_color)

        self.spn_header_h = QSpinBox()
        self.spn_header_h.setRange(50, 600)
        self.spn_header_h.setSingleStep(2)
        self.spn_header_h.setValue(250)
        self.spn_header_h.setSuffix(" px")
        self.spn_header_h.valueChanged.connect(self._schedule_preview)
        tl.addRow("Chiều cao header (shrink):", self.spn_header_h)

        # Per-clip custom header text (multi-line: Enter = xuống dòng)
        self.txt_custom_text = QPlainTextEdit()
        self.txt_custom_text.setPlaceholderText("Để trống = dùng 'PART 1', 'PART 2'…\n(Enter để xuống dòng, mỗi Part riêng biệt)")
        self.txt_custom_text.setMaximumHeight(60)
        self.txt_custom_text.textChanged.connect(self._on_custom_header_changed)
        tl.addRow("Text tuỳ chỉnh (clip):", self.txt_custom_text)

        # Opacity (watermark only)
        self.lbl_opacity = QLabel("Độ trong suốt:")
        self.spn_opacity = QDoubleSpinBox()
        self.spn_opacity.setRange(0.05, 1.0)
        self.spn_opacity.setSingleStep(0.05)
        self.spn_opacity.setValue(0.35)
        self.spn_opacity.valueChanged.connect(self._schedule_preview)
        tl.addRow(self.lbl_opacity, self.spn_opacity)

        fl.addWidget(tg)

        # Subtitle
        sg = QGroupBox("Subtitle style (áp dụng khi xuất)")
        sl = QFormLayout(sg)
        self.cmb_sub_style = QComboBox()
        for key, label in SUBTITLE_STYLES.items():
            self.cmb_sub_style.addItem(label, key)
        self.cmb_sub_style.currentIndexChanged.connect(self._on_sub_style_changed)
        sl.addRow("Kiểu subtitle:", self.cmb_sub_style)

        # Font family
        self.cmb_sub_font = QComboBox()
        for display, family in _SUBTITLE_FONTS:
            self.cmb_sub_font.addItem(display, family)
        self.cmb_sub_font.currentIndexChanged.connect(self._schedule_preview)
        sl.addRow("Phông chữ:", self.cmb_sub_font)

        # Bold / Italic on same row
        style_row = QHBoxLayout()
        self.chk_sub_bold = QCheckBox("Bold")
        self.chk_sub_bold.setChecked(True)
        self.chk_sub_bold.toggled.connect(self._schedule_preview)
        self.chk_sub_italic = QCheckBox("Italic")
        self.chk_sub_italic.setChecked(False)
        self.chk_sub_italic.toggled.connect(self._schedule_preview)
        self.chk_sub_underline = QCheckBox("Underline")
        self.chk_sub_underline.setChecked(False)
        self.chk_sub_underline.toggled.connect(self._schedule_preview)
        style_row.addWidget(self.chk_sub_bold)
        style_row.addWidget(self.chk_sub_italic)
        style_row.addWidget(self.chk_sub_underline)
        style_row.addStretch()
        sl.addRow("Kiểu chữ:", style_row)

        self.spn_sub_fontsize = QSpinBox()
        self.spn_sub_fontsize.setRange(20, 150)
        self.spn_sub_fontsize.setValue(65)
        self.spn_sub_fontsize.valueChanged.connect(self._schedule_preview)
        sl.addRow("Cỡ chữ (px):", self.spn_sub_fontsize)
        self.cmb_sub_color = QComboBox()
        self.cmb_sub_color.addItems(_COLORS)
        self.cmb_sub_color.setCurrentText("white")
        self.cmb_sub_color.currentTextChanged.connect(self._schedule_preview)
        sl.addRow("Màu chữ:", self.cmb_sub_color)
        self.cmb_sub_highlight = QComboBox()
        self.cmb_sub_highlight.addItems(_COLORS)
        self.cmb_sub_highlight.setCurrentText("yellow")
        self.cmb_sub_highlight.currentTextChanged.connect(self._schedule_preview)
        sl.addRow("Màu highlight (karaoke):", self.cmb_sub_highlight)
        self.cmb_sub_pos = QComboBox()
        for key, label in _POSITIONS.items():
            self.cmb_sub_pos.addItem(label, key)
        self.cmb_sub_pos.setCurrentText("Dưới cùng")
        self.cmb_sub_pos.currentIndexChanged.connect(self._on_sub_pos_changed)
        sl.addRow("Vị trí:", self.cmb_sub_pos)

        # MarginV slider: fine-tune subtitle distance from edge
        self.sld_sub_margin_v = QSlider(Qt.Orientation.Horizontal)
        self.sld_sub_margin_v.setRange(0, 400)
        self.sld_sub_margin_v.setValue(0)
        self.sld_sub_margin_v.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.sld_sub_margin_v.setTickInterval(50)
        self.sld_sub_margin_v.valueChanged.connect(self._on_sub_margin_changed)
        self.lbl_sub_margin_v = QLabel("MarginV: tự động")
        self.lbl_sub_margin_v.setStyleSheet("color:#888; font-size:10px;")
        sl.addRow("Khoảng cách (px):", self.sld_sub_margin_v)
        sl.addRow(self.lbl_sub_margin_v)

        # Bar-style controls (subtitle on coloured background bar)
        self.cmb_sub_bg_color = QComboBox()
        self.cmb_sub_bg_color.addItems(_COLORS)
        self.cmb_sub_bg_color.setCurrentText("white")
        self.cmb_sub_bg_color.currentTextChanged.connect(self._schedule_preview)
        self.lbl_sub_bg_color = QLabel("Màu nền bar:")
        sl.addRow(self.lbl_sub_bg_color, self.cmb_sub_bg_color)

        self.spn_sub_bar_h = QSpinBox()
        self.spn_sub_bar_h.setRange(50, 600)
        self.spn_sub_bar_h.setSingleStep(2)
        self.spn_sub_bar_h.setValue(200)
        self.spn_sub_bar_h.setSuffix(" px")
        self.spn_sub_bar_h.valueChanged.connect(self._schedule_preview)
        self.lbl_sub_bar_h = QLabel("Chiều cao bar:")
        sl.addRow(self.lbl_sub_bar_h, self.spn_sub_bar_h)

        # Per-clip custom subtitle text (multi-line: Enter = xuống dòng)
        self.txt_sub_custom = QPlainTextEdit()
        self.txt_sub_custom.setPlaceholderText("Để trống = dùng subtitle từ phiên âm\n(Enter để xuống dòng)")
        self.txt_sub_custom.setMaximumHeight(60)
        self.txt_sub_custom.textChanged.connect(self._on_sub_custom_changed)
        sl.addRow("Text cố định (clip):", self.txt_sub_custom)
        self.lbl_sub_custom_hint = QLabel("💡 Chọn Part ở danh sách bên dưới để nhập text riêng")
        self.lbl_sub_custom_hint.setStyleSheet("color:#6c7086; font-size:10px;")
        self.lbl_sub_custom_hint.setWordWrap(True)
        sl.addRow(self.lbl_sub_custom_hint)

        lbl_info = QLabel(
            "⚠️ Bắt buộc tạo subtitle trước khi xuất:\n"
            "Tab 2 → Phiên âm xong → bấm\n"
            "\"Tạo subtitle cho tất cả Parts\""
        )
        lbl_info.setStyleSheet(
            "color:#f9e2af; font-size:11px; background:#313244;"
            "border-radius:4px; padding:5px;"
        )
        lbl_info.setWordWrap(True)
        self._subtitle_info_label = lbl_info
        sl.addRow(lbl_info)

        # Live indicator showing how many clips have subtitle files
        self.lbl_sub_file_status = QLabel("📋 Chưa kiểm tra")
        self.lbl_sub_file_status.setStyleSheet("color:#888; font-size:10px;")
        sl.addRow(self.lbl_sub_file_status)
        fl.addWidget(sg)

        ai_title_g = QGroupBox("AI viet title cho Parts")
        self._ai_title_group = ai_title_g
        ai_title_l = QFormLayout(ai_title_g)
        self.cmb_ai_title_provider = QComboBox()
        self.cmb_ai_title_provider.addItem("Google Gemini", "gemini")
        self.cmb_ai_title_provider.addItem("Groq", "groq")
        self.cmb_ai_title_provider.addItem("OpenRouter", "openrouter")
        self.cmb_ai_title_provider.addItem("Ollama local", "ollama")
        self.cmb_ai_title_provider.currentIndexChanged.connect(self._on_ai_title_provider_changed)
        ai_title_l.addRow("Provider:", self.cmb_ai_title_provider)
        self.cmb_ai_title_model = QComboBox()
        ai_title_l.addRow("Model:", self.cmb_ai_title_model)
        self.chk_ai_title_bottom = QCheckBox("Tao title duoi vao Text co dinh khi khong co sub")
        self.chk_ai_title_bottom.setChecked(True)
        ai_title_l.addRow(self.chk_ai_title_bottom)
        self.btn_ai_titles = QPushButton("AI viet title cho tat ca Parts")
        self.btn_ai_titles.clicked.connect(self._start_ai_titles)
        ai_title_l.addRow(self.btn_ai_titles)
        self.lbl_ai_title_status = QLabel("")
        self.lbl_ai_title_status.setStyleSheet("color:#888; font-size:10px;")
        self.lbl_ai_title_status.setWordWrap(True)
        ai_title_l.addRow(self.lbl_ai_title_status)
        fl.addWidget(ai_title_g)
        self._on_ai_title_provider_changed()

        # Encoding
        eg = QGroupBox("Cài đặt mã hóa")
        el = QFormLayout(eg)

        nvenc_ok = is_nvenc_available()
        self.chk_gpu = QCheckBox(
            "⚡ Dùng GPU (NVIDIA NVENC) — nhanh hơn 3-10x"
            if nvenc_ok else
            "⚡ GPU (NVENC) — không khả dụng"
        )
        self.chk_gpu.setChecked(False)
        self.chk_gpu.setEnabled(nvenc_ok)
        self.chk_gpu.toggled.connect(self._on_gpu_toggled)
        el.addRow(self.chk_gpu)

        self.cmb_preset = QComboBox()
        self._populate_presets(gpu=False)
        el.addRow("Preset FFmpeg:", self.cmb_preset)
        self.spn_crf = QSpinBox()
        self.spn_crf.setRange(0, 51)
        self.spn_crf.setValue(23)
        self.lbl_crf = QLabel("CRF (thấp = tốt hơn):")
        el.addRow(self.lbl_crf, self.spn_crf)
        fl.addWidget(eg)

        # Output folder
        og = QGroupBox("Thư mục xuất")
        ol = QHBoxLayout(og)
        self.txt_out_dir = QLineEdit()
        self.txt_out_dir.setPlaceholderText("Mặc định: output/<project>/exports/")
        btn_browse = QPushButton("📁 Chọn")
        btn_browse.clicked.connect(self._browse_output)
        ol.addWidget(self.txt_out_dir)
        ol.addWidget(btn_browse)
        fl.addWidget(og)

        self.chk_captions = QCheckBox("📝 Tự tạo caption & hashtag cho từng Part")
        self.chk_captions.setChecked(True)
        fl.addWidget(self.chk_captions)
        export_parts_g = QGroupBox("Parts sẽ xuất")
        self._export_parts_group = export_parts_g
        export_parts_l = QVBoxLayout(export_parts_g)
        btn_parts_row = QHBoxLayout()
        self.btn_export_all_parts = QPushButton("Chọn tất cả")
        self.btn_export_all_parts.clicked.connect(lambda: self._set_all_export_parts(True))
        self.btn_export_no_parts = QPushButton("Bỏ chọn")
        self.btn_export_no_parts.clicked.connect(lambda: self._set_all_export_parts(False))
        btn_parts_row.addWidget(self.btn_export_all_parts)
        btn_parts_row.addWidget(self.btn_export_no_parts)
        export_parts_l.addLayout(btn_parts_row)
        self.lst_export_parts = QListWidget()
        self.lst_export_parts.setMaximumHeight(140)
        self.lst_export_parts.itemChanged.connect(self._on_export_part_changed)
        self.lst_export_parts.currentRowChanged.connect(self._on_export_part_current_changed)
        export_parts_l.addWidget(self.lst_export_parts)
        fl.addWidget(export_parts_g)

        fl.addStretch()

        scroll.setWidget(fw)
        top_splitter.addWidget(scroll)

        # ── Right: live preview (chiếm ~50% width) ────────────────────────
        pp = QWidget()
        pp.setMinimumWidth(280)
        pp.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        ppl = QVBoxLayout(pp)
        ppl.setContentsMargins(6, 4, 4, 4)
        ppl.setSpacing(6)

        hdr = QHBoxLayout()
        hdr.addWidget(QLabel("🔍 Preview hình ảnh xuất bản"))
        hdr.addStretch()
        ppl.addLayout(hdr)

        self.lbl_export_source = QLabel("Chưa có video nguồn xuất")
        self.lbl_export_source.setWordWrap(True)
        self.lbl_export_source.setStyleSheet(
            "color:#89b4fa; background:#181825; border:1px solid #313244;"
            "border-radius:4px; padding:6px;"
        )
        ppl.addWidget(self.lbl_export_source)

        self.btn_play_export_source = QPushButton(
            "▶ Mở video nguồn xuất để nghe thử"
        )
        self.btn_play_export_source.setEnabled(False)
        self.btn_play_export_source.clicked.connect(
            self._open_export_source_video
        )
        ppl.addWidget(self.btn_play_export_source)

        ppl.addWidget(QLabel("Hiển thị Part:"))
        self.cmb_preview_clip = QComboBox()
        self.cmb_preview_clip.addItem("PART 1 (mặc định)", "PART 1")
        self.cmb_preview_clip.currentIndexChanged.connect(self._on_preview_clip_changed)
        ppl.addWidget(self.cmb_preview_clip)

        # Crop adjustment widget (only visible in custom_crop mode)
        self._crop_adjust_widget = _CropAdjustWidget(label="Chon custom crop de hien thi khung cat")
        self._crop_adjust_widget.setVisible(False)
        self._crop_adjust_widget.crop_changed.connect(self._on_crop_drag)
        ppl.addWidget(self._crop_adjust_widget)

        self._wm_adjust_widget = _CropAdjustWidget(
            lock_aspect=False,
            label="Bat xu ly watermark de chon vung",
        )
        self._wm_adjust_widget.setVisible(False)
        self._wm_adjust_widget.crop_changed.connect(self._on_wm_drag)
        ppl.addWidget(self._wm_adjust_widget)

        # Preview label fills remaining vertical space
        self._preview_lbl = _PreviewLabel()
        self._preview_lbl.text_dragged.connect(self._on_text_dragged)
        ppl.addWidget(self._preview_lbl, 1)   # stretch=1 → fills space

        self.lbl_thumb_status = QLabel("Chưa tải thumbnail")
        self.lbl_thumb_status.setStyleSheet("color:#888; font-size:10px;")
        ppl.addWidget(self.lbl_thumb_status)

        top_splitter.addWidget(pp)
        top_splitter.setSizes([1, 1])   # equal 50 / 50
        root.addWidget(top_splitter, 1)

        # ── Bottom: export controls ────────────────────────────────────────
        self.btn_export = QPushButton("🚀 BẮT ĐẦU XUẤT VIDEO")
        self.btn_export.setFixedHeight(48)
        self.btn_export.setStyleSheet(
            "background: #7aa2f7; color: #1e1e2e; font-size: 16px; font-weight: bold;"
        )
        self.btn_export.clicked.connect(self._start_export)
        root.addWidget(self.btn_export)

        self.lbl_export_status = QLabel("")
        self.lbl_export_status.setStyleSheet("color: #cdd6f4;")
        root.addWidget(self.lbl_export_status)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        root.addWidget(self.progress_bar)

        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setMaximumHeight(130)
        self.txt_log.setStyleSheet("background: #111; color: #aaa; font-size: 11px;")
        root.addWidget(self.txt_log)

        # Set initial visibility
        self._on_part_mode_changed()
        self._on_sub_style_changed()
        self._on_remove_wm_toggled(False)
        self._on_background_music_toggled(False)

    # ─── Public API ────────────────────────────────────────────────────────────

    def _is_review_export(self) -> bool:
        """Return True when export must use the voiced timeline artifact.

        ``review_voice_video`` is the same completed video shown by the preview
        and is the canonical input for every voice workflow.  Timed dubbing
        keeps its own ``workflow_mode`` value, so gating this on mode ==
        ``review`` incorrectly fell back to the original clip (and its audio)
        during export.
        """
        return bool(
            self._project
            and getattr(self._project, "review_voice_video", "")
            and Path(self._project.review_voice_video).exists()
        )

    def _review_export_clip(self) -> Clip | None:
        if not self._is_review_export():
            return None
        duration = float(
            getattr(self._project, "review_video_duration", 0.0) or 0.0
        )
        if duration <= 0:
            metadata = get_video_metadata(self._project.review_voice_video) or {}
            duration = float(metadata.get("duration", 0.0) or 0.0)
        if duration <= 0:
            duration = sum(
                clip.duration for clip in self._project.clips if clip.enabled
            )
        review_clip = Clip(
            id="review-final",
            index=0,
            start_time=0.0,
            end_time=max(0.1, duration),
            enabled=True,
            part_text="VIDEO REVIEW",
            source_video=self._project.review_voice_video,
            subtitle_file=(
                getattr(self._project, "narration_subtitle_file", "") or ""
            ),
        )
        review_clip.is_review_master = True
        return review_clip

    def _apply_workflow_mode_ui(self):
        review = self._is_review_export()
        self._part_text_group.setVisible(not review)
        self._ai_title_group.setVisible(not review)
        self.chk_captions.setVisible(not review)
        self._export_parts_group.setTitle(
            "Video sẽ xuất" if review else "Parts sẽ xuất"
        )
        self.btn_export_all_parts.setVisible(not review)
        self.btn_export_no_parts.setVisible(not review)
        if review:
            self.chk_part_text.setChecked(False)
            self.chk_captions.setChecked(False)
            self.btn_export.setText("🚀 XUẤT VIDEO ĐÃ LỒNG TIẾNG")
            source = Path(self._project.review_voice_video)
            self.lbl_export_source.setText(
                f"✅ Nguồn xuất: {source.name} — video đã lồng tiếng"
            )
            self.lbl_export_source.setToolTip(str(source.resolve()))
            self.btn_play_export_source.setText(
                "▶ Phát video đã lồng tiếng để kiểm tra"
            )
            self.btn_play_export_source.setEnabled(True)
            self._subtitle_info_label.setText(
                "✅ Phụ đề ở bước này dùng một track toàn timeline được tạo "
                "từ giọng đọc cuối. PART text luôn tắt trong luồng review."
            )
        else:
            self.btn_export.setText("🚀 BẮT ĐẦU XUẤT VIDEO")
            source_path = (
                getattr(self._project, "source_video", "")
                if self._project else ""
            )
            source = Path(source_path) if source_path else None
            self.lbl_export_source.setText(
                f"Nguồn xuất: {source.name} — video gốc"
                if source else "Chưa có video nguồn xuất"
            )
            self.lbl_export_source.setToolTip(
                str(source.resolve()) if source else ""
            )
            self.btn_play_export_source.setText(
                "▶ Mở video nguồn xuất để nghe thử"
            )
            self.btn_play_export_source.setEnabled(
                bool(source and source.exists())
            )
            self._subtitle_info_label.setText(
                "Tạo subtitle trước khi xuất ở bước Phiên âm/Phụ đề."
            )

    def _open_export_source_video(self):
        clip = self._current_preview_clip()
        source_path = (
            getattr(clip, "source_video", "") if clip else ""
        ) or (
            getattr(self._project, "source_video", "")
            if self._project else ""
        )
        if not source_path or not Path(source_path).exists():
            QMessageBox.warning(
                self,
                "Thiếu video",
                "Không tìm thấy video nguồn sẽ dùng để xuất.",
            )
            return
        if not QDesktopServices.openUrl(
            QUrl.fromLocalFile(str(Path(source_path).resolve()))
        ):
            QMessageBox.warning(
                self,
                "Không mở được video",
                f"Không thể mở file:\n{source_path}",
            )

    def load_project(self, project: Project):
        self._project = project
        cfg = project.export_config

        idx = self.cmb_platform.findText(cfg.platform)
        if idx >= 0:
            self.cmb_platform.setCurrentIndex(idx)

        for i in range(self.cmb_aspect.count()):
            if self.cmb_aspect.itemData(i) == cfg.aspect_mode:
                self.cmb_aspect.setCurrentIndex(i); break

        self.chk_part_text.setChecked(cfg.part_text_enabled)

        for i in range(self.cmb_part_mode.count()):
            if self.cmb_part_mode.itemData(i) == cfg.part_text_mode:
                self.cmb_part_mode.setCurrentIndex(i); break

        for i in range(self.cmb_part_pos.count()):
            if self.cmb_part_pos.itemData(i) == cfg.part_text_position:
                self.cmb_part_pos.setCurrentIndex(i); break

        self.spn_fontsize.setValue(cfg.part_text_fontsize)
        self.spn_header_h.setValue(cfg.header_height)
        saved_part_font = getattr(cfg, "part_text_font", "Arial") or "Arial"
        for i in range(self.cmb_part_font.count()):
            if self.cmb_part_font.itemData(i) == saved_part_font:
                self.cmb_part_font.setCurrentIndex(i); break
        self.chk_part_bold.setChecked(getattr(cfg, "part_text_bold", True))
        self.chk_part_italic.setChecked(getattr(cfg, "part_text_italic", False))
        self.chk_part_underline.setChecked(getattr(cfg, "part_text_underline", False))

        # Color fields (were missing in previous version)
        if cfg.part_text_color in _COLORS:
            self.cmb_text_color.setCurrentText(cfg.part_text_color)
        if (cfg.part_text_bg_color or "black") in _COLORS:
            self.cmb_bg_color.setCurrentText(cfg.part_text_bg_color or "black")

        gpu = getattr(cfg, "use_gpu", False)
        self.chk_gpu.blockSignals(True)
        self.chk_gpu.setChecked(gpu)
        self.chk_gpu.blockSignals(False)
        self._populate_presets(gpu=gpu)
        self.cmb_preset.setCurrentText(cfg.preset)
        self.spn_crf.setValue(cfg.crf)
        # txt_custom_text is now per-clip — loaded in _on_export_part_current_changed
        self.spn_opacity.setValue(getattr(cfg, "watermark_opacity", 0.35))

        # Crop
        cx = getattr(cfg, "crop_x", 0.5)
        cy = getattr(cfg, "crop_y", 0.5)
        cw = getattr(cfg, "crop_w", 1.0)
        ch = getattr(cfg, "crop_h", 1.0)
        self._crop_w = cw
        self._crop_h = ch
        self._sld_crop_x.blockSignals(True)
        self._sld_crop_y.blockSignals(True)
        self._sld_crop_x.setValue(int(cx * 100))
        self._sld_crop_y.setValue(int(cy * 100))
        self._lbl_crop_xy.setText(f"X: {int(cx*100)}% / Y: {int(cy*100)}%")
        self._sld_crop_x.blockSignals(False)
        self._sld_crop_y.blockSignals(False)

        # Background image
        self._bg_image_path = getattr(cfg, "background_image", "") or ""
        self._lbl_bgimg.setText(
            Path(self._bg_image_path).name if self._bg_image_path else "(chưa chọn ảnh)"
        )
        self._crop_adjust_widget.set_aspect(cfg.width, cfg.height)
        self._crop_adjust_widget.set_crop(cx, cy, cw, ch)

        self._bg_music_path = (
            getattr(cfg, "background_music_path", "") or ""
        )
        self.lbl_background_music.setText(
            Path(self._bg_music_path).name
            if self._bg_music_path else "(chưa chọn nhạc)"
        )
        self.lbl_background_music.setToolTip(self._bg_music_path)
        self.spn_background_music_volume.setValue(
            int(getattr(cfg, "background_music_volume", 15) or 0)
        )
        self.chk_background_music.blockSignals(True)
        self.chk_background_music.setChecked(
            bool(getattr(cfg, "background_music_enabled", False))
        )
        self.chk_background_music.blockSignals(False)
        self._on_background_music_toggled(
            self.chk_background_music.isChecked()
        )

        # Loading a checked value emits ``toggled``.  That handler can start a
        # thumbnail QThread before the rest of the project state is ready, and
        # later setup below requests the same thumbnail again.  Block the
        # signal here and apply the completed state once at the end.
        self.chk_remove_wm.blockSignals(True)
        self.chk_remove_wm.setChecked(
            getattr(cfg, "remove_watermark_enabled", False)
        )
        self.chk_remove_wm.blockSignals(False)
        for i in range(self.cmb_remove_wm_mode.count()):
            if self.cmb_remove_wm_mode.itemData(i) == getattr(cfg, "remove_watermark_mode", "blur"):
                self.cmb_remove_wm_mode.setCurrentIndex(i); break
        self.spn_remove_wm_strength.setValue(getattr(cfg, "remove_watermark_strength", 3))
        wm_color = getattr(cfg, "remove_watermark_color", "black") or "black"
        if wm_color in _COLORS:
            self.cmb_remove_wm_color.setCurrentText(wm_color)
        self._wm_x = getattr(cfg, "remove_watermark_x", 0.82)
        self._wm_y = getattr(cfg, "remove_watermark_y", 0.88)
        self._wm_w = getattr(cfg, "remove_watermark_w", 0.16)
        self._wm_h = getattr(cfg, "remove_watermark_h", 0.08)
        saved_regions = getattr(cfg, "remove_watermark_regions", None) or []
        self._wm_regions = [
            {
                "x": float(region.get("x", 0.82)),
                "y": float(region.get("y", 0.88)),
                "w": float(region.get("w", 0.16)),
                "h": float(region.get("h", 0.08)),
            }
            for region in saved_regions
            if isinstance(region, dict)
        ]
        if not self._wm_regions:
            self._wm_regions = [{
                "x": self._wm_x, "y": self._wm_y,
                "w": self._wm_w, "h": self._wm_h,
            }]
        self._wm_region_index = 0
        self._refresh_wm_region_combo()
        self._load_selected_wm_region()
        self._wm_adjust_widget.set_crop(self._wm_x, self._wm_y, self._wm_w, self._wm_h)
        self._update_wm_label()
        self._on_remove_wm_toggled(self.chk_remove_wm.isChecked())

        for i in range(self.cmb_sub_style.count()):
            if self.cmb_sub_style.itemData(i) == cfg.subtitle_style:
                self.cmb_sub_style.setCurrentIndex(i); break

        # Font family
        saved_font = getattr(cfg, "subtitle_font", "Arial") or "Arial"
        for i in range(self.cmb_sub_font.count()):
            if self.cmb_sub_font.itemData(i) == saved_font:
                self.cmb_sub_font.setCurrentIndex(i); break

        self.chk_sub_bold.setChecked(getattr(cfg, "subtitle_bold", True))
        self.chk_sub_italic.setChecked(getattr(cfg, "subtitle_italic", False))
        self.chk_sub_underline.setChecked(getattr(cfg, "subtitle_underline", False))
        self.spn_sub_fontsize.setValue(cfg.subtitle_fontsize)
        self.cmb_sub_color.setCurrentText(cfg.subtitle_color)
        self.cmb_sub_highlight.setCurrentText(cfg.subtitle_highlight_color)
        for i in range(self.cmb_sub_pos.count()):
            if self.cmb_sub_pos.itemData(i) == cfg.subtitle_position:
                self.cmb_sub_pos.setCurrentIndex(i); break

        # Subtitle margin_v slider
        saved_mv = getattr(cfg, "subtitle_margin_v", 0)
        self.sld_sub_margin_v.setValue(saved_mv)
        self._on_sub_margin_changed(saved_mv)

        # Drag-and-drop y positions
        self._part_text_y_pct = getattr(cfg, "part_text_y_pct", -1.0)
        self._subtitle_y_pct = getattr(cfg, "subtitle_y_pct", -1.0)

        # Bar-style settings
        saved_bg = getattr(cfg, "subtitle_bg_color", "white") or "white"
        if saved_bg in _COLORS:
            self.cmb_sub_bg_color.setCurrentText(saved_bg)
        self.spn_sub_bar_h.setValue(getattr(cfg, "subtitle_bar_height", 200))
        self._on_sub_style_changed()

        out_dir = project.global_output_dir or str(Path(project.output_dir) / "exports")
        self.txt_out_dir.setText(out_dir)

        self._apply_workflow_mode_ui()

        # Populate clip preview selector
        self.cmb_preview_clip.blockSignals(True)
        self.cmb_preview_clip.clear()
        review_clip = self._review_export_clip()
        if review_clip:
            self.cmb_preview_clip.addItem(
                f"Video đã lồng tiếng ({Path(review_clip.source_video).name})",
                "VIDEO REVIEW",
            )
        else:
            for clip in project.clips:
                self.cmb_preview_clip.addItem(clip.part_text, clip.part_text)
        if not project.clips and not review_clip:
            self.cmb_preview_clip.addItem("PART 1 (mặc định)", "PART 1")
        self.cmb_preview_clip.blockSignals(False)
        self._populate_export_parts(project)

        # Update subtitle status badge
        self._update_sub_file_status()

        # Trigger aspect mode show/hide and thumbnail load
        self._on_aspect_mode_changed()
        if getattr(project.export_config, "pre_crop_enabled", False):
            self._load_wide_thumbnail()
        self._load_thumbnail()

    def refresh_clips(self, project: Project):
        """Refresh only the clip selector without resetting any export settings.

        Called by main_window when the editor tab modifies clips so the
        preview-part selector stays in sync without wiping user's export choices.
        """
        self._project = project
        self._apply_workflow_mode_ui()
        self.cmb_preview_clip.blockSignals(True)
        self.cmb_preview_clip.clear()
        review_clip = self._review_export_clip()
        if review_clip:
            self.cmb_preview_clip.addItem(
                f"Video đã lồng tiếng ({Path(review_clip.source_video).name})",
                "VIDEO REVIEW",
            )
        else:
            for clip in project.clips:
                self.cmb_preview_clip.addItem(clip.part_text, clip.part_text)
        if not project.clips and not review_clip:
            self.cmb_preview_clip.addItem("PART 1 (mặc định)", "PART 1")
        self.cmb_preview_clip.blockSignals(False)
        self._populate_export_parts(project)
        self._update_sub_file_status()
        if getattr(project.export_config, "pre_crop_enabled", False):
            self._load_wide_thumbnail()
        self._schedule_preview()

    def _populate_export_parts(self, project: Project):
        if not hasattr(self, "lst_export_parts"):
            return
        self.lst_export_parts.blockSignals(True)
        self.lst_export_parts.clear()
        review_clip = self._review_export_clip()
        clips = [review_clip] if review_clip else project.clips
        for clip in clips:
            label = (
                f"Video đã lồng tiếng  ({clip.duration:.1f}s)"
                if review_clip else f"{clip.part_text}  ({clip.duration:.1f}s)"
            )
            item = QListWidgetItem(label)
            item.setData(
                Qt.ItemDataRole.UserRole,
                "review-final" if review_clip else clip.index,
            )
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Checked if clip.enabled else Qt.CheckState.Unchecked
            )
            self.lst_export_parts.addItem(item)
        self.lst_export_parts.blockSignals(False)
        if self.lst_export_parts.count() and self.lst_export_parts.currentRow() < 0:
            self.lst_export_parts.setCurrentRow(0)
        # Load first clip's custom header + subtitle
        if project.clips and not review_clip:
            self.txt_custom_text.blockSignals(True)
            self.txt_custom_text.setPlainText(
                getattr(project.clips[0], "custom_header", "") or ""
            )
            self.txt_custom_text.blockSignals(False)
            self.txt_sub_custom.blockSignals(True)
            self.txt_sub_custom.setPlainText(
                getattr(project.clips[0], "custom_subtitle", "") or ""
            )
            self.txt_sub_custom.blockSignals(False)

    def _selected_export_clips(self) -> list:
        if not self._project:
            return []
        review_clip = self._review_export_clip()
        if review_clip:
            if (
                self.lst_export_parts.count()
                and self.lst_export_parts.item(0).checkState()
                != Qt.CheckState.Checked
            ):
                return []
            return [review_clip]
        if not hasattr(self, "lst_export_parts") or self.lst_export_parts.count() == 0:
            return [c for c in self._project.clips if c.enabled]

        selected_indexes = set()
        for row in range(self.lst_export_parts.count()):
            item = self.lst_export_parts.item(row)
            if item.checkState() == Qt.CheckState.Checked:
                selected_indexes.add(item.data(Qt.ItemDataRole.UserRole))
        return [c for c in self._project.clips if c.index in selected_indexes]

    def _set_all_export_parts(self, checked: bool):
        state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        self.lst_export_parts.blockSignals(True)
        for row in range(self.lst_export_parts.count()):
            item = self.lst_export_parts.item(row)
            item.setCheckState(state)
            if self._project and not self._is_review_export():
                idx = item.data(Qt.ItemDataRole.UserRole)
                for clip in self._project.clips:
                    if clip.index == idx:
                        clip.enabled = checked
                        break
        self.lst_export_parts.blockSignals(False)
        if self._project and not self._is_review_export():
            save_project(self._project)

    def _on_export_part_changed(self, item: QListWidgetItem):
        if not self._project:
            return
        if self._is_review_export():
            return
        idx = item.data(Qt.ItemDataRole.UserRole)
        checked = item.checkState() == Qt.CheckState.Checked
        for clip in self._project.clips:
            if clip.index == idx:
                clip.enabled = checked
                break
        save_project(self._project)

    def _on_export_part_current_changed(self, row: int):
        if row < 0:
            return
        item = self.lst_export_parts.item(row)
        if not item:
            return
        idx = item.data(Qt.ItemDataRole.UserRole)
        if self._is_review_export():
            self.cmb_preview_clip.setCurrentIndex(0)
            self.txt_custom_text.clear()
            self.txt_sub_custom.clear()
            self._schedule_preview()
            return
        if self._project:
            for clip in self._project.clips:
                if clip.index == idx:
                    combo_idx = self.cmb_preview_clip.findData(clip.part_text)
                    if combo_idx >= 0:
                        self.cmb_preview_clip.setCurrentIndex(combo_idx)
                    # Load per-clip custom header text
                    self.txt_custom_text.blockSignals(True)
                    self.txt_custom_text.setPlainText(
                        getattr(clip, "custom_header", "") or ""
                    )
                    self.txt_custom_text.blockSignals(False)
                    # Load per-clip custom subtitle text
                    self.txt_sub_custom.blockSignals(True)
                    self.txt_sub_custom.setPlainText(
                        getattr(clip, "custom_subtitle", "") or ""
                    )
                    self.txt_sub_custom.blockSignals(False)
                    self._schedule_preview()
                    break

    def _on_preview_clip_changed(self, _idx=None):
        """Sync the export parts list when preview clip dropdown changes."""
        part_text = self.cmb_preview_clip.currentData()
        if not part_text or not self._project:
            self._schedule_preview()
            return
        if self._is_review_export():
            self.lst_export_parts.blockSignals(True)
            if self.lst_export_parts.count():
                self.lst_export_parts.setCurrentRow(0)
            self.lst_export_parts.blockSignals(False)
            self._load_thumbnail()
            if self.cmb_aspect.currentData() == "custom_crop":
                self._load_wide_thumbnail()
            self._schedule_preview()
            return
        # Find matching row in the export parts list and select it
        for row in range(self.lst_export_parts.count()):
            item = self.lst_export_parts.item(row)
            idx = item.data(Qt.ItemDataRole.UserRole)
            for clip in self._project.clips:
                if clip.index == idx and clip.part_text == part_text:
                    self.lst_export_parts.blockSignals(True)
                    self.lst_export_parts.setCurrentRow(row)
                    self.lst_export_parts.blockSignals(False)
                    # Load this clip's custom header + subtitle text
                    self.txt_custom_text.blockSignals(True)
                    self.txt_custom_text.setPlainText(
                        getattr(clip, "custom_header", "") or ""
                    )
                    self.txt_custom_text.blockSignals(False)
                    self.txt_sub_custom.blockSignals(True)
                    self.txt_sub_custom.setPlainText(
                        getattr(clip, "custom_subtitle", "") or ""
                    )
                    self.txt_sub_custom.blockSignals(False)
                    self._load_thumbnail()
                    if self.cmb_aspect.currentData() == "custom_crop":
                        self._load_wide_thumbnail()
                    self._schedule_preview()
                    return
        self._schedule_preview()

    def _current_preview_clip(self):
        if not self._project:
            return None
        review_clip = self._review_export_clip()
        if review_clip:
            return review_clip
        part_text = self.cmb_preview_clip.currentData()
        row = self.lst_export_parts.currentRow()
        if 0 <= row < self.lst_export_parts.count():
            idx = self.lst_export_parts.item(row).data(Qt.ItemDataRole.UserRole)
            for clip in self._project.clips:
                if clip.index == idx:
                    return clip
        for clip in self._project.clips:
            if clip.part_text == part_text:
                return clip
        return self._project.clips[0] if self._project.clips else None

    def _populate_presets(self, gpu: bool):
        """Populate preset combo box for CPU (libx264) or GPU (NVENC)."""
        self.cmb_preset.clear()
        if gpu:
            # NVENC presets: p1 (fastest) → p7 (best quality)
            self.cmb_preset.addItems(["p1", "p2", "p3", "p4", "p5", "p6", "p7"])
            self.cmb_preset.setCurrentText("p4")
        else:
            self.cmb_preset.addItems([
                "ultrafast", "superfast", "veryfast", "faster", "fast", "medium",
            ])
            self.cmb_preset.setCurrentText("fast")

    def _on_gpu_toggled(self, checked: bool):
        """Switch between CPU and GPU encoder presets."""
        self._populate_presets(gpu=checked)
        if checked:
            self.lbl_crf.setText("CQ (thấp = tốt hơn):")
            self.spn_crf.setRange(0, 51)
            if self.spn_crf.value() > 30:
                self.spn_crf.setValue(23)
        else:
            self.lbl_crf.setText("CRF (thấp = tốt hơn):")

    def _on_sub_custom_changed(self):
        """Save custom subtitle text to the currently selected clip."""
        if not self._project:
            return
        if self._is_review_export():
            return
        text = self.txt_sub_custom.toPlainText()
        row = self.lst_export_parts.currentRow()
        if row < 0:
            return
        item = self.lst_export_parts.item(row)
        if not item:
            return
        idx = item.data(Qt.ItemDataRole.UserRole)
        for clip in self._project.clips:
            if clip.index == idx:
                clip.custom_subtitle = text.strip()
                save_project(self._project)
                break
        self._schedule_preview()

    def _on_custom_header_changed(self):
        """Save custom header text to the currently selected clip."""
        if not self._project:
            return
        if self._is_review_export():
            return
        text = self.txt_custom_text.toPlainText()
        row = self.lst_export_parts.currentRow()
        if row < 0:
            return
        item = self.lst_export_parts.item(row)
        if not item:
            return
        idx = item.data(Qt.ItemDataRole.UserRole)
        for clip in self._project.clips:
            if clip.index == idx:
                clip.custom_header = text.strip()
                save_project(self._project)
                break
        self._schedule_preview()

    # ─── Preview ───────────────────────────────────────────────────────────────

    def _update_sub_file_status(self):
        """Refresh the subtitle-file badge showing how many clips have .srt/.ass files."""
        if not self._project:
            self.lbl_sub_file_status.setText("📋 Chưa có project")
            self.lbl_sub_file_status.setStyleSheet("color:#888; font-size:10px;")
            return
        if self._is_review_export():
            subtitle = (
                getattr(self._project, "narration_subtitle_file", "") or ""
            )
            if subtitle and Path(subtitle).exists():
                self.lbl_sub_file_status.setText(
                    f"✅ Subtitle toàn video: {Path(subtitle).name}"
                )
                self.lbl_sub_file_status.setStyleSheet(
                    "color:#a6e3a1; font-size:10px; font-weight:bold;"
                )
            else:
                self.lbl_sub_file_status.setText(
                    "❌ Chưa có subtitle toàn video từ giọng đọc"
                )
                self.lbl_sub_file_status.setStyleSheet(
                    "color:#f38ba8; font-size:10px; font-weight:bold;"
                )
            return
        if not self._project.clips:
            self.lbl_sub_file_status.setText("📋 Chưa có phân cảnh")
            self.lbl_sub_file_status.setStyleSheet("color:#888; font-size:10px;")
            return

        total = len(self._project.clips)
        have = sum(
            1 for c in self._project.clips
            if c.subtitle_file and Path(c.subtitle_file).exists()
        )

        if have == 0:
            text = f"❌ 0/{total} Parts có file subtitle"
            style = "color:#f38ba8; font-size:10px; font-weight:bold;"
        elif have < total:
            text = f"⚠️ {have}/{total} Parts có file subtitle"
            style = "color:#f9e2af; font-size:10px; font-weight:bold;"
        else:
            text = f"✅ {have}/{total} Parts có file subtitle"
            style = "color:#a6e3a1; font-size:10px; font-weight:bold;"

        self.lbl_sub_file_status.setText(text)
        self.lbl_sub_file_status.setStyleSheet(style)

    def _clip_has_real_subtitle(self, clip) -> bool:
        """True when exporting this clip would use an existing subtitle file."""
        if clip.subtitle_file and Path(clip.subtitle_file).exists():
            return True
        cfg = self._project.export_config if self._project else None
        global_sub = getattr(cfg, "global_subtitle_file", "") if cfg else ""
        return bool(
            cfg
            and getattr(cfg, "subtitle_enabled", True)
            and global_sub
            and Path(global_sub).exists()
        )

    def _load_thumbnail(self):
        if not self._project:
            return
        clip = self._current_preview_clip()
        source_video = (
            getattr(clip, "source_video", "") if clip else ""
        ) or self._project.source_video
        clip_idx = getattr(clip, "index", 0) if clip else 0
        thumb_path = str(
            Path(self._project.output_dir) / "previews" / f"thumb_part{clip_idx:02d}.jpg"
        )
        if Path(thumb_path).exists():
            self._on_thumb_ready(thumb_path)
        else:
            # Several UI signals can request the same preview while a project
            # is being loaded.  Replacing the only Python reference to a live
            # QThread makes Qt abort the whole application.
            if self._thumb_worker and self._thumb_worker.isRunning():
                return
            dur = getattr(clip, "duration", 0) or self._project.video_metadata.get("duration") or 60
            offset = min(30.0, float(dur) * 0.1)
            self._thumb_worker = ThumbnailWorker(
                source_video, thumb_path, offset
            )
            self._thumb_worker.finished.connect(self._on_thumb_ready)
            self._thumb_worker.start()
            self.lbl_thumb_status.setText("⏳ Đang tạo thumbnail…")

    def _on_thumb_ready(self, path: str):
        if path and Path(path).exists():
            self._thumb_pixmap = QPixmap(path)
            self.lbl_thumb_status.setText("✅ Thumbnail sẵn sàng")
        else:
            self._thumb_pixmap = None
            self.lbl_thumb_status.setText("⚠️ Không tạo được thumbnail")
        self._update_preview()

    def _load_wide_thumbnail(self):
        """Load wide (original-aspect-ratio) thumbnail for the crop-adjust widget."""
        if not self._project:
            return
        clip = self._current_preview_clip()
        source_video = (
            getattr(clip, "source_video", "") if clip else ""
        ) or self._project.source_video
        clip_idx = getattr(clip, "index", 0) if clip else 0
        wide_path = str(
            Path(self._project.output_dir) / "previews" / f"wide_thumb_part{clip_idx:02d}.jpg"
        )
        if Path(wide_path).exists():
            self._on_wide_thumb_ready(wide_path)
            return
        # Watermark, aspect and pre-crop setup may all request this image in
        # one event cycle.  Keep the running worker alive instead of replacing
        # it ("QThread: Destroyed while thread is still running").
        if self._wide_thumb_worker and self._wide_thumb_worker.isRunning():
            return
        dur = getattr(clip, "duration", 0) or self._project.video_metadata.get("duration") or 60
        offset = min(30.0, float(dur) * 0.1)
        self._wide_thumb_worker = WideThumbnailWorker(
            source_video, wide_path, offset
        )
        self._wide_thumb_worker.finished.connect(self._on_wide_thumb_ready)
        self._wide_thumb_worker.start()
        self.lbl_thumb_status.setText("⏳ Đang tạo wide thumbnail…")

    def _on_wide_thumb_ready(self, path: str):
        if path and Path(path).exists():
            self._wide_pixmap = QPixmap(path)
            self._crop_adjust_widget.set_thumbnail(self._wide_pixmap)
            self._wm_adjust_widget.set_thumbnail(self._wide_pixmap)
        else:
            self._wide_pixmap = None
        self._update_preview()

    def _schedule_preview(self, *_):
        self._preview_timer.start(220)

    def _pre_cropped_pixmap(self, pix: QPixmap | None, cfg) -> QPixmap | None:
        if not pix or pix.isNull():
            return pix
        if not getattr(cfg, "pre_crop_enabled", False):
            return pix
        ww, wh = pix.width(), pix.height()
        cw = max(2, int(ww * max(0.05, min(1.0, getattr(cfg, "pre_crop_w", 1.0)))))
        ch = max(2, int(wh * max(0.05, min(1.0, getattr(cfg, "pre_crop_h", 1.0)))))
        cx = max(0.0, min(1.0, getattr(cfg, "pre_crop_x", 0.5)))
        cy = max(0.0, min(1.0, getattr(cfg, "pre_crop_y", 0.5)))
        x = int(max(0, ww - cw) * cx)
        y = int(max(0, wh - ch) * cy)
        cropped = pix.copy(x, y, cw, ch)
        return cropped if not cropped.isNull() else pix

    def _preview_thumb(self, cfg) -> QPixmap | None:
        if getattr(cfg, "pre_crop_enabled", False) and self._wide_pixmap and not self._wide_pixmap.isNull():
            return self._pre_cropped_pixmap(self._wide_pixmap, cfg)
        return self._thumb_pixmap

    def _update_preview(self):
        cfg = self._collect_config()
        part_text = self.cmb_preview_clip.currentData() or "PART 1"
        thumb_src = self._preview_thumb(cfg)
        wide_src = self._pre_cropped_pixmap(self._wide_pixmap, cfg)

        canvas = QPixmap(_PW, _PH)
        canvas.fill(QColor("#111"))
        painter = QPainter(canvas)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)

        # ── Draw thumbnail ────────────────────────────────────────────────
        if cfg.aspect_mode == "custom_crop" and wide_src and not wide_src.isNull():
            # Show the portion that would be cropped from the wide source
            ww = wide_src.width()
            wh = wide_src.height()
            cx_pct = max(0.0, min(1.0, cfg.crop_x))
            cy_pct = max(0.0, min(1.0, cfg.crop_y))
            cbox_w = max(2, int(ww * max(0.05, min(1.0, getattr(cfg, "crop_w", 1.0)))))
            cbox_h = max(2, int(wh * max(0.05, min(1.0, getattr(cfg, "crop_h", 1.0)))))
            x_off = int((ww - cbox_w) * cx_pct)
            y_off = int((wh - cbox_h) * cy_pct)
            cropped = wide_src.copy(x_off, y_off, cbox_w, cbox_h)
            if not cropped.isNull():
                thumb = cropped.scaled(
                    _PW, _PH,
                    Qt.AspectRatioMode.IgnoreAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                painter.drawPixmap(0, 0, thumb)
        elif cfg.aspect_mode == "keep_ratio" and thumb_src and not thumb_src.isNull():
            # Keep original ratio — scale to fit, centred with black bars
            thumb = thumb_src.scaled(
                _PW, _PH,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            ox = (_PW - thumb.width()) // 2
            oy = (_PH - thumb.height()) // 2
            painter.drawPixmap(ox, oy, thumb)
        elif cfg.aspect_mode == "black_bg" and thumb_src and not thumb_src.isNull():
            # Black background — scale to fit with black bars (already black canvas)
            thumb = thumb_src.scaled(
                _PW, _PH,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            ox = (_PW - thumb.width()) // 2
            oy = (_PH - thumb.height()) // 2
            painter.drawPixmap(ox, oy, thumb)
        elif cfg.aspect_mode == "blur_bg" and thumb_src and not thumb_src.isNull():
            # Blur background — zoomed blurred copy behind the fitted video
            bg = thumb_src.scaled(
                _PW, _PH,
                Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                Qt.TransformationMode.SmoothTransformation,
            )
            bx = (bg.width() - _PW) // 2
            by = (bg.height() - _PH) // 2
            painter.drawPixmap(0, 0, bg, bx, by, _PW, _PH)
            # Darken to simulate blur (real blur is expensive in Qt)
            overlay = QColor(0, 0, 0)
            overlay.setAlphaF(0.55)
            painter.fillRect(0, 0, _PW, _PH, overlay)
            # Draw fitted video on top
            fg = thumb_src.scaled(
                _PW, _PH,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            fx = (_PW - fg.width()) // 2
            fy = (_PH - fg.height()) // 2
            painter.drawPixmap(fx, fy, fg)
        elif cfg.aspect_mode == "image_bg":
            # Custom image background covering the canvas, with the video fitted
            bgpx = self._get_bg_pixmap()
            if bgpx and not bgpx.isNull():
                bg = bgpx.scaled(
                    _PW, _PH,
                    Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                    Qt.TransformationMode.SmoothTransformation,
                )
                bx = (bg.width() - _PW) // 2
                by = (bg.height() - _PH) // 2
                painter.drawPixmap(0, 0, bg, bx, by, _PW, _PH)
            else:
                painter.fillRect(0, 0, _PW, _PH, QColor("#444"))
            if thumb_src and not thumb_src.isNull():
                scale = _PW / cfg.width
                is_shrink = cfg.part_text_enabled and cfg.part_text_mode == "shrink"
                hh = max(0, int(cfg.header_height * scale)) if is_shrink else 0
                bar_h = (int(getattr(cfg, "subtitle_bar_height", 0) * scale)
                         if cfg.subtitle_style in ("bar", "shrink_bar") else 0)
                area_h = max(10, _PH - hh - bar_h)
                fg = thumb_src.scaled(
                    _PW, area_h,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                fx = (_PW - fg.width()) // 2
                fy = hh + (area_h - fg.height()) // 2
                painter.drawPixmap(fx, fy, fg)
        elif thumb_src and not thumb_src.isNull():
            # Center crop (default) — fill and crop
            thumb = thumb_src.scaled(
                _PW, _PH,
                Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                Qt.TransformationMode.SmoothTransformation,
            )
            ox = (thumb.width() - _PW) // 2
            oy = (thumb.height() - _PH) // 2
            painter.drawPixmap(0, 0, thumb, ox, oy, _PW, _PH)

        # ── Collect text bounding rects for drag-and-drop ────────────────
        if getattr(cfg, "remove_watermark_enabled", False):
            regions = getattr(cfg, "remove_watermark_regions", None) or [{
                "x": cfg.remove_watermark_x, "y": cfg.remove_watermark_y,
                "w": cfg.remove_watermark_w, "h": cfg.remove_watermark_h,
            }]
            for index, region in enumerate(regions):
                mode = cfg.remove_watermark_mode or "blur"
                norm_x = max(0.0, min(1.0, region.get("x", 0.82)))
                norm_y = max(0.0, min(1.0, region.get("y", 0.88)))
                norm_w = max(0.05, min(1.0, region.get("w", 0.16)))
                norm_h = max(0.05, min(1.0, region.get("h", 0.08)))
                if mode == "delogo":
                    norm_x, norm_y, norm_w, norm_h = _refine_delogo_region(
                        norm_x,
                        norm_y,
                        norm_w,
                        norm_h,
                        getattr(cfg, "remove_watermark_strength", 3),
                    )
                rw = max(8, int(_PW * norm_w))
                rh = max(8, int(_PH * norm_h))
                rx = int(max(0, _PW - rw) * norm_x)
                ry = int(max(0, _PH - rh) * norm_y)
                if mode == "cover":
                    cover = _to_qcolor(getattr(cfg, "remove_watermark_color", "black") or "black")
                    cover.setAlphaF(0.9)
                    painter.fillRect(QRect(rx, ry, rw, rh), cover)
                elif mode == "delogo":
                    # Qt preview cannot reproduce FFmpeg's edge interpolation
                    # exactly.  Show a light translucent guide; the exported
                    # video reconstructs the selected area from its edges.
                    patch_color = QColor(137, 180, 250, 45)
                    painter.fillRect(QRect(rx, ry, rw, rh), patch_color)
                else:
                    shade = QColor(0, 0, 0)
                    strength = max(1, min(5, getattr(cfg, "remove_watermark_strength", 3)))
                    shade.setAlphaF(min(0.75, (0.20 + strength * 0.10) if mode == "blur" else (0.30 + strength * 0.10)))
                    painter.fillRect(QRect(rx, ry, rw, rh), shade)
                color = "#89b4fa" if index == self._wm_region_index else "#f9e2af"
                painter.setPen(QPen(QColor(color), 2 if index == self._wm_region_index else 1, Qt.PenStyle.DashLine))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawRect(QRect(rx, ry, rw, rh))

        text_rects: dict[str, QRect] = {}

        # ── Draw PART text overlay ────────────────────────────────────────
        if cfg.part_text_enabled:
            scale = _PW / cfg.width
            fs = max(7, int(cfg.part_text_fontsize * scale))
            # Per-clip custom header from the text field (synced with current clip)
            text = self.txt_custom_text.toPlainText().strip() or part_text
            text_lines = text.split("\n") if "\n" in text else [text]
            mode = cfg.part_text_mode or "overlay"
            tc = _to_qcolor(cfg.part_text_color or "white")
            bc = _to_qcolor(cfg.part_text_bg_color or "black")

            font = QFont(getattr(cfg, "part_text_font", "Arial") or "Arial")
            font.setPointSize(fs)
            font.setBold(getattr(cfg, "part_text_bold", True))
            font.setItalic(getattr(cfg, "part_text_italic", False))
            font.setUnderline(getattr(cfg, "part_text_underline", False))
            painter.setFont(font)
            fm = QFontMetrics(font)
            tw = max(fm.horizontalAdvance(ln) for ln in text_lines)
            th = fm.height()
            total_th = th * len(text_lines)
            cx = (_PW - tw) // 2

            # Y position: manual drag or preset
            part_y_pct = getattr(cfg, "part_text_y_pct", -1.0)
            pos = cfg.part_text_position or "top"
            if part_y_pct >= 0:
                cy = int(part_y_pct * _PH)
            else:
                cy = (int(_PH * 0.06) + th if pos == "top"
                      else _PH - int(_PH * 0.06) if pos == "bottom"
                      else _PH // 2 + th // 2)

            if mode == "shrink":
                hh = max(20, int(cfg.header_height * scale))
                if cfg.aspect_mode != "image_bg":
                    # image_bg shows the picture behind the title (no solid band)
                    painter.fillRect(QRect(0, 0, _PW, hh), bc)
                # Title is draggable in every shrink mode: dragged → free
                # position; otherwise centred inside the header band.
                if part_y_pct >= 0:
                    y0 = cy
                else:
                    y0 = (hh - total_th) // 2 + th - 2
                painter.setPen(tc)
                for li, line in enumerate(text_lines):
                    lx = max(4, (_PW - fm.horizontalAdvance(line)) // 2)
                    painter.drawText(lx, y0 + li * th, line)
                text_rects["part"] = QRect(cx - 4, y0 - th - 4, tw + 8, total_th + 8)

            elif mode == "watermark":
                op = getattr(cfg, "watermark_opacity", 0.35)
                tc2 = QColor(tc); tc2.setAlphaF(op)
                sc = QColor(0, 0, 0); sc.setAlphaF(min(op * 0.8, 0.9))
                for li, line in enumerate(text_lines):
                    ly = cy + li * th
                    lx = max(4, (_PW - fm.horizontalAdvance(line)) // 2)
                    painter.setPen(sc); painter.drawText(lx + 1, ly + 1, line)
                    painter.setPen(tc2); painter.drawText(lx, ly, line)
                text_rects["part"] = QRect(cx - 4, cy - th - 4, tw + 8, total_th + 8)

            elif mode == "lower_third":
                bh = max(16, int(_PH * 0.085))
                bg = QColor(bc); bg.setAlphaF(0.82)
                painter.fillRect(QRect(0, _PH - bh, _PW, bh), bg)
                painter.setPen(tc)
                y0 = _PH - int(bh * 0.28) - (len(text_lines) - 1) * th
                for li, line in enumerate(text_lines):
                    lx = max(4, (_PW - fm.horizontalAdvance(line)) // 2)
                    painter.drawText(lx, y0 + li * th, line)

            elif mode == "outline":
                for li, line in enumerate(text_lines):
                    ly = cy + li * th
                    lx = max(4, (_PW - fm.horizontalAdvance(line)) // 2)
                    painter.setPen(QColor(0, 0, 0))
                    for dx, dy in [(-1,-1),(1,-1),(-1,1),(1,1),(0,-1),(0,1),(-1,0),(1,0)]:
                        painter.drawText(lx + dx, ly + dy, line)
                    painter.setPen(tc); painter.drawText(lx, ly, line)
                text_rects["part"] = QRect(cx - 4, cy - th - 4, tw + 8, total_th + 8)

            else:  # overlay (default)
                pad = 4
                bg = QColor(0, 0, 0); bg.setAlphaF(0.55)
                painter.fillRect(QRect(cx - pad, cy - th - pad, tw + pad * 2, total_th + pad * 2), bg)
                painter.setPen(tc)
                for li, line in enumerate(text_lines):
                    lx = max(4, (_PW - fm.horizontalAdvance(line)) // 2)
                    painter.drawText(lx, cy + li * th, line)
                text_rects["part"] = QRect(cx - pad, cy - th - pad, tw + pad * 2, total_th + pad * 2)

            # Draw drag handle indicator for draggable modes
            if "part" in text_rects:
                r = text_rects["part"]
                pen = QPen(QColor("#7aa2f7"), 1, Qt.PenStyle.DashLine)
                painter.setPen(pen)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawRect(r.adjusted(-2, -2, 2, 2))

        # ── Draw subtitle preview ─────────────────────────────────────────
        sub_style = cfg.subtitle_style if cfg.subtitle_style != "none" else None
        # Read custom subtitle from the text field (which is synced with the
        # currently previewed clip via _on_preview_clip_changed)
        custom_sub = self.txt_sub_custom.toPlainText().strip()
        # Show subtitle preview when: style selected OR custom text entered
        if sub_style or custom_sub:
            sub_scale = _PW / cfg.width
            sub_fs = max(8, int(self.spn_sub_fontsize.value() * sub_scale))
            sample = custom_sub if custom_sub else "Sample subtitle text here"
            if sub_style == "word":
                sample = (sample.split() or ["Sample"])[0]
            # For multi-line preview, use first line for layout calculation
            sample_lines = sample.split("\n") if "\n" in sample else [sample]
            sample_display = sample_lines[0]  # first line for position calc

            sfont = QFont(self.cmb_sub_font.currentData() or "Arial")
            sfont.setPointSize(sub_fs)
            if self.chk_sub_bold.isChecked():
                sfont.setBold(True)
            if self.chk_sub_italic.isChecked():
                sfont.setItalic(True)
            if self.chk_sub_underline.isChecked():
                sfont.setUnderline(True)
            painter.setFont(sfont)
            sfm = QFontMetrics(sfont)
            stw = max(sfm.horizontalAdvance(ln) for ln in sample_lines)
            sth = sfm.height()
            scx = max(4, (_PW - stw) // 2)

            sub_pos = cfg.subtitle_position or "bottom"
            sub_tc = _to_qcolor(self.cmb_sub_color.currentText())
            sub_hl = _to_qcolor(self.cmb_sub_highlight.currentText())

            # Y position: manual drag or preset
            sub_y_pct = getattr(cfg, "subtitle_y_pct", -1.0)

            if sub_style in ("bar", "shrink_bar"):
                # ── Bar mode: text on coloured background bar ─────────
                bar_h = max(20, int(getattr(cfg, "subtitle_bar_height", 200) * sub_scale))
                bar_bg = _to_qcolor(getattr(cfg, "subtitle_bg_color", "white"))
                if sub_y_pct >= 0:
                    bar_y = max(0, int(sub_y_pct * _PH) - bar_h // 2)
                elif sub_pos == "top":
                    bar_y = 0
                elif sub_pos == "center":
                    bar_y = (_PH - bar_h) // 2
                else:
                    bar_y = _PH - bar_h
                painter.fillRect(QRect(0, bar_y, _PW, bar_h), bar_bg)
                total_text_h = sth * len(sample_lines)
                text_y_start = bar_y + (bar_h - total_text_h) // 2 + sth
                painter.setPen(sub_tc)
                for li, line in enumerate(sample_lines):
                    lx = max(4, (_PW - sfm.horizontalAdvance(line)) // 2)
                    painter.drawText(lx, text_y_start + li * sth, line)
                text_rects["sub"] = QRect(0, bar_y, _PW, bar_h)
            else:
                # ── Standard subtitle (plain/karaoke/custom-only) ─────
                if sub_y_pct >= 0:
                    scy = int(sub_y_pct * _PH)
                else:
                    margin_v_px = getattr(cfg, "subtitle_margin_v", 0)
                    if margin_v_px > 0:
                        mv_scaled = int(margin_v_px * (_PH / cfg.height))
                    else:
                        mv_scaled = int(_PH * 0.04)

                    if sub_pos == "top":
                        scy = mv_scaled + sth
                    elif sub_pos == "bottom":
                        scy = _PH - mv_scaled
                    else:
                        scy = _PH // 2 + sth // 2

                # Draw each line with black outline
                for li, line in enumerate(sample_lines):
                    ly = scy + li * sth
                    lx = max(4, (_PW - sfm.horizontalAdvance(line)) // 2)
                    painter.setPen(QColor(0, 0, 0))
                    for dx, dy in [(-1,-1),(1,-1),(-1,1),(1,1),(0,-1),(0,1),(-1,0),(1,0)]:
                        painter.drawText(lx + dx, ly + dy, line)

                    if sub_style == "karaoke" and li == 0:
                        words = line.split()
                        first = words[0] if words else ""
                        rest = " " + " ".join(words[1:]) if len(words) > 1 else ""
                        fw = sfm.horizontalAdvance(first)
                        painter.setPen(sub_hl)
                        painter.drawText(lx, ly, first)
                        painter.setPen(sub_tc)
                        painter.drawText(lx + fw, ly, rest)
                    else:
                        painter.setPen(sub_tc)
                        painter.drawText(lx, ly, line)

                text_rects["sub"] = QRect(
                    scx - 4, scy - sth - 4, stw + 8, sth + 8
                )

            # Draw drag handle for subtitle
            if "sub" in text_rects:
                r = text_rects["sub"]
                pen = QPen(QColor("#a6e3a1"), 1, Qt.PenStyle.DashLine)
                painter.setPen(pen)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawRect(r.adjusted(-2, -2, 2, 2))

        painter.end()
        self._preview_lbl.set_text_rects(text_rects)
        self._preview_lbl.update_raw(canvas)

    # ─── Form helpers ──────────────────────────────────────────────────────────

    def _on_aspect_mode_changed(self, _idx=None):
        mode = self.cmb_aspect.currentData() or "center_crop"
        is_custom = (mode == "custom_crop")
        self._crop_group.setVisible(is_custom)
        self._crop_adjust_widget.setVisible(is_custom)
        self._bgimg_group.setVisible(mode == "image_bg")
        pre_crop = bool(self._project and getattr(self._project.export_config, "pre_crop_enabled", False))
        if (is_custom or self.chk_remove_wm.isChecked() or pre_crop) and self._project:
            self._load_wide_thumbnail()
        self._schedule_preview()

    def _update_wm_label(self):
        self.lbl_remove_wm_xy.setText(
            f"Vùng {self._wm_region_index + 1}/{max(1, len(self._wm_regions))} — "
            f"X: {int(self._wm_x*100)}% / Y: {int(self._wm_y*100)}% / "
            f"W: {int(self._wm_w*100)}% / H: {int(self._wm_h*100)}%"
        )

    def _refresh_wm_region_combo(self):
        self.cmb_wm_region.blockSignals(True)
        self.cmb_wm_region.clear()
        for index in range(len(self._wm_regions)):
            self.cmb_wm_region.addItem(f"Vùng {index + 1}", index)
        if self._wm_regions:
            self._wm_region_index = max(
                0, min(self._wm_region_index, len(self._wm_regions) - 1)
            )
            self.cmb_wm_region.setCurrentIndex(self._wm_region_index)
        self.cmb_wm_region.blockSignals(False)
        self.btn_remove_wm_region.setEnabled(
            self.chk_remove_wm.isChecked() and len(self._wm_regions) > 1
        )

    def _load_selected_wm_region(self):
        if not self._wm_regions:
            return
        region = self._wm_regions[self._wm_region_index]
        self._wm_x = max(0.0, min(1.0, float(region.get("x", 0.82))))
        self._wm_y = max(0.0, min(1.0, float(region.get("y", 0.88))))
        self._wm_w = max(0.05, min(1.0, float(region.get("w", 0.16))))
        self._wm_h = max(0.05, min(1.0, float(region.get("h", 0.08))))
        self._wm_adjust_widget.set_crop(
            self._wm_x, self._wm_y, self._wm_w, self._wm_h
        )
        self._update_wm_label()

    def _on_wm_region_changed(self, index: int):
        if 0 <= index < len(self._wm_regions):
            self._wm_region_index = index
            self._load_selected_wm_region()
            self._schedule_preview()

    def _add_wm_region(self):
        presets = ((0.10, 0.10), (0.82, 0.10), (0.10, 0.88), (0.82, 0.88))
        x, y = presets[len(self._wm_regions) % len(presets)]
        self._wm_regions.append({"x": x, "y": y, "w": 0.16, "h": 0.08})
        self._wm_region_index = len(self._wm_regions) - 1
        self._refresh_wm_region_combo()
        self._load_selected_wm_region()
        self._schedule_preview()

    def _remove_wm_region(self):
        if len(self._wm_regions) <= 1:
            return
        self._wm_regions.pop(self._wm_region_index)
        self._wm_region_index = min(
            self._wm_region_index, len(self._wm_regions) - 1
        )
        self._refresh_wm_region_combo()
        self._load_selected_wm_region()
        self._schedule_preview()

    def _on_remove_wm_toggled(self, checked: bool):
        self.cmb_remove_wm_mode.setEnabled(checked)
        self.spn_remove_wm_strength.setEnabled(checked)
        self.cmb_remove_wm_color.setEnabled(checked)
        self.cmb_wm_region.setEnabled(checked)
        self.btn_add_wm_region.setEnabled(checked)
        self.btn_remove_wm_region.setEnabled(checked and len(self._wm_regions) > 1)
        self._wm_adjust_widget.setVisible(checked)
        if checked and self._project:
            self._load_wide_thumbnail()
        self._schedule_preview()

    def _on_pick_bg_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Chọn ảnh nền", "",
            "Ảnh (*.png *.jpg *.jpeg *.bmp *.webp)",
        )
        if path:
            self._bg_image_path = path
            self._lbl_bgimg.setText(Path(path).name)
            if self._project:
                self._project.export_config.background_image = path
            self._schedule_preview()

    def _on_clear_bg_image(self):
        self._bg_image_path = ""
        self._lbl_bgimg.setText("(chưa chọn ảnh)")
        if self._project:
            self._project.export_config.background_image = ""
        self._schedule_preview()

    def _on_background_music_toggled(self, checked: bool):
        self.btn_pick_background_music.setEnabled(checked)
        self.btn_clear_background_music.setEnabled(
            checked and bool(self._bg_music_path)
        )
        self.spn_background_music_volume.setEnabled(checked)

    def _on_pick_background_music(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Chọn nhạc nền",
            "",
            "Audio (*.mp3 *.wav *.m4a *.aac *.flac *.ogg *.opus);;"
            "Tất cả file (*)",
        )
        if not path:
            return
        self._bg_music_path = path
        self.lbl_background_music.setText(Path(path).name)
        self.lbl_background_music.setToolTip(path)
        self.btn_clear_background_music.setEnabled(True)

    def _on_clear_background_music(self):
        self._bg_music_path = ""
        self.lbl_background_music.setText("(chưa chọn nhạc)")
        self.lbl_background_music.setToolTip("")
        self.btn_clear_background_music.setEnabled(False)

    def _get_bg_pixmap(self):
        """Return a cached QPixmap of the chosen background image (or None)."""
        path = getattr(self, "_bg_image_path", "") or ""
        if not path or not Path(path).exists():
            return None
        if getattr(self, "_bgimg_cache_path", None) != path:
            self._bgimg_cache = QPixmap(path)
            self._bgimg_cache_path = path
        return self._bgimg_cache

    def _on_sub_style_changed(self, _idx=None):
        style = self.cmb_sub_style.currentData() or "none"
        is_bar = style in ("bar", "shrink_bar")
        self.cmb_sub_bg_color.setVisible(is_bar)
        self.lbl_sub_bg_color.setVisible(is_bar)
        self.spn_sub_bar_h.setVisible(is_bar)
        self.lbl_sub_bar_h.setVisible(is_bar)
        self.cmb_sub_bg_color.setEnabled(is_bar)
        self._schedule_preview()

    def _on_ai_title_provider_changed(self, _idx=None):
        provider = self.cmb_ai_title_provider.currentData() or "gemini"
        self.cmb_ai_title_model.clear()
        models = {
            "gemini": GEMINI_FREE_MODELS,
            "groq": GROQ_FREE_MODELS,
            "openrouter": OPENROUTER_FREE_MODELS,
            "ollama": OLLAMA_SUGGESTED,
        }.get(provider, GEMINI_FREE_MODELS)
        for mid, label in models:
            self.cmb_ai_title_model.addItem(label, mid)

    def _start_ai_titles(self):
        if not self._project or not self._project.clips:
            QMessageBox.warning(self, "Chua co Parts", "Can tao/chia Parts truoc.")
            return
        provider = self.cmb_ai_title_provider.currentData() or "gemini"
        if provider != "ollama" and not load_api_key(provider):
            QMessageBox.warning(
                self,
                "Thieu API key",
                f"Chua co API key cho {provider}. Vao Cai dat -> API Keys.",
            )
            return
        self.btn_ai_titles.setEnabled(False)
        self.lbl_ai_title_status.setText("Dang goi AI...")
        self._ai_title_worker = _AITitleWorker(
            self._project,
            provider,
            self.cmb_ai_title_model.currentData() or "",
            self.chk_ai_title_bottom.isChecked(),
        )
        self._ai_title_worker.status.connect(self.lbl_ai_title_status.setText)
        self._ai_title_worker.finished.connect(self._on_ai_titles_finished)
        self._ai_title_worker.start()

    def _on_ai_titles_finished(self, ok: bool, result):
        self.btn_ai_titles.setEnabled(True)
        if not ok:
            self.lbl_ai_title_status.setText(str(result))
            QMessageBox.critical(self, "Loi AI title", str(result))
            return
        by_index = {int(item.get("index", 0)): item for item in result.get("parts", [])}
        applied = 0
        for clip in self._project.clips:
            item = by_index.get(clip.index)
            if not item:
                continue
            top = (item.get("top_title") or "").strip()
            bottom = (item.get("bottom_title") or "").strip()
            if top:
                clip.custom_header = top
            if (
                self.chk_ai_title_bottom.isChecked()
                and bottom
                and not self._clip_has_real_subtitle(clip)
            ):
                clip.custom_subtitle = bottom
            applied += 1
        save_project(self._project)
        self.refresh_clips(self._project)
        self.lbl_ai_title_status.setText(f"Da ap dung title AI cho {applied} Parts.")
        self._schedule_preview()

    def _on_part_mode_changed(self, _idx=None):
        mode = self.cmb_part_mode.currentData() or "overlay"
        self.lbl_opacity.setVisible(mode == "watermark")
        self.spn_opacity.setVisible(mode == "watermark")
        self.spn_header_h.setEnabled(mode == "shrink")

    def _on_crop_slider_changed(self):
        cx = self._sld_crop_x.value() / 100.0
        cy = self._sld_crop_y.value() / 100.0
        self._lbl_crop_xy.setText(
            f"X: {self._sld_crop_x.value()}% / Y: {self._sld_crop_y.value()}%"
        )
        self._crop_adjust_widget.set_crop(cx, cy, 1.0, 1.0)
        self._schedule_preview()

    def _on_crop_drag(self, cx: float, cy: float, cw: float, ch: float):
        self._crop_w = cw
        self._crop_h = ch
        """Crop widget was dragged — sync sliders without re-emitting valueChanged."""
        self._sld_crop_x.blockSignals(True)
        self._sld_crop_y.blockSignals(True)
        self._sld_crop_x.setValue(int(cx * 100))
        self._sld_crop_y.setValue(int(cy * 100))
        self._lbl_crop_xy.setText(
            f"X: {int(cx*100)}% / Y: {int(cy*100)}% / Zoom: {int(100 / max(cw, 0.01))}%"
        )
        self._sld_crop_x.blockSignals(False)
        self._sld_crop_y.blockSignals(False)
        self._schedule_preview()

    def _on_wm_drag(self, cx: float, cy: float, cw: float, ch: float):
        self._wm_x = max(0.0, min(1.0, cx))
        self._wm_y = max(0.0, min(1.0, cy))
        self._wm_w = max(0.05, min(1.0, cw))
        self._wm_h = max(0.05, min(1.0, ch))
        if self._wm_regions and 0 <= self._wm_region_index < len(self._wm_regions):
            self._wm_regions[self._wm_region_index] = {
                "x": self._wm_x, "y": self._wm_y,
                "w": self._wm_w, "h": self._wm_h,
            }
        self._update_wm_label()
        self._schedule_preview()

    def _on_sub_margin_changed(self, val: int):
        if val == 0:
            self.lbl_sub_margin_v.setText("MarginV: tự động (~4% chiều cao)")
        else:
            self.lbl_sub_margin_v.setText(f"MarginV: {val} px")
        self._schedule_preview()

    def _on_part_pos_changed(self, _idx=None):
        """Reset manual drag position when user selects a preset position."""
        self._part_text_y_pct = -1.0
        self._schedule_preview()

    def _on_sub_pos_changed(self, _idx=None):
        """Reset manual drag position when user selects a preset position."""
        self._subtitle_y_pct = -1.0
        self._schedule_preview()

    def _on_text_dragged(self, name: str, y_pct: float):
        """Handle drag-and-drop from the preview label."""
        if name == "part":
            self._part_text_y_pct = y_pct
        elif name == "sub":
            self._subtitle_y_pct = y_pct
        self._schedule_preview()

    def _toggle_part_text(self, state: bool):
        for w in (self.cmb_part_mode, self.cmb_part_pos, self.spn_fontsize,
                  self.cmb_part_font, self.chk_part_bold, self.chk_part_italic,
                  self.chk_part_underline,
                  self.cmb_text_color, self.cmb_bg_color, self.spn_header_h,
                  self.txt_custom_text, self.spn_opacity):
            w.setEnabled(state)

    def _browse_output(self):
        d = QFileDialog.getExistingDirectory(self, "Chọn thư mục xuất")
        if d:
            self.txt_out_dir.setText(d)

    def _collect_config(self) -> ExportConfig:
        cfg = ExportConfig()
        cfg.platform = self.cmb_platform.currentText()
        cfg.aspect_mode = self.cmb_aspect.currentData() or "center_crop"
        cfg.part_text_enabled = self.chk_part_text.isChecked()
        cfg.part_text_mode = self.cmb_part_mode.currentData() or "overlay"
        cfg.part_text_position = self.cmb_part_pos.currentData() or "top"
        cfg.part_text_font = self.cmb_part_font.currentData() or "Arial"
        cfg.part_text_bold = self.chk_part_bold.isChecked()
        cfg.part_text_italic = self.chk_part_italic.isChecked()
        cfg.part_text_underline = self.chk_part_underline.isChecked()
        cfg.part_text_fontsize = self.spn_fontsize.value()
        cfg.part_text_color = self.cmb_text_color.currentText()
        cfg.part_text_bg_color = self.cmb_bg_color.currentText()
        header_h = self.spn_header_h.value()
        if header_h % 2:
            header_h -= 1
        cfg.header_height = max(50, header_h)
        # watermark_text is now per-clip (clip.custom_header), not global
        cfg.watermark_opacity = self.spn_opacity.value()
        cfg.crop_x = self._sld_crop_x.value() / 100.0
        cfg.crop_y = self._sld_crop_y.value() / 100.0
        cfg.crop_w = self._crop_w
        cfg.crop_h = self._crop_h
        cfg.remove_watermark_enabled = self.chk_remove_wm.isChecked()
        cfg.remove_watermark_mode = self.cmb_remove_wm_mode.currentData() or "blur"
        cfg.remove_watermark_x = self._wm_x
        cfg.remove_watermark_y = self._wm_y
        cfg.remove_watermark_w = self._wm_w
        cfg.remove_watermark_h = self._wm_h
        cfg.remove_watermark_strength = self.spn_remove_wm_strength.value()
        cfg.remove_watermark_color = self.cmb_remove_wm_color.currentText()
        cfg.remove_watermark_regions = [dict(region) for region in self._wm_regions]
        cfg.background_image = getattr(self, "_bg_image_path", "") or ""
        cfg.background_music_enabled = self.chk_background_music.isChecked()
        cfg.background_music_path = self._bg_music_path
        cfg.background_music_volume = self.spn_background_music_volume.value()
        cfg.use_gpu = self.chk_gpu.isChecked()
        cfg.preset = self.cmb_preset.currentText()
        cfg.crf = self.spn_crf.value()
        cfg.subtitle_style = self.cmb_sub_style.currentData() or "none"
        cfg.subtitle_font = self.cmb_sub_font.currentData() or "Arial"
        cfg.subtitle_bold = self.chk_sub_bold.isChecked()
        cfg.subtitle_italic = self.chk_sub_italic.isChecked()
        cfg.subtitle_underline = self.chk_sub_underline.isChecked()
        cfg.subtitle_fontsize = self.spn_sub_fontsize.value()
        cfg.subtitle_color = self.cmb_sub_color.currentText()
        cfg.subtitle_highlight_color = self.cmb_sub_highlight.currentText()
        cfg.subtitle_position = self.cmb_sub_pos.currentData() or "bottom"
        cfg.subtitle_margin_v = self.sld_sub_margin_v.value()
        cfg.subtitle_bg_color = self.cmb_sub_bg_color.currentText()
        bar_h = self.spn_sub_bar_h.value()
        if bar_h % 2:
            bar_h -= 1
        cfg.subtitle_bar_height = max(50, bar_h)
        cfg.part_text_y_pct = self._part_text_y_pct
        cfg.subtitle_y_pct = self._subtitle_y_pct
        if self._project:
            src = self._project.export_config
            cfg.subtitle_enabled = src.subtitle_enabled
            cfg.global_subtitle_file = src.global_subtitle_file
            cfg.pre_crop_enabled = getattr(src, "pre_crop_enabled", False)
            cfg.pre_crop_x = getattr(src, "pre_crop_x", 0.5)
            cfg.pre_crop_y = getattr(src, "pre_crop_y", 0.5)
            cfg.pre_crop_w = getattr(src, "pre_crop_w", 1.0)
            cfg.pre_crop_h = getattr(src, "pre_crop_h", 1.0)
        if self._is_review_export():
            cfg.part_text_enabled = False
            cfg.watermark_text = ""
            cfg.global_subtitle_file = (
                getattr(self._project, "narration_subtitle_file", "") or ""
            )
            cfg.subtitle_enabled = bool(
                cfg.global_subtitle_file
                and Path(cfg.global_subtitle_file).exists()
                and cfg.subtitle_style != "none"
            )
        return cfg

    # ─── Export ────────────────────────────────────────────────────────────────

    def _start_export(self):
        if not self._project:
            QMessageBox.warning(self, "Chưa có project", "Vui lòng tạo project và chia Parts trước.")
            return
        clips_to_export = self._selected_export_clips()
        if not clips_to_export:
            QMessageBox.warning(
                self,
                "Không có video",
                "Chưa có video review hợp lệ hoặc video chưa được chọn để xuất.",
            )
            return

        cfg = self._collect_config()

        if cfg.background_music_enabled:
            music_path = Path(cfg.background_music_path)
            if not cfg.background_music_path or not music_path.is_file():
                QMessageBox.warning(
                    self,
                    "Chưa chọn nhạc nền",
                    "Bạn đã bật nhạc nền nhưng file nhạc không tồn tại. "
                    "Hãy chọn lại file nhạc hoặc tắt tùy chọn nhạc nền.",
                )
                return

        # ── Pre-export subtitle check ──────────────────────────────────────
        if cfg.subtitle_style and cfg.subtitle_style != "none":
            no_sub = [
                c for c in clips_to_export
                if not (c.subtitle_file and Path(c.subtitle_file).exists())
            ]
            has_global = (
                cfg.subtitle_enabled
                and cfg.global_subtitle_file
                and Path(cfg.global_subtitle_file).exists()
            )
            if no_sub and not has_global:
                if len(no_sub) == len(clips_to_export):
                    # None of the clips have subtitles — strong warning
                    reply = QMessageBox.warning(
                        self,
                        "⚠️ Chưa có file subtitle",
                        "Bạn đã chọn kiểu subtitle nhưng CHƯA tạo file subtitle "
                        "cho bất kỳ Part nào.\n\n"
                        "Cách tạo subtitle:\n"
                        "  1. Vào Bước 6 (Phụ đề)\n"
                        "  2. Phiên âm giọng đọc\n"
                        "  3. Bấm \"Tạo subtitle toàn video\"\n\n"
                        "Tiếp tục xuất sẽ KHÔNG có subtitle trong video.",
                        QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
                        QMessageBox.StandardButton.Cancel,
                    )
                    if reply != QMessageBox.StandardButton.Ok:
                        return
                else:
                    # Some clips missing
                    names = ", ".join(c.part_text for c in no_sub)
                    reply = QMessageBox.question(
                        self,
                        "⚠️ Một số Part chưa có subtitle",
                        f"Các Part sau chưa có file subtitle:\n{names}\n\n"
                        "Những Part này sẽ xuất KHÔNG có subtitle.\n"
                        "Tiếp tục?",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                        QMessageBox.StandardButton.Cancel,
                    )
                    if reply != QMessageBox.StandardButton.Yes:
                        return

        self._project.export_config = cfg
        save_project(self._project)

        out_dir = self.txt_out_dir.text().strip() or str(
            Path(self._project.output_dir) / "exports"
        )
        self.btn_export.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, len(clips_to_export))
        self.progress_bar.setValue(0)
        self.txt_log.clear()

        # Summary line — include subtitle info so user can confirm
        sub_info = cfg.subtitle_style if cfg.subtitle_style != "none" else "không có sub"
        if self._is_review_export():
            self.txt_log.append(
                f"▶ Xuất 1 video review • PART text: tắt • "
                f"Aspect: {cfg.aspect_mode} • Sub: {sub_info}"
            )
            self.txt_log.append(
                f"🎤 Nguồn âm thanh: "
                f"{Path(clips_to_export[0].source_video).name} "
                "(video đã lồng tiếng)"
            )
        else:
            self.txt_log.append(
                f"▶ Xuất {len(clips_to_export)} Parts • "
                f"Chế độ: {cfg.part_text_mode} • Aspect: {cfg.aspect_mode} • "
                f"Sub: {sub_info}"
            )

        if cfg.background_music_enabled:
            self.txt_log.append(
                f"🎵 Nhạc nền: {Path(cfg.background_music_path).name} • "
                f"{cfg.background_music_volume}% • tự lặp theo video"
            )

        self._worker = ExportWorker(self._project, clips_to_export, cfg, out_dir)
        self._worker.clip_started.connect(self._on_clip_started)
        self._worker.clip_progress.connect(self._on_clip_progress)
        self._worker.clip_done.connect(self._on_clip_done)
        self._worker.all_done.connect(self._on_all_done)
        self._worker.start()

    def _on_clip_started(self, idx: int, part_text: str):
        label = "video review" if self._is_review_export() else part_text
        self.lbl_export_status.setText(f"⏳ Đang xuất {label}...")
        self.txt_log.append(f"→ Bắt đầu xuất {label}...")
        # Show custom subtitle info for debugging
        if self._project:
            for clip in self._project.clips:
                if clip.index == idx:
                    custom = getattr(clip, "custom_subtitle", "") or ""
                    if custom:
                        self.txt_log.append(f"  📝 Text cố định: '{custom}'")
                    break

    def _on_clip_progress(self, idx: int, elapsed: float, total: float):
        if total > 0:
            pct = int(elapsed / total * 100)
            label = "REVIEW" if self._is_review_export() else f"PART {idx}"
            self.lbl_export_status.setText(
                f"⏳ {label}: {pct}% ({elapsed:.1f}s / {total:.1f}s)"
            )

    def _on_clip_done(self, idx: int, ok: bool, result: str):
        if ok and "\n⚠️" in result:
            # GPU→CPU fallback: path + warning on separate lines
            parts = result.split("\n", 1)
            label = "REVIEW" if self._is_review_export() else f"PART {idx}"
            self.txt_log.append(f"✅ {label}: {Path(parts[0]).name}")
            self.txt_log.append(parts[1])  # ⚠️ warning line
        else:
            icon = "✅" if ok else "❌"
            label = "REVIEW" if self._is_review_export() else f"PART {idx}"
            self.txt_log.append(f"{icon} {label}: {Path(result).name if ok else result}")
        if ok:
            self.progress_bar.setValue(self.progress_bar.value() + 1)
        sb = self.txt_log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _on_all_done(self, exported: list):
        self.btn_export.setEnabled(True)
        enabled_count = (
            1 if self._is_review_export()
            else len([c for c in self._project.clips if c.enabled])
        )
        self.lbl_export_status.setText(
            (
                f"✅ Xuất xong {len(exported)} video review."
                if self._is_review_export()
                else f"✅ Xuất xong {len(exported)}/{enabled_count} Parts."
            )
        )
        save_export_history(self._project, exported)

        if self.chk_captions.isChecked() and exported:
            platform = self.cmb_platform.currentText()
            captions = [
                generate_caption(self._project, clip, platform)
                for clip in self._project.clips if clip.enabled
            ]
            save_captions(self._project, captions)
            self.txt_log.append(f"📝 Đã tạo {len(captions)} caption trong captions/")

        save_project(self._project)
        self.export_finished.emit(exported)

        if exported:
            import os
            out_dir = self.txt_out_dir.text()
            reply = QMessageBox.question(
                self, "Xuất xong",
                f"✅ Xuất thành công {len(exported)} video.\nThư mục: {out_dir}\n\nMở thư mục?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes and out_dir:
                os.startfile(out_dir)
