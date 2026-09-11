"""Entry point for AI Movie Short & Review Studio."""

import sys
import os
from pathlib import Path


def _base_dir() -> Path:
    """Folder where user-writable data (data/, output/, tools/) should live.

    • Frozen (PyInstaller .exe) → the folder containing the executable.
    • Dev run                   → the project root (this file's folder).
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).parent


def _resource_path(rel: str) -> Path:
    """Path to a bundled, read-only resource (e.g. assets/icon.ico)."""
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        return Path(meipass) / rel
    return Path(__file__).parent / rel


BASE_DIR = _base_dir()
# Ensure working directory is the app base so relative paths resolve correctly
os.chdir(BASE_DIR)

# Add project root to Python path (dev mode only; frozen apps use the bundle)
if not getattr(sys, "frozen", False):
    sys.path.insert(0, str(BASE_DIR))

# ── Silence Qt-internal FFmpeg AV1 hwaccel spam ─────────────────────────────
# Qt6 multimedia uses a bundled FFmpeg that tries D3D11 hardware-accelerated
# AV1 decoding on every frame; if the GPU doesn't support it the C-level
# FFmpeg logger prints "[av1] Failed setup for format d3d11..." to raw stderr.
# Fix: load Qt's avutil DLL via ctypes and set AV_LOG_FATAL (8) so WARNING/
# ERROR messages from the in-process FFmpeg are silenced.
# This does NOT affect our subprocess FFmpeg (separate process).
def _silence_qt_ffmpeg_av1_spam():
    import ctypes
    from pathlib import Path as _Path

    AV_LOG_FATAL = 8  # suppresses WARNING(24) and ERROR(16) messages

    candidates: list[str] = []
    # 1. PyQt6 ships its own Qt6/bin/avutil-*.dll — highest priority
    try:
        import PyQt6 as _pq6
        qt_bin = _Path(_pq6.__file__).parent / "Qt6" / "bin"
        candidates += [str(p) for p in sorted(qt_bin.glob("avutil-*.dll"), reverse=True)]
    except Exception:
        pass
    # 2. Fallback: try common version numbers on the system PATH
    candidates += [f"avutil-{v}.dll" for v in range(62, 54, -1)]
    candidates.append("avutil.dll")

    for path in candidates:
        try:
            lib = ctypes.CDLL(path)
            lib.av_log_set_level.argtypes = [ctypes.c_int]
            lib.av_log_set_level.restype = None
            lib.av_log_set_level(AV_LOG_FATAL)
            return  # success
        except Exception:
            continue

_silence_qt_ffmpeg_av1_spam()
del _silence_qt_ffmpeg_av1_spam

from src.utils.logger import setup_logger

logger = setup_logger("AMSR")
logger.info("=" * 60)
logger.info("AI Movie Short & Review Studio starting...")


def check_python_version():
    if sys.version_info < (3, 10):
        print("❌ Yêu cầu Python 3.10 trở lên.")
        sys.exit(1)


def main():
    check_python_version()

    try:
        from PyQt6.QtWidgets import QApplication
        from PyQt6.QtCore import Qt
        from PyQt6.QtGui import QFont, QIcon
    except ImportError:
        print("❌ PyQt6 chưa được cài. Chạy: pip install PyQt6")
        sys.exit(1)

    app = QApplication(sys.argv)
    app.setApplicationName("AI Movie Short & Review Studio")
    app.setApplicationVersion("2.0.0")

    # App / window icon
    icon_path = _resource_path("assets/icon.ico")
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))

    font = QFont("Segoe UI", 10)
    app.setFont(font)

    from src.ui.main_window import MainWindow
    window = MainWindow()
    window.show()

    logger.info("App window shown.")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
