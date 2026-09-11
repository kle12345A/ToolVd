# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for AI Movie Short & Review Studio (onedir build)."""

import importlib.util
from pathlib import Path

from PyInstaller.utils.hooks import collect_all

APP_NAME = "AI Movie Studio"

datas = [("assets", "assets")]
binaries = []
hiddenimports = []

# Keep the lightweight downloader helpers beside the packaged app. FFmpeg is
# still managed/downloaded separately because its binaries are much larger.
for tool_name in ("yt-dlp.exe", "gallery-dl.exe"):
    tool_path = Path("tools") / tool_name
    if tool_path.exists():
        datas.append((str(tool_path), "tools"))

# Heavy packages imported lazily (faster-whisper, edge-tts, ...) must be
# collected explicitly so PyInstaller bundles their code + data + DLLs.
_collect = [
    "faster_whisper",
    "ctranslate2",
    "onnxruntime",
    "tokenizers",
    "huggingface_hub",
    "edge_tts",
    "aiohttp",
    "av",
    "piper",
    "pathvalidate",
]

# VoxCPM is an optional multi-GB install. If it was installed before building,
# include its runtime stack; otherwise keep the normal distribution lightweight.
if importlib.util.find_spec("voxcpm") is not None:
    _collect += [
        "voxcpm",
        "torch",
        "torchaudio",
        "torchcodec",
        "transformers",
        "soundfile",
        "librosa",
        "modelscope",
        "funasr",
    ]
for pkg in _collect:
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception as exc:  # pragma: no cover
        print(f"[spec] collect_all skipped for {pkg}: {exc}")

a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib", "PySide6", "PyQt5"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,            # windowed app (no console window)
    disable_windowed_traceback=False,
    icon="assets/icon.ico",
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name=APP_NAME,
)
