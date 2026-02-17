# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for Poker GTO Assistant.
Builds a standalone application with all dependencies bundled.

Usage:
    pyinstaller poker_gto.spec
"""

import os
import sys
import platform
import shutil

block_cipher = None

# --- Determine Tesseract paths to bundle ---
tesseract_binaries = []
tesseract_datas = []

if platform.system() == "Windows":
    # Common Windows install paths
    tess_candidates = [
        os.path.join(os.environ.get("PROGRAMFILES", ""), "Tesseract-OCR"),
        os.path.join(os.environ.get("PROGRAMFILES(X86)", ""), "Tesseract-OCR"),
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "Tesseract-OCR"),
    ]
    # Also try finding tesseract on PATH (e.g. Chocolatey shimmed installs)
    tess_which = shutil.which("tesseract")
    if tess_which:
        tess_candidates.insert(0, os.path.dirname(os.path.realpath(tess_which)))
    for tess_dir in tess_candidates:
        if os.path.isfile(os.path.join(tess_dir, "tesseract.exe")):
            print(f"[spec] Found Tesseract at: {tess_dir}")
            # Bundle the tesseract executable
            tesseract_binaries.append(
                (os.path.join(tess_dir, "tesseract.exe"), "tesseract")
            )
            # Bundle required DLLs
            for dll in os.listdir(tess_dir):
                if dll.endswith(".dll"):
                    tesseract_binaries.append(
                        (os.path.join(tess_dir, dll), "tesseract")
                    )
            # Bundle tessdata (language files)
            tessdata = os.path.join(tess_dir, "tessdata")
            if os.path.isdir(tessdata):
                print(f"[spec] Bundling tessdata from: {tessdata}")
                print(f"[spec] tessdata contents: {os.listdir(tessdata)}")
                tesseract_datas.append((tessdata, "tesseract/tessdata"))
            else:
                print(f"[spec] WARNING: tessdata not found at {tessdata}")
                # Search for tessdata in common Chocolatey / system locations
                fallback_tessdata = [
                    os.path.join(os.environ.get("PROGRAMDATA", ""), "chocolatey", "lib", "tesseract", "tools", "tessdata"),
                    os.path.join(os.environ.get("PROGRAMDATA", ""), "chocolatey", "lib", "tesseract", "tools", "tesseract-ocr", "tessdata"),
                ]
                for fb in fallback_tessdata:
                    if os.path.isdir(fb):
                        print(f"[spec] Found tessdata at fallback: {fb}")
                        tesseract_datas.append((fb, "tesseract/tessdata"))
                        break
                else:
                    print("[spec] WARNING: No tessdata found — OCR will not work in bundled build!")
            break
    else:
        print("[spec] WARNING: Tesseract executable not found on this system")

elif platform.system() == "Darwin":
    # macOS via Homebrew
    brew_tess = shutil.which("tesseract")
    if brew_tess:
        tesseract_binaries.append((brew_tess, "tesseract"))
        # Find tessdata
        for tessdata_path in [
            "/usr/local/share/tessdata",
            "/opt/homebrew/share/tessdata",
            "/usr/local/share/tesseract-ocr/4.00/tessdata",
        ]:
            if os.path.isdir(tessdata_path):
                tesseract_datas.append((tessdata_path, "tesseract/tessdata"))
                break

else:
    # Linux
    tess_path = shutil.which("tesseract")
    if tess_path:
        tesseract_binaries.append((tess_path, "tesseract"))
        for tessdata_path in [
            "/usr/share/tesseract-ocr/4.00/tessdata",
            "/usr/share/tesseract-ocr/5/tessdata",
            "/usr/share/tessdata",
        ]:
            if os.path.isdir(tessdata_path):
                tesseract_datas.append((tessdata_path, "tesseract/tessdata"))
                break


a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=tesseract_binaries,
    datas=tesseract_datas,
    hiddenimports=[
        "PyQt6",
        "PyQt6.QtCore",
        "PyQt6.QtGui",
        "PyQt6.QtWidgets",
        "cv2",
        "numpy",
        "PIL",
        "PIL.Image",
        "pytesseract",
        "mss",
        "mss.tools",
        # Windows window detection (pywin32)
        "win32gui",
        "win32con",
        "win32api",
        "pywintypes",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "tkinter",
        "matplotlib",
        "scipy",
        "pandas",
        "IPython",
        "jupyter",
        "notebook",
    ],
    noarchive=False,
    optimize=0,
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="PokerGTOAssistant",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # No console window — GUI app
    icon=None,      # Add icon path here if you have one, e.g. "assets/icon.ico"
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="PokerGTOAssistant",
)
