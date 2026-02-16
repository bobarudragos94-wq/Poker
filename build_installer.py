#!/usr/bin/env python3
"""
Build script for Poker GTO Assistant installer.

This script:
  1. Installs build dependencies (PyInstaller)
  2. Verifies Tesseract OCR is available (will be bundled automatically)
  3. Runs PyInstaller to create a standalone application
  4. (Windows only) Optionally creates an InnoSetup-based installer .exe

Usage:
    python build_installer.py              # Build standalone folder
    python build_installer.py --onefile    # Build single executable
    python build_installer.py --clean      # Clean previous builds first

Requirements:
    - Python 3.8+
    - Tesseract OCR installed on the build machine
      (it gets bundled into the output so end users don't need it)
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SPEC_FILE = os.path.join(ROOT_DIR, "poker_gto.spec")
DIST_DIR = os.path.join(ROOT_DIR, "dist")
BUILD_DIR = os.path.join(ROOT_DIR, "build")


def run(cmd: list[str], check: bool = True):
    print(f"  > {' '.join(cmd)}")
    return subprocess.run(cmd, check=check)


def ensure_pyinstaller():
    """Install PyInstaller if not available."""
    try:
        import PyInstaller  # noqa: F401
        print("[OK] PyInstaller is installed")
    except ImportError:
        print("[..] Installing PyInstaller...")
        run([sys.executable, "-m", "pip", "install", "pyinstaller"])
        print("[OK] PyInstaller installed")


def check_tesseract() -> bool:
    """Check that Tesseract is installed on the build machine."""
    tess = shutil.which("tesseract")
    if tess:
        print(f"[OK] Tesseract found: {tess}")
        return True

    # Windows: check common paths
    if platform.system() == "Windows":
        for prog in [os.environ.get("PROGRAMFILES", ""),
                      os.environ.get("PROGRAMFILES(X86)", ""),
                      os.environ.get("LOCALAPPDATA", "")]:
            candidate = os.path.join(prog, "Tesseract-OCR", "tesseract.exe")
            if os.path.isfile(candidate):
                print(f"[OK] Tesseract found: {candidate}")
                return True

    print("[!!] Tesseract OCR not found on this machine.")
    print("     The build will still work, but Tesseract will NOT be bundled.")
    print("     Install Tesseract first for a fully self-contained build:")
    if platform.system() == "Windows":
        print("       https://github.com/UB-Mannheim/tesseract/wiki")
    elif platform.system() == "Darwin":
        print("       brew install tesseract")
    else:
        print("       sudo apt install tesseract-ocr")
    return False


def clean():
    """Remove previous build artifacts."""
    for d in [DIST_DIR, BUILD_DIR]:
        if os.path.isdir(d):
            print(f"  Removing {d}")
            shutil.rmtree(d)
    print("[OK] Cleaned build directories")


def build(onefile: bool = False):
    """Run PyInstaller to produce the application."""
    if onefile:
        # Single-file mode: override the spec and use command-line flags
        cmd = [
            sys.executable, "-m", "PyInstaller",
            "--noconfirm",
            "--onefile",
            "--windowed",
            "--name", "PokerGTOAssistant",
            os.path.join(ROOT_DIR, "main.py"),
        ]
    else:
        # Use the spec file (bundles Tesseract, more control)
        cmd = [
            sys.executable, "-m", "PyInstaller",
            "--noconfirm",
            SPEC_FILE,
        ]

    run(cmd)

    # Show output location
    if onefile:
        if platform.system() == "Windows":
            exe = os.path.join(DIST_DIR, "PokerGTOAssistant.exe")
        else:
            exe = os.path.join(DIST_DIR, "PokerGTOAssistant")
        print()
        print("=" * 60)
        print(f"  BUILD COMPLETE — single executable:")
        print(f"  {exe}")
        print("=" * 60)
    else:
        out_dir = os.path.join(DIST_DIR, "PokerGTOAssistant")
        print()
        print("=" * 60)
        print(f"  BUILD COMPLETE — application folder:")
        print(f"  {out_dir}")
        if platform.system() == "Windows":
            print(f"  Run: {os.path.join(out_dir, 'PokerGTOAssistant.exe')}")
        else:
            print(f"  Run: {os.path.join(out_dir, 'PokerGTOAssistant')}")
        print()
        print("  To distribute, zip the entire folder and share it.")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Build Poker GTO Assistant installer")
    parser.add_argument("--onefile", action="store_true",
                        help="Build a single executable instead of a folder")
    parser.add_argument("--clean", action="store_true",
                        help="Clean previous build artifacts before building")
    args = parser.parse_args()

    print()
    print("=" * 60)
    print("  Poker GTO Assistant — Installer Builder")
    print("=" * 60)
    print()

    if args.clean:
        clean()
        print()

    ensure_pyinstaller()
    has_tess = check_tesseract()
    print()

    if not has_tess:
        resp = input("Continue without bundling Tesseract? [y/N] ").strip().lower()
        if resp != "y":
            print("Aborted. Install Tesseract and try again.")
            sys.exit(1)

    print("[..] Building application...")
    print()
    build(onefile=args.onefile)


if __name__ == "__main__":
    main()
