# Poker GTO Assistant

Real-time GTO (Game Theory Optimal) decision assistant for PokerStars tournaments. Captures your table screen, detects cards, position, and stack sizes via OCR, then displays optimal action recommendations as an overlay.

## Installation

### Option A: Download the pre-built installer (recommended)

Download the latest release for your platform from the **Releases** page — no Python or Tesseract installation required. Just extract/install and run.

### Option B: Run from source

1. **Install Python 3.8+** — [python.org](https://www.python.org/downloads/)

2. **Install Tesseract OCR**
   - **Windows**: Download from [UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki)
   - **macOS**: `brew install tesseract`
   - **Linux**: `sudo apt install tesseract-ocr`

3. **Install dependencies and run**
   ```bash
   pip install -r requirements.txt
   python main.py
   ```

### CLI options

```
python main.py              # Launch GUI with overlay
python main.py --manual     # Terminal-based manual input mode
python main.py --verbose    # Enable debug logging
python main.py --config path/to/config.json
```

## Building the installer yourself

The build script bundles Python, all dependencies, **and** Tesseract OCR into a standalone application that users can run without installing anything.

### Prerequisites (build machine only)

- Python 3.8+
- Tesseract OCR installed (it gets bundled into the output)

### Build commands

```bash
# Build a standalone folder (recommended — fastest build)
python build_installer.py

# Build a single .exe / binary
python build_installer.py --onefile

# Clean previous builds first
python build_installer.py --clean
```

The output will appear in `dist/PokerGTOAssistant/`. Zip it up and share — recipients just run the executable.

## Configuration

On first launch, default screen regions match an 800x600 PokerStars 6-max table. Adjust regions for your resolution and theme by editing `config.json` (created on first save) or via the GUI settings.

## Project structure

```
main.py                  # Entry point
config.py                # App configuration & table regions
gui/                     # PyQt6 GUI (main window + overlay)
screen_reader/           # Screen capture, OCR, card detection
decision/                # GTO advisor (preflop + postflop)
gto_engine/              # Core GTO: ranges, equity, ICM, hand eval
tests/                   # Unit tests (pytest)
build_installer.py       # Build script for standalone installer
poker_gto.spec           # PyInstaller spec file
```
