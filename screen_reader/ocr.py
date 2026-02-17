"""
OCR module for reading text from PokerStars table regions.
Reads stack sizes, pot amounts, blind levels, and player names.
"""

import re
import logging
import os
import sys
import platform
from typing import Optional, Tuple

import numpy as np

try:
    import pytesseract
except ImportError:
    pytesseract = None

try:
    from PIL import Image, ImageFilter, ImageEnhance
except ImportError:
    Image = None

try:
    import cv2
except ImportError:
    cv2 = None

logger = logging.getLogger(__name__)


def _configure_bundled_tesseract():
    """Auto-detect and configure Tesseract from a PyInstaller bundle."""
    if pytesseract is None:
        return

    # When running from a PyInstaller bundle, sys._MEIPASS points to the
    # temporary folder where bundled files are extracted.
    base_path = getattr(sys, "_MEIPASS", None)
    if base_path is None:
        return

    tess_dir = os.path.join(base_path, "tesseract")

    if platform.system() == "Windows":
        tess_exe = os.path.join(tess_dir, "tesseract.exe")
    else:
        tess_exe = os.path.join(tess_dir, "tesseract")

    if os.path.isfile(tess_exe):
        pytesseract.pytesseract.tesseract_cmd = tess_exe
        # Point TESSDATA_PREFIX to the folder containing eng.traineddata.
        # Tesseract 5.x looks for *.traineddata directly in TESSDATA_PREFIX,
        # while 4.x appends /tessdata automatically — set to the tessdata
        # dir itself to work with both versions.
        tessdata = os.path.join(tess_dir, "tessdata")
        if os.path.isdir(tessdata):
            os.environ["TESSDATA_PREFIX"] = tessdata
        elif os.path.isfile(os.path.join(tess_dir, "eng.traineddata")):
            # Fallback: traineddata files sitting directly in tess_dir
            os.environ["TESSDATA_PREFIX"] = tess_dir
        else:
            logger.error("Bundled tesseract found at %s but no tessdata! "
                         "Contents of tess_dir: %s",
                         tess_exe, os.listdir(tess_dir) if os.path.isdir(tess_dir) else "<missing>")
        logger.info("Using bundled Tesseract: %s (TESSDATA_PREFIX=%s)",
                     tess_exe, os.environ.get("TESSDATA_PREFIX", "<not set>"))


_configure_bundled_tesseract()


class OCRReader:
    """Reads text from screen regions using Tesseract OCR."""

    def __init__(self, language: str = "eng", psm: int = 7):
        """
        Args:
            language: Tesseract language code.
            psm: Page segmentation mode (7 = single line, 8 = single word).
        """
        self.language = language
        self.psm = psm

        if pytesseract is None:
            logger.warning("pytesseract not installed - OCR will not work")

    def preprocess_for_ocr(self, img: np.ndarray,
                           invert: bool = True,
                           threshold: bool = True,
                           scale: float = 2.0) -> np.ndarray:
        """
        Preprocess an image region for better OCR accuracy.
        PokerStars uses light text on dark backgrounds.
        """
        if cv2 is None:
            return img

        processed = img.copy()

        # Scale up for better OCR accuracy
        if scale != 1.0:
            h, w = processed.shape[:2]
            processed = cv2.resize(processed, (int(w * scale), int(h * scale)),
                                   interpolation=cv2.INTER_CUBIC)

        # Convert to grayscale
        if len(processed.shape) == 3:
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)

        # Invert if needed (PokerStars has light text on dark bg)
        if invert:
            mean_val = np.mean(processed)
            if mean_val < 128:  # Dark background
                processed = cv2.bitwise_not(processed)

        # Apply threshold for clean binary image
        if threshold:
            processed = cv2.GaussianBlur(processed, (3, 3), 0)
            _, processed = cv2.threshold(processed, 0, 255,
                                         cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return processed

    def read_text(self, img: np.ndarray, whitelist: str = "",
                  preprocess: bool = True) -> str:
        """
        Read text from a numpy image array.

        Args:
            img: Image as numpy array (BGR or grayscale).
            whitelist: Characters to restrict OCR to.
            preprocess: Whether to apply preprocessing.

        Returns:
            Recognized text string.
        """
        if pytesseract is None:
            return ""

        if preprocess:
            img = self.preprocess_for_ocr(img)

        # Build tesseract config
        config = f"--psm {self.psm}"
        if whitelist:
            config += f" -c tessedit_char_whitelist={whitelist}"

        try:
            if len(img.shape) == 2:
                pil_img = Image.fromarray(img)
            else:
                pil_img = Image.fromarray(img[:, :, ::-1])  # BGR to RGB

            text = pytesseract.image_to_string(pil_img, lang=self.language,
                                               config=config)
            return text.strip()
        except Exception as e:
            logger.error("OCR failed: %s", e)
            return ""

    def read_number(self, img: np.ndarray) -> Optional[float]:
        """
        Read a numeric value from an image (stack size, pot, bet).
        Handles formats like: 1,234  $1.5k  12.5M  1500  BB 15
        """
        whitelist = "0123456789,.$kKmMBb. "
        text = self.read_text(img, whitelist=whitelist)

        if not text:
            return None

        return self.parse_number(text)

    @staticmethod
    def parse_number(text: str) -> Optional[float]:
        """Parse a number string with various poker formats."""
        text = text.strip().upper()

        # Remove common prefixes/suffixes
        text = text.replace("$", "").replace("BB", "").replace(",", "")
        text = text.replace(" ", "")

        if not text:
            return None

        try:
            # Handle K/M suffixes (e.g., "1.5K" = 1500)
            multiplier = 1
            if text.endswith("K"):
                multiplier = 1000
                text = text[:-1]
            elif text.endswith("M"):
                multiplier = 1000000
                text = text[:-1]

            return float(text) * multiplier
        except ValueError:
            # Try extracting just digits
            digits = re.findall(r'[\d.]+', text)
            if digits:
                try:
                    return float(digits[0]) * multiplier
                except ValueError:
                    pass
            return None

    def read_blind_level(self, img: np.ndarray) -> Optional[Tuple[float, float, float]]:
        """
        Read blind level from the table header.
        Returns (small_blind, big_blind, ante) or None.

        Common tournament formats:
          "100/200"
          "100/200 ante 25"
          "Level 5: 100/200"
          "Level 5: 100/200/25"
          "Blinds 100/200 Ante 25"
        """
        text = self.read_text(img, whitelist="0123456789/: anteLevelBblid")

        if not text:
            return None

        logger.debug("Blind OCR text: '%s'", text)

        # Strategy 1: look for explicit "X/Y" or "X/Y/Z" pattern (most reliable)
        slash_match = re.search(r'(\d[\d,]*)\s*/\s*(\d[\d,]*)(?:\s*/\s*(\d[\d,]*))?', text)
        if slash_match:
            try:
                sb = float(slash_match.group(1).replace(",", ""))
                bb = float(slash_match.group(2).replace(",", ""))
                ante = float(slash_match.group(3).replace(",", "")) if slash_match.group(3) else 0
                # Sanity: sb should be <= bb
                if sb > bb:
                    sb, bb = bb, sb
                # Check for separate "ante N" after the slash pattern
                if ante == 0:
                    ante_match = re.search(r'ante\s*(\d[\d,]*)', text, re.IGNORECASE)
                    if ante_match:
                        ante = float(ante_match.group(1).replace(",", ""))
                return (sb, bb, ante)
            except (ValueError, IndexError):
                pass

        # Strategy 2: extract all numbers and use heuristics
        numbers = re.findall(r'\d[\d,]*', text.replace(",", ""))
        if len(numbers) >= 2:
            try:
                # Skip leading "Level N" — the level number is typically small
                # compared to the actual blind values
                vals = [float(n) for n in numbers]

                # If first number is much smaller than the rest, it's likely "Level N"
                if len(vals) >= 3 and vals[0] < vals[1] * 0.1:
                    vals = vals[1:]

                if len(vals) >= 2:
                    sb = vals[0]
                    bb = vals[1]
                    ante = 0.0
                    # Third number: ante if it's smaller than bb
                    if len(vals) >= 3 and vals[2] < bb:
                        ante = vals[2]
                    # Sanity: sb should be <= bb
                    if sb > bb:
                        sb, bb = bb, sb
                    return (sb, bb, ante)
            except (ValueError, IndexError):
                pass

        return None

    def detect_action_buttons(self, img: np.ndarray) -> dict:
        """
        Detect which action buttons are visible (Fold, Check, Call, Raise, All-in).
        Returns dict of {action: is_visible}.
        """
        text = self.read_text(img, preprocess=True).upper()

        return {
            "fold": "FOLD" in text,
            "check": "CHECK" in text,
            "call": "CALL" in text,
            "raise": "RAISE" in text or "BET" in text,
            "allin": "ALL" in text or "ALLIN" in text,
        }
