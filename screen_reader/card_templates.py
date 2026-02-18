"""
Programmatic card template generation for template-matching-based card detection.

Generates reference images of each rank character (2-9, T, J, Q, K, A) and
each suit symbol at multiple sizes and font styles.  These are used by
CardDetector for cv2.matchTemplate() instead of relying on Tesseract OCR.

Templates are rendered once on first use and cached in memory.
"""

import logging
from typing import Dict, List, Tuple

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = None
    ImageDraw = None
    ImageFont = None

logger = logging.getLogger(__name__)

RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']

# Unicode suit characters for rendering
SUIT_CHARS = {
    's': '\u2660',  # Spade
    'h': '\u2665',  # Heart
    'd': '\u2666',  # Diamond
    'c': '\u2663',  # Club
}

# Template heights to generate (pixels).  During matching we'll also
# rescale dynamically, but having multiple base sizes improves accuracy
# across different card/window resolutions.
TEMPLATE_HEIGHTS = [20, 28, 36, 48, 60]


class TemplateCache:
    """Generates and caches rank/suit template images."""

    def __init__(self):
        self._rank_templates: Dict[str, List[np.ndarray]] = {}
        self._suit_templates: Dict[str, List[np.ndarray]] = {}
        self._built = False

    def build(self):
        """Generate all templates.  Called once on first use."""
        if self._built:
            return
        self._rank_templates = self._generate_rank_templates()
        self._suit_templates = self._generate_suit_templates()
        self._built = True
        total_rank = sum(len(v) for v in self._rank_templates.values())
        total_suit = sum(len(v) for v in self._suit_templates.values())
        logger.info("Template cache built: %d rank templates, %d suit templates",
                     total_rank, total_suit)

    @property
    def rank_templates(self) -> Dict[str, List[np.ndarray]]:
        self.build()
        return self._rank_templates

    @property
    def suit_templates(self) -> Dict[str, List[np.ndarray]]:
        self.build()
        return self._suit_templates

    def _generate_rank_templates(self) -> Dict[str, List[np.ndarray]]:
        """Generate binary template images for each rank character.

        Each template is a white-on-black image of the rank character,
        rendered at multiple sizes.  For matching, these will be compared
        against a binarized crop of the card's top-left corner.
        """
        templates = {}
        for rank in RANKS:
            templates[rank] = []
            # Use the display character (10 -> "10", etc.)
            display_char = '10' if rank == 'T' else rank
            for height in TEMPLATE_HEIGHTS:
                imgs = self._render_character(display_char, height)
                templates[rank].extend(imgs)
        return templates

    def _generate_suit_templates(self) -> Dict[str, List[np.ndarray]]:
        """Generate binary template images for each suit symbol."""
        templates = {}
        for suit, char in SUIT_CHARS.items():
            templates[suit] = []
            for height in TEMPLATE_HEIGHTS:
                imgs = self._render_character(char, height)
                templates[suit].extend(imgs)
        return templates

    @staticmethod
    def _render_character(char: str, target_height: int) -> List[np.ndarray]:
        """Render a character as white-on-black binary image(s).

        Tries multiple fonts/approaches to maximize matching robustness:
        1. PIL TrueType rendering with available system fonts
        2. OpenCV putText as fallback

        Returns a list of numpy arrays (grayscale, 0 or 255).
        """
        results = []

        # --- Approach 1: PIL rendering (higher quality, anti-aliased) ---
        if Image is not None and ImageDraw is not None:
            pil_imgs = _render_with_pil(char, target_height)
            results.extend(pil_imgs)

        # --- Approach 2: OpenCV putText (always available) ---
        if cv2 is not None:
            cv_imgs = _render_with_cv2(char, target_height)
            results.extend(cv_imgs)

        return results


def _render_with_pil(char: str, target_height: int) -> List[np.ndarray]:
    """Render a character using PIL.  Tries several fonts."""
    results = []
    fonts_to_try = _get_available_fonts(target_height)

    for font in fonts_to_try:
        try:
            # Create a temporary image to measure text size
            tmp = Image.new('L', (target_height * 3, target_height * 3), 0)
            draw = ImageDraw.Draw(tmp)
            bbox = draw.textbbox((0, 0), char, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]

            if tw < 2 or th < 2:
                continue

            # Render at measured size with padding
            pad = max(2, target_height // 8)
            img = Image.new('L', (tw + pad * 2, th + pad * 2), 0)
            draw = ImageDraw.Draw(img)
            draw.text((pad - bbox[0], pad - bbox[1]), char, fill=255, font=font)

            arr = np.array(img, dtype=np.uint8)

            # Binarize (remove anti-aliasing)
            _, binary = cv2.threshold(arr, 100, 255, cv2.THRESH_BINARY)

            # Crop to tight bounding box
            binary = _tight_crop(binary)
            if binary is not None and binary.shape[0] >= 4 and binary.shape[1] >= 3:
                results.append(binary)

                # Also generate a slightly bold variant
                kernel = np.ones((2, 2), np.uint8)
                bold = cv2.dilate(binary, kernel, iterations=1)
                bold = _tight_crop(bold)
                if bold is not None:
                    results.append(bold)
        except Exception:
            continue

    return results


def _render_with_cv2(char: str, target_height: int) -> List[np.ndarray]:
    """Render a character using OpenCV putText.  Simpler but always works."""
    if cv2 is None:
        return []

    results = []
    # Try multiple OpenCV fonts
    fonts = [
        cv2.FONT_HERSHEY_SIMPLEX,
        cv2.FONT_HERSHEY_DUPLEX,
        cv2.FONT_HERSHEY_COMPLEX,
    ]
    # Also try different thicknesses
    for font_face in fonts:
        for thickness in [1, 2]:
            try:
                scale = target_height / 30.0
                (tw, th), baseline = cv2.getTextSize(char, font_face, scale, thickness)
                if tw < 2 or th < 2:
                    continue

                pad = max(2, target_height // 8)
                img = np.zeros((th + baseline + pad * 2, tw + pad * 2), dtype=np.uint8)
                cv2.putText(img, char, (pad, th + pad), font_face,
                            scale, 255, thickness, cv2.LINE_AA)

                _, binary = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
                binary = _tight_crop(binary)
                if binary is not None and binary.shape[0] >= 4 and binary.shape[1] >= 3:
                    results.append(binary)
            except Exception:
                continue

    return results


def _get_available_fonts(target_height: int) -> list:
    """Get a list of PIL fonts to try, falling back gracefully."""
    fonts = []
    if ImageFont is None:
        return fonts

    font_size = max(10, int(target_height * 0.9))

    # Try common system fonts
    font_names = [
        "DejaVuSans-Bold.ttf",
        "DejaVuSans.ttf",
        "arial.ttf",
        "Arial.ttf",
        "LiberationSans-Bold.ttf",
        "LiberationSans-Regular.ttf",
        "FreeSansBold.ttf",
        "FreeSans.ttf",
    ]
    for name in font_names:
        try:
            fonts.append(ImageFont.truetype(name, font_size))
        except (OSError, IOError):
            continue

    # Always include the built-in default font as a last resort
    try:
        fonts.append(ImageFont.load_default())
    except Exception:
        pass

    return fonts


def _tight_crop(img: np.ndarray) -> np.ndarray:
    """Crop a binary image to its tight bounding box of non-zero pixels."""
    if img is None or img.size == 0:
        return None

    coords = cv2.findNonZero(img)
    if coords is None:
        return None

    x, y, w, h = cv2.boundingRect(coords)
    if w < 2 or h < 2:
        return None

    return img[y:y+h, x:x+w].copy()


# Module-level singleton cache
_cache = TemplateCache()


def get_rank_templates() -> Dict[str, List[np.ndarray]]:
    """Get the global rank template cache."""
    return _cache.rank_templates


def get_suit_templates() -> Dict[str, List[np.ndarray]]:
    """Get the global suit template cache."""
    return _cache.suit_templates
