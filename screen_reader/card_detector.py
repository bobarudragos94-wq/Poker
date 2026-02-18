"""
Card detection module using template matching and color analysis.
Detects card rank and suit from PokerStars table screenshots.

Architecture
------------
1. **Rank detection**: Multi-scale template matching against programmatically
   generated reference images of each rank character (2-9, T, J, Q, K, A).
   This replaces the previous Tesseract OCR approach which was unreliable
   on tiny card images.

2. **Suit detection**: HSV color analysis with wide thresholds, plus
   shape-based disambiguation for 2-color decks.

3. **Card finding**: Contour-based detection of white rectangles on the
   table felt, used by TableStateReader to locate cards dynamically.
"""

import logging
from typing import Optional, Tuple, List, Dict

from dataclasses import dataclass

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from PIL import Image
except ImportError:
    Image = None

logger = logging.getLogger(__name__)

RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
SUITS = ['s', 'h', 'd', 'c']  # spades, hearts, diamonds, clubs

# ---------------------------------------------------------------------------
# HSV color ranges for suit detection.
#
# 4-color deck (PokerStars default):
#   Spades = black, Hearts = red, Diamonds = BLUE, Clubs = GREEN
# 2-color deck (classic):
#   Spades = black, Hearts = red, Diamonds = RED, Clubs = BLACK
#
# Ranges are deliberately wide to handle various PokerStars themes,
# anti-aliased edges, and slight color variations.
# ---------------------------------------------------------------------------
SUIT_COLORS_HSV = {
    's': {  # Spades — black/dark gray
        'lower': np.array([0, 0, 0]),
        'upper': np.array([180, 80, 100]),
    },
    'h': {  # Hearts — red (wraps around H=0/180)
        'lower': np.array([0, 70, 70]),
        'upper': np.array([15, 255, 255]),
        'lower2': np.array([155, 70, 70]),
        'upper2': np.array([180, 255, 255]),
    },
    'd': {  # Diamonds — blue (4-color) or red (2-color, handled by shape)
        'lower': np.array([90, 50, 50]),
        'upper': np.array([135, 255, 255]),
    },
    'c': {  # Clubs — green (4-color) or black (2-color, handled by shape)
        'lower': np.array([30, 50, 50]),
        'upper': np.array([90, 255, 255]),
    },
}


@dataclass
class Card:
    """Represents a playing card."""
    rank: str  # '2'-'9', 'T', 'J', 'Q', 'K', 'A'
    suit: str  # 's', 'h', 'd', 'c'

    def __str__(self):
        return f"{self.rank}{self.suit}"

    def __repr__(self):
        return f"Card('{self.rank}{self.suit}')"

    def __eq__(self, other):
        if isinstance(other, Card):
            return self.rank == other.rank and self.suit == other.suit
        return False

    def __hash__(self):
        return hash((self.rank, self.suit))

    @property
    def pretty(self) -> str:
        suit_symbols = {'s': '\u2660', 'h': '\u2665', 'd': '\u2666', 'c': '\u2663'}
        return f"{self.rank}{suit_symbols.get(self.suit, self.suit)}"

    @classmethod
    def from_string(cls, s: str) -> "Card":
        """Parse a card from string like 'As', 'Th', '2c'."""
        s = s.strip()
        if len(s) != 2:
            raise ValueError(f"Invalid card string: {s}")
        rank = s[0].upper()
        suit = s[1].lower()
        if rank not in RANKS:
            raise ValueError(f"Invalid rank: {rank}")
        if suit not in SUITS:
            raise ValueError(f"Invalid suit: {suit}")
        return cls(rank=rank, suit=suit)

    @property
    def rank_value(self) -> int:
        """Numeric rank value (2=2, ..., A=14)."""
        rank_values = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6,
                       '7': 7, '8': 8, '9': 9, 'T': 10, 'J': 11,
                       'Q': 12, 'K': 13, 'A': 14}
        return rank_values.get(self.rank, 0)


# ---------------------------------------------------------------------------
# CardDetector — the main detection class
# ---------------------------------------------------------------------------

class CardDetector:
    """Detects playing cards from screen captures of PokerStars.

    Uses template matching for rank detection and color analysis for suit
    detection.  Designed to work without external asset files — templates
    are generated programmatically on first use.
    """

    def __init__(self, match_threshold: float = 0.55):
        self.match_threshold = match_threshold
        self._templates_loaded = False
        self._rank_templates: Dict[str, List[np.ndarray]] = {}
        self._suit_templates: Dict[str, List[np.ndarray]] = {}

    def _ensure_templates(self):
        """Lazy-load templates on first use."""
        if self._templates_loaded:
            return
        from .card_templates import get_rank_templates, get_suit_templates
        self._rank_templates = get_rank_templates()
        self._suit_templates = get_suit_templates()
        self._templates_loaded = True

    def detect_card(self, img: np.ndarray) -> Optional[Card]:
        """
        Detect a single card from an image region.

        Args:
            img: Card image region as numpy array (BGR).
                 Expected to be a tightly-cropped single card.

        Returns:
            Card object or None if no card detected.
        """
        if img is None or img.size == 0:
            return None

        if cv2 is None:
            return None

        # Check if the region actually contains a card
        if not self._is_card_present(img):
            logger.debug("No card present in %dx%d region", img.shape[1], img.shape[0])
            return None

        self._ensure_templates()

        rank = self._detect_rank(img)
        suit = self._detect_suit(img)

        if rank and suit:
            logger.debug("Card detected: %s%s", rank, suit)
            return Card(rank=rank, suit=suit)

        logger.debug("Partial detection: rank=%s suit=%s on %dx%d",
                     rank, suit, img.shape[1], img.shape[0])
        return None

    def detect_multiple_cards(self, images: List[np.ndarray]) -> List[Optional[Card]]:
        """Detect cards from multiple image regions."""
        return [self.detect_card(img) for img in images]

    # ------------------------------------------------------------------
    # Card presence check
    # ------------------------------------------------------------------

    @staticmethod
    def _is_card_present(img: np.ndarray) -> bool:
        """Check if a card is present in the region.

        Cards are white/light rectangles.  The table felt is green (cash)
        or blue/navy (tournaments).  We check that the region has enough
        bright pixels and is not predominantly felt-colored.
        """
        if cv2 is None or img is None:
            return False

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # White/light pixels (card background): low saturation, high value
        white_mask = cv2.inRange(hsv, np.array([0, 0, 120]), np.array([180, 100, 255]))
        white_ratio = np.sum(white_mask > 0) / white_mask.size

        if white_ratio > 0.06:
            return True

        # Check it's not all felt (green + blue combined)
        green_mask = cv2.inRange(hsv, np.array([25, 30, 30]), np.array([95, 255, 220]))
        blue_mask = cv2.inRange(hsv, np.array([90, 20, 15]), np.array([145, 255, 220]))
        felt_ratio = (np.sum(green_mask > 0) + np.sum(blue_mask > 0)) / green_mask.size

        if felt_ratio < 0.35:
            # Region is not felt — could be a card with non-white background
            light_mask = cv2.inRange(hsv, np.array([0, 0, 70]), np.array([180, 255, 255]))
            light_ratio = np.sum(light_mask > 0) / light_mask.size
            if light_ratio > 0.15:
                return True

        return False

    # ------------------------------------------------------------------
    # Rank detection via template matching
    # ------------------------------------------------------------------

    def _detect_rank(self, img: np.ndarray) -> Optional[str]:
        """Detect the rank of a card using shape-based template matching.

        Strategy:
        1. Crop the top-left corner where the rank character is printed
        2. Binarize using multiple thresholds
        3. Isolate the character by finding its tight bounding box
        4. Resize both character and templates to a standard size
        5. Compare using pixel overlap (IoU-like) metric

        This avoids the sliding-window problem where a simple template
        like 'J' matches sub-regions of other characters.
        """
        h, w = img.shape[:2]
        if h < 8 or w < 8:
            return None

        # Crop top-left region where the rank character lives.
        # Only take top ~30% to avoid the suit symbol below.
        rank_region = img[0:int(h * 0.30), 0:int(w * 0.50)]
        if rank_region.size == 0:
            return None

        gray = cv2.cvtColor(rank_region, cv2.COLOR_BGR2GRAY)

        # Scale up for better resolution
        rh, rw = gray.shape[:2]
        scale = max(2, min(6, 80 // max(rh, 1)))
        scaled = cv2.resize(gray, (rw * scale, rh * scale),
                            interpolation=cv2.INTER_CUBIC)

        # Try multiple binarization strategies
        binaries = self._make_binaries(scaled)

        best_rank = None
        best_score = 0.0

        for binary in binaries:
            # Isolate the character: find the largest connected component
            char_img = self._isolate_character(binary)
            if char_img is None:
                continue

            rank, score = self._match_character(char_img)
            if rank and score > best_score:
                best_score = score
                best_rank = rank

        if best_rank and best_score >= self.match_threshold:
            logger.debug("Rank matched: %s (score=%.3f)", best_rank, best_score)
            return best_rank

        logger.debug("Rank match failed: best=%s score=%.3f (threshold=%.2f)",
                     best_rank, best_score, self.match_threshold)
        return None

    def _match_character(self, char_img: np.ndarray) -> Tuple[Optional[str], float]:
        """Compare an isolated character image against all rank templates.

        Both the character and each template are resized to a standard
        height (MATCH_H) before comparison.  The score is based on pixel
        overlap (similar to IoU), which is much more discriminating than
        sliding-window correlation.
        """
        MATCH_H = 40  # Standard comparison height
        ch, cw = char_img.shape[:2]
        if ch < 3 or cw < 2:
            return None, 0.0

        # Resize character to standard height, preserving aspect ratio
        char_scale = MATCH_H / ch
        char_resized = cv2.resize(char_img,
                                   (max(3, int(cw * char_scale)), MATCH_H),
                                   interpolation=cv2.INTER_AREA)
        _, char_resized = cv2.threshold(char_resized, 127, 255, cv2.THRESH_BINARY)

        char_w = char_resized.shape[1]

        best_rank = None
        best_score = 0.0

        for rank, templates in self._rank_templates.items():
            for tmpl in templates:
                th, tw = tmpl.shape[:2]
                if th < 3 or tw < 2:
                    continue

                # Resize template to same standard height
                tmpl_scale = MATCH_H / th
                tmpl_resized = cv2.resize(tmpl,
                                           (max(3, int(tw * tmpl_scale)), MATCH_H),
                                           interpolation=cv2.INTER_AREA)
                _, tmpl_resized = cv2.threshold(tmpl_resized, 127, 255,
                                                 cv2.THRESH_BINARY)

                tmpl_w = tmpl_resized.shape[1]

                # Skip if aspect ratios are very different
                if char_w > 0 and tmpl_w > 0:
                    ratio = char_w / tmpl_w
                    if ratio < 0.4 or ratio > 2.5:
                        continue

                # Pad both to the same width for comparison
                max_w = max(char_w, tmpl_w)
                char_padded = np.zeros((MATCH_H, max_w), dtype=np.uint8)
                tmpl_padded = np.zeros((MATCH_H, max_w), dtype=np.uint8)

                # Center horizontally
                cx = (max_w - char_w) // 2
                tx = (max_w - tmpl_w) // 2
                char_padded[:, cx:cx + char_w] = char_resized
                tmpl_padded[:, tx:tx + tmpl_w] = tmpl_resized

                # Score: pixel overlap metric
                # intersection / union (IoU on binary pixels)
                char_bool = char_padded > 127
                tmpl_bool = tmpl_padded > 127

                intersection = np.sum(char_bool & tmpl_bool)
                union = np.sum(char_bool | tmpl_bool)

                if union == 0:
                    continue

                iou = intersection / union

                # Also compute a correlation score on the padded images
                # for finer discrimination
                if char_padded.std() > 0 and tmpl_padded.std() > 0:
                    corr = np.corrcoef(char_padded.flatten().astype(float),
                                        tmpl_padded.flatten().astype(float))[0, 1]
                    if np.isnan(corr):
                        corr = 0.0
                else:
                    corr = 0.0

                # Combined score: weighted average of IoU and correlation
                score = 0.6 * iou + 0.4 * max(0, corr)

                if score > best_score:
                    best_score = score
                    best_rank = rank

        return best_rank, best_score

    @staticmethod
    def _isolate_character(binary: np.ndarray) -> Optional[np.ndarray]:
        """Find and isolate the rank character(s) from a binarized image.

        For single-character ranks (A, K, Q, J, 2-9), finds the largest
        connected component.  For multi-character ranks (10/T), merges
        nearby components that are at similar vertical positions.
        """
        if binary is None or binary.size == 0:
            return None

        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8)

        if num_labels <= 1:
            return None  # Only background

        # Collect significant components (skip tiny noise)
        min_area = max(8, binary.size * 0.002)
        components = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                components.append({
                    'label': i,
                    'x': stats[i, cv2.CC_STAT_LEFT],
                    'y': stats[i, cv2.CC_STAT_TOP],
                    'w': stats[i, cv2.CC_STAT_WIDTH],
                    'h': stats[i, cv2.CC_STAT_HEIGHT],
                    'area': area,
                    'cy': centroids[i][1],
                })

        if not components:
            return None

        # Sort by area (largest first)
        components.sort(key=lambda c: c['area'], reverse=True)
        best = components[0]

        # Check for multi-character rank: if there's another significant
        # component at a similar vertical position and close horizontally,
        # merge them (handles "10" rendering)
        merged_labels = [best['label']]
        merge_x1 = best['x']
        merge_y1 = best['y']
        merge_x2 = best['x'] + best['w']
        merge_y2 = best['y'] + best['h']

        for comp in components[1:]:
            # Must be at a similar vertical center (within 40% of best's height)
            cy_diff = abs(comp['cy'] - best['cy'])
            if cy_diff > best['h'] * 0.4:
                continue
            # Must be horizontally close (within best's width)
            gap = max(0, comp['x'] - merge_x2, merge_x1 - (comp['x'] + comp['w']))
            if gap > best['w'] * 0.5:
                continue
            # Must be a significant fraction of the best component
            if comp['area'] < best['area'] * 0.15:
                continue

            merged_labels.append(comp['label'])
            merge_x1 = min(merge_x1, comp['x'])
            merge_y1 = min(merge_y1, comp['y'])
            merge_x2 = max(merge_x2, comp['x'] + comp['w'])
            merge_y2 = max(merge_y2, comp['y'] + comp['h'])

        w = merge_x2 - merge_x1
        h = merge_y2 - merge_y1
        if w < 3 or h < 4:
            return None

        # Create a combined mask of all merged components
        region = labels[merge_y1:merge_y2, merge_x1:merge_x2]
        combined_mask = np.zeros_like(region, dtype=np.uint8)
        for label in merged_labels:
            combined_mask[region == label] = 255

        return combined_mask

    @staticmethod
    def _make_binaries(scaled_gray: np.ndarray) -> List[np.ndarray]:
        """Generate binarized versions of a scaled grayscale image."""
        results = []

        # OTSU — inverted (dark text on light background = standard cards)
        _, otsu_inv = cv2.threshold(scaled_gray, 0, 255,
                                     cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        results.append(otsu_inv)

        # Fixed thresholds (inverted)
        for thresh in [100, 140, 180]:
            _, binary_inv = cv2.threshold(scaled_gray, thresh, 255,
                                           cv2.THRESH_BINARY_INV)
            results.append(binary_inv)

        # Adaptive threshold
        adaptive = cv2.adaptiveThreshold(scaled_gray, 255,
                                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY_INV, 15, 5)
        results.append(adaptive)

        return results

    # ------------------------------------------------------------------
    # Suit detection via color analysis + shape
    # ------------------------------------------------------------------

    def _detect_suit(self, img: np.ndarray) -> Optional[str]:
        """Detect the suit using color analysis on the suit symbol area.

        Strategy:
        1. Crop the suit area (below rank, in top portion of card)
        2. Count pixels matching each suit's color range
        3. For 4-color decks, the dominant color identifies the suit
        4. For 2-color decks, fall back to shape analysis
        """
        h, w = img.shape[:2]

        # The suit symbol sits below the rank text.  Crop the relevant area.
        # Try both the suit-symbol area and a wider area for robustness.
        regions_to_try = [
            img[int(h * 0.22):int(h * 0.55), 0:int(w * 0.50)],  # Standard suit position
            img[int(h * 0.30):int(h * 0.60), 0:int(w * 0.45)],  # Lower suit position
            img[0:int(h * 0.55), 0:int(w * 0.50)],               # Full rank+suit area
        ]

        for region in regions_to_try:
            if region.size == 0:
                continue
            suit = self._color_detect_suit(region)
            if suit:
                return suit

        return None

    def _color_detect_suit(self, region: np.ndarray) -> Optional[str]:
        """Color-based suit detection on a single region."""
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

        # Mask out the card background (white/light pixels)
        # Only analyze colored/dark pixels that belong to the suit symbol
        bg_mask = cv2.inRange(hsv, np.array([0, 0, 150]), np.array([180, 40, 255]))
        fg_mask = cv2.bitwise_not(bg_mask)
        fg_pixel_count = np.sum(fg_mask > 0)

        if fg_pixel_count < 8:
            return None

        # Count pixels matching each suit color
        suit_scores = {}
        for suit, color_range in SUIT_COLORS_HSV.items():
            mask = cv2.inRange(hsv, color_range['lower'], color_range['upper'])
            if 'lower2' in color_range:
                mask2 = cv2.inRange(hsv, color_range['lower2'], color_range['upper2'])
                mask = cv2.bitwise_or(mask, mask2)
            # Only count foreground pixels
            mask = cv2.bitwise_and(mask, fg_mask)
            suit_scores[suit] = np.sum(mask > 0)

        total_matched = sum(suit_scores.values())
        if total_matched == 0:
            return self._detect_suit_by_brightness(region)

        best_suit = max(suit_scores, key=suit_scores.get)
        best_count = suit_scores[best_suit]

        # Confidence check: best suit should have a significant lead
        sorted_scores = sorted(suit_scores.values(), reverse=True)
        if len(sorted_scores) >= 2 and sorted_scores[0] > 0:
            dominance = sorted_scores[0] / max(total_matched, 1)
        else:
            dominance = 1.0

        logger.debug("Suit scores: %s (dominance=%.2f)",
                     {k: v for k, v in suit_scores.items() if v > 0}, dominance)

        # 4-color deck: clear winner
        if dominance > 0.5:
            return best_suit

        # 2-color deck: need shape analysis to disambiguate
        if best_suit == 'h':
            # Red: could be hearts or diamonds
            shape = self._detect_suit_by_shape(region)
            if shape in ('h', 'd'):
                return shape
            return 'h'

        if best_suit == 's':
            # Black: could be spades or clubs
            shape = self._detect_suit_by_shape(region)
            if shape in ('s', 'c'):
                return shape
            return 's'

        return best_suit

    @staticmethod
    def _detect_suit_by_shape(region: np.ndarray) -> Optional[str]:
        """Distinguish suits by the shape of the suit symbol.

        Shape heuristics:
        - Hearts (heart): wider at top, pointed at bottom
        - Diamonds (diamond): pointed at top and bottom, widest at middle
        - Spades (spade): pointed at top, wider at bottom
        - Clubs (club): three round lobes, stem at bottom

        Uses pixel distribution analysis to classify.
        """
        if cv2 is None:
            return None

        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

        h, w = mask.shape
        if h < 6 or w < 6:
            return None

        total = np.sum(mask > 0)
        if total < 15:
            return None

        # Divide into vertical thirds
        third_h = h // 3
        top = mask[:third_h, :]
        mid = mask[third_h:third_h * 2, :]
        bot = mask[third_h * 2:, :]

        top_px = np.sum(top > 0)
        mid_px = np.sum(mid > 0)
        bot_px = np.sum(bot > 0)

        # Width at each third
        top_widths = [np.sum(mask[r, :] > 0) for r in range(third_h)]
        mid_widths = [np.sum(mask[r, :] > 0) for r in range(third_h, third_h * 2)]
        bot_widths = [np.sum(mask[r, :] > 0) for r in range(third_h * 2, h)]

        max_top_w = max(top_widths) if top_widths else 0
        max_mid_w = max(mid_widths) if mid_widths else 0
        max_bot_w = max(bot_widths) if bot_widths else 0

        # Determine if the symbol color is red or black
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        red_mask1 = cv2.inRange(hsv, np.array([0, 60, 60]), np.array([15, 255, 255]))
        red_mask2 = cv2.inRange(hsv, np.array([155, 60, 60]), np.array([180, 255, 255]))
        red_count = np.sum(red_mask1 > 0) + np.sum(red_mask2 > 0)
        is_red = red_count > total * 0.25

        if is_red:
            # Hearts vs Diamonds
            # Diamond: widest at middle, narrow at top and bottom
            # Heart: wide at top (two bumps), narrow at bottom (point)
            if max_mid_w > max_top_w and max_mid_w > max_bot_w * 1.1:
                return 'd'
            if top_px > bot_px:
                return 'h'
            # Diamond shape is vertically symmetric
            if abs(top_px - bot_px) < total * 0.15:
                return 'd'
            return 'h'
        else:
            # Spades vs Clubs
            # Spade: pointed top, wide bottom body, thin stem
            # Club: three round lobes at top/sides, thin stem at bottom
            if max_top_w < max_mid_w and max_bot_w < max_mid_w:
                # Widest at middle - could be spade body
                return 's'
            if top_px > bot_px * 1.1:
                return 'c'  # Club has more mass at top (three lobes)
            return 's'

    @staticmethod
    def _detect_suit_by_brightness(region: np.ndarray) -> Optional[str]:
        """Last-resort suit detection using brightness and saturation analysis."""
        if cv2 is None:
            return None

        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

        # Mask out very bright pixels (card background)
        fg_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 160]))
        if np.sum(fg_mask > 0) < 8:
            return None

        fg_hue = hsv[:, :, 0][fg_mask > 0]
        fg_sat = hsv[:, :, 1][fg_mask > 0]
        fg_val = hsv[:, :, 2][fg_mask > 0]

        avg_sat = np.mean(fg_sat)
        avg_val = np.mean(fg_val)
        avg_hue = np.mean(fg_hue)

        # Low saturation + low value = black (spades)
        if avg_sat < 50 and avg_val < 80:
            return 's'
        # Red hue
        if avg_hue < 15 or avg_hue > 160:
            return 'h'
        # Green hue
        if 30 < avg_hue < 90:
            return 'c'
        # Blue hue
        if 90 < avg_hue < 135:
            return 'd'

        return 's'  # Default to spades (black)

    # ------------------------------------------------------------------
    # Card localization (finding card rectangles on the table)
    # ------------------------------------------------------------------

    @staticmethod
    def find_card_rectangles(img: np.ndarray,
                              search_region: tuple = None,
                              min_card_w: int = 15,
                              max_card_w: int = 120,
                              min_card_h: int = 20,
                              max_card_h: int = 160,
                              ) -> List[Tuple[int, int, int, int]]:
        """Find card-shaped white rectangles in an image.

        This is used to dynamically locate cards on the table instead of
        relying on hardcoded pixel coordinates.

        Args:
            img: Full table image (BGR).
            search_region: (x, y, w, h) to restrict search area, or None
                           for full image.
            min_card_w, max_card_w: Expected card width range.
            min_card_h, max_card_h: Expected card height range.

        Returns:
            List of (x, y, w, h) bounding boxes in image coordinates.
        """
        if cv2 is None or img is None:
            return []

        if search_region:
            sx, sy, sw, sh = search_region
            roi = img[sy:sy+sh, sx:sx+sw]
            offset_x, offset_y = sx, sy
        else:
            roi = img
            offset_x, offset_y = 0, 0

        if roi.size == 0:
            return []

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Multi-threshold card finding
        all_rects = []

        # Strategy 1: HSV white detection (low sat, high value)
        for s_max, v_min in [(100, 130), (120, 100), (150, 80)]:
            mask = cv2.inRange(hsv, np.array([0, 0, v_min]),
                                np.array([180, s_max, 255]))
            rects = _extract_card_rects(mask, min_card_w, max_card_w,
                                         min_card_h, max_card_h)
            all_rects.extend(rects)
            if len(rects) >= 2:
                break

        # Strategy 2: Grayscale brightness
        if len(all_rects) < 2:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            for thresh in [150, 120, 90]:
                _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
                rects = _extract_card_rects(binary, min_card_w, max_card_w,
                                             min_card_h, max_card_h)
                all_rects.extend(rects)
                if len(rects) >= 2:
                    break

        # Deduplicate overlapping rectangles
        all_rects = _deduplicate_rects(all_rects)

        # Offset back to image coordinates
        return [(x + offset_x, y + offset_y, w, h) for x, y, w, h in all_rects]


# ---------------------------------------------------------------------------
# Helper functions (module-level)
# ---------------------------------------------------------------------------

def _extract_card_rects(mask: np.ndarray,
                         min_w: int, max_w: int,
                         min_h: int, max_h: int) -> List[Tuple[int, int, int, int]]:
    """Extract card-shaped rectangles from a binary mask.

    Handles both individual cards and merged card blobs (two adjacent
    cards that appear as one wide rectangle).
    """
    if cv2 is None:
        return []

    # Minimal morphological cleanup — only 1 iteration of CLOSE to avoid
    # bridging the narrow gap between adjacent cards
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

    rects = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = h / w if w > 0 else 0
        area = cv2.contourArea(cnt)
        bbox_area = w * h
        fill = area / bbox_area if bbox_area > 0 else 0

        # Single card: right size and taller-than-wide
        if min_w <= w <= max_w and min_h <= h <= max_h:
            if 0.8 <= aspect <= 3.0 and fill > 0.3:
                rects.append((x, y, w, h))
                continue

        # Merged pair: too wide for a single card but right height.
        # Split into two halves.
        if w > max_w and min_h <= h <= max_h and fill > 0.3:
            half_w = w // 2
            if min_w <= half_w <= max_w:
                rects.append((x, y, half_w, h))
                rects.append((x + half_w, y, w - half_w, h))
                continue

        # Wide blob that could be 2 cards (aspect too low for single card)
        if w > min_w * 1.5 and min_h <= h <= max_h and aspect < 0.8 and fill > 0.3:
            half_w = w // 2
            if min_w * 0.5 <= half_w <= max_w:
                rects.append((x, y, half_w, h))
                rects.append((x + half_w, y, w - half_w, h))

    return rects


def _deduplicate_rects(rects: List[Tuple[int, int, int, int]],
                        overlap_thresh: float = 0.4) -> List[Tuple[int, int, int, int]]:
    """Remove duplicate/overlapping rectangles, keeping the largest."""
    if len(rects) <= 1:
        return rects

    # Sort by area (largest first)
    rects = sorted(rects, key=lambda r: r[2] * r[3], reverse=True)
    keep = []

    for rect in rects:
        is_dup = False
        for kept in keep:
            if _iou(rect, kept) > overlap_thresh:
                is_dup = True
                break
        if not is_dup:
            keep.append(rect)

    return keep


def _iou(r1: Tuple[int, int, int, int],
          r2: Tuple[int, int, int, int]) -> float:
    """Intersection over Union of two rectangles."""
    x1 = max(r1[0], r2[0])
    y1 = max(r1[1], r2[1])
    x2 = min(r1[0] + r1[2], r2[0] + r2[2])
    y2 = min(r1[1] + r1[3], r2[1] + r2[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    area1 = r1[2] * r1[3]
    area2 = r2[2] * r2[3]
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Convenience functions (backward-compatible API)
# ---------------------------------------------------------------------------

def parse_hand(hand_str: str) -> Tuple[Card, Card]:
    """Parse a hand string like 'AhKs' or 'Ah Ks' into two Card objects."""
    hand_str = hand_str.strip().replace(" ", "")
    if len(hand_str) != 4:
        raise ValueError(f"Invalid hand string: {hand_str}")
    return (Card.from_string(hand_str[:2]), Card.from_string(hand_str[2:]))


def hand_to_string(card1: Card, card2: Card, include_suits: bool = True) -> str:
    """Convert two cards to a hand notation string."""
    if include_suits:
        return f"{card1}{card2}"

    r1, r2 = card1.rank_value, card2.rank_value
    if r1 < r2:
        r1, r2 = r2, r1
        card1, card2 = card2, card1

    if card1.rank == card2.rank:
        return f"{card1.rank}{card2.rank}"
    elif card1.suit == card2.suit:
        return f"{card1.rank}{card2.rank}s"
    else:
        return f"{card1.rank}{card2.rank}o"
