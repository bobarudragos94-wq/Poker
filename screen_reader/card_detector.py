"""
Card detection module using color analysis and pattern recognition.
Detects card rank and suit from PokerStars table screenshots.
"""

import logging
from typing import Optional, Tuple, List
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

# PokerStars color ranges (HSV) for suit detection.
#
# 4-color deck (PokerStars default for online play):
#   Spades = black, Hearts = red, Diamonds = BLUE, Clubs = GREEN
#
# 2-color deck (classic):
#   Spades = black, Hearts = red, Diamonds = RED, Clubs = BLACK
#
# We detect all four 4-color ranges first.  If the best match is red we
# disambiguate hearts vs diamonds using the suit symbol shape (see
# ``_detect_suit``).  This handles both deck types.

SUIT_COLORS_HSV_4COLOR = {
    's': {  # Spades - black/dark gray
        'lower': np.array([0, 0, 0]),
        'upper': np.array([180, 50, 80]),
    },
    'h': {  # Hearts - red
        'lower': np.array([0, 100, 100]),
        'upper': np.array([10, 255, 255]),
        'lower2': np.array([160, 100, 100]),
        'upper2': np.array([180, 255, 255]),
    },
    'd': {  # Diamonds - blue (4-color only)
        'lower': np.array([100, 80, 80]),
        'upper': np.array([130, 255, 255]),
    },
    'c': {  # Clubs - green (4-color only)
        'lower': np.array([35, 80, 80]),
        'upper': np.array([85, 255, 255]),
    },
}

# Backward-compatible alias
SUIT_COLORS_HSV = SUIT_COLORS_HSV_4COLOR

# Rank detection via OCR character mapping
RANK_OCR_MAP = {
    '2': '2', '3': '3', '4': '4', '5': '5', '6': '6',
    '7': '7', '8': '8', '9': '9', '0': 'T', '10': 'T',
    'T': 'T', 'J': 'J', 'Q': 'Q', 'K': 'K', 'A': 'A',
    '1': 'A',  # Sometimes OCR reads A as 1
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


class CardDetector:
    """Detects playing cards from screen captures of PokerStars."""

    def __init__(self, match_threshold: float = 0.75):
        self.match_threshold = match_threshold
        self._ocr = None

    def _get_ocr(self):
        """Lazy-load OCR reader."""
        if self._ocr is None:
            from .ocr import OCRReader
            self._ocr = OCRReader(psm=10)  # Single character mode
        return self._ocr

    def detect_card(self, img: np.ndarray) -> Optional[Card]:
        """
        Detect a single card from an image region.
        Uses a combination of OCR for rank and color analysis for suit.

        Args:
            img: Card image region as numpy array (BGR).

        Returns:
            Card object or None if no card detected.
        """
        if img is None or img.size == 0:
            logger.debug("Card detect: empty image")
            return None

        # Check if the region actually contains a card (not empty/green felt)
        if not self._is_card_present(img):
            logger.debug("Card detect: no card present (%dx%d region)", img.shape[1], img.shape[0])
            return None

        rank = self._detect_rank(img)
        suit = self._detect_suit(img)

        if rank and suit:
            logger.debug("Card detected: %s%s", rank, suit)
            return Card(rank=rank, suit=suit)

        logger.debug("Card detect: partial match rank=%s suit=%s", rank, suit)
        return None

    def detect_multiple_cards(self, images: List[np.ndarray]) -> List[Optional[Card]]:
        """Detect cards from multiple image regions."""
        return [self.detect_card(img) for img in images]

    def _is_card_present(self, img: np.ndarray) -> bool:
        """
        Check if a card is present in the image region.
        Cards are white/light colored; empty spaces are table felt.
        Uses multiple strategies to handle different PokerStars themes
        (green felt for cash games, blue/navy felt for tournaments).
        """
        if cv2 is None or img is None:
            return False

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Strategy 1: Check for white/light pixels (card background)
        # Cards are mostly white with low saturation and high value.
        # Use a generous threshold to handle slight off-whites and shadows.
        lower_white = np.array([0, 0, 130])
        upper_white = np.array([180, 80, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        white_ratio = np.sum(white_mask > 0) / white_mask.size

        if white_ratio > 0.08:
            logger.debug("Card present (white): ratio=%.3f", white_ratio)
            return True

        # Strategy 2: Check that this is NOT table felt (green OR blue).
        # PokerStars uses green felt for cash and blue/navy for tournaments.
        # Green felt: H=30-90
        lower_green = np.array([30, 40, 40])
        upper_green = np.array([90, 255, 200])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        green_ratio = np.sum(green_mask > 0) / green_mask.size

        # Blue felt (common in PokerStars tournaments): H=90-140
        lower_blue = np.array([90, 30, 20])
        upper_blue = np.array([140, 255, 200])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        blue_ratio = np.sum(blue_mask > 0) / blue_mask.size

        felt_ratio = green_ratio + blue_ratio

        # If region is mostly NOT felt, a card may be present
        if felt_ratio < 0.30:
            lower_light = np.array([0, 0, 80])
            upper_light = np.array([180, 255, 255])
            light_mask = cv2.inRange(hsv, lower_light, upper_light)
            light_ratio = np.sum(light_mask > 0) / light_mask.size

            if light_ratio > 0.20:
                logger.debug("Card present (non-felt): green=%.3f blue=%.3f light=%.3f",
                             green_ratio, blue_ratio, light_ratio)
                return True

        logger.debug("No card present: white=%.3f green=%.3f blue=%.3f",
                     white_ratio, green_ratio, blue_ratio)
        return False

    def _detect_rank(self, img: np.ndarray) -> Optional[str]:
        """Detect the rank of a card using OCR on the top-left corner.

        Tries multiple preprocessing strategies to handle different card
        styles, theme colours, and image qualities.
        """
        if cv2 is None:
            return None

        h, w = img.shape[:2]
        if h < 10 or w < 10:
            return None

        # The rank character is printed in the top-left corner of the card.
        # Crop generously to ensure the character is included even if the
        # contour bounding box is slightly off.
        rank_region = img[1:int(h * 0.40), 1:int(w * 0.45)]

        if rank_region.size == 0:
            return None

        gray = cv2.cvtColor(rank_region, cv2.COLOR_BGR2GRAY)

        # Scale up for better OCR accuracy
        scale = max(3, 60 // max(gray.shape[0], 1))  # aim for ~60px tall
        gray = cv2.resize(gray, (gray.shape[1] * scale, gray.shape[0] * scale),
                          interpolation=cv2.INTER_CUBIC)

        # Try multiple thresholding strategies — the card background can be
        # white (standard) or tinted (themed tables).
        ocr = self._get_ocr()
        for thresh_method in ("otsu", "fixed_low", "fixed_high"):
            if thresh_method == "otsu":
                _, binary = cv2.threshold(gray, 0, 255,
                                          cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            elif thresh_method == "fixed_low":
                _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
            else:
                _, binary = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY_INV)

            text = ocr.read_text(binary, whitelist="23456789TJQKA10",
                                 preprocess=False)
            text = text.strip().upper()

            if text in RANK_OCR_MAP:
                return RANK_OCR_MAP[text]
            if text and text[0] in RANK_OCR_MAP:
                return RANK_OCR_MAP[text[0]]

        logger.debug("Rank OCR failed on %dx%d region", w, h)
        return None

    def _detect_suit(self, img: np.ndarray) -> Optional[str]:
        """
        Detect the suit of a card using color analysis.

        Works with both 4-color and 2-color decks:
        - 4-color: Spades=black, Hearts=red, Diamonds=BLUE, Clubs=GREEN
        - 2-color: Spades=black, Hearts=red, Diamonds=RED,  Clubs=BLACK

        For the 2-color deck, red suits (hearts/diamonds) and black suits
        (spades/clubs) are distinguished by the shape of the suit symbol.
        """
        if cv2 is None:
            return None

        h, w = img.shape[:2]

        # The suit symbol is below the rank, in the top portion
        suit_region = img[int(h * 0.25):int(h * 0.55), 2:int(w * 0.45)]

        if suit_region.size == 0:
            return None

        hsv = cv2.cvtColor(suit_region, cv2.COLOR_BGR2HSV)

        # Count pixels matching each 4-color suit
        suit_scores = {}
        for suit, color_range in SUIT_COLORS_HSV_4COLOR.items():
            mask = cv2.inRange(hsv, color_range['lower'], color_range['upper'])
            if 'lower2' in color_range:
                mask2 = cv2.inRange(hsv, color_range['lower2'], color_range['upper2'])
                mask = cv2.bitwise_or(mask, mask2)
            suit_scores[suit] = np.sum(mask > 0)

        if not suit_scores or max(suit_scores.values()) == 0:
            return self._detect_suit_by_average_color(suit_region)

        best_suit = max(suit_scores, key=suit_scores.get)

        # If the best match is red (hearts) but blue/green scored zero,
        # the user likely has a 2-color deck.  Try to distinguish hearts
        # from diamonds using the shape of the suit symbol.
        if best_suit == 'h' and suit_scores.get('d', 0) == 0 and suit_scores.get('c', 0) == 0:
            shape_suit = self._detect_suit_by_shape(suit_region)
            if shape_suit in ('h', 'd'):
                best_suit = shape_suit

        # Similarly, if black is dominant and no green/blue was found,
        # try to distinguish spades from clubs by shape.
        if best_suit == 's' and suit_scores.get('d', 0) == 0 and suit_scores.get('c', 0) == 0:
            shape_suit = self._detect_suit_by_shape(suit_region)
            if shape_suit in ('s', 'c'):
                best_suit = shape_suit

        return best_suit

    def _detect_suit_by_shape(self, suit_region: np.ndarray) -> Optional[str]:
        """
        Distinguish suits by the shape of the suit symbol.
        Used for 2-color decks where hearts/diamonds are both red
        and spades/clubs are both black.

        Shape heuristics:
        - Hearts (♥): wider at top, pointed at bottom → more pixels in top half
        - Diamonds (♦): pointed at top and bottom → vertically symmetric
        - Spades (♠): pointed at top, wider at bottom → more pixels in bottom half
        - Clubs (♣): three lobes → roughly symmetric with wider top
        """
        if cv2 is None:
            return None

        gray = cv2.cvtColor(suit_region, cv2.COLOR_BGR2GRAY)
        # Isolate non-white pixels (the suit symbol itself)
        _, mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

        h, w = mask.shape
        if h < 4 or w < 4:
            return None

        total = np.sum(mask > 0)
        if total < 10:
            return None

        mid_y = h // 2
        top_pixels = np.sum(mask[:mid_y, :] > 0)
        bot_pixels = np.sum(mask[mid_y:, :] > 0)

        # Vertical symmetry: ratio of top-half to bottom-half pixels
        if total > 0:
            top_ratio = top_pixels / total
        else:
            return None

        # Check horizontal extent of the widest row in top vs bottom
        top_widths = [np.sum(mask[r, :] > 0) for r in range(mid_y)]
        bot_widths = [np.sum(mask[r, :] > 0) for r in range(mid_y, h)]
        max_top_w = max(top_widths) if top_widths else 0
        max_bot_w = max(bot_widths) if bot_widths else 0

        # Detect suit color (red or black) to narrow candidates
        hsv = cv2.cvtColor(suit_region, cv2.COLOR_BGR2HSV)
        red_mask = cv2.inRange(hsv, np.array([0, 80, 80]), np.array([10, 255, 255]))
        red_mask2 = cv2.inRange(hsv, np.array([160, 80, 80]), np.array([180, 255, 255]))
        is_red = (np.sum(red_mask > 0) + np.sum(red_mask2 > 0)) > total * 0.3

        if is_red:
            # Hearts vs Diamonds
            # ♥: wider at top → top_ratio > 0.5, max_top_w > max_bot_w
            # ♦: symmetric diamond → top_ratio ≈ 0.5, max widths at center
            if top_ratio > 0.55 or max_top_w > max_bot_w * 1.2:
                return 'h'
            else:
                return 'd'
        else:
            # Spades vs Clubs
            # ♠: pointed top, wide bottom → bot_pixels > top_pixels
            # ♣: three lobes at top → top_pixels ≥ bot_pixels
            if top_ratio > 0.45 and max_top_w >= max_bot_w:
                return 'c'
            else:
                return 's'

    def _detect_suit_by_average_color(self, img: np.ndarray) -> Optional[str]:
        """Fallback suit detection using average color of the suit symbol."""
        if cv2 is None:
            return None

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Mask out white/light pixels (card background)
        mask = cv2.inRange(hsv, np.array([0, 30, 0]), np.array([180, 255, 200]))

        if np.sum(mask > 0) < 10:
            return None

        # Get average hue of the colored pixels
        hue_values = hsv[:, :, 0][mask > 0]
        sat_values = hsv[:, :, 1][mask > 0]
        val_values = hsv[:, :, 2][mask > 0]

        avg_hue = np.mean(hue_values)
        avg_sat = np.mean(sat_values)
        avg_val = np.mean(val_values)

        # Classify based on average hue
        if avg_sat < 40:
            return 's'  # Spades (low saturation = black/gray)
        elif avg_hue < 15 or avg_hue > 160:
            return 'h'  # Hearts (red hue)
        elif 35 < avg_hue < 85:
            return 'c'  # Clubs (green hue)
        elif 100 < avg_hue < 130:
            return 'd'  # Diamonds (blue hue)

        return None


def parse_hand(hand_str: str) -> Tuple[Card, Card]:
    """
    Parse a hand string like 'AhKs' or 'Ah Ks' into two Card objects.
    """
    hand_str = hand_str.strip().replace(" ", "")
    if len(hand_str) != 4:
        raise ValueError(f"Invalid hand string: {hand_str}")
    return (Card.from_string(hand_str[:2]), Card.from_string(hand_str[2:]))


def hand_to_string(card1: Card, card2: Card, include_suits: bool = True) -> str:
    """Convert two cards to a hand notation string."""
    if include_suits:
        return f"{card1}{card2}"

    # Return range notation (e.g., "AKs", "AKo", "AA")
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
