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

# PokerStars color ranges (HSV) for suit detection
# These are calibrated for the default PokerStars theme
SUIT_COLORS_HSV = {
    's': {  # Spades - black/dark gray
        'lower': np.array([0, 0, 0]),
        'upper': np.array([180, 50, 80]),
    },
    'h': {  # Hearts - red
        'lower': np.array([0, 100, 100]),
        'upper': np.array([10, 255, 255]),
        'lower2': np.array([160, 100, 100]),  # Red wraps in HSV
        'upper2': np.array([180, 255, 255]),
    },
    'd': {  # Diamonds - blue
        'lower': np.array([100, 80, 80]),
        'upper': np.array([130, 255, 255]),
    },
    'c': {  # Clubs - green
        'lower': np.array([35, 80, 80]),
        'upper': np.array([85, 255, 255]),
    },
}

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
        Cards are white/light colored; empty spaces are green felt.
        """
        if cv2 is None or img is None:
            return False

        # Convert to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Check for white/light pixels (card background)
        # Cards are mostly white with S < 50 and V > 180
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([180, 50, 255])
        mask = cv2.inRange(hsv, lower_white, upper_white)

        white_ratio = np.sum(mask > 0) / mask.size
        return white_ratio > 0.15  # At least 15% white pixels = card present

    def _detect_rank(self, img: np.ndarray) -> Optional[str]:
        """Detect the rank of a card using OCR on the top-left corner."""
        if cv2 is None:
            return None

        h, w = img.shape[:2]

        # The rank is in the top-left corner of the card
        # Crop to approximately the top-left 40% x 35%
        rank_region = img[2:int(h * 0.35), 2:int(w * 0.40)]

        if rank_region.size == 0:
            return None

        # Preprocess: convert to grayscale, threshold
        gray = cv2.cvtColor(rank_region, cv2.COLOR_BGR2GRAY)

        # Scale up
        gray = cv2.resize(gray, (gray.shape[1] * 3, gray.shape[0] * 3),
                          interpolation=cv2.INTER_CUBIC)

        # Threshold to get dark text on white card
        _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

        # Use OCR to read the rank character
        ocr = self._get_ocr()
        text = ocr.read_text(binary, whitelist="23456789TJQKA10", preprocess=False)
        text = text.strip().upper()

        # Map OCR result to rank
        if text in RANK_OCR_MAP:
            return RANK_OCR_MAP[text]

        # Try first character only
        if text and text[0] in RANK_OCR_MAP:
            return RANK_OCR_MAP[text[0]]

        return None

    def _detect_suit(self, img: np.ndarray) -> Optional[str]:
        """
        Detect the suit of a card using color analysis.
        PokerStars uses colored suit symbols:
        - Spades: black
        - Hearts: red
        - Diamonds: blue
        - Clubs: green
        """
        if cv2 is None:
            return None

        h, w = img.shape[:2]

        # The suit symbol is below the rank, in the top portion
        # Look at the area below the rank text
        suit_region = img[int(h * 0.25):int(h * 0.55), 2:int(w * 0.45)]

        if suit_region.size == 0:
            return None

        hsv = cv2.cvtColor(suit_region, cv2.COLOR_BGR2HSV)

        # Count pixels matching each suit color
        suit_scores = {}

        for suit, color_range in SUIT_COLORS_HSV.items():
            mask = cv2.inRange(hsv, color_range['lower'], color_range['upper'])
            if 'lower2' in color_range:
                mask2 = cv2.inRange(hsv, color_range['lower2'], color_range['upper2'])
                mask = cv2.bitwise_or(mask, mask2)
            suit_scores[suit] = np.sum(mask > 0)

        if not suit_scores or max(suit_scores.values()) == 0:
            # Fallback: detect based on average color of non-white pixels
            return self._detect_suit_by_average_color(suit_region)

        best_suit = max(suit_scores, key=suit_scores.get)
        return best_suit

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
