"""
Table state reader - combines screen capture, OCR, and card detection
to build a complete picture of the current game state.
"""

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from .capture import ScreenCapture
from .ocr import OCRReader
from .card_detector import CardDetector, Card
from config import AppConfig, TableRegions, BASELINE_WIDTH, BASELINE_HEIGHT

logger = logging.getLogger(__name__)


@dataclass
class PlayerState:
    """State of a single player at the table."""
    seat: int
    stack: float = 0.0
    current_bet: float = 0.0
    is_active: bool = False       # Has cards / in the hand
    is_dealer: bool = False
    is_hero: bool = False
    name: str = ""


@dataclass
class GameState:
    """Complete snapshot of the current game state."""
    # Hero's cards
    hero_cards: List[Card] = field(default_factory=list)

    # Board cards (0-5 cards)
    board_cards: List[Card] = field(default_factory=list)

    # Pot and betting
    pot_size: float = 0.0
    current_bet: float = 0.0      # Current bet to call

    # Blinds
    small_blind: float = 0.0
    big_blind: float = 0.0
    ante: float = 0.0

    # Hero info
    hero_stack: float = 0.0
    hero_position: str = ""        # "BTN", "SB", "BB", "UTG", etc.
    hero_seat: int = 0

    # Table info
    num_players: int = 6
    active_players: int = 6        # Players still in the hand
    dealer_seat: int = -1

    # All player states
    players: List[PlayerState] = field(default_factory=list)

    # Available actions
    can_fold: bool = False
    can_check: bool = False
    can_call: bool = False
    can_raise: bool = False
    can_allin: bool = False
    call_amount: float = 0.0

    # Game phase
    street: str = "preflop"        # "preflop", "flop", "turn", "river"

    # Timestamp
    timestamp: float = 0.0

    @property
    def hero_stack_bb(self) -> float:
        """Hero's stack in big blinds."""
        if self.big_blind > 0:
            return self.hero_stack / self.big_blind
        return 0.0

    @property
    def pot_bb(self) -> float:
        """Pot size in big blinds."""
        if self.big_blind > 0:
            return self.pot_size / self.big_blind
        return 0.0

    @property
    def m_ratio(self) -> float:
        """Harrington's M-ratio (orbits until blinded out)."""
        cost_per_orbit = (self.small_blind + self.big_blind +
                          self.ante * self.num_players)
        if cost_per_orbit > 0:
            return self.hero_stack / cost_per_orbit
        return float('inf')

    @property
    def hand_notation(self) -> str:
        """Hero's hand in standard notation (e.g., 'AKs', 'QQ')."""
        if len(self.hero_cards) != 2:
            return ""
        c1, c2 = self.hero_cards
        # Put higher card first
        if c1.rank_value < c2.rank_value:
            c1, c2 = c2, c1
        if c1.rank == c2.rank:
            return f"{c1.rank}{c2.rank}"
        elif c1.suit == c2.suit:
            return f"{c1.rank}{c2.rank}s"
        else:
            return f"{c1.rank}{c2.rank}o"

    @property
    def is_preflop(self) -> bool:
        return len(self.board_cards) == 0

    @property
    def is_flop(self) -> bool:
        return len(self.board_cards) == 3

    @property
    def is_turn(self) -> bool:
        return len(self.board_cards) == 4

    @property
    def is_river(self) -> bool:
        return len(self.board_cards) == 5

    def get_street(self) -> str:
        n = len(self.board_cards)
        if n == 0:
            return "preflop"
        elif n == 3:
            return "flop"
        elif n == 4:
            return "turn"
        elif n == 5:
            return "river"
        return "unknown"


class TableStateReader:
    """
    Reads the complete table state from screen capture.
    Combines ScreenCapture, OCR, and CardDetector.
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self.capture = ScreenCapture(config.pokerstars_window_title)
        self.ocr = OCRReader(language=config.ocr_language, psm=config.ocr_psm)
        self.card_detector = CardDetector(match_threshold=config.card_match_threshold)
        self.regions = config.regions
        self._last_state: Optional[GameState] = None
        # Cached table-felt bounding box (x, y, w, h) within the captured image
        self._table_bbox: Optional[Tuple[int, int, int, int]] = None
        self._table_bbox_miss_count: int = 0
        self._debug_frame_count: int = 0
        # Set to True (or pass --verbose) to dump cropped regions to disk
        self.debug_save_regions: bool = logging.getLogger().isEnabledFor(logging.DEBUG)

    @staticmethod
    def _detect_table_felt(img: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Detect the green felt table area within the captured window image.

        PokerStars tables have a large green/teal oval felt area.  When the
        window aspect ratio differs from the baseline 800x600 (4:3), the
        client adds dark padding.  Finding the actual table bounds lets us
        scale the baseline regions accurately.

        Returns (x, y, w, h) of the bounding rectangle of the felt area,
        or None if detection fails.
        """
        if cv2 is None:
            return None

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # PokerStars felt is green/teal: H 35-90, S 30-255, V 40-220
        # This deliberately wide range covers the various PokerStars themes
        lower = np.array([25, 25, 30])
        upper = np.array([95, 255, 220])
        mask = cv2.inRange(hsv, lower, upper)

        # Morphological close to fill small gaps in the felt
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find the largest contour (should be the table)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        img_area = img.shape[0] * img.shape[1]

        # The felt should cover a significant portion of the image (>15%)
        if area < img_area * 0.15:
            return None

        x, y, w, h = cv2.boundingRect(largest)
        return (x, y, w, h)

    def read_state(self) -> Optional[GameState]:
        """
        Capture the current table and read the full game state.
        Returns a GameState object or None if capture fails.
        """
        # Capture full table
        table_img = self.capture.capture_full_table()
        if table_img is None:
            logger.warning("Failed to capture table screenshot")
            return self._last_state

        h_img, w_img = table_img.shape[:2]

        # Detect the actual table felt area to handle aspect-ratio differences
        # between the captured window and the 800x600 baseline.
        felt_bbox = self._detect_table_felt(table_img)
        if felt_bbox is not None:
            self._table_bbox = felt_bbox
            self._table_bbox_miss_count = 0
            logger.debug("Table felt detected at %s (image %dx%d)", felt_bbox, w_img, h_img)
        else:
            self._table_bbox_miss_count += 1
            if self._table_bbox_miss_count > 20:
                self._table_bbox = None  # stale, drop it
            if self._table_bbox is not None:
                logger.debug("Table felt not detected this frame, using cached bbox")
            else:
                logger.debug("Table felt not detected; falling back to full image")

        logger.debug("Captured table image: %dx%d (baseline %dx%d, scale %.2fx%.2f)",
                      w_img, h_img, BASELINE_WIDTH, BASELINE_HEIGHT,
                      w_img / BASELINE_WIDTH, h_img / BASELINE_HEIGHT)

        state = GameState(timestamp=time.time())

        # Save full-table debug screenshot on first few frames
        self._debug_frame_count += 1
        if self._debug_frame_count <= 3:
            self._save_debug_image(table_img, f"full_table_{self._debug_frame_count}")

        # Read hero's cards
        state.hero_cards = self._read_hero_cards(table_img)
        if not state.hero_cards:
            logger.debug("No hero cards detected (image %dx%d)", w_img, h_img)

        # Read board cards
        state.board_cards = self._read_board_cards(table_img)
        state.street = state.get_street()

        # Read pot size
        state.pot_size = self._read_pot(table_img)

        # Read blind levels
        blinds = self._read_blinds(table_img)
        if blinds:
            state.small_blind, state.big_blind, state.ante = blinds
        elif self._last_state:
            # Carry over blind info from last state
            state.small_blind = self._last_state.small_blind
            state.big_blind = self._last_state.big_blind
            state.ante = self._last_state.ante

        # Read player stacks and detect who is active
        state.players = self._read_players(table_img)
        state.num_players = self.config.num_players

        # Find dealer button
        state.dealer_seat = self._find_dealer(table_img)

        # Set hero info
        hero_seat = self.config.hero_seat
        state.hero_seat = hero_seat
        for p in state.players:
            if p.seat == hero_seat:
                state.hero_stack = p.stack
                p.is_hero = True
                break

        # Determine hero position
        if state.dealer_seat >= 0:
            state.hero_position = self._get_position(
                hero_seat, state.dealer_seat, state.num_players
            )

        # Count active players
        state.active_players = sum(1 for p in state.players if p.is_active)

        # Read available actions
        actions = self._read_actions(table_img)
        state.can_fold = actions.get("fold", False)
        state.can_check = actions.get("check", False)
        state.can_call = actions.get("call", False)
        state.can_raise = actions.get("raise", False)
        state.can_allin = actions.get("allin", False)

        # Calculate call amount
        if state.players:
            max_bet = max((p.current_bet for p in state.players), default=0)
            hero_bet = next((p.current_bet for p in state.players
                             if p.seat == hero_seat), 0)
            state.call_amount = max_bet - hero_bet
            state.current_bet = max_bet

        self._last_state = state
        return state

    def _save_debug_image(self, img: np.ndarray, name: str):
        """Save a debug image to disk (only when debug logging is on)."""
        if not self.debug_save_regions or cv2 is None:
            return
        import sys as _sys
        if getattr(_sys, "frozen", False):
            base = os.path.dirname(_sys.executable)
        else:
            base = os.path.dirname(os.path.abspath(__file__))
        debug_dir = os.path.join(base, "debug_regions")
        os.makedirs(debug_dir, exist_ok=True)
        path = os.path.join(debug_dir, f"{name}.png")
        try:
            cv2.imwrite(path, img)
        except Exception:
            pass

    def _read_hero_cards(self, img: np.ndarray) -> List[Card]:
        """Read hero's hole cards."""
        cards = []
        for idx, region in enumerate([self.regions.hero_card1, self.regions.hero_card2]):
            card_img = self._crop_region(img, region)
            if card_img is not None:
                if self._debug_frame_count < 3:
                    self._save_debug_image(card_img, f"hero_card{idx}")
                card = self.card_detector.detect_card(card_img)
                if card:
                    cards.append(card)
        return cards

    def _read_board_cards(self, img: np.ndarray) -> List[Card]:
        """Read community cards (flop, turn, river)."""
        cards = []
        board_regions = [
            self.regions.board_card1, self.regions.board_card2,
            self.regions.board_card3, self.regions.board_card4,
            self.regions.board_card5,
        ]
        for region in board_regions:
            card_img = self._crop_region(img, region)
            if card_img is not None:
                card = self.card_detector.detect_card(card_img)
                if card:
                    cards.append(card)
                else:
                    break  # No more cards on board
        return cards

    def _read_pot(self, img: np.ndarray) -> float:
        """Read the pot size."""
        pot_img = self._crop_region(img, self.regions.pot_area)
        if pot_img is not None:
            value = self.ocr.read_number(pot_img)
            return value if value is not None else 0.0
        return 0.0

    def _read_blinds(self, img: np.ndarray) -> Optional[Tuple[float, float, float]]:
        """Read blind level information."""
        blind_img = self._crop_region(img, self.regions.blind_info)
        if blind_img is not None:
            if self._debug_frame_count <= 3:
                self._save_debug_image(blind_img, "blind_info")
            return self.ocr.read_blind_level(blind_img)
        return None

    def _read_players(self, img: np.ndarray) -> List[PlayerState]:
        """Read all player states (stacks, bets, active status)."""
        players = []
        for seat in range(self.config.num_players):
            player = PlayerState(seat=seat)

            # Read stack
            if seat in self.regions.player_stacks:
                stack_img = self._crop_region(img, self.regions.player_stacks[seat])
                if stack_img is not None:
                    stack = self.ocr.read_number(stack_img)
                    if stack is not None:
                        player.stack = stack
                        player.is_active = True

            # Read current bet
            if seat in self.regions.player_bets:
                bet_img = self._crop_region(img, self.regions.player_bets[seat])
                if bet_img is not None:
                    bet = self.ocr.read_number(bet_img)
                    if bet is not None:
                        player.current_bet = bet

            players.append(player)

        return players

    def _find_dealer(self, img: np.ndarray) -> int:
        """
        Find which seat has the dealer button.
        Looks for the 'D' button image at each seat's dealer position.
        """
        if not cv2:
            return self._last_state.dealer_seat if self._last_state else 0

        best_seat = -1
        best_score = 0

        for seat, region in self.regions.dealer_positions.items():
            btn_img = self._crop_region(img, region)
            if btn_img is None:
                continue

            # The dealer button is typically a white/yellow circle with 'D'
            hsv = cv2.cvtColor(btn_img, cv2.COLOR_BGR2HSV)

            # Look for yellow/white circular button
            lower_yellow = np.array([15, 80, 150])
            upper_yellow = np.array([35, 255, 255])
            lower_white = np.array([0, 0, 200])
            upper_white = np.array([180, 30, 255])

            mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
            mask_white = cv2.inRange(hsv, lower_white, upper_white)
            mask = cv2.bitwise_or(mask_yellow, mask_white)

            score = np.sum(mask > 0) / mask.size
            if score > best_score and score > 0.15:
                best_score = score
                best_seat = seat

        if best_seat >= 0:
            return best_seat

        # Fallback: return last known dealer seat
        if self._last_state and self._last_state.dealer_seat >= 0:
            return self._last_state.dealer_seat
        return 0

    def _read_actions(self, img: np.ndarray) -> dict:
        """Read available action buttons."""
        action_img = self._crop_region(img, self.regions.action_buttons)
        if action_img is not None:
            return self.ocr.detect_action_buttons(action_img)
        return {}

    @staticmethod
    def _scale_region(region: Tuple[int, int, int, int],
                      img_w: int, img_h: int,
                      table_bbox: Optional[Tuple[int, int, int, int]] = None,
                      ) -> Tuple[int, int, int, int]:
        """Scale a region from the 800x600 baseline to the actual image size.

        If *table_bbox* is provided the region is scaled relative to that
        sub-rectangle (the detected green-felt area) instead of the full
        captured image.  This correctly handles windows whose aspect ratio
        differs from the 4:3 baseline (e.g. 1920x1140).
        """
        bx, by, bw, bh = table_bbox if table_bbox else (0, 0, img_w, img_h)
        x, y, w, h = region
        sx = bw / BASELINE_WIDTH
        sy = bh / BASELINE_HEIGHT
        return (int(bx + x * sx), int(by + y * sy), int(w * sx), int(h * sy))

    def _crop_region(self, img: np.ndarray,
                     region: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """Crop a region from the full table image, scaling from baseline 800x600."""
        h_img, w_img = img.shape[:2]
        # Scale region, using the detected table-felt bbox when available
        x, y, w, h = self._scale_region(region, w_img, h_img,
                                         table_bbox=self._table_bbox)
        # Clamp to image boundaries
        x = max(0, min(x, w_img - 1))
        y = max(0, min(y, h_img - 1))
        x2 = min(x + w, w_img)
        y2 = min(y + h, h_img)
        if x2 <= x or y2 <= y:
            return None
        return img[y:y2, x:x2].copy()

    @staticmethod
    def _get_position(hero_seat: int, dealer_seat: int, num_players: int) -> str:
        """Determine hero's position name."""
        if num_players <= 2:
            positions = ["BTN/SB", "BB"]
        elif num_players <= 6:
            positions = ["BTN", "SB", "BB", "UTG", "MP", "CO"]
        else:
            positions = ["BTN", "SB", "BB", "UTG", "UTG+1", "UTG+2",
                         "MP", "HJ", "CO"]

        offset = (hero_seat - dealer_seat) % num_players
        if offset < len(positions):
            return positions[offset]
        return f"Seat{hero_seat}"

    def set_manual_state(self, hero_cards: str = "", position: str = "",
                         stack_bb: float = 0, big_blind: float = 0,
                         pot: float = 0, board: str = "",
                         num_players: int = 6, active_players: int = 6) -> GameState:
        """
        Create a GameState from manual input (for testing or when OCR fails).
        hero_cards: e.g., "AhKs"
        board: e.g., "Th9h2c" or "Th 9h 2c"
        """
        state = GameState(timestamp=time.time())

        # Parse hero cards
        if hero_cards:
            from .card_detector import parse_hand
            try:
                c1, c2 = parse_hand(hero_cards)
                state.hero_cards = [c1, c2]
            except ValueError:
                pass

        # Parse board
        if board:
            board = board.replace(" ", "")
            for i in range(0, len(board), 2):
                if i + 1 < len(board):
                    try:
                        state.board_cards.append(Card.from_string(board[i:i+2]))
                    except ValueError:
                        pass
            state.street = state.get_street()

        state.big_blind = big_blind
        state.small_blind = big_blind / 2 if big_blind else 0
        state.hero_stack = stack_bb * big_blind if big_blind else stack_bb
        state.hero_position = position
        state.pot_size = pot
        state.num_players = num_players
        state.active_players = active_players

        self._last_state = state
        return state
