"""
Table state reader - combines screen capture, OCR, and card detection
to build a complete picture of the current game state.
"""

import logging
import os
import re
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
        self._no_cards_count = 0

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
        logger.debug("Captured table image: %dx%d (baseline %dx%d, scale %.2fx%.2f)",
                      w_img, h_img, BASELINE_WIDTH, BASELINE_HEIGHT,
                      w_img / BASELINE_WIDTH, h_img / BASELINE_HEIGHT)

        state = GameState(timestamp=time.time())

        # Read hero's cards
        state.hero_cards = self._read_hero_cards(table_img)
        if not state.hero_cards:
            self._no_cards_count += 1
            if self._no_cards_count <= 3 or self._no_cards_count % 20 == 0:
                diag = self._diagnose_hero_region(table_img)
                logger.info("No hero cards detected (image %dx%d, attempt #%d) "
                            "- %s "
                            "- use 'Save Debug Screenshot' to check region alignment",
                            w_img, h_img, self._no_cards_count, diag)
            if self._no_cards_count in (1, 3, 50):
                self._save_hero_debug_images(table_img, self._no_cards_count)
            if self._no_cards_count >= 30:
                logger.info("Hero cards not found after %d attempts - "
                            "re-detecting window...", self._no_cards_count)
                self.capture.invalidate_window()
                self._no_cards_count = 0
        else:
            self._no_cards_count = 0

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

    # ------------------------------------------------------------------
    # Hero card detection
    # ------------------------------------------------------------------

    def _read_hero_cards(self, img: np.ndarray) -> List[Card]:
        """Read hero's hole cards.

        Strategy (ordered by reliability):
        1. Dynamic contour search — find card-shaped white rectangles in
           the bottom-center area of the table.
        2. Fixed region fallback — use the configured pixel coordinates.
        3. Mixed — if dynamic found 1 card, try fixed regions for the other.
        """
        h_img, w_img = img.shape[:2]

        # Method 1: Dynamic contour-based card finding
        # Search the bottom-center of the table where hero cards live
        search = (
            int(w_img * 0.15),   # x
            int(h_img * 0.45),   # y  (below community cards)
            int(w_img * 0.70),   # width
            int(h_img * 0.40),   # height  (extend to bottom of table)
        )
        # Expected card sizes proportional to image
        min_w = max(12, int(w_img * 0.02))
        max_w = max(40, int(w_img * 0.10))
        min_h = max(16, int(h_img * 0.04))
        max_h = max(60, int(h_img * 0.20))

        card_rects = CardDetector.find_card_rectangles(
            img, search_region=search,
            min_card_w=min_w, max_card_w=max_w,
            min_card_h=min_h, max_card_h=max_h,
        )

        logger.debug("Dynamic card search found %d rectangles in hero area", len(card_rects))

        # Try to detect cards from found rectangles
        if card_rects:
            # Sort by x position (left to right) and pick the best pair
            # closest to horizontal center
            center_x = w_img / 2
            card_rects.sort(key=lambda r: abs(r[0] + r[2] / 2 - center_x))

            # Try pairs
            dynamic_cards = self._detect_cards_from_rects(img, card_rects[:6])
            if len(dynamic_cards) == 2:
                return dynamic_cards

        # Method 2: Fixed regions from config
        fixed_cards = []
        for region in [self.regions.hero_card1, self.regions.hero_card2]:
            card_img = self._crop_region(img, region)
            if card_img is not None:
                card = self.card_detector.detect_card(card_img)
                if card:
                    fixed_cards.append(card)

        if len(fixed_cards) == 2:
            return fixed_cards

        # Method 3: Mix dynamic + fixed if one method found 1 card
        all_found = []
        seen = set()
        for card in (dynamic_cards if card_rects else []) + fixed_cards:
            key = (card.rank, card.suit)
            if key not in seen:
                seen.add(key)
                all_found.append(card)

        if len(all_found) >= 2:
            return all_found[:2]
        if len(all_found) == 1:
            return all_found

        return []

    def _detect_cards_from_rects(self, img: np.ndarray,
                                  rects: List[Tuple[int, int, int, int]]) -> List[Card]:
        """Detect cards from a list of bounding rectangles."""
        h_img, w_img = img.shape[:2]
        cards = []
        for rx, ry, rw, rh in rects:
            y1 = max(0, ry)
            y2 = min(h_img, ry + rh)
            x1 = max(0, rx)
            x2 = min(w_img, rx + rw)
            card_img = img[y1:y2, x1:x2]
            if card_img.size > 0:
                card = self.card_detector.detect_card(card_img)
                if card:
                    cards.append(card)
                    logger.debug("Found %s at (%d,%d,%d,%d)", card, rx, ry, rw, rh)
            if len(cards) >= 2:
                break
        return cards

    # ------------------------------------------------------------------
    # Board card detection
    # ------------------------------------------------------------------

    def _read_board_cards(self, img: np.ndarray) -> List[Card]:
        """Read community cards (flop, turn, river).

        Uses both dynamic contour search in the board area and fixed
        regions as a fallback.
        """
        h_img, w_img = img.shape[:2]

        # Method 1: Dynamic search in the board area (center of table)
        board_search = (
            int(w_img * 0.20),   # x
            int(h_img * 0.25),   # y
            int(w_img * 0.60),   # width
            int(h_img * 0.25),   # height
        )
        min_w = max(12, int(w_img * 0.02))
        max_w = max(40, int(w_img * 0.10))
        min_h = max(16, int(h_img * 0.04))
        max_h = max(60, int(h_img * 0.20))

        card_rects = CardDetector.find_card_rectangles(
            img, search_region=board_search,
            min_card_w=min_w, max_card_w=max_w,
            min_card_h=min_h, max_card_h=max_h,
        )

        if card_rects:
            # Sort left to right
            card_rects.sort(key=lambda r: r[0])
            dynamic_cards = []
            for rx, ry, rw, rh in card_rects[:5]:
                y1 = max(0, ry)
                y2 = min(h_img, ry + rh)
                x1 = max(0, rx)
                x2 = min(w_img, rx + rw)
                card_img = img[y1:y2, x1:x2]
                if card_img.size > 0:
                    card = self.card_detector.detect_card(card_img)
                    if card:
                        dynamic_cards.append(card)
            if dynamic_cards:
                return dynamic_cards

        # Method 2: Fixed regions
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

    # ------------------------------------------------------------------
    # OCR-based readings (pot, blinds, players, actions)
    # ------------------------------------------------------------------

    def _read_pot(self, img: np.ndarray) -> float:
        """Read the pot size."""
        pot_img = self._crop_region(img, self.regions.pot_area)
        if pot_img is not None:
            value = self.ocr.read_number(pot_img)
            return value if value is not None else 0.0
        return 0.0

    def _read_blinds(self, img: np.ndarray) -> Optional[Tuple[float, float, float]]:
        """Read blind level information.

        Tries parsing from the PokerStars window title first (most reliable
        for tournaments).  Falls back to OCR on the table image.
        """
        title = self.capture.refresh_window_title()
        if title:
            result = self._parse_blinds_from_title(title)
            if result:
                return result

        blind_img = self._crop_region(img, self.regions.blind_info)
        if blind_img is not None:
            return self.ocr.read_blind_level(blind_img)
        return None

    @staticmethod
    def _parse_blinds_from_title(title: str) -> Optional[Tuple[float, float, float]]:
        """Parse blind level from a PokerStars window title."""
        if not title:
            return None

        match = re.search(
            r'(\d[\d,]*)\s*/\s*(\d[\d,]*)\s*(?:ante\s+(\d[\d,]*))?',
            title, re.IGNORECASE,
        )
        if match:
            try:
                sb = float(match.group(1).replace(",", ""))
                bb = float(match.group(2).replace(",", ""))
                ante = float(match.group(3).replace(",", "")) if match.group(3) else 0.0
                if sb > bb:
                    sb, bb = bb, sb
                logger.debug("Parsed blinds from title: sb=%.0f bb=%.0f ante=%.0f",
                             sb, bb, ante)
                return (sb, bb, ante)
            except (ValueError, IndexError):
                pass
        return None

    def _read_players(self, img: np.ndarray) -> List[PlayerState]:
        """Read all player states (stacks, bets, active status)."""
        players = []
        for seat in range(self.config.num_players):
            player = PlayerState(seat=seat)

            if seat in self.regions.player_stacks:
                stack_img = self._crop_region(img, self.regions.player_stacks[seat])
                if stack_img is not None:
                    stack = self.ocr.read_number(stack_img)
                    if stack is not None:
                        player.stack = stack
                        player.is_active = True

            if seat in self.regions.player_bets:
                bet_img = self._crop_region(img, self.regions.player_bets[seat])
                if bet_img is not None:
                    bet = self.ocr.read_number(bet_img)
                    if bet is not None:
                        player.current_bet = bet

            players.append(player)

        return players

    def _find_dealer(self, img: np.ndarray) -> int:
        """Find which seat has the dealer button."""
        if not cv2:
            return self._last_state.dealer_seat if self._last_state else 0

        best_seat = -1
        best_score = 0

        for seat, region in self.regions.dealer_positions.items():
            btn_img = self._crop_region(img, region)
            if btn_img is None:
                continue

            hsv = cv2.cvtColor(btn_img, cv2.COLOR_BGR2HSV)

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

        if self._last_state and self._last_state.dealer_seat >= 0:
            return self._last_state.dealer_seat
        return 0

    def _read_actions(self, img: np.ndarray) -> dict:
        """Read available action buttons."""
        action_img = self._crop_region(img, self.regions.action_buttons)
        if action_img is not None:
            return self.ocr.detect_action_buttons(action_img)
        return {}

    # ------------------------------------------------------------------
    # Debug tools
    # ------------------------------------------------------------------

    def save_debug_screenshot(self, path: str = "debug_regions.png") -> Optional[str]:
        """Save a debug screenshot with all regions drawn as colored rectangles."""
        if cv2 is None:
            logger.error("cv2 not available for debug screenshot")
            return None

        table_img = self.capture.capture_full_table()
        if table_img is None:
            logger.error("Cannot capture table for debug screenshot")
            return None

        h_img, w_img = table_img.shape[:2]
        debug_img = table_img.copy()

        # Draw all configured regions
        region_defs = [
            ("hero_card1", self.regions.hero_card1, (0, 255, 0)),
            ("hero_card2", self.regions.hero_card2, (0, 255, 0)),
            ("board1", self.regions.board_card1, (255, 255, 0)),
            ("board2", self.regions.board_card2, (255, 255, 0)),
            ("board3", self.regions.board_card3, (255, 255, 0)),
            ("board4", self.regions.board_card4, (255, 255, 0)),
            ("board5", self.regions.board_card5, (255, 255, 0)),
            ("pot", self.regions.pot_area, (0, 165, 255)),
            ("blinds", self.regions.blind_info, (255, 0, 255)),
            ("actions", self.regions.action_buttons, (0, 255, 255)),
        ]
        for seat, region in self.regions.player_stacks.items():
            region_defs.append((f"stack{seat}", region, (255, 128, 0)))
        for seat, region in self.regions.player_bets.items():
            region_defs.append((f"bet{seat}", region, (128, 0, 255)))
        for seat, region in self.regions.dealer_positions.items():
            region_defs.append((f"D{seat}", region, (0, 128, 255)))

        for label, region, color in region_defs:
            x, y, w, h = self._scale_region(region, w_img, h_img)
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(debug_img, label, (x, y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Also draw the dynamic search areas
        # Hero search area
        hsx = int(w_img * 0.20)
        hsy = int(h_img * 0.45)
        hsw = int(w_img * 0.60)
        hsh = int(h_img * 0.30)
        cv2.rectangle(debug_img, (hsx, hsy), (hsx + hsw, hsy + hsh), (0, 200, 200), 1)
        cv2.putText(debug_img, "hero_search", (hsx, hsy - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 200), 1)

        # Board search area
        bsx = int(w_img * 0.20)
        bsy = int(h_img * 0.25)
        bsw = int(w_img * 0.60)
        bsh = int(h_img * 0.25)
        cv2.rectangle(debug_img, (bsx, bsy), (bsx + bsw, bsy + bsh), (200, 200, 0), 1)
        cv2.putText(debug_img, "board_search", (bsx, bsy - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 0), 1)

        # Also run dynamic card finding and draw found rectangles
        hero_rects = CardDetector.find_card_rectangles(
            table_img, search_region=(hsx, hsy, hsw, hsh),
            min_card_w=max(12, int(w_img * 0.02)),
            max_card_w=max(40, int(w_img * 0.10)),
            min_card_h=max(16, int(h_img * 0.04)),
            max_card_h=max(60, int(h_img * 0.20)),
        )
        for i, (rx, ry, rw, rh) in enumerate(hero_rects):
            cv2.rectangle(debug_img, (rx, ry), (rx + rw, ry + rh), (255, 0, 255), 2)
            cv2.putText(debug_img, f"found_hero{i}", (rx, ry - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)

        board_rects = CardDetector.find_card_rectangles(
            table_img, search_region=(bsx, bsy, bsw, bsh),
            min_card_w=max(12, int(w_img * 0.02)),
            max_card_w=max(40, int(w_img * 0.10)),
            min_card_h=max(16, int(h_img * 0.04)),
            max_card_h=max(60, int(h_img * 0.20)),
        )
        for i, (rx, ry, rw, rh) in enumerate(board_rects):
            cv2.rectangle(debug_img, (rx, ry), (rx + rw, ry + rh), (255, 128, 255), 2)
            cv2.putText(debug_img, f"found_board{i}", (rx, ry - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 128, 255), 1)

        cv2.imwrite(path, debug_img)
        logger.info("Debug screenshot saved to %s (%dx%d) "
                     "[hero_rects=%d, board_rects=%d]",
                     path, w_img, h_img, len(hero_rects), len(board_rects))
        return path

    def _diagnose_hero_region(self, img: np.ndarray) -> str:
        """Return a short diagnostic string about the hero card region contents."""
        try:
            if cv2 is None:
                return "cv2 unavailable"
            h_img, w_img = img.shape[:2]
            region = self.regions.hero_card1
            x, y, w, h = self._scale_region(region, w_img, h_img)
            card_img = self._crop_region(img, region)
            if card_img is None:
                return f"region ({x},{y},{w},{h}) out of bounds"
            hsv = cv2.cvtColor(card_img, cv2.COLOR_BGR2HSV)
            avg_h = float(np.mean(hsv[:, :, 0]))
            avg_s = float(np.mean(hsv[:, :, 1]))
            avg_v = float(np.mean(hsv[:, :, 2]))
            white_mask = cv2.inRange(hsv, np.array([0, 0, 130]),
                                     np.array([180, 80, 255]))
            green_mask = cv2.inRange(hsv, np.array([30, 40, 40]),
                                     np.array([90, 255, 200]))
            blue_mask = cv2.inRange(hsv, np.array([90, 30, 20]),
                                    np.array([140, 255, 200]))
            white_pct = np.sum(white_mask > 0) / white_mask.size * 100
            green_pct = np.sum(green_mask > 0) / green_mask.size * 100
            blue_pct = np.sum(blue_mask > 0) / blue_mask.size * 100
            return (f"region ({x},{y},{w},{h}) "
                    f"avgHSV=({avg_h:.0f},{avg_s:.0f},{avg_v:.0f}) "
                    f"white={white_pct:.0f}% green={green_pct:.0f}% blue={blue_pct:.0f}%")
        except Exception as e:
            return f"diag error: {e}"

    def _save_hero_debug_images(self, img: np.ndarray, attempt: int):
        """Save debug images showing what the hero card search area contains."""
        try:
            if cv2 is None:
                return
            debug_dir = os.path.join(os.getcwd(), "debug")
            os.makedirs(debug_dir, exist_ok=True)

            h_img, w_img = img.shape[:2]

            # Save the search area
            sx1 = int(w_img * 0.20)
            sx2 = int(w_img * 0.80)
            sy1 = int(h_img * 0.45)
            sy2 = int(h_img * 0.75)
            search_crop = img[sy1:sy2, sx1:sx2].copy()

            # Draw fixed-region rectangles
            for i, region in enumerate([self.regions.hero_card1,
                                         self.regions.hero_card2]):
                x, y, w, h = self._scale_region(region, w_img, h_img)
                rx, ry = x - sx1, y - sy1
                cv2.rectangle(search_crop, (rx, ry), (rx + w, ry + h),
                              (0, 255, 0), 2)
                cv2.putText(search_crop, f"fixed{i+1}", (rx, ry - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            cv2.imwrite(os.path.join(debug_dir, "hero_search.png"), search_crop)

            # Save fixed-region crops
            for i, region in enumerate([self.regions.hero_card1,
                                         self.regions.hero_card2]):
                crop = self._crop_region(img, region)
                if crop is not None:
                    cv2.imwrite(os.path.join(debug_dir,
                                f"hero_card{i+1}_fixed.png"), crop)

            # Save full table with annotations
            self.save_debug_screenshot(
                os.path.join(debug_dir, f"debug_attempt_{attempt}.png")
            )

            logger.info("Hero debug images saved to %s/", debug_dir)
        except Exception as e:
            logger.debug("Failed to save hero debug images: %s", e)

    # ------------------------------------------------------------------
    # Region utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _scale_region(region: Tuple[int, int, int, int],
                      img_w: int, img_h: int) -> Tuple[int, int, int, int]:
        """Scale a region from the 800x600 baseline to the actual image size."""
        x, y, w, h = region
        sx = img_w / BASELINE_WIDTH
        sy = img_h / BASELINE_HEIGHT
        return (int(x * sx), int(y * sy), int(w * sx), int(h * sy))

    @staticmethod
    def _crop_region(img: np.ndarray, region: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """Crop a region from the full table image, scaling from baseline 800x600."""
        h_img, w_img = img.shape[:2]
        x, y, w, h = TableStateReader._scale_region(region, w_img, h_img)
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
        """Create a GameState from manual input."""
        state = GameState(timestamp=time.time())

        if hero_cards:
            from .card_detector import parse_hand
            try:
                c1, c2 = parse_hand(hero_cards)
                state.hero_cards = [c1, c2]
            except ValueError:
                pass

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
