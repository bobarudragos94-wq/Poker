"""
Configuration for the Poker GTO Assistant.
Screen regions are defined as (x, y, width, height) relative to the PokerStars window.
Users should calibrate these for their specific screen resolution and PokerStars theme.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional
import json
import os

# Region type: (x, y, width, height) relative to table window
Region = Tuple[int, int, int, int]


# Baseline resolution the default regions are calibrated for
BASELINE_WIDTH = 800
BASELINE_HEIGHT = 600


@dataclass
class TableRegions:
    """Screen regions for a 6-max PokerStars table (default 800x600 layout)."""

    # Hero's hole cards (two cards at the bottom center)
    hero_card1: Region = (310, 405, 50, 70)
    hero_card2: Region = (360, 405, 50, 70)

    # Community cards (flop, turn, river - centered)
    board_card1: Region = (230, 230, 50, 70)
    board_card2: Region = (285, 230, 50, 70)
    board_card3: Region = (340, 230, 50, 70)
    board_card4: Region = (395, 230, 50, 70)
    board_card5: Region = (450, 230, 50, 70)

    # Pot size (center of the table above board cards)
    pot_area: Region = (320, 195, 160, 25)

    # Hero stack (below hero's cards)
    hero_stack: Region = (320, 480, 120, 20)

    # Blind level / tournament info
    blind_info: Region = (10, 5, 200, 20)

    # Dealer button position (6 possible locations for 6-max)
    # These are approximate center points for the dealer chip
    dealer_positions: Dict[int, Region] = field(default_factory=lambda: {
        0: (430, 360, 30, 30),   # Seat 0 (hero / bottom center)
        1: (175, 360, 30, 30),   # Seat 1 (bottom left)
        2: (100, 220, 30, 30),   # Seat 2 (top left)
        3: (265, 115, 30, 30),   # Seat 3 (top center-left)
        4: (505, 115, 30, 30),   # Seat 4 (top center-right)
        5: (660, 220, 30, 30),   # Seat 5 (top right)
    })

    # Player stacks (6-max seats, hero is seat 0)
    player_stacks: Dict[int, Region] = field(default_factory=lambda: {
        0: (320, 480, 120, 20),
        1: (85, 420, 120, 20),
        2: (30, 250, 120, 20),
        3: (200, 120, 120, 20),
        4: (470, 120, 120, 20),
        5: (620, 250, 120, 20),
    })

    # Action buttons area (fold, check/call, raise at bottom)
    action_buttons: Region = (450, 530, 340, 50)

    # Bet amounts for each player
    player_bets: Dict[int, Region] = field(default_factory=lambda: {
        0: (350, 375, 100, 20),
        1: (195, 340, 100, 20),
        2: (145, 245, 100, 20),
        3: (270, 165, 100, 20),
        4: (450, 165, 100, 20),
        5: (580, 245, 100, 20),
    })

    # Player name/status areas (to detect who is active / has cards)
    player_active: Dict[int, Region] = field(default_factory=lambda: {
        0: (320, 460, 120, 20),
        1: (85, 400, 120, 20),
        2: (30, 230, 120, 20),
        3: (200, 100, 120, 20),
        4: (470, 100, 120, 20),
        5: (620, 230, 120, 20),
    })


@dataclass
class AppConfig:
    """Main application configuration."""

    # Screen capture settings
    capture_interval_ms: int = 500          # How often to capture (milliseconds)
    pokerstars_window_title: str = "PokerStars"

    # Table layout
    table_type: str = "6max"                # "6max", "9max", or "headsup"
    regions: TableRegions = field(default_factory=TableRegions)

    # OCR settings
    ocr_language: str = "eng"
    ocr_psm: int = 7                        # Page segmentation mode (single line)
    ocr_whitelist_numbers: str = "0123456789,.$kKmMBb "
    ocr_whitelist_cards: str = "23456789TJQKA♠♥♦♣shdc"

    # Card detection
    card_match_threshold: float = 0.75      # Template match confidence threshold
    use_template_matching: bool = True       # True = template match, False = OCR only

    # GTO engine settings
    tournament_mode: bool = True            # Tournament (ICM) vs Cash game
    hero_seat: int = 0                      # Hero's seat index
    num_players: int = 6                    # Table size

    # Display settings
    overlay_opacity: float = 0.85
    overlay_position: Tuple[int, int] = (50, 50)
    overlay_size: Tuple[int, int] = (350, 500)
    font_size: int = 12
    show_ranges: bool = True
    show_equity: bool = True
    show_ev: bool = True

    # Colors
    color_fold: str = "#e74c3c"
    color_call: str = "#f39c12"
    color_raise: str = "#2ecc71"
    color_allin: str = "#9b59b6"

    def save(self, path: str = "config.json"):
        """Save config to JSON file."""
        data = {
            "capture_interval_ms": self.capture_interval_ms,
            "pokerstars_window_title": self.pokerstars_window_title,
            "table_type": self.table_type,
            "tournament_mode": self.tournament_mode,
            "hero_seat": self.hero_seat,
            "num_players": self.num_players,
            "overlay_opacity": self.overlay_opacity,
            "overlay_position": list(self.overlay_position),
            "overlay_size": list(self.overlay_size),
            "font_size": self.font_size,
            "show_ranges": self.show_ranges,
            "show_equity": self.show_equity,
            "show_ev": self.show_ev,
            "card_match_threshold": self.card_match_threshold,
            "capture_interval_ms": self.capture_interval_ms,
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str = "config.json") -> "AppConfig":
        """Load config from JSON file."""
        config = cls()
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            for key, value in data.items():
                if hasattr(config, key):
                    if key == "overlay_position":
                        value = tuple(value)
                    elif key == "overlay_size":
                        value = tuple(value)
                    setattr(config, key, value)
        return config


# Position names for different table sizes
POSITIONS = {
    "6max": {
        0: ["UTG", "MP", "CO", "BTN", "SB", "BB"],
    },
    "9max": {
        0: ["UTG", "UTG+1", "UTG+2", "MP", "MP+1", "HJ", "CO", "BTN", "SB", "BB"],
    },
    "headsup": {
        0: ["BTN/SB", "BB"],
    },
}


def get_position_name(seat: int, dealer_seat: int, num_players: int,
                      table_type: str = "6max") -> str:
    """Get position name for a given seat relative to the dealer."""
    if table_type == "6max":
        positions = ["BTN", "SB", "BB", "UTG", "MP", "CO"]
    elif table_type == "9max":
        positions = ["BTN", "SB", "BB", "UTG", "UTG+1", "UTG+2",
                     "MP", "HJ", "CO"]
    else:
        positions = ["BTN/SB", "BB"]

    offset = (seat - dealer_seat) % num_players
    if offset < len(positions):
        return positions[offset]
    return f"Seat{seat}"
