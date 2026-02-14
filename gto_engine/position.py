"""
Position utilities for tournament poker.
Handles position naming, ordering, and positional advantages.
"""

from typing import List, Optional


class PositionManager:
    """Manages position-related calculations and utilities."""

    POSITIONS_6MAX = ["UTG", "MP", "CO", "BTN", "SB", "BB"]
    POSITIONS_9MAX = ["UTG", "UTG+1", "UTG+2", "MP", "HJ", "CO", "BTN", "SB", "BB"]
    POSITIONS_HU = ["BTN/SB", "BB"]

    # Position advantage ranking (higher = better position, more info)
    POSITION_ADVANTAGE = {
        "BTN": 1.0,
        "CO": 0.85,
        "MP": 0.65,
        "HJ": 0.65,
        "UTG+2": 0.55,
        "UTG+1": 0.50,
        "UTG": 0.45,
        "SB": 0.30,   # Worst position (first to act postflop)
        "BB": 0.40,    # Better than SB (last to act preflop, gets discount)
        "BTN/SB": 0.60,  # HU button
    }

    # Opening sizing recommendations by position
    OPEN_SIZING = {
        "UTG": 2.0,   # Tighter sizing, less positional advantage
        "MP": 2.0,
        "CO": 2.3,
        "BTN": 2.5,
        "SB": 3.0,    # Bigger from SB (out of position postflop)
    }

    @staticmethod
    def get_positions(num_players: int) -> List[str]:
        """Get position names for a given table size."""
        if num_players <= 2:
            return PositionManager.POSITIONS_HU[:num_players]
        elif num_players <= 6:
            return PositionManager.POSITIONS_6MAX[-num_players:]
        else:
            return PositionManager.POSITIONS_9MAX[-num_players:]

    @staticmethod
    def seat_to_position(seat: int, dealer_seat: int,
                         num_players: int) -> str:
        """Convert a seat number to a position name."""
        positions = PositionManager.get_positions(num_players)
        # Dealer is BTN, then SB, BB, etc.
        btn_idx = positions.index("BTN") if "BTN" in positions else 0
        offset = (seat - dealer_seat) % num_players
        # Map offset to position
        pos_idx = (btn_idx + offset) % len(positions)
        return positions[pos_idx] if pos_idx < len(positions) else f"Seat{seat}"

    @staticmethod
    def is_in_position(hero_pos: str, villain_pos: str) -> bool:
        """
        Determine if hero acts after villain postflop.
        Returns True if hero has positional advantage.
        """
        hero_adv = PositionManager.POSITION_ADVANTAGE.get(hero_pos, 0.5)
        villain_adv = PositionManager.POSITION_ADVANTAGE.get(villain_pos, 0.5)
        return hero_adv > villain_adv

    @staticmethod
    def get_open_sizing(position: str, stack_bb: float) -> float:
        """
        Get recommended open-raise sizing in big blinds.
        Adjusts for stack depth.
        """
        base = PositionManager.OPEN_SIZING.get(position, 2.2)

        # Short-stacked: use smaller sizing or just jam
        if stack_bb < 12:
            return stack_bb  # Just jam
        elif stack_bb < 20:
            return min(base, 2.0)  # Smaller opens
        elif stack_bb < 30:
            return base
        else:
            return base + 0.2  # Slightly larger deep

    @staticmethod
    def get_3bet_sizing(position: str, open_size: float,
                        is_in_position: bool, stack_bb: float) -> float:
        """Get recommended 3-bet sizing."""
        if stack_bb < 25:
            # Short enough to just jam as a 3-bet
            return stack_bb

        if is_in_position:
            multiplier = 2.8
        else:
            multiplier = 3.5

        size = open_size * multiplier

        # Don't make 3-bet more than ~40% of stack (just jam instead)
        if size > stack_bb * 0.4:
            return stack_bb

        return size

    @staticmethod
    def is_early_position(position: str) -> bool:
        """Check if position is early (tighter ranges needed)."""
        return position in ("UTG", "UTG+1", "UTG+2")

    @staticmethod
    def is_late_position(position: str) -> bool:
        """Check if position is late (wider ranges allowed)."""
        return position in ("CO", "BTN")

    @staticmethod
    def is_blind(position: str) -> bool:
        """Check if position is a blind."""
        return position in ("SB", "BB")
