"""
Push/Fold engine for short-stacked tournament play.
Based on Nash equilibrium calculations and ICM-adjusted ranges.
"""

from typing import List, Optional, Tuple
from .ranges import RangeManager


class PushFoldEngine:
    """Determines push/fold decisions for short-stacked play."""

    def __init__(self):
        self.ranges = RangeManager()

    def should_push(self, hand: str, position: str, stack_bb: float,
                    num_players: int = 6, has_antes: bool = False,
                    icm_factor: float = 1.0) -> Tuple[bool, float]:
        """
        Determine if hero should push all-in.

        Args:
            hand: Hand notation (e.g., "AKs", "TT")
            position: Position string (e.g., "BTN", "CO")
            stack_bb: Stack size in big blinds
            num_players: Players at the table
            has_antes: Whether antes are in play
            icm_factor: ICM adjustment (< 1.0 = tighter)

        Returns:
            Tuple of (should_push: bool, confidence: float 0-1)
        """
        pos = self.ranges._normalize_position(position)
        divisor = self.ranges.POSITION_DIVISORS.get(pos, 8.0)

        # Get Nash threshold
        threshold = self.ranges.PUSH_FOLD_HU.get(hand, 0)
        adjusted_threshold = threshold / divisor

        # Ante adjustment (wider with antes)
        if has_antes:
            adjusted_threshold *= 1.25

        # ICM adjustment
        adjusted_threshold *= icm_factor

        should_push = stack_bb <= adjusted_threshold

        # Calculate confidence (how clearly this is a push)
        if adjusted_threshold <= 0:
            confidence = 0.0
        elif stack_bb <= adjusted_threshold * 0.5:
            confidence = 1.0  # Very clear push
        elif stack_bb <= adjusted_threshold:
            # Linear scale from 0.5 to 1.0
            confidence = 0.5 + 0.5 * (1 - stack_bb / adjusted_threshold)
        elif stack_bb <= adjusted_threshold * 1.5:
            # Marginal - not quite a push
            confidence = 0.3 * (1 - (stack_bb - adjusted_threshold) /
                                (adjusted_threshold * 0.5))
        else:
            confidence = 0.0

        return should_push, max(0.0, min(1.0, confidence))

    def should_call_shove(self, hand: str, shover_stack_bb: float,
                          pot_bb: float = 0, num_behind: int = 0,
                          icm_factor: float = 1.0) -> Tuple[bool, float]:
        """
        Determine if hero should call an opponent's all-in.

        Args:
            hand: Hand notation
            shover_stack_bb: All-in amount in big blinds
            pot_bb: Dead money in pot (blinds + antes)
            num_behind: Number of players yet to act behind
            icm_factor: ICM adjustment

        Returns:
            Tuple of (should_call: bool, confidence: float 0-1)
        """
        threshold = self.ranges.CALL_VS_SHOVE.get(hand, 0)

        # Adjust for players behind
        if num_behind > 0:
            threshold /= (1 + num_behind * 0.5)

        # ICM adjustment
        threshold *= icm_factor

        # Pot odds bonus (more dead money = looser call)
        if pot_bb > 0 and shover_stack_bb > 0:
            pot_odds_bonus = pot_bb / shover_stack_bb
            threshold *= (1 + pot_odds_bonus * 0.1)

        should_call = shover_stack_bb <= threshold

        if threshold <= 0:
            confidence = 0.0
        elif shover_stack_bb <= threshold * 0.6:
            confidence = 1.0
        elif shover_stack_bb <= threshold:
            confidence = 0.5 + 0.5 * (1 - shover_stack_bb / threshold)
        else:
            confidence = 0.0

        return should_call, max(0.0, min(1.0, confidence))

    def get_pushing_range(self, position: str, stack_bb: float,
                          has_antes: bool = False,
                          icm_factor: float = 1.0) -> List[str]:
        """
        Get the full list of hands that should be pushed from a position.

        Returns:
            List of hand notations that are pushes.
        """
        pushes = []
        for hand, threshold in self.ranges.PUSH_FOLD_HU.items():
            should, _ = self.should_push(hand, position, stack_bb,
                                         has_antes=has_antes,
                                         icm_factor=icm_factor)
            if should:
                pushes.append(hand)
        return pushes

    def get_push_fold_summary(self, position: str, stack_bb: float,
                              has_antes: bool = False,
                              icm_factor: float = 1.0) -> dict:
        """
        Get a summary of push/fold strategy for current situation.

        Returns:
            Dict with push range, percentage, and zone info.
        """
        pushing_hands = self.get_pushing_range(position, stack_bb,
                                               has_antes, icm_factor)

        # Calculate approximate range percentage
        total_combos = 0
        for h in pushing_hands:
            if len(h) == 2:  # Pair
                total_combos += 6
            elif h.endswith('s'):
                total_combos += 4
            else:
                total_combos += 12
        range_pct = (total_combos / 1326) * 100

        # M-zone classification
        if stack_bb > 20:
            zone = "green"
            zone_desc = "Full play - push/fold not recommended"
        elif stack_bb > 10:
            zone = "yellow"
            zone_desc = "Raise or fold. Push with tight range."
        elif stack_bb > 6:
            zone = "orange"
            zone_desc = "Push/fold zone. Shove or fold."
        elif stack_bb > 1:
            zone = "red"
            zone_desc = "Critical! Push any playable hand."
        else:
            zone = "dead"
            zone_desc = "Desperate. Push any two cards."

        return {
            "pushing_hands": pushing_hands,
            "range_pct": range_pct,
            "total_combos": total_combos,
            "zone": zone,
            "zone_desc": zone_desc,
            "stack_bb": stack_bb,
            "position": position,
        }
