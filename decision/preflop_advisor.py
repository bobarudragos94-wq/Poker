"""
Preflop decision advisor using GTO ranges and tournament adjustments.
"""

from typing import Optional, List, Tuple
from dataclasses import dataclass

from screen_reader.table_state import GameState
from gto_engine.ranges import RangeManager, hand_in_range, range_percentage
from gto_engine.push_fold import PushFoldEngine
from gto_engine.equity import EquityCalculator
from gto_engine.position import PositionManager


@dataclass
class PreflopDecision:
    """Result of preflop analysis."""
    action: str             # "FOLD", "CALL", "RAISE", "ALL-IN"
    confidence: float       # 0.0 - 1.0
    reasoning: str          # Human-readable explanation
    sizing: float = 0.0     # Recommended bet size in BB
    hand_notation: str = ""
    position: str = ""
    stack_bb: float = 0.0
    in_range: bool = False
    range_pct: float = 0.0
    ev_estimate: float = 0.0
    alternative: str = ""   # Secondary recommendation


class PreflopAdvisor:
    """Provides GTO-based preflop decisions for tournament play."""

    def __init__(self):
        self.ranges = RangeManager()
        self.push_fold = PushFoldEngine()
        self.equity_calc = EquityCalculator()
        self.position_mgr = PositionManager()

    def advise(self, state: GameState, icm_factor: float = 1.0) -> PreflopDecision:
        """
        Generate a preflop recommendation based on the current game state.

        Args:
            state: Current game state from screen reader.
            icm_factor: ICM adjustment factor (< 1.0 = tighten for bubble).

        Returns:
            PreflopDecision with recommended action.
        """
        hand = state.hand_notation
        position = state.hero_position
        stack_bb = state.hero_stack_bb
        has_antes = state.ante > 0

        if not hand or not position:
            return PreflopDecision(
                action="WAIT",
                confidence=0.0,
                reasoning="Waiting for cards and position data...",
                hand_notation=hand,
                position=position,
                stack_bb=stack_bb,
            )

        # Check if this is push/fold territory
        if stack_bb <= 15:
            return self._push_fold_decision(hand, position, stack_bb,
                                            has_antes, icm_factor, state)

        # Check if facing an all-in from another player
        if state.current_bet > 0 and self._is_facing_allin(state):
            return self._facing_allin_decision(hand, position, stack_bb,
                                               state, icm_factor)

        # Check if facing a raise (3bet situation)
        if state.current_bet > state.big_blind * 3:
            return self._facing_raise_decision(hand, position, stack_bb,
                                               state, icm_factor)

        # Standard open-raise decision (RFI)
        return self._open_raise_decision(hand, position, stack_bb,
                                         has_antes, icm_factor, state)

    def _push_fold_decision(self, hand: str, position: str, stack_bb: float,
                            has_antes: bool, icm_factor: float,
                            state: GameState) -> PreflopDecision:
        """Decision for short-stacked push/fold play."""
        should_push, confidence = self.push_fold.should_push(
            hand, position, stack_bb, has_antes=has_antes, icm_factor=icm_factor
        )

        summary = self.push_fold.get_push_fold_summary(
            position, stack_bb, has_antes, icm_factor
        )

        if should_push:
            return PreflopDecision(
                action="ALL-IN",
                confidence=confidence,
                reasoning=(
                    f"PUSH {hand} from {position} at {stack_bb:.1f}BB. "
                    f"{summary['zone'].upper()} zone ({summary['zone_desc']}). "
                    f"Push range is ~{summary['range_pct']:.1f}% of hands."
                ),
                sizing=stack_bb,
                hand_notation=hand,
                position=position,
                stack_bb=stack_bb,
                in_range=True,
                range_pct=summary['range_pct'],
            )
        else:
            return PreflopDecision(
                action="FOLD",
                confidence=confidence,
                reasoning=(
                    f"FOLD {hand} from {position} at {stack_bb:.1f}BB. "
                    f"Not in push range ({summary['zone'].upper()} zone). "
                    f"Push range is ~{summary['range_pct']:.1f}% from here."
                ),
                hand_notation=hand,
                position=position,
                stack_bb=stack_bb,
                in_range=False,
                range_pct=summary['range_pct'],
            )

    def _open_raise_decision(self, hand: str, position: str, stack_bb: float,
                             has_antes: bool, icm_factor: float,
                             state: GameState) -> PreflopDecision:
        """Decision for opening the pot (raise first in)."""
        in_range = self.ranges.hand_in_open_range(hand, position, stack_bb)
        open_range = self.ranges.get_open_range(position, stack_bb)
        range_pct = range_percentage(open_range) if open_range else 0

        # Get sizing
        sizing = self.position_mgr.get_open_sizing(position, stack_bb)

        # ICM tightening
        if icm_factor < 0.9:
            # Tighten range for ICM spots
            if not in_range:
                return PreflopDecision(
                    action="FOLD",
                    confidence=0.8,
                    reasoning=(
                        f"FOLD {hand} from {position}. "
                        f"Not in opening range ({range_pct:.1f}%). "
                        f"ICM pressure suggests tighter play."
                    ),
                    hand_notation=hand,
                    position=position,
                    stack_bb=stack_bb,
                    in_range=False,
                    range_pct=range_pct,
                )

        if in_range:
            # Estimate EV
            equity = self.equity_calc.preflop_hand_equity_estimate(
                hand, num_opponents=max(1, state.active_players - 1)
            )

            return PreflopDecision(
                action="RAISE",
                confidence=0.8 if equity > 0.5 else 0.6,
                reasoning=(
                    f"RAISE {hand} from {position} to {sizing:.1f}BB. "
                    f"In opening range ({range_pct:.1f}% from {position}). "
                    f"Estimated equity: {equity*100:.1f}%."
                ),
                sizing=sizing,
                hand_notation=hand,
                position=position,
                stack_bb=stack_bb,
                in_range=True,
                range_pct=range_pct,
                ev_estimate=equity,
            )
        else:
            return PreflopDecision(
                action="FOLD",
                confidence=0.7,
                reasoning=(
                    f"FOLD {hand} from {position}. "
                    f"Not in opening range ({range_pct:.1f}% from {position}). "
                    f"Hand is too weak to open from this position."
                ),
                hand_notation=hand,
                position=position,
                stack_bb=stack_bb,
                in_range=False,
                range_pct=range_pct,
            )

    def _facing_raise_decision(self, hand: str, position: str, stack_bb: float,
                               state: GameState,
                               icm_factor: float) -> PreflopDecision:
        """Decision when facing an open raise (call, 3-bet, or fold)."""
        raise_size_bb = state.current_bet / state.big_blind if state.big_blind > 0 else 0

        # Check 3-bet range
        # We need to guess opener's position based on bet
        opener_pos = self._estimate_opener_position(state)
        three_bet_range = self.ranges.get_3bet_range(position, opener_pos)
        in_3bet_range = hand_in_range(hand, three_bet_range) if three_bet_range else False

        if in_3bet_range:
            sizing = self.position_mgr.get_3bet_sizing(
                position, raise_size_bb,
                self.position_mgr.is_in_position(position, opener_pos),
                stack_bb
            )

            if sizing >= stack_bb:
                return PreflopDecision(
                    action="ALL-IN",
                    confidence=0.85,
                    reasoning=(
                        f"ALL-IN with {hand} vs {opener_pos} raise. "
                        f"3-bet would be > 40% of stack - jam is better. "
                        f"In 3-bet range vs {opener_pos}."
                    ),
                    sizing=stack_bb,
                    hand_notation=hand,
                    position=position,
                    stack_bb=stack_bb,
                    in_range=True,
                )

            return PreflopDecision(
                action="RAISE",
                confidence=0.75,
                reasoning=(
                    f"3-BET {hand} to {sizing:.1f}BB vs {opener_pos} open. "
                    f"In 3-bet range from {position} vs {opener_pos}."
                ),
                sizing=sizing,
                hand_notation=hand,
                position=position,
                stack_bb=stack_bb,
                in_range=True,
                alternative="Call is also acceptable with this hand.",
            )

        # Check calling range
        call_range = self.ranges.CALL_RANGES.get("deep", {}).get(
            f"{position}_vs_{opener_pos}", ""
        )
        in_call_range = hand_in_range(hand, call_range) if call_range else False

        if in_call_range:
            pot_odds = self.equity_calc.pot_odds(
                state.pot_size, state.call_amount
            )

            return PreflopDecision(
                action="CALL",
                confidence=0.65,
                reasoning=(
                    f"CALL {hand} vs {opener_pos} raise ({raise_size_bb:.1f}BB). "
                    f"In calling range. Pot odds: {pot_odds*100:.1f}%. "
                    f"Play postflop in position." if self.position_mgr.is_in_position(
                        position, opener_pos
                    ) else f"CALL {hand} vs {opener_pos} raise. In calling range."
                ),
                hand_notation=hand,
                position=position,
                stack_bb=stack_bb,
                in_range=True,
            )

        return PreflopDecision(
            action="FOLD",
            confidence=0.7,
            reasoning=(
                f"FOLD {hand} vs {opener_pos} raise ({raise_size_bb:.1f}BB). "
                f"Not in 3-bet or calling range from {position}."
            ),
            hand_notation=hand,
            position=position,
            stack_bb=stack_bb,
            in_range=False,
        )

    def _facing_allin_decision(self, hand: str, position: str, stack_bb: float,
                               state: GameState,
                               icm_factor: float) -> PreflopDecision:
        """Decision when facing an all-in."""
        shove_size_bb = state.current_bet / state.big_blind if state.big_blind > 0 else stack_bb

        should_call, confidence = self.push_fold.should_call_shove(
            hand, shove_size_bb, state.pot_bb,
            num_behind=0,  # Simplified
            icm_factor=icm_factor,
        )

        equity = self.equity_calc.preflop_hand_equity_estimate(hand, num_opponents=1)
        pot_odds = self.equity_calc.pot_odds(state.pot_size, state.call_amount)

        if should_call:
            return PreflopDecision(
                action="CALL",
                confidence=confidence,
                reasoning=(
                    f"CALL all-in with {hand}. "
                    f"Equity ~{equity*100:.1f}% vs shoving range. "
                    f"Pot odds: {pot_odds*100:.1f}%. "
                    f"{'ICM favors calling.' if icm_factor >= 1.0 else 'ICM pressure - borderline.'}"
                ),
                hand_notation=hand,
                position=position,
                stack_bb=stack_bb,
                in_range=True,
                ev_estimate=equity,
            )
        else:
            return PreflopDecision(
                action="FOLD",
                confidence=confidence,
                reasoning=(
                    f"FOLD {hand} vs all-in ({shove_size_bb:.1f}BB). "
                    f"Equity ~{equity*100:.1f}% not enough. "
                    f"Need {pot_odds*100:.1f}% to call."
                ),
                hand_notation=hand,
                position=position,
                stack_bb=stack_bb,
                in_range=False,
                ev_estimate=equity,
            )

    def _is_facing_allin(self, state: GameState) -> bool:
        """Check if we're facing an all-in bet."""
        for player in state.players:
            if not player.is_hero and player.current_bet > 0:
                if player.current_bet >= player.stack * 0.95:
                    return True
        return False

    def _estimate_opener_position(self, state: GameState) -> str:
        """Estimate the opener's position based on who has bet."""
        for player in state.players:
            if not player.is_hero and player.current_bet > state.big_blind:
                # This player raised - estimate their position
                if state.dealer_seat >= 0:
                    return PositionManager.seat_to_position(
                        player.seat, state.dealer_seat, state.num_players
                    )
        return "CO"  # Default assumption
