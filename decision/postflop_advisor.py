"""
Postflop decision advisor using GTO principles.
Analyzes board texture, hand strength, draws, and optimal actions.
"""

from typing import Optional, List, Tuple
from dataclasses import dataclass

from screen_reader.table_state import GameState
from screen_reader.card_detector import Card
from gto_engine.hand_evaluator import HandEvaluator
from gto_engine.equity import EquityCalculator


@dataclass
class PostflopDecision:
    """Result of postflop analysis."""
    action: str                 # "FOLD", "CHECK", "CALL", "BET", "RAISE", "ALL-IN"
    confidence: float           # 0.0 - 1.0
    reasoning: str              # Explanation
    sizing: float = 0.0        # Recommended bet/raise size in BB
    sizing_pct: float = 0.0    # As percentage of pot
    hand_strength: str = ""     # "monster", "strong", "medium", "weak", "draw"
    hand_name: str = ""         # e.g., "Two Pair", "Flush Draw"
    equity: float = 0.0        # Estimated equity
    outs: int = 0              # Number of outs (for draws)
    pot_odds: float = 0.0      # Pot odds needed
    spr: float = 0.0           # Stack to pot ratio
    board_texture: str = ""    # "dry", "wet", "monotone", "paired"


# C-bet sizing and frequency guidelines by board texture
CBET_GUIDELINES = {
    'A_high_dry':    {'freq': 0.80, 'sizing': 0.33, 'desc': 'A-high dry board'},
    'A_high_wet':    {'freq': 0.65, 'sizing': 0.50, 'desc': 'A-high wet board'},
    'K_high_dry':    {'freq': 0.70, 'sizing': 0.33, 'desc': 'K-high dry board'},
    'K_high_wet':    {'freq': 0.55, 'sizing': 0.50, 'desc': 'K-high wet board'},
    'Q_high_dry':    {'freq': 0.65, 'sizing': 0.33, 'desc': 'Q-high dry board'},
    'mid_dry':       {'freq': 0.55, 'sizing': 0.33, 'desc': 'Mid dry board'},
    'mid_connected': {'freq': 0.35, 'sizing': 0.67, 'desc': 'Mid connected board'},
    'low_dry':       {'freq': 0.45, 'sizing': 0.33, 'desc': 'Low dry board'},
    'low_connected': {'freq': 0.27, 'sizing': 0.67, 'desc': 'Low connected board'},
    'monotone':      {'freq': 0.30, 'sizing': 0.33, 'desc': 'Monotone board'},
    'paired':        {'freq': 0.65, 'sizing': 0.33, 'desc': 'Paired board'},
}


class PostflopAdvisor:
    """Provides GTO-based postflop decisions."""

    def __init__(self):
        self.evaluator = HandEvaluator()
        self.equity_calc = EquityCalculator()

    def advise(self, state: GameState, is_aggressor: bool = True,
               num_opponents: int = 1) -> PostflopDecision:
        """
        Generate a postflop recommendation.

        Args:
            state: Current game state.
            is_aggressor: Whether hero was the preflop raiser.
            num_opponents: Number of opponents in the hand.

        Returns:
            PostflopDecision with recommended action.
        """
        if len(state.hero_cards) != 2 or len(state.board_cards) < 3:
            return PostflopDecision(
                action="WAIT",
                confidence=0.0,
                reasoning="Waiting for board cards...",
            )

        # Evaluate hand strength
        rank, kickers, hand_name = self.evaluator.evaluate(
            state.hero_cards, state.board_cards
        )
        strength = self.evaluator.hand_strength_category(rank)

        # Count outs and draws
        outs_info = self.equity_calc.count_outs(state.hero_cards, state.board_cards)
        total_outs = outs_info.get('total_estimated', 0)

        # Calculate equity
        streets_left = 2 if state.is_flop else (1 if state.is_turn else 0)
        if total_outs > 0 and streets_left > 0:
            draw_equity = self.equity_calc.quick_equity_estimate(total_outs, streets_left)
        else:
            draw_equity = 0.0

        # Monte Carlo equity (if we have time - use quick estimate for draws)
        mc_equity = self.equity_calc.monte_carlo_equity(
            state.hero_cards, state.board_cards,
            num_opponents=num_opponents, iterations=2000
        )

        # SPR
        spr = self.equity_calc.stack_to_pot_ratio(state.hero_stack, state.pot_size)

        # Board texture
        texture = self._analyze_board_texture(state.board_cards)

        # Pot odds if facing a bet
        pot_odds = 0.0
        if state.call_amount > 0:
            pot_odds = self.equity_calc.pot_odds(state.pot_size, state.call_amount)

        # Now determine the best action
        if state.call_amount > 0:
            # Facing a bet
            decision = self._facing_bet_decision(
                state, rank, hand_name, strength, mc_equity,
                total_outs, draw_equity, pot_odds, spr, texture,
                num_opponents
            )
        elif is_aggressor:
            # We have the initiative - consider c-betting
            decision = self._cbet_decision(
                state, rank, hand_name, strength, mc_equity,
                total_outs, draw_equity, spr, texture, num_opponents
            )
        else:
            # We don't have initiative - check or donk bet
            decision = self._check_or_lead_decision(
                state, rank, hand_name, strength, mc_equity,
                total_outs, draw_equity, spr, texture, num_opponents
            )

        # Fill in common fields
        decision.hand_name = hand_name
        decision.hand_strength = strength
        decision.equity = mc_equity
        decision.outs = total_outs
        decision.pot_odds = pot_odds
        decision.spr = spr
        decision.board_texture = texture

        return decision

    def _facing_bet_decision(self, state, rank, hand_name, strength,
                             equity, outs, draw_equity, pot_odds, spr,
                             texture, num_opponents) -> PostflopDecision:
        """Decision when facing a bet."""
        call_amount_bb = state.call_amount / state.big_blind if state.big_blind else 0
        pot_bb = state.pot_bb

        # Monster hand - raise for value
        if strength in ("monster", "very_strong"):
            raise_size = self._calculate_raise_size(state, 0.75)
            return PostflopDecision(
                action="RAISE",
                confidence=0.9,
                reasoning=(
                    f"RAISE with {hand_name}! Strong hand on {texture} board. "
                    f"Equity: {equity*100:.1f}%. Raise for value."
                ),
                sizing=raise_size,
                sizing_pct=0.75,
            )

        # Strong hand - call or raise
        if strength == "strong":
            if spr < 3:
                return PostflopDecision(
                    action="ALL-IN",
                    confidence=0.8,
                    reasoning=(
                        f"ALL-IN with {hand_name}. Low SPR ({spr:.1f}), "
                        f"committed with this hand. Equity: {equity*100:.1f}%."
                    ),
                    sizing=state.hero_stack_bb,
                )
            return PostflopDecision(
                action="CALL",
                confidence=0.75,
                reasoning=(
                    f"CALL with {hand_name}. Equity: {equity*100:.1f}% "
                    f"vs pot odds {pot_odds*100:.1f}%. "
                    f"Good hand, keep the pot manageable."
                ),
            )

        # Medium hand - check pot odds
        if strength == "medium":
            if equity > pot_odds + 0.05:
                return PostflopDecision(
                    action="CALL",
                    confidence=0.6,
                    reasoning=(
                        f"CALL with {hand_name}. Equity {equity*100:.1f}% > "
                        f"pot odds {pot_odds*100:.1f}%. Marginal but profitable."
                    ),
                )
            else:
                return PostflopDecision(
                    action="FOLD",
                    confidence=0.55,
                    reasoning=(
                        f"FOLD {hand_name}. Equity {equity*100:.1f}% "
                        f"doesn't justify pot odds {pot_odds*100:.1f}%."
                    ),
                )

        # Drawing hand
        if outs >= 8:  # Strong draw (flush draw, OESD)
            if draw_equity > pot_odds:
                if draw_equity > 0.35 and spr > 3:
                    raise_size = self._calculate_raise_size(state, 0.67)
                    return PostflopDecision(
                        action="RAISE",
                        confidence=0.65,
                        reasoning=(
                            f"SEMI-BLUFF RAISE with {outs} outs ({draw_equity*100:.1f}% equity). "
                            f"Strong draw + fold equity makes this profitable."
                        ),
                        sizing=raise_size,
                        sizing_pct=0.67,
                    )
                return PostflopDecision(
                    action="CALL",
                    confidence=0.7,
                    reasoning=(
                        f"CALL with {outs} outs draw. "
                        f"Equity {draw_equity*100:.1f}% > pot odds {pot_odds*100:.1f}%."
                    ),
                )
            elif outs >= 12:
                # Combo draw - call even at slight negative
                return PostflopDecision(
                    action="CALL",
                    confidence=0.55,
                    reasoning=(
                        f"CALL combo draw ({outs} outs). Implied odds justify the call."
                    ),
                )

        if outs >= 4 and outs < 8:  # Weak draw (gutshot)
            if draw_equity > pot_odds:
                return PostflopDecision(
                    action="CALL",
                    confidence=0.5,
                    reasoning=(
                        f"CALL with {outs} outs. Barely getting correct odds "
                        f"({draw_equity*100:.1f}% vs {pot_odds*100:.1f}%)."
                    ),
                )

        # Weak hand - fold
        return PostflopDecision(
            action="FOLD",
            confidence=0.7,
            reasoning=(
                f"FOLD {hand_name}. Equity {equity*100:.1f}% too low "
                f"vs bet. Need {pot_odds*100:.1f}% to continue."
            ),
        )

    def _cbet_decision(self, state, rank, hand_name, strength,
                       equity, outs, draw_equity, spr, texture,
                       num_opponents) -> PostflopDecision:
        """Decision for continuation betting (preflop aggressor)."""
        # Get c-bet guideline for this board texture
        guideline = CBET_GUIDELINES.get(texture, {'freq': 0.50, 'sizing': 0.50})
        cbet_freq = guideline['freq']
        cbet_sizing = guideline['sizing']

        # Adjust for number of opponents
        if num_opponents > 1:
            cbet_freq *= 0.7  # Less c-betting multiway
            cbet_sizing = max(cbet_sizing, 0.50)  # Larger when betting multiway

        bet_size = state.pot_size * cbet_sizing
        bet_size_bb = bet_size / state.big_blind if state.big_blind > 0 else 0

        # Strong hands - always c-bet for value
        if strength in ("monster", "very_strong", "strong"):
            return PostflopDecision(
                action="BET",
                confidence=0.85,
                reasoning=(
                    f"C-BET {cbet_sizing*100:.0f}% pot ({bet_size_bb:.1f}BB) "
                    f"with {hand_name} for value. "
                    f"Equity: {equity*100:.1f}% on {texture} board."
                ),
                sizing=bet_size_bb,
                sizing_pct=cbet_sizing,
            )

        # Medium hands - c-bet on favorable textures
        if strength == "medium":
            if cbet_freq >= 0.55:  # Favorable board
                return PostflopDecision(
                    action="BET",
                    confidence=0.65,
                    reasoning=(
                        f"C-BET {cbet_sizing*100:.0f}% pot ({bet_size_bb:.1f}BB) "
                        f"with {hand_name}. {texture} board favors our range "
                        f"(c-bet freq ~{cbet_freq*100:.0f}%)."
                    ),
                    sizing=bet_size_bb,
                    sizing_pct=cbet_sizing,
                )
            else:
                return PostflopDecision(
                    action="CHECK",
                    confidence=0.55,
                    reasoning=(
                        f"CHECK {hand_name}. {texture} board doesn't strongly "
                        f"favor c-betting (freq ~{cbet_freq*100:.0f}%). "
                        f"Consider checking to control pot."
                    ),
                )

        # Draws - semi-bluff c-bet
        if outs >= 8:
            return PostflopDecision(
                action="BET",
                confidence=0.65,
                reasoning=(
                    f"SEMI-BLUFF c-bet {cbet_sizing*100:.0f}% pot with {outs} outs. "
                    f"Draw equity {draw_equity*100:.1f}% plus fold equity."
                ),
                sizing=bet_size_bb,
                sizing_pct=cbet_sizing,
            )

        # Weak hands on favorable boards - bluff c-bet at low frequency
        if cbet_freq >= 0.55 and equity > 0.20:
            return PostflopDecision(
                action="BET",
                confidence=0.45,
                reasoning=(
                    f"C-BET bluff {cbet_sizing*100:.0f}% pot on {texture} board. "
                    f"Range advantage supports c-betting. "
                    f"Fold to any resistance."
                ),
                sizing=bet_size_bb,
                sizing_pct=cbet_sizing,
                alternative="Check is also fine - this is a marginal bluff spot.",
            )

        # Default: check
        return PostflopDecision(
            action="CHECK",
            confidence=0.6,
            reasoning=(
                f"CHECK with {hand_name}. "
                f"Board texture ({texture}) doesn't favor c-betting "
                f"with this holding."
            ),
        )

    def _check_or_lead_decision(self, state, rank, hand_name, strength,
                                equity, outs, draw_equity, spr, texture,
                                num_opponents) -> PostflopDecision:
        """Decision when we're the caller (no initiative)."""
        # Very strong - consider check-raise
        if strength in ("monster", "very_strong"):
            if spr > 2:
                raise_size = self._calculate_raise_size(state, 0.75)
                return PostflopDecision(
                    action="CHECK",
                    confidence=0.75,
                    reasoning=(
                        f"CHECK with {hand_name} to trap. "
                        f"Plan to check-raise if opponent bets. "
                        f"Equity: {equity*100:.1f}%."
                    ),
                    alternative="Check-raise if opponent bets.",
                )

        # Strong draws - check with plan to check-raise
        if outs >= 12:
            return PostflopDecision(
                action="CHECK",
                confidence=0.6,
                reasoning=(
                    f"CHECK {outs}-out draw. Plan to check-raise as semi-bluff "
                    f"if opponent bets small, or call if appropriate."
                ),
                alternative="Check-raise or call depending on bet size.",
            )

        # Default: check and evaluate
        return PostflopDecision(
            action="CHECK",
            confidence=0.6,
            reasoning=(
                f"CHECK with {hand_name} ({strength}). "
                f"No initiative - evaluate opponent's action."
            ),
        )

    def _analyze_board_texture(self, board: List[Card]) -> str:
        """Classify the board texture."""
        if not board:
            return "unknown"

        ranks = [c.rank_value for c in board]
        suits = [c.suit for c in board]
        max_rank = max(ranks)
        min_rank = min(ranks)

        # Check for monotone (all same suit)
        if len(set(suits)) == 1:
            return "monotone"

        # Check for paired board
        rank_counts = {}
        for r in ranks:
            rank_counts[r] = rank_counts.get(r, 0) + 1
        if max(rank_counts.values()) >= 2:
            return "paired"

        # Check connectivity
        sorted_ranks = sorted(ranks)
        max_gap = max(sorted_ranks[i+1] - sorted_ranks[i]
                      for i in range(len(sorted_ranks)-1))
        is_connected = max_gap <= 2 and (max_rank - min_rank) <= 4

        # Check for flush draw potential
        suit_counts = {}
        for s in suits:
            suit_counts[s] = suit_counts.get(s, 0) + 1
        has_flush_draw = max(suit_counts.values()) >= 2

        is_wet = is_connected or has_flush_draw

        # Classify by high card
        if max_rank >= 14:  # Ace high
            return "A_high_wet" if is_wet else "A_high_dry"
        elif max_rank >= 13:  # King high
            return "K_high_wet" if is_wet else "K_high_dry"
        elif max_rank >= 12:  # Queen high
            return "Q_high_dry"  # Simplified
        elif max_rank >= 9:  # Mid board
            return "mid_connected" if is_connected else "mid_dry"
        else:  # Low board
            return "low_connected" if is_connected else "low_dry"

    @staticmethod
    def _calculate_raise_size(state: GameState, pot_fraction: float) -> float:
        """Calculate raise size in BB."""
        if state.big_blind <= 0:
            return 0
        raise_amount = state.pot_size * pot_fraction
        return raise_amount / state.big_blind
