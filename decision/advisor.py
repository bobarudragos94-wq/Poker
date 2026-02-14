"""
Main GTO Advisor - orchestrates preflop and postflop decisions.
"""

from dataclasses import dataclass, field
from typing import Optional, List

from screen_reader.table_state import GameState
from gto_engine.icm import ICMCalculator
from gto_engine.equity import EquityCalculator
from .preflop_advisor import PreflopAdvisor, PreflopDecision
from .postflop_advisor import PostflopAdvisor, PostflopDecision


@dataclass
class Decision:
    """Unified decision output."""
    action: str                    # "FOLD", "CHECK", "CALL", "RAISE", "BET", "ALL-IN"
    confidence: float              # 0.0 - 1.0
    reasoning: str                 # Main explanation
    details: List[str] = field(default_factory=list)  # Additional detail lines
    sizing_bb: float = 0.0        # Recommended size in BB
    sizing_pct: float = 0.0       # Recommended size as pot %
    equity: float = 0.0           # Estimated equity
    hand_name: str = ""           # Hand description
    position: str = ""
    stack_bb: float = 0.0
    m_ratio: float = 0.0
    street: str = ""
    alternative: str = ""         # Alternative play
    icm_note: str = ""            # ICM-specific note

    @property
    def color(self) -> str:
        """Get display color for the action."""
        colors = {
            "FOLD": "#e74c3c",
            "CHECK": "#3498db",
            "CALL": "#f39c12",
            "RAISE": "#2ecc71",
            "BET": "#2ecc71",
            "ALL-IN": "#9b59b6",
            "WAIT": "#95a5a6",
        }
        return colors.get(self.action, "#ffffff")

    @property
    def confidence_str(self) -> str:
        """Human-readable confidence level."""
        if self.confidence >= 0.85:
            return "Very High"
        elif self.confidence >= 0.70:
            return "High"
        elif self.confidence >= 0.55:
            return "Medium"
        elif self.confidence >= 0.40:
            return "Low"
        else:
            return "Very Low"


class GTOAdvisor:
    """
    Main advisor that combines all components to produce recommendations.
    """

    def __init__(self, tournament_mode: bool = True):
        self.preflop = PreflopAdvisor()
        self.postflop = PostflopAdvisor()
        self.icm = ICMCalculator()
        self.equity_calc = EquityCalculator()
        self.tournament_mode = tournament_mode
        self._last_decision: Optional[Decision] = None

    def analyze(self, state: GameState,
                payouts: Optional[List[float]] = None,
                all_stacks: Optional[List[float]] = None) -> Decision:
        """
        Analyze the current game state and produce a recommendation.

        Args:
            state: Current table state from screen reader.
            payouts: Tournament payout structure (for ICM).
            all_stacks: All player chip stacks (for ICM).

        Returns:
            Decision object with recommendation.
        """
        if not state.hero_cards:
            return Decision(
                action="WAIT",
                confidence=0.0,
                reasoning="No cards detected. Make sure PokerStars table is visible.",
                street=state.street,
                position=state.hero_position,
                stack_bb=state.hero_stack_bb,
                m_ratio=state.m_ratio,
            )

        # Calculate ICM factor if in tournament mode
        icm_factor = 1.0
        icm_note = ""
        if self.tournament_mode and payouts and all_stacks:
            hero_idx = state.hero_seat
            icm_factor = self.icm.icm_adjustment_factor(
                state.hero_stack, all_stacks, payouts, hero_idx
            )
            bf = self.icm.bubble_factor(
                state.hero_stack, all_stacks, payouts, hero_idx
            )
            if bf > 1.5:
                icm_note = f"ICM pressure (bubble factor {bf:.1f}x) - play tighter!"
            elif bf > 1.2:
                icm_note = f"Mild ICM pressure (bubble factor {bf:.1f}x)"

        # Route to preflop or postflop advisor
        if state.is_preflop:
            decision = self._preflop_decision(state, icm_factor, icm_note)
        else:
            decision = self._postflop_decision(state, icm_note)

        self._last_decision = decision
        return decision

    def _preflop_decision(self, state: GameState, icm_factor: float,
                          icm_note: str) -> Decision:
        """Generate preflop decision."""
        result = self.preflop.advise(state, icm_factor=icm_factor)

        details = []
        details.append(f"Hand: {state.hand_notation} | Position: {state.hero_position}")
        details.append(f"Stack: {state.hero_stack_bb:.1f}BB | M-ratio: {state.m_ratio:.1f}")

        if state.hero_stack_bb <= 15:
            details.append(f"SHORT STACK - Push/fold mode")
        if state.active_players:
            details.append(f"Players in hand: {state.active_players}")

        if result.range_pct > 0:
            details.append(f"Opening range from {state.hero_position}: ~{result.range_pct:.1f}%")

        return Decision(
            action=result.action,
            confidence=result.confidence,
            reasoning=result.reasoning,
            details=details,
            sizing_bb=result.sizing,
            equity=result.ev_estimate,
            hand_name=state.hand_notation,
            position=state.hero_position,
            stack_bb=state.hero_stack_bb,
            m_ratio=state.m_ratio,
            street="preflop",
            alternative=result.alternative,
            icm_note=icm_note,
        )

    def _postflop_decision(self, state: GameState, icm_note: str) -> Decision:
        """Generate postflop decision."""
        result = self.postflop.advise(state)

        details = []
        details.append(f"Hand: {result.hand_name} | Board: {self._format_board(state)}")
        details.append(f"Equity: {result.equity*100:.1f}% | SPR: {result.spr:.1f}")

        if result.outs > 0:
            details.append(f"Outs: {result.outs} | Draw equity: {result.equity*100:.1f}%")

        if result.pot_odds > 0:
            details.append(f"Pot odds: {result.pot_odds*100:.1f}%")

        details.append(f"Board texture: {result.board_texture}")

        sizing_pct = result.sizing_pct if result.sizing_pct else 0

        return Decision(
            action=result.action,
            confidence=result.confidence,
            reasoning=result.reasoning,
            details=details,
            sizing_bb=result.sizing,
            sizing_pct=sizing_pct,
            equity=result.equity,
            hand_name=result.hand_name,
            position=state.hero_position,
            stack_bb=state.hero_stack_bb,
            m_ratio=state.m_ratio,
            street=state.street,
            alternative=result.alternative if hasattr(result, 'alternative') else "",
            icm_note=icm_note,
        )

    @staticmethod
    def _format_board(state: GameState) -> str:
        """Format board cards for display."""
        if not state.board_cards:
            return "---"
        return " ".join(c.pretty for c in state.board_cards)

    @property
    def last_decision(self) -> Optional[Decision]:
        return self._last_decision
