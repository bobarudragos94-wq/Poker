"""Tests for the decision engine."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from screen_reader.table_state import GameState
from screen_reader.card_detector import Card
from decision.advisor import GTOAdvisor, Decision
from decision.preflop_advisor import PreflopAdvisor
from decision.postflop_advisor import PostflopAdvisor


class TestGTOAdvisor:
    """Integration tests for the full advisor pipeline."""

    def setup_method(self):
        self.advisor = GTOAdvisor(tournament_mode=True)

    def _make_state(self, cards, position, stack_bb, bb=100,
                    board=None, pot=0, active=6):
        """Helper to create a GameState."""
        state = GameState()
        state.hero_cards = cards
        state.hero_position = position
        state.big_blind = bb
        state.small_blind = bb / 2
        state.hero_stack = stack_bb * bb
        state.hero_seat = 0
        state.num_players = 6
        state.active_players = active
        state.pot_size = pot
        if board:
            state.board_cards = board
            state.street = state.get_street()
        return state

    def test_premium_hand_utg_raises(self):
        state = self._make_state(
            [Card('A', 's'), Card('A', 'h')], "UTG", 50
        )
        decision = self.advisor.analyze(state)
        assert decision.action in ("RAISE", "ALL-IN")
        assert decision.confidence >= 0.6

    def test_trash_hand_utg_folds(self):
        state = self._make_state(
            [Card('7', 's'), Card('2', 'h')], "UTG", 50
        )
        decision = self.advisor.analyze(state)
        assert decision.action == "FOLD"

    def test_short_stack_push(self):
        state = self._make_state(
            [Card('A', 's'), Card('T', 'h')], "BTN", 8
        )
        decision = self.advisor.analyze(state)
        assert decision.action == "ALL-IN"

    def test_short_stack_fold_trash(self):
        state = self._make_state(
            [Card('3', 's'), Card('2', 'h')], "UTG", 8
        )
        decision = self.advisor.analyze(state)
        assert decision.action == "FOLD"

    def test_btn_opens_wide(self):
        state = self._make_state(
            [Card('9', 's'), Card('8', 's')], "BTN", 40
        )
        decision = self.advisor.analyze(state)
        assert decision.action == "RAISE"

    def test_no_cards_waits(self):
        state = self._make_state([], "BTN", 40)
        decision = self.advisor.analyze(state)
        assert decision.action == "WAIT"

    def test_postflop_strong_hand(self):
        state = self._make_state(
            [Card('A', 's'), Card('A', 'h')], "BTN", 40,
            board=[Card('A', 'd'), Card('K', 'c'), Card('5', 's')],
            pot=600
        )
        decision = self.advisor.analyze(state)
        assert decision.action in ("BET", "RAISE", "ALL-IN")

    def test_decision_has_details(self):
        state = self._make_state(
            [Card('K', 's'), Card('Q', 's')], "CO", 30
        )
        decision = self.advisor.analyze(state)
        assert len(decision.details) > 0
        assert decision.position == "CO"
        assert decision.street == "preflop"

    def test_decision_color(self):
        d = Decision(action="FOLD", confidence=0.8, reasoning="test")
        assert d.color == "#e74c3c"

        d = Decision(action="RAISE", confidence=0.8, reasoning="test")
        assert d.color == "#2ecc71"


class TestPreflopAdvisor:
    """Tests for preflop-specific decisions."""

    def setup_method(self):
        self.advisor = PreflopAdvisor()

    def test_push_fold_with_low_stack(self):
        state = GameState()
        state.hero_cards = [Card('A', 'h'), Card('5', 's')]
        state.hero_position = "BTN"
        state.big_blind = 100
        state.small_blind = 50
        state.hero_stack = 600  # 6BB
        state.num_players = 6
        state.active_players = 6

        result = self.advisor.advise(state)
        assert result.action == "ALL-IN"  # A5s from BTN at 6bb should push

    def test_medium_stack_open(self):
        state = GameState()
        state.hero_cards = [Card('A', 's'), Card('K', 'h')]
        state.hero_position = "MP"
        state.big_blind = 100
        state.small_blind = 50
        state.hero_stack = 3000  # 30BB
        state.num_players = 6
        state.active_players = 6

        result = self.advisor.advise(state)
        assert result.action == "RAISE"
        assert result.sizing > 0


class TestPostflopAdvisor:
    """Tests for postflop decisions."""

    def setup_method(self):
        self.advisor = PostflopAdvisor()

    def test_analyze_board_texture_dry(self):
        board = [Card('A', 's'), Card('7', 'h'), Card('2', 'c')]
        texture = self.advisor._analyze_board_texture(board)
        assert "dry" in texture or "A_high" in texture

    def test_analyze_board_texture_monotone(self):
        board = [Card('9', 's'), Card('6', 's'), Card('3', 's')]
        texture = self.advisor._analyze_board_texture(board)
        assert texture == "monotone"

    def test_analyze_board_texture_paired(self):
        board = [Card('K', 's'), Card('K', 'h'), Card('5', 'c')]
        texture = self.advisor._analyze_board_texture(board)
        assert texture == "paired"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
