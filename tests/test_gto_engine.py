"""Tests for the GTO engine core components."""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gto_engine.ranges import RangeManager, hand_in_range, expand_range, range_percentage
from gto_engine.hand_evaluator import HandEvaluator
from gto_engine.equity import EquityCalculator
from gto_engine.push_fold import PushFoldEngine
from gto_engine.icm import ICMCalculator
from gto_engine.position import PositionManager
from screen_reader.card_detector import Card


class TestRangeManager:
    """Tests for range parsing and management."""

    def setup_method(self):
        self.rm = RangeManager()

    def test_hand_in_range_pair_plus(self):
        assert hand_in_range("AA", "TT+")
        assert hand_in_range("KK", "TT+")
        assert hand_in_range("TT", "TT+")
        assert not hand_in_range("99", "TT+")

    def test_hand_in_range_suited_plus(self):
        assert hand_in_range("AKs", "ATs+")
        assert hand_in_range("AJs", "ATs+")
        assert hand_in_range("ATs", "ATs+")
        assert not hand_in_range("A9s", "ATs+")

    def test_hand_in_range_offsuit_plus(self):
        assert hand_in_range("AKo", "AJo+")
        assert hand_in_range("AQo", "AJo+")
        assert not hand_in_range("ATo", "AJo+")

    def test_hand_in_range_pair_span(self):
        assert hand_in_range("88", "TT-77")
        assert hand_in_range("77", "TT-77")
        assert hand_in_range("TT", "TT-77")
        assert not hand_in_range("66", "TT-77")
        assert not hand_in_range("JJ", "TT-77")

    def test_hand_in_range_direct_match(self):
        assert hand_in_range("JTs", "JTs")
        assert not hand_in_range("JTo", "JTs")

    def test_hand_in_range_complex(self):
        range_str = "TT+, ATs+, KQs, AJo+"
        assert hand_in_range("AA", range_str)
        assert hand_in_range("AKs", range_str)
        assert hand_in_range("KQs", range_str)
        assert hand_in_range("AKo", range_str)
        assert not hand_in_range("99", range_str)
        assert not hand_in_range("A9s", range_str)
        assert not hand_in_range("KJs", range_str)

    def test_expand_range(self):
        hands = expand_range("AA")
        assert "AA" in hands
        assert len(hands) == 1

        hands = expand_range("TT+")
        assert "TT" in hands
        assert "AA" in hands
        assert "KK" in hands
        assert len(hands) == 5  # TT, JJ, QQ, KK, AA

    def test_range_percentage(self):
        # AA = 6/1326 = ~0.45%
        pct = range_percentage("AA")
        assert 0.4 < pct < 0.5

    def test_get_open_range_deep_utg(self):
        r = self.rm.get_open_range("UTG", 50)
        assert r != ""
        assert hand_in_range("AA", r)
        assert hand_in_range("AQo", r)

    def test_get_open_range_deep_btn(self):
        r = self.rm.get_open_range("BTN", 50)
        assert r != ""
        # BTN range should be wider than UTG
        utg_range = self.rm.get_open_range("UTG", 50)
        btn_pct = range_percentage(r)
        utg_pct = range_percentage(utg_range)
        assert btn_pct > utg_pct

    def test_hand_in_open_range(self):
        assert self.rm.hand_in_open_range("AA", "UTG", 50)
        assert self.rm.hand_in_open_range("22", "BTN", 50)
        assert not self.rm.hand_in_open_range("32o", "UTG", 50)


class TestHandEvaluator:
    """Tests for hand evaluation."""

    def setup_method(self):
        self.eval = HandEvaluator()

    def test_royal_flush(self):
        hole = [Card('A', 's'), Card('K', 's')]
        board = [Card('Q', 's'), Card('J', 's'), Card('T', 's'),
                 Card('2', 'h'), Card('3', 'c')]
        rank, _, name = self.eval.evaluate(hole, board)
        assert rank == 9
        assert name == "Royal Flush"

    def test_straight_flush(self):
        hole = [Card('9', 'h'), Card('8', 'h')]
        board = [Card('7', 'h'), Card('6', 'h'), Card('5', 'h'),
                 Card('2', 'c'), Card('3', 'd')]
        rank, _, name = self.eval.evaluate(hole, board)
        assert rank == 8

    def test_four_of_a_kind(self):
        hole = [Card('A', 's'), Card('A', 'h')]
        board = [Card('A', 'd'), Card('A', 'c'), Card('K', 's'),
                 Card('2', 'h'), Card('3', 'c')]
        rank, _, name = self.eval.evaluate(hole, board)
        assert rank == 7

    def test_full_house(self):
        hole = [Card('K', 's'), Card('K', 'h')]
        board = [Card('K', 'd'), Card('Q', 'c'), Card('Q', 's'),
                 Card('2', 'h'), Card('3', 'c')]
        rank, _, name = self.eval.evaluate(hole, board)
        assert rank == 6

    def test_flush(self):
        hole = [Card('A', 's'), Card('J', 's')]
        board = [Card('9', 's'), Card('5', 's'), Card('2', 's'),
                 Card('K', 'h'), Card('3', 'c')]
        rank, _, name = self.eval.evaluate(hole, board)
        assert rank == 5

    def test_straight(self):
        hole = [Card('8', 's'), Card('7', 'h')]
        board = [Card('6', 'd'), Card('5', 'c'), Card('4', 's'),
                 Card('K', 'h'), Card('2', 'c')]
        rank, _, name = self.eval.evaluate(hole, board)
        assert rank == 4

    def test_two_pair(self):
        hole = [Card('A', 's'), Card('K', 'h')]
        board = [Card('A', 'd'), Card('K', 'c'), Card('2', 's'),
                 Card('7', 'h'), Card('3', 'c')]
        rank, _, name = self.eval.evaluate(hole, board)
        assert rank == 2

    def test_one_pair(self):
        hole = [Card('A', 's'), Card('K', 'h')]
        board = [Card('A', 'd'), Card('9', 'c'), Card('5', 's'),
                 Card('7', 'h'), Card('3', 'c')]
        rank, _, name = self.eval.evaluate(hole, board)
        assert rank == 1

    def test_high_card(self):
        hole = [Card('A', 's'), Card('K', 'h')]
        board = [Card('9', 'd'), Card('7', 'c'), Card('5', 's'),
                 Card('3', 'h'), Card('2', 'c')]
        rank, _, name = self.eval.evaluate(hole, board)
        assert rank == 0

    def test_hand_comparison(self):
        # Full house beats flush
        result = self.eval.compare_hands((6, [13, 12]), (5, [14, 13, 9, 5, 2]))
        assert result == 1

    def test_preflop_hand_rank(self):
        # AA should rank highest
        aa = self.eval.preflop_hand_rank(Card('A', 's'), Card('A', 'h'))
        kk = self.eval.preflop_hand_rank(Card('K', 's'), Card('K', 'h'))
        assert aa > kk

        # Suited > offsuit for same ranks
        aks = self.eval.preflop_hand_rank(Card('A', 's'), Card('K', 's'))
        ako = self.eval.preflop_hand_rank(Card('A', 's'), Card('K', 'h'))
        assert aks > ako


class TestEquityCalculator:
    """Tests for equity calculations."""

    def setup_method(self):
        self.calc = EquityCalculator()

    def test_pot_odds(self):
        odds = self.calc.pot_odds(100, 50)
        assert abs(odds - 1/3) < 0.01

    def test_pot_odds_ratio(self):
        ratio = self.calc.pot_odds_ratio(100, 50)
        assert abs(ratio - 3.0) < 0.01

    def test_quick_equity_flush_draw(self):
        eq = self.calc.quick_equity_estimate(9, 2)  # Flush draw, flop
        assert 0.34 < eq < 0.38

    def test_quick_equity_gutshot(self):
        eq = self.calc.quick_equity_estimate(4, 1)  # Gutshot, turn
        assert 0.07 < eq < 0.09

    def test_expected_value_positive(self):
        ev = self.calc.expected_value(0.40, 200, 100)
        assert ev > 0  # 0.40 * 200 - 0.60 * 100 = 20

    def test_expected_value_negative(self):
        ev = self.calc.expected_value(0.20, 200, 100)
        assert ev < 0  # 0.20 * 200 - 0.80 * 100 = -40

    def test_bluff_breakeven(self):
        be = self.calc.bluff_breakeven(100, 100)
        assert abs(be - 0.5) < 0.01  # Pot-size bet needs 50% folds

    def test_mdf(self):
        mdf = self.calc.minimum_defense_frequency(100, 100)
        assert abs(mdf - 0.5) < 0.01  # Pot-size bet: defend 50%

    def test_monte_carlo_aa_vs_random(self):
        hero = [Card('A', 's'), Card('A', 'h')]
        equity = self.calc.monte_carlo_equity(hero, [], num_opponents=1, iterations=3000)
        assert equity > 0.80  # AA ~85% vs random

    def test_spr(self):
        spr = self.calc.stack_to_pot_ratio(1000, 100)
        assert spr == 10.0

    def test_m_ratio(self):
        m = self.calc.m_ratio(10000, 50, 100, ante=10, num_players=6)
        expected = 10000 / (50 + 100 + 60)  # = 47.6
        assert abs(m - expected) < 0.1


class TestPushFoldEngine:
    """Tests for push/fold decisions."""

    def setup_method(self):
        self.pf = PushFoldEngine()

    def test_aa_always_push(self):
        should, conf = self.pf.should_push("AA", "UTG", 10)
        assert should
        assert conf > 0.5

    def test_72o_fold_from_utg(self):
        should, conf = self.pf.should_push("72o", "UTG", 10)
        assert not should

    def test_short_stack_push_btn(self):
        should, _ = self.pf.should_push("K9s", "BTN", 8)
        assert should  # K9s threshold / BTN divisor should be pushable at 8bb

    def test_call_vs_shove_aa(self):
        should, conf = self.pf.should_call_shove("AA", 20)
        assert should
        assert conf > 0.8

    def test_push_range_summary(self):
        summary = self.pf.get_push_fold_summary("BTN", 8)
        assert summary['zone'] in ('orange', 'yellow', 'red')
        assert len(summary['pushing_hands']) > 0
        assert summary['range_pct'] > 0


class TestICMCalculator:
    """Tests for ICM calculations."""

    def test_icm_equal_stacks(self):
        stacks = [1000, 1000, 1000]
        payouts = [50, 30, 20]
        equities = ICMCalculator.icm_exact(stacks, payouts)

        # Equal stacks should give equal equity
        for eq in equities:
            assert abs(eq - 33.33) < 0.5

    def test_icm_chip_leader(self):
        stacks = [3000, 1000, 1000]
        payouts = [50, 30, 20]
        equities = ICMCalculator.icm_exact(stacks, payouts)

        # Chip leader should have most equity but less than chip %
        assert equities[0] > equities[1]
        # With 60% of chips, equity should be less than $60
        assert equities[0] < 60

    def test_icm_diminishing_returns(self):
        stacks = [5000, 5000]
        payouts = [70, 30]
        equities = ICMCalculator.icm_exact(stacks, payouts)
        # Equal stacks: each gets $50
        assert abs(equities[0] - 50) < 1
        assert abs(equities[1] - 50) < 1

    def test_bubble_factor(self):
        stacks = [5000, 3000, 2000, 1000]
        payouts = [40, 30, 20]  # 3 paid, 4 players (bubble!)
        bf = ICMCalculator.bubble_factor(3000, stacks, payouts, hero_index=1)
        assert bf > 1.0  # Should have positive bubble pressure


class TestPositionManager:
    """Tests for position utilities."""

    def test_get_positions_6max(self):
        positions = PositionManager.get_positions(6)
        assert len(positions) == 6
        assert "BTN" in positions
        assert "BB" in positions

    def test_is_in_position(self):
        assert PositionManager.is_in_position("BTN", "SB")
        assert PositionManager.is_in_position("CO", "UTG")
        assert not PositionManager.is_in_position("SB", "BTN")

    def test_open_sizing(self):
        # BTN sizing should be larger than UTG
        btn_size = PositionManager.get_open_sizing("BTN", 50)
        utg_size = PositionManager.get_open_sizing("UTG", 50)
        assert btn_size >= utg_size

    def test_short_stack_sizing(self):
        # Very short stack should just jam
        size = PositionManager.get_open_sizing("BTN", 8)
        assert size == 8  # Full stack


class TestCardDetection:
    """Tests for card parsing and representation."""

    def test_card_from_string(self):
        card = Card.from_string("As")
        assert card.rank == "A"
        assert card.suit == "s"
        assert card.rank_value == 14

    def test_card_equality(self):
        c1 = Card("A", "s")
        c2 = Card("A", "s")
        c3 = Card("A", "h")
        assert c1 == c2
        assert c1 != c3

    def test_card_pretty(self):
        card = Card("A", "s")
        assert "\u2660" in card.pretty  # Contains spade symbol

    def test_invalid_card(self):
        with pytest.raises(ValueError):
            Card.from_string("Xz")


class TestScreenCapture:
    """Tests for window detection patterns."""

    def test_is_pokerstars_window_tournament(self):
        from screen_reader.capture import ScreenCapture
        # Tournament table titles should match
        assert ScreenCapture._is_pokerstars_window(
            "Tournament #3932271610 Table 1 - $0.25+$0.00 - No Limit Hold'em"
        )
        assert ScreenCapture._is_pokerstars_window(
            "Tournament #12345 Table 3 - Turbo"
        )

    def test_is_pokerstars_window_spin_and_go(self):
        from screen_reader.capture import ScreenCapture
        assert ScreenCapture._is_pokerstars_window(
            "Spin & Go #98765 Table 1 - No Limit Hold'em"
        )

    def test_is_pokerstars_window_not_matching(self):
        from screen_reader.capture import ScreenCapture
        # Regular non-poker windows should not match
        assert not ScreenCapture._is_pokerstars_window("Google Chrome")
        assert not ScreenCapture._is_pokerstars_window("Microsoft Word")
        assert not ScreenCapture._is_pokerstars_window("Tournament Bracket")
        assert not ScreenCapture._is_pokerstars_window("")

    def test_invalidate_window(self):
        from screen_reader.capture import ScreenCapture
        sc = ScreenCapture()
        sc.set_window_rect(100, 200, 800, 600)
        assert sc.is_window_found()
        sc.invalidate_window()
        assert not sc.is_window_found()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
