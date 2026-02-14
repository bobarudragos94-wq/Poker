"""
Equity calculator for poker hands.
Provides quick estimates using the rule of 2/4 and Monte Carlo simulation.
"""

import random
from typing import List, Tuple, Optional, Set
from itertools import combinations

from screen_reader.card_detector import Card, RANKS, SUITS
from .hand_evaluator import HandEvaluator


def build_deck(exclude: List[Card] = None) -> List[Card]:
    """Build a standard 52-card deck, optionally excluding known cards."""
    exclude_set = set()
    if exclude:
        exclude_set = {(c.rank, c.suit) for c in exclude}

    deck = []
    for rank in RANKS:
        for suit in SUITS:
            if (rank, suit) not in exclude_set:
                deck.append(Card(rank=rank, suit=suit))
    return deck


class EquityCalculator:
    """Calculate hand equity using various methods."""

    def __init__(self):
        self.evaluator = HandEvaluator()

    def monte_carlo_equity(self, hero_cards: List[Card],
                           board: List[Card] = None,
                           num_opponents: int = 1,
                           iterations: int = 5000) -> float:
        """
        Calculate hero's equity via Monte Carlo simulation.

        Args:
            hero_cards: Hero's 2 hole cards.
            board: Known board cards (0-5).
            num_opponents: Number of opposing players.
            iterations: Number of simulations.

        Returns:
            Equity as float between 0 and 1.
        """
        if board is None:
            board = []

        known_cards = list(hero_cards) + list(board)
        deck = build_deck(exclude=known_cards)

        wins = 0
        ties = 0
        total = 0

        cards_needed_for_board = 5 - len(board)

        for _ in range(iterations):
            random.shuffle(deck)
            idx = 0

            # Deal remaining board cards
            sim_board = list(board)
            for _ in range(cards_needed_for_board):
                sim_board.append(deck[idx])
                idx += 1

            # Evaluate hero's hand
            hero_rank, hero_kickers, _ = self.evaluator.evaluate(hero_cards, sim_board)

            # Deal and evaluate opponent hands
            hero_wins = True
            is_tie = False

            for _ in range(num_opponents):
                opp_cards = [deck[idx], deck[idx + 1]]
                idx += 2

                opp_rank, opp_kickers, _ = self.evaluator.evaluate(opp_cards, sim_board)

                result = self.evaluator.compare_hands(
                    (hero_rank, hero_kickers), (opp_rank, opp_kickers)
                )

                if result < 0:
                    hero_wins = False
                    is_tie = False
                    break
                elif result == 0:
                    is_tie = True

            if hero_wins and not is_tie:
                wins += 1
            elif is_tie:
                ties += 1
            total += 1

        if total == 0:
            return 0.5

        return (wins + ties * 0.5) / total

    @staticmethod
    def quick_equity_estimate(outs: int, streets_remaining: int) -> float:
        """
        Quick equity estimate using the Rule of 2 and 4.
        - Flop (2 cards to come): outs * 4  (accurate for <= 8 outs)
        - Turn (1 card to come): outs * 2
        """
        if streets_remaining == 2:
            if outs <= 8:
                return min(outs * 4 / 100, 1.0)
            else:
                return min((outs * 3 + 8) / 100, 1.0)
        else:
            return min(outs * 2 / 100, 1.0)

    @staticmethod
    def precise_equity(outs: int, cards_remaining: int, streets: int) -> float:
        """
        Precise equity based on outs.
        equity = 1 - ((remaining - outs) / remaining) ^ streets
        """
        if cards_remaining <= 0 or outs >= cards_remaining:
            return 1.0
        miss_one = (cards_remaining - outs) / cards_remaining
        return 1.0 - (miss_one ** streets)

    @staticmethod
    def count_outs(hero_cards: List[Card], board: List[Card]) -> dict:
        """
        Count the number of outs for common draws.
        Returns a dict of {draw_type: num_outs}.
        """
        if not board:
            return {}

        all_cards = hero_cards + board
        hero_ranks = {c.rank_value for c in hero_cards}
        hero_suits = {c.suit for c in hero_cards}
        board_ranks = [c.rank_value for c in board]
        board_suits = [c.suit for c in board]

        outs = {}

        # Flush draw (4 cards of same suit)
        suit_counts = {}
        for c in all_cards:
            suit_counts[c.suit] = suit_counts.get(c.suit, 0) + 1
        for suit, count in suit_counts.items():
            if count == 4:
                outs['flush_draw'] = 9  # 13 - 4 = 9 remaining suited cards
                break

        # Straight draws
        all_values = sorted(set(c.rank_value for c in all_cards))

        # Open-ended straight draw (OESD)
        for i in range(len(all_values) - 3):
            window = all_values[i:i+4]
            if window[-1] - window[0] == 3 and len(window) == 4:
                # 4 consecutive values - open ended
                if window[0] > 2 and window[-1] < 14:  # Not at the edges
                    outs['oesd'] = 8
                    break

        # Gutshot
        for i in range(len(all_values) - 3):
            window = all_values[i:i+4]
            if window[-1] - window[0] == 4:
                # Missing one card in the middle
                expected = set(range(window[0], window[-1] + 1))
                actual = set(window)
                if len(expected - actual) == 1:
                    outs['gutshot'] = 4
                    break

        # Overcards
        if board:
            max_board = max(board_ranks)
            overcard_count = sum(1 for r in hero_ranks if r > max_board)
            if overcard_count > 0:
                outs['overcards'] = overcard_count * 3

        # Total unique outs (rough estimate, may overlap)
        total = 0
        for draw_type, count in outs.items():
            total += count
        # Cap at reasonable maximum (accounting for overlap)
        outs['total_estimated'] = min(total, 20)

        return outs

    @staticmethod
    def pot_odds(pot: float, bet: float) -> float:
        """
        Calculate pot odds as equity percentage needed to call.
        pot_odds = call / (pot + call)
        """
        if pot + bet <= 0:
            return 0.0
        return bet / (pot + bet)

    @staticmethod
    def pot_odds_ratio(pot: float, bet: float) -> float:
        """Return pot odds as a ratio (e.g., 3.0 means 3:1)."""
        if bet <= 0:
            return float('inf')
        return (pot + bet) / bet

    @staticmethod
    def expected_value(equity: float, pot_if_win: float, cost: float) -> float:
        """
        Calculate expected value of a call.
        EV = (equity * pot_if_win) - ((1 - equity) * cost)
        """
        return (equity * pot_if_win) - ((1 - equity) * cost)

    @staticmethod
    def fold_equity_ev(fold_pct: float, pot: float, bet: float,
                       equity_when_called: float) -> float:
        """
        EV including fold equity.
        EV = (F * P) + (1-F) * [(E * (P+B)) - ((1-E) * B)]
        """
        fold_ev = fold_pct * pot
        call_ev = (1 - fold_pct) * (
            (equity_when_called * (pot + bet)) -
            ((1 - equity_when_called) * bet)
        )
        return fold_ev + call_ev

    @staticmethod
    def bluff_breakeven(bet: float, pot: float) -> float:
        """Minimum fold frequency needed for a 0% equity bluff to break even."""
        if pot + bet <= 0:
            return 1.0
        return bet / (pot + bet)

    @staticmethod
    def minimum_defense_frequency(bet: float, pot: float) -> float:
        """How often you must continue to prevent auto-profit bluffs."""
        if pot + bet <= 0:
            return 0.0
        return pot / (pot + bet)

    @staticmethod
    def stack_to_pot_ratio(effective_stack: float, pot: float) -> float:
        """SPR = effective stack / pot."""
        if pot <= 0:
            return float('inf')
        return effective_stack / pot

    @staticmethod
    def m_ratio(stack: float, sb: float, bb: float,
                ante: float = 0, num_players: int = 6) -> float:
        """Harrington's M-ratio (orbits until blinded out)."""
        cost_per_orbit = sb + bb + (ante * num_players)
        if cost_per_orbit <= 0:
            return float('inf')
        return stack / cost_per_orbit

    @staticmethod
    def preflop_hand_equity_estimate(hand_notation: str, num_opponents: int = 1) -> float:
        """
        Quick preflop equity estimate based on hand category.
        These are approximate and based on preflop all-in equity tables.
        """
        # Base equities vs 1 random hand
        premium_hands = {
            'AA': 0.85, 'KK': 0.82, 'QQ': 0.80, 'JJ': 0.77, 'TT': 0.75,
            'AKs': 0.67, 'AKo': 0.65, 'AQs': 0.66, 'AQo': 0.63,
            'AJs': 0.65, 'AJo': 0.63, 'KQs': 0.63, 'KQo': 0.61,
        }

        if hand_notation in premium_hands:
            base_eq = premium_hands[hand_notation]
        else:
            # Estimate based on hand type
            if len(hand_notation) == 2:  # Pair
                rank_val = RANKS.index(hand_notation[0])
                base_eq = 0.50 + (rank_val / 13) * 0.35
            elif hand_notation.endswith('s'):
                r1 = RANKS.index(hand_notation[0])
                r2 = RANKS.index(hand_notation[1])
                base_eq = 0.35 + ((r1 + r2) / 26) * 0.30
            else:
                r1 = RANKS.index(hand_notation[0])
                r2 = RANKS.index(hand_notation[1])
                base_eq = 0.32 + ((r1 + r2) / 26) * 0.28

        # Adjust for multiple opponents (equity drops)
        if num_opponents > 1:
            # Rough approximation: equity vs N opponents
            base_eq = base_eq ** (1 + (num_opponents - 1) * 0.35)

        return min(max(base_eq, 0.0), 1.0)
