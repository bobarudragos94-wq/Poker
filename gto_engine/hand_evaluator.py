"""
Hand evaluator for Texas Hold'em.
Evaluates the best 5-card hand from 7 cards (2 hole + 5 board).
"""

from typing import List, Tuple, Optional
from itertools import combinations
from screen_reader.card_detector import Card

# Hand rankings (higher = better)
HAND_RANKS = {
    "high_card": 0,
    "one_pair": 1,
    "two_pair": 2,
    "three_of_a_kind": 3,
    "straight": 4,
    "flush": 5,
    "full_house": 6,
    "four_of_a_kind": 7,
    "straight_flush": 8,
    "royal_flush": 9,
}

HAND_NAMES = {v: k.replace("_", " ").title() for k, v in HAND_RANKS.items()}


class HandEvaluator:
    """Evaluates poker hands and determines hand strength."""

    @staticmethod
    def evaluate(hole_cards: List[Card], board: List[Card]) -> Tuple[int, List[int], str]:
        """
        Evaluate the best 5-card hand.

        Args:
            hole_cards: List of 2 Cards.
            board: List of 3-5 Cards.

        Returns:
            Tuple of (hand_rank, kickers, hand_name)
            hand_rank: 0-9 (higher is better)
            kickers: List of card values for tiebreaking
            hand_name: Human-readable name
        """
        all_cards = hole_cards + board

        if len(all_cards) < 5:
            # Not enough cards to make a hand
            values = sorted([c.rank_value for c in all_cards], reverse=True)
            return (0, values, "High Card")

        best_rank = -1
        best_kickers = []
        best_name = "High Card"

        # Evaluate all 5-card combinations
        for combo in combinations(all_cards, 5):
            rank, kickers, name = HandEvaluator._evaluate_five(list(combo))
            if (rank, kickers) > (best_rank, best_kickers):
                best_rank = rank
                best_kickers = kickers
                best_name = name

        return (best_rank, best_kickers, best_name)

    @staticmethod
    def _evaluate_five(cards: List[Card]) -> Tuple[int, List[int], str]:
        """Evaluate exactly 5 cards."""
        values = sorted([c.rank_value for c in cards], reverse=True)
        suits = [c.suit for c in cards]

        is_flush = len(set(suits)) == 1
        is_straight, straight_high = HandEvaluator._check_straight(values)

        # Count value frequencies
        freq = {}
        for v in values:
            freq[v] = freq.get(v, 0) + 1

        counts = sorted(freq.values(), reverse=True)
        # Values sorted by frequency then by rank
        freq_sorted = sorted(freq.keys(), key=lambda x: (freq[x], x), reverse=True)

        # Royal flush
        if is_flush and is_straight and straight_high == 14:
            return (9, [14], "Royal Flush")

        # Straight flush
        if is_flush and is_straight:
            return (8, [straight_high], "Straight Flush")

        # Four of a kind
        if counts == [4, 1]:
            quad_val = [v for v, c in freq.items() if c == 4][0]
            kicker = [v for v, c in freq.items() if c == 1][0]
            return (7, [quad_val, kicker], "Four of a Kind")

        # Full house
        if counts == [3, 2]:
            trip_val = [v for v, c in freq.items() if c == 3][0]
            pair_val = [v for v, c in freq.items() if c == 2][0]
            return (6, [trip_val, pair_val], "Full House")

        # Flush
        if is_flush:
            return (5, values, "Flush")

        # Straight
        if is_straight:
            return (4, [straight_high], "Straight")

        # Three of a kind
        if counts == [3, 1, 1]:
            trip_val = [v for v, c in freq.items() if c == 3][0]
            kickers = sorted([v for v, c in freq.items() if c == 1], reverse=True)
            return (3, [trip_val] + kickers, "Three of a Kind")

        # Two pair
        if counts == [2, 2, 1]:
            pairs = sorted([v for v, c in freq.items() if c == 2], reverse=True)
            kicker = [v for v, c in freq.items() if c == 1][0]
            return (2, pairs + [kicker], "Two Pair")

        # One pair
        if counts == [2, 1, 1, 1]:
            pair_val = [v for v, c in freq.items() if c == 2][0]
            kickers = sorted([v for v, c in freq.items() if c == 1], reverse=True)
            return (1, [pair_val] + kickers, "One Pair")

        # High card
        return (0, values, "High Card")

    @staticmethod
    def _check_straight(values: List[int]) -> Tuple[bool, int]:
        """Check if sorted values form a straight. Returns (is_straight, high_card)."""
        unique = sorted(set(values), reverse=True)
        if len(unique) < 5:
            return False, 0

        # Normal straight check
        if unique[0] - unique[4] == 4:
            return True, unique[0]

        # Wheel (A-2-3-4-5): A=14, 5,4,3,2
        if set(unique) == {14, 5, 4, 3, 2}:
            return True, 5  # 5-high straight

        return False, 0

    @staticmethod
    def hand_strength_category(rank: int) -> str:
        """Categorize hand strength for display."""
        if rank >= 7:
            return "monster"      # Quads+
        elif rank >= 5:
            return "very_strong"  # Flush+
        elif rank >= 3:
            return "strong"       # Trips+
        elif rank >= 2:
            return "medium"       # Two pair
        elif rank >= 1:
            return "marginal"     # One pair
        else:
            return "weak"         # High card

    @staticmethod
    def preflop_hand_rank(card1: Card, card2: Card) -> float:
        """
        Rate a preflop hand from 0 to 1 (1 = best).
        Based on Sklansky-Karlson rankings adapted for tournament play.
        """
        r1 = card1.rank_value
        r2 = card2.rank_value
        if r1 < r2:
            r1, r2 = r2, r1

        is_pair = r1 == r2
        is_suited = card1.suit == card2.suit

        # Base score from card values
        if is_pair:
            score = 0.5 + (r1 / 14) * 0.5  # Pairs: 0.57 (22) to 1.0 (AA)
        else:
            high_score = r1 / 14
            low_score = r2 / 14
            gap = r1 - r2

            score = (high_score * 0.6 + low_score * 0.3)

            # Connectivity bonus
            if gap == 1:
                score += 0.05
            elif gap == 2:
                score += 0.03

            # Suited bonus
            if is_suited:
                score += 0.06

            # Penalty for big gaps
            if gap > 4:
                score -= 0.05 * (gap - 4)

        return max(0.0, min(1.0, score))

    @staticmethod
    def compare_hands(hand1: Tuple[int, List[int]], hand2: Tuple[int, List[int]]) -> int:
        """
        Compare two evaluated hands.
        Returns: 1 if hand1 wins, -1 if hand2 wins, 0 if tie.
        """
        rank1, kickers1 = hand1
        rank2, kickers2 = hand2

        if rank1 > rank2:
            return 1
        elif rank1 < rank2:
            return -1

        # Same rank - compare kickers
        for k1, k2 in zip(kickers1, kickers2):
            if k1 > k2:
                return 1
            elif k1 < k2:
                return -1

        return 0  # Tie
