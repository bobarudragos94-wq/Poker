"""
Independent Chip Model (ICM) calculations for tournament poker.
Calculates $EV equity based on chip stacks and payout structure.
"""

import random
import math
from typing import List, Optional, Tuple
from itertools import permutations


class ICMCalculator:
    """Calculates ICM equity and adjustments for tournament decisions."""

    @staticmethod
    def icm_exact(stacks: List[float], payouts: List[float]) -> List[float]:
        """
        Exact ICM calculation using Malmuth-Harville model.
        Enumerates all finishing order permutations.

        WARNING: O(n!) complexity. Only use for n <= 9 players.

        Args:
            stacks: Chip counts for each player.
            payouts: Prize amounts for 1st, 2nd, 3rd, etc.

        Returns:
            Dollar equity for each player.
        """
        total = sum(stacks)
        n = len(stacks)
        n_pay = min(len(payouts), n)
        equities = [0.0] * n

        if n > 9:
            return ICMCalculator.icm_monte_carlo(stacks, payouts)

        for perm in permutations(range(n)):
            prob = 1.0
            remaining = total
            for pos in range(n):
                player = perm[pos]
                if remaining <= 0:
                    break
                prob *= stacks[player] / remaining
                remaining -= stacks[player]

            for pos in range(n_pay):
                equities[perm[pos]] += prob * payouts[pos]

        return equities

    @staticmethod
    def icm_monte_carlo(stacks: List[float], payouts: List[float],
                        iterations: int = 100000) -> List[float]:
        """
        Monte Carlo ICM approximation for larger player counts.

        Args:
            stacks: Chip counts for each player.
            payouts: Prize amounts.
            iterations: Number of simulations.

        Returns:
            Estimated dollar equity for each player.
        """
        total = sum(stacks)
        n = len(stacks)
        n_pay = min(len(payouts), n)
        equity_sums = [0.0] * n

        for _ in range(iterations):
            remaining_players = list(range(n))
            remaining_chips = float(total)

            for place in range(n):
                if not remaining_players:
                    break

                # Weighted random selection
                weights = [stacks[p] / remaining_chips for p in remaining_players]
                r = random.random()
                cumulative = 0.0
                chosen_idx = 0
                for idx, w in enumerate(weights):
                    cumulative += w
                    if r <= cumulative:
                        chosen_idx = idx
                        break

                player = remaining_players[chosen_idx]
                if place < n_pay:
                    equity_sums[player] += payouts[place]

                remaining_chips -= stacks[player]
                remaining_players.pop(chosen_idx)

        return [e / iterations for e in equity_sums]

    @staticmethod
    def icm_adjustment_factor(hero_stack: float, all_stacks: List[float],
                              payouts: List[float], hero_index: int = 0) -> float:
        """
        Calculate ICM pressure factor.
        Returns a multiplier (0.5-1.5) that indicates how much
        to tighten (< 1.0) or loosen (> 1.0) compared to chip-EV.

        < 1.0 = tighten ranges (ICM pressure, near bubble)
        = 1.0 = play chip-EV
        > 1.0 = can be more aggressive (chip leader applying pressure)
        """
        if not payouts or len(all_stacks) <= 1:
            return 1.0

        n = len(all_stacks)
        total = sum(all_stacks)
        hero_pct = hero_stack / total if total > 0 else 0

        # Calculate current ICM equity
        current_equity = ICMCalculator._quick_icm(all_stacks, payouts)
        if not current_equity:
            return 1.0

        hero_eq = current_equity[hero_index]
        chip_eq = hero_pct * sum(payouts)

        # ICM factor: ratio of ICM equity to chip equity
        # If ICM equity > chip equity, player has more to lose -> tighten
        if chip_eq > 0:
            factor = chip_eq / hero_eq if hero_eq > 0 else 1.0
        else:
            factor = 1.0

        # Clamp to reasonable range
        return max(0.5, min(1.5, factor))

    @staticmethod
    def bubble_factor(hero_stack: float, all_stacks: List[float],
                      payouts: List[float], hero_index: int = 0) -> float:
        """
        Calculate bubble factor - how much more ICM equity we risk
        by going all-in compared to what we gain.

        Higher bubble factor = more ICM pressure = play tighter.
        Typical values: 1.0 (no pressure) to 3.0+ (extreme bubble).
        """
        if not payouts or len(all_stacks) <= 2:
            return 1.0

        n = len(all_stacks)
        total = sum(all_stacks)

        # Current equities
        current_eq = ICMCalculator._quick_icm(all_stacks, payouts)
        if not current_eq:
            return 1.0

        # Simulate doubling up (winning an average pot)
        avg_opponent_stack = (total - hero_stack) / (n - 1) if n > 1 else 0
        pot_won = min(hero_stack, avg_opponent_stack)

        win_stacks = list(all_stacks)
        win_stacks[hero_index] += pot_won
        # Remove the smallest other stack as if they busted
        other_indices = [i for i in range(n) if i != hero_index]
        if other_indices:
            smallest_idx = min(other_indices, key=lambda i: all_stacks[i])
            lose_stacks = [s for i, s in enumerate(all_stacks) if i != smallest_idx]
            win_eq = ICMCalculator._quick_icm(win_stacks, payouts)
        else:
            win_eq = current_eq

        # Simulate busting
        lose_stacks_sim = [s for i, s in enumerate(all_stacks) if i != hero_index]
        lose_eq = ICMCalculator._quick_icm(lose_stacks_sim, payouts)

        if not win_eq or not lose_eq:
            return 1.0

        # What we gain vs what we lose
        equity_gained = win_eq[hero_index] - current_eq[hero_index]
        equity_lost = current_eq[hero_index]  # We lose everything if we bust

        if equity_gained > 0:
            return equity_lost / equity_gained
        return 2.0  # Default medium pressure

    @staticmethod
    def _quick_icm(stacks: List[float], payouts: List[float]) -> Optional[List[float]]:
        """Quick ICM approximation using a simplified model."""
        n = len(stacks)
        if n == 0:
            return None
        if n <= 9:
            return ICMCalculator.icm_exact(stacks, payouts)
        return ICMCalculator.icm_monte_carlo(stacks, payouts, iterations=10000)

    @staticmethod
    def suggested_range_tightening(bubble_factor_val: float) -> float:
        """
        Convert bubble factor to percentage range tightening.
        Returns a value 0-0.5 (0% to 50% tighter than normal).
        """
        if bubble_factor_val <= 1.0:
            return 0.0
        elif bubble_factor_val <= 1.5:
            return 0.10  # 10% tighter
        elif bubble_factor_val <= 2.0:
            return 0.20  # 20% tighter
        elif bubble_factor_val <= 3.0:
            return 0.35  # 35% tighter
        else:
            return 0.50  # 50% tighter (extreme bubble)

    @staticmethod
    def chip_ev_to_icm_ev(chip_ev: float, bubble_factor_val: float) -> float:
        """
        Adjust a chip-EV decision for ICM.
        A positive chip-EV play might be ICM-negative near the bubble.
        """
        if bubble_factor_val <= 1.0:
            return chip_ev

        # ICM-adjusted EV considers that losing is worse than winning is good
        if chip_ev >= 0:
            return chip_ev / bubble_factor_val
        else:
            return chip_ev * bubble_factor_val
