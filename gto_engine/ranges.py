"""
GTO preflop ranges for tournament poker.
Includes opening ranges, 3-bet ranges, and calling ranges by position and stack depth.
"""

from typing import Dict, List, Set, Tuple, Optional


# All 169 unique starting hands
def _generate_all_hands() -> List[str]:
    """Generate all 169 unique starting hands."""
    ranks = "AKQJT98765432"
    hands = []
    # Pairs
    for r in ranks:
        hands.append(f"{r}{r}")
    # Suited combos
    for i, r1 in enumerate(ranks):
        for r2 in ranks[i+1:]:
            hands.append(f"{r1}{r2}s")
    # Offsuit combos
    for i, r1 in enumerate(ranks):
        for r2 in ranks[i+1:]:
            hands.append(f"{r1}{r2}o")
    return hands


ALL_HANDS = _generate_all_hands()

# Rank ordering for comparisons
RANK_ORDER = {'A': 14, 'K': 13, 'Q': 12, 'J': 11, 'T': 10,
              '9': 9, '8': 8, '7': 7, '6': 6, '5': 5,
              '4': 4, '3': 3, '2': 2}


def hand_in_range(hand: str, range_str: str) -> bool:
    """
    Check if a specific hand is within a range string.
    hand: e.g., "AKs", "TT", "J9o"
    range_str: e.g., "TT+, ATs+, KQs, AJo+"
    """
    hand = hand.strip()
    range_parts = [r.strip() for r in range_str.split(",")]

    for part in range_parts:
        part = part.strip()
        if not part:
            continue

        if _hand_matches_part(hand, part):
            return True

    return False


def _hand_matches_part(hand: str, part: str) -> bool:
    """Check if a hand matches a single range component."""
    # Remove whitespace
    part = part.strip()

    # Direct match
    if hand == part:
        return True

    # Handle '+' notation
    if part.endswith('+'):
        base = part[:-1]
        return _hand_gte(hand, base)

    # Handle '-' range notation (e.g., "TT-77", "ATs-A7s")
    if '-' in part:
        parts = part.split('-')
        if len(parts) == 2:
            return _hand_in_span(hand, parts[0].strip(), parts[1].strip())

    return False


def _hand_gte(hand: str, base: str) -> bool:
    """Check if hand is >= base in standard notation with '+'."""
    # Pair+: e.g., "TT+" means TT, JJ, QQ, KK, AA
    if len(base) == 2 and base[0] == base[1]:
        if len(hand) == 2 and hand[0] == hand[1]:
            return RANK_ORDER.get(hand[0], 0) >= RANK_ORDER.get(base[0], 0)
        return False

    # Suited+: e.g., "ATs+" means ATs, AJs, AQs, AKs
    if len(base) == 3 and base[2] == 's':
        if len(hand) == 3 and hand[2] == 's' and hand[0] == base[0]:
            return RANK_ORDER.get(hand[1], 0) >= RANK_ORDER.get(base[1], 0)
        return False

    # Offsuit+: e.g., "ATo+" means ATo, AJo, AQo, AKo
    if len(base) == 3 and base[2] == 'o':
        if len(hand) == 3 and hand[2] == 'o' and hand[0] == base[0]:
            return RANK_ORDER.get(hand[1], 0) >= RANK_ORDER.get(base[1], 0)
        return False

    return False


def _hand_in_span(hand: str, high: str, low: str) -> bool:
    """Check if hand is between high and low (inclusive)."""
    # Pair range: "TT-77"
    if len(high) == 2 and high[0] == high[1]:
        if len(hand) == 2 and hand[0] == hand[1]:
            val = RANK_ORDER.get(hand[0], 0)
            return (RANK_ORDER.get(low[0], 0) <= val <= RANK_ORDER.get(high[0], 0))
        return False

    # Suited range: "ATs-A7s"
    if len(high) == 3 and high[2] == 's':
        if len(hand) == 3 and hand[2] == 's' and hand[0] == high[0]:
            val = RANK_ORDER.get(hand[1], 0)
            return (RANK_ORDER.get(low[1], 0) <= val <= RANK_ORDER.get(high[1], 0))
        return False

    # Offsuit range: "ATo-A7o"
    if len(high) == 3 and high[2] == 'o':
        if len(hand) == 3 and hand[2] == 'o' and hand[0] == high[0]:
            val = RANK_ORDER.get(hand[1], 0)
            return (RANK_ORDER.get(low[1], 0) <= val <= RANK_ORDER.get(high[1], 0))
        return False

    return False


def expand_range(range_str: str) -> Set[str]:
    """Expand a range string into a set of individual hands."""
    result = set()
    for hand in ALL_HANDS:
        if hand_in_range(hand, range_str):
            result.add(hand)
    return result


def range_percentage(range_str: str) -> float:
    """Calculate what percentage of all hands a range covers."""
    hands = expand_range(range_str)
    # Weight: pairs = 6 combos, suited = 4, offsuit = 12
    total_combos = 0
    for h in hands:
        if len(h) == 2:  # Pair
            total_combos += 6
        elif h[2] == 's':
            total_combos += 4
        else:
            total_combos += 12
    return (total_combos / 1326) * 100


class RangeManager:
    """Manages GTO preflop ranges for different positions and stack depths."""

    # === OPENING RANGES (Raise First In) by position and stack depth ===
    # Stack depth categories: "deep" (40bb+), "mid" (20-40bb), "short" (10-20bb)

    OPEN_RANGES = {
        "deep": {  # 40bb+
            "UTG":  "77+, A9s+, A5s, KTs+, QTs+, JTs, T9s, 98s, AQo+",
            "MP":   "55+, A4s+, K9s+, Q9s+, J9s+, T9s, 98s, AJo+, KQo",
            "CO":   "22+, A2s+, K7s+, Q8s+, J8s+, T8s+, 97s+, 86s+, 75s+, 64s+, 54s, A9o+, KJo+, QJo",
            "BTN":  "22+, A2s+, K2s+, Q4s+, J7s+, T6s+, 96s+, 85s+, 75s+, 64s+, 53s+, 43s, A2o+, K7o+, Q8o+, J8o+, T8o+, 97o+, 87o",
            "SB":   "22+, A2s+, K2s+, Q4s+, J6s+, T6s+, 96s+, 85s+, 74s+, 64s+, 53s+, 43s, A2o+, K5o+, Q8o+, J9o+, T9o+",
        },
        "mid": {  # 20-40bb
            "UTG":  "55+, A3s+, K8s+, Q9s+, J9s+, T8s+, 98s, ATo+, KTo+, QJo",
            "MP":   "55+, A2s+, K7s+, Q8s+, J8s+, T8s+, 98s, A9o+, KTo+, QTo+, JTo",
            "CO":   "33+, A2s+, K3s+, Q5s+, J7s+, T7s+, 97s+, 86s+, 76s+, A5o+, K9o+, QTo+, JTo",
            "BTN":  "22+, A2s+, K2s+, Q3s+, J5s+, T5s+, 96s+, 86s+, 75s+, 64s+, 54s, A2o+, K7o+, Q8o+, J8o+, T8o+, 98o",
            "SB":   "22+, A2s+, K2s+, Q2s+, J4s+, T5s+, 95s+, 85s+, 75s+, 64s+, 54s, A2o+, K4o+, Q7o+, J8o+, T8o+, 98o",
        },
        "short": {  # 10-20bb (raise/jam)
            "UTG":  "44+, A4s+, K9s+, QTs+, J9s+, T9s, A9o+, KJo+",
            "MP":   "33+, A2s+, K9s+, Q9s+, J9s+, T9s, A9o+, KJo+, QJo",
            "CO":   "22+, A2s+, K6s+, Q8s+, J8s+, T8s+, 98s, A3o+, KTo+, QJo, JTo",
            "BTN":  "22+, A2s+, K2s+, Q5s+, J6s+, T6s+, 97s+, 87s, A2o+, K7o+, QTo+, JTo",
            "SB":   "22+, A2s+, K2s+, Q2s+, J3s+, T5s+, 95s+, 85s+, 75s+, 64s+, 54s, A2o+, K3o+, Q7o+, J8o+, T8o+",
        },
    }

    # === 3-BET RANGES (vs opener from specific position) ===
    THREE_BET_RANGES = {
        "BB_vs_UTG": "QQ+, AKs, A5s-A4s",
        "BB_vs_MP":  "JJ+, AKs, AKo, A5s-A3s, 76s, 65s",
        "BB_vs_CO":  "TT+, AQs+, AKo, A5s-A2s, K9s-K7s, Q9s, J9s, 87s, 76s, 65s, 54s",
        "BB_vs_BTN": "TT+, AJs+, KQs, AQo+, A9s-A2s, K9s-K5s, Q9s-Q7s, J9s-J8s, T8s-T7s, 97s, 87s, 76s, 65s, 54s",
        "SB_vs_UTG": "QQ+, AKs, AKo, A5s-A4s",
        "SB_vs_CO":  "TT+, AQs+, AKo, A5s-A3s, KJs-K9s, QJs",
        "SB_vs_BTN": "TT+, AJs+, KQs, AQo+, A9s-A2s, KTs-K7s, QTs-Q9s, JTs-J9s, T9s, 98s, 87s",
        "BTN_vs_UTG": "QQ+, AKs, AQo, A5s-A4s",
        "BTN_vs_CO":  "JJ+, AKs, AKo, AQo, A5s-A3s, KQs, KJs",
        "CO_vs_UTG":  "QQ+, AKs, AQo, A5s",
        "CO_vs_MP":   "QQ+, AKs, AKo, AQo, A5s-A4s",
    }

    # === 4-BET RANGES ===
    FOUR_BET_RANGES = {
        "UTG_vs_3bet": "KK+, AKo, A5s",
        "MP_vs_3bet":  "QQ+, AKs, AKo, A5s, A4s",
        "CO_vs_3bet":  "QQ+, AKs, AKo, A5s-A3s, AQo",
        "BTN_vs_3bet": "JJ+, AKs, AKo, A5s-A2s, AQo, KQs",
    }

    # === CALLING RANGES (vs open raise) ===
    CALL_RANGES = {
        "deep": {
            "BB_vs_UTG": "22-66, ATs-A2s, KJs-K9s, QJs-Q9s, JTs-J9s, T9s, 98s, 87s, 76s, 65s, 54s, AJo-ATo, KQo-KJo",
            "BB_vs_CO":  "22-99, A9s-A2s, KTs-K5s, QTs-Q7s, JTs-J8s, T8s+, 98s, 87s, 76s, 65s, ATo-A7o, KJo-K9o, QJo-QTo, JTo",
            "BB_vs_BTN": "22-99, A9s-A2s, K9s-K3s, Q9s-Q5s, J8s-J6s, T7s+, 97s+, 86s+, 76s, 65s, 54s, ATo-A3o, KTo-K7o, QTo-Q9o, JTo-J9o, T9o",
        },
    }

    # === PUSH/FOLD CHART (Max BB to shove, from SB heads-up Nash) ===
    # For multi-way, divide by position divisor
    PUSH_FOLD_HU = {
        # Pairs
        'AA': 500, 'KK': 500, 'QQ': 500, 'JJ': 500, 'TT': 500,
        '99': 500, '88': 500, '77': 500, '66': 500, '55': 500,
        '44': 500, '33': 500, '22': 20,
        # Suited
        'AKs': 500, 'AQs': 500, 'AJs': 500, 'ATs': 500,
        'A9s': 20, 'A8s': 20, 'A7s': 20, 'A6s': 20,
        'A5s': 20, 'A4s': 20, 'A3s': 20, 'A2s': 20,
        'KQs': 20, 'KJs': 20, 'KTs': 20, 'K9s': 20,
        'K8s': 17, 'K7s': 15, 'K6s': 14, 'K5s': 13,
        'K4s': 12, 'K3s': 11, 'K2s': 10,
        'QJs': 20, 'QTs': 20, 'Q9s': 16, 'Q8s': 13,
        'Q7s': 11, 'Q6s': 10, 'Q5s': 9.5, 'Q4s': 9,
        'Q3s': 8, 'Q2s': 7.5,
        'JTs': 20, 'J9s': 16, 'J8s': 13, 'J7s': 10,
        'J6s': 8, 'J5s': 7, 'J4s': 6.5, 'J3s': 5.5, 'J2s': 5,
        'T9s': 18, 'T8s': 14, 'T7s': 10, 'T6s': 7.5,
        'T5s': 6, 'T4s': 5.5, 'T3s': 5, 'T2s': 4.5,
        '98s': 16, '97s': 12, '96s': 8, '95s': 6, '94s': 4.5,
        '87s': 14, '86s': 9, '85s': 6, '84s': 4,
        '76s': 13, '75s': 8, '74s': 5,
        '65s': 12, '64s': 7, '63s': 5,
        '54s': 11, '53s': 7, '52s': 4,
        '43s': 8, '42s': 4, '32s': 4,
        # Offsuit
        'AKo': 500, 'AQo': 500, 'AJo': 500, 'ATo': 20,
        'A9o': 20, 'A8o': 20, 'A7o': 18, 'A6o': 16,
        'A5o': 18, 'A4o': 16, 'A3o': 15, 'A2o': 14,
        'KQo': 20, 'KJo': 20, 'KTo': 17, 'K9o': 13,
        'K8o': 10, 'K7o': 8, 'K6o': 7, 'K5o': 6.5,
        'K4o': 6, 'K3o': 5.5, 'K2o': 5,
        'QJo': 17, 'QTo': 13, 'Q9o': 9, 'Q8o': 7,
        'Q7o': 5.5, 'Q6o': 5, 'Q5o': 4.5, 'Q4o': 4, 'Q3o': 3.5,
        'JTo': 14, 'J9o': 9, 'J8o': 6.5, 'J7o': 5,
        'J6o': 4, 'J5o': 3.5, 'J4o': 3,
        'T9o': 11, 'T8o': 7, 'T7o': 5, 'T6o': 3.5,
        '98o': 9, '97o': 6, '96o': 4,
        '87o': 8, '86o': 5, '85o': 3,
        '76o': 7, '75o': 4, '74o': 2.5,
        '65o': 6, '64o': 3.5,
        '54o': 5.5, '53o': 3, '52o': 2,
        '43o': 3.5, '42o': 2, '32o': 1.7,
    }

    # Position divisors for adjusting HU push/fold to full table
    POSITION_DIVISORS = {
        'SB':  1.0,
        'BTN': 2.0,
        'CO':  4.0,
        'MP':  6.0,  # HJ equivalent in 6-max
        'UTG': 8.0,
    }

    # Calling ranges vs all-in (max BB to call)
    CALL_VS_SHOVE = {
        'AA': 500, 'KK': 500, 'QQ': 500, 'JJ': 500, 'TT': 500,
        '99': 20, '88': 20, '77': 20, '66': 16, '55': 14,
        '44': 12, '33': 10, '22': 8,
        'AKs': 500, 'AQs': 500, 'AJs': 20, 'ATs': 20,
        'A9s': 17, 'A8s': 15, 'A7s': 13, 'A6s': 12,
        'A5s': 13, 'A4s': 12, 'A3s': 11, 'A2s': 10,
        'KQs': 20, 'KJs': 18, 'KTs': 16, 'K9s': 13,
        'K8s': 10, 'K7s': 8, 'K6s': 7, 'K5s': 6,
        'QJs': 16, 'QTs': 14, 'Q9s': 10, 'Q8s': 7,
        'JTs': 16, 'J9s': 11, 'J8s': 7,
        'T9s': 13, 'T8s': 8,
        '98s': 10, '97s': 7,
        '87s': 9, '86s': 6,
        '76s': 8, '75s': 5,
        '65s': 7, '54s': 6,
        'AKo': 500, 'AQo': 20, 'AJo': 18, 'ATo': 15,
        'A9o': 12, 'A8o': 10, 'A7o': 9, 'A6o': 7,
        'A5o': 9, 'A4o': 8, 'A3o': 7, 'A2o': 6,
        'KQo': 16, 'KJo': 13, 'KTo': 10, 'K9o': 7,
        'QJo': 9, 'QTo': 7,
        'JTo': 7,
    }

    def get_open_range(self, position: str, stack_bb: float) -> str:
        """Get the opening range for a position and stack depth."""
        depth = self._stack_depth_category(stack_bb)
        pos = self._normalize_position(position)
        return self.OPEN_RANGES.get(depth, {}).get(pos, "")

    def get_3bet_range(self, position: str, vs_position: str) -> str:
        """Get 3-bet range from position vs opener position."""
        pos = self._normalize_position(position)
        vs = self._normalize_position(vs_position)
        key = f"{pos}_vs_{vs}"
        return self.THREE_BET_RANGES.get(key, "")

    def get_4bet_range(self, position: str) -> str:
        """Get 4-bet range for a position."""
        pos = self._normalize_position(position)
        key = f"{pos}_vs_3bet"
        return self.FOUR_BET_RANGES.get(key, "")

    def is_push(self, hand: str, position: str, stack_bb: float,
                has_antes: bool = False) -> bool:
        """Check if a hand should be pushed all-in (short stack)."""
        pos = self._normalize_position(position)
        divisor = self.POSITION_DIVISORS.get(pos, 8.0)

        threshold = self.PUSH_FOLD_HU.get(hand, 0)
        adjusted = threshold / divisor

        if has_antes:
            adjusted *= 1.25  # Widen with antes

        return stack_bb <= adjusted

    def is_call_vs_shove(self, hand: str, shover_stack_bb: float,
                         num_players_behind: int = 0) -> bool:
        """Check if we should call an all-in."""
        threshold = self.CALL_VS_SHOVE.get(hand, 0)
        # Tighten when players are behind us
        if num_players_behind > 0:
            threshold /= (1 + num_players_behind * 0.5)
        return shover_stack_bb <= threshold

    def hand_in_open_range(self, hand: str, position: str, stack_bb: float) -> bool:
        """Check if a hand is in the opening range."""
        range_str = self.get_open_range(position, stack_bb)
        return hand_in_range(hand, range_str) if range_str else False

    @staticmethod
    def _stack_depth_category(stack_bb: float) -> str:
        if stack_bb >= 40:
            return "deep"
        elif stack_bb >= 20:
            return "mid"
        else:
            return "short"

    @staticmethod
    def _normalize_position(pos: str) -> str:
        """Normalize position strings."""
        pos = pos.upper().replace("+", "").replace("/", "_")
        mapping = {
            "UTG": "UTG", "UTG+1": "UTG", "UTG+2": "MP",
            "MP": "MP", "MP+1": "MP", "HJ": "MP",
            "LJ": "MP", "LOJACK": "MP", "HIJACK": "MP",
            "CO": "CO", "CUTOFF": "CO",
            "BTN": "BTN", "BUTTON": "BTN", "BU": "BTN", "D": "BTN",
            "SB": "SB", "SMALL BLIND": "SB",
            "BB": "BB", "BIG BLIND": "BB",
            "BTN_SB": "SB",
        }
        return mapping.get(pos, pos)
