#!/usr/bin/env python3
"""
Poker GTO Assistant - Real-time GTO decisions for PokerStars tournaments.

This app captures your PokerStars table screen in real-time, detects your
cards, position, stack size, and board using OCR and computer vision,
then provides optimal GTO-based recommendations via an overlay.

Usage:
    python main.py              # Launch GUI
    python main.py --manual     # Manual input mode (terminal)
    python main.py --help       # Show help
"""

import sys
import os
import argparse
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import AppConfig


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def run_gui(config: AppConfig):
    """Launch the GUI application."""
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtGui import QFont
    from gui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("Poker GTO Assistant")
    app.setStyle("Fusion")

    # Set default font
    font = QFont("Segoe UI", 10)
    app.setFont(font)

    window = MainWindow(config)
    window.show()

    sys.exit(app.exec())


def run_manual_mode():
    """Run in terminal-based manual input mode."""
    from screen_reader.table_state import TableStateReader, GameState
    from decision.advisor import GTOAdvisor
    from screen_reader.card_detector import Card

    config = AppConfig()
    advisor = GTOAdvisor(tournament_mode=True)
    reader = TableStateReader(config)

    print("=" * 60)
    print("  POKER GTO ASSISTANT - Manual Mode")
    print("=" * 60)
    print()

    while True:
        try:
            print("-" * 40)
            hand = input("Your hand (e.g., AhKs) or 'q' to quit: ").strip()
            if hand.lower() == 'q':
                break

            position = input("Position (UTG/MP/CO/BTN/SB/BB): ").strip().upper()
            stack_bb = float(input("Stack in BB: ").strip())
            bb_size = float(input("Big blind size: ").strip() or "100")
            pot = float(input("Pot size (0 for preflop): ").strip() or "0")
            board = input("Board cards (blank for preflop): ").strip()
            num_players = int(input("Number of players (default 6): ").strip() or "6")

            state = reader.set_manual_state(
                hero_cards=hand,
                position=position,
                stack_bb=stack_bb,
                big_blind=bb_size,
                pot=pot,
                board=board,
                num_players=num_players,
            )

            decision = advisor.analyze(state)

            print()
            print(f"  ACTION:     {decision.action}")
            print(f"  Confidence: {decision.confidence_str} ({decision.confidence*100:.0f}%)")
            if decision.sizing_bb > 0:
                print(f"  Sizing:     {decision.sizing_bb:.1f}BB", end="")
                if decision.sizing_pct:
                    print(f" ({decision.sizing_pct*100:.0f}% pot)", end="")
                print()
            print(f"  Reasoning:  {decision.reasoning}")
            for detail in decision.details:
                print(f"    - {detail}")
            if decision.icm_note:
                print(f"  ICM:        {decision.icm_note}")
            if decision.alternative:
                print(f"  Alt:        {decision.alternative}")
            print()

        except (ValueError, KeyboardInterrupt) as e:
            if isinstance(e, KeyboardInterrupt):
                print("\nGoodbye!")
                break
            print(f"  Invalid input: {e}")
            print()


def main():
    parser = argparse.ArgumentParser(
        description="Poker GTO Assistant - Real-time tournament decisions"
    )
    parser.add_argument("--manual", "-m", action="store_true",
                        help="Run in terminal-based manual mode")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable debug logging")
    parser.add_argument("--config", "-c", type=str, default="config.json",
                        help="Path to config file")

    args = parser.parse_args()
    setup_logging(args.verbose)

    if args.manual:
        run_manual_mode()
    else:
        config = AppConfig.load(args.config)
        run_gui(config)


if __name__ == "__main__":
    main()
