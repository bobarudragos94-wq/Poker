"""
Main application window for the Poker GTO Assistant.
Provides configuration, manual input mode, and controls the overlay.
"""

import sys
import logging
from typing import Optional

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox,
    QLineEdit, QGroupBox, QFormLayout, QCheckBox, QTabWidget,
    QTextEdit, QSlider, QFrame, QMessageBox, QApplication
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QFont, QColor, QIcon

from config import AppConfig
from screen_reader.table_state import TableStateReader, GameState
from decision.advisor import GTOAdvisor, Decision
from .overlay import OverlayWidget

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """Main application window with controls and manual input."""

    def __init__(self, config: AppConfig):
        super().__init__()
        self.config = config
        self.advisor = GTOAdvisor(tournament_mode=config.tournament_mode)
        self.table_reader = TableStateReader(config)
        self.overlay: Optional[OverlayWidget] = None

        # Scanning timer
        self.scan_timer = QTimer()
        self.scan_timer.timeout.connect(self._on_scan_tick)

        self._is_scanning = False
        self._consecutive_failures = 0

        self._setup_ui()
        self.setWindowTitle("Poker GTO Assistant")
        self.setMinimumSize(500, 700)
        self.resize(520, 750)

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(10)

        # Stylesheet
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1a1a2e;
            }
            QWidget {
                background-color: #1a1a2e;
                color: #ecf0f1;
            }
            QGroupBox {
                border: 1px solid #34495e;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 16px;
                font-weight: bold;
                color: #ecf0f1;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QPushButton {
                background-color: #2c3e50;
                color: #ecf0f1;
                border: 1px solid #34495e;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #34495e;
            }
            QPushButton:pressed {
                background-color: #1abc9c;
            }
            QPushButton#startBtn {
                background-color: #27ae60;
                font-size: 14px;
                padding: 12px;
            }
            QPushButton#startBtn:hover {
                background-color: #2ecc71;
            }
            QPushButton#stopBtn {
                background-color: #c0392b;
                font-size: 14px;
                padding: 12px;
            }
            QPushButton#stopBtn:hover {
                background-color: #e74c3c;
            }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
                background-color: #2c3e50;
                color: #ecf0f1;
                border: 1px solid #34495e;
                border-radius: 4px;
                padding: 4px 8px;
            }
            QTabWidget::pane {
                border: 1px solid #34495e;
                border-radius: 4px;
            }
            QTabBar::tab {
                background-color: #2c3e50;
                color: #bdc3c7;
                padding: 8px 16px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: #34495e;
                color: #ecf0f1;
            }
            QTextEdit {
                background-color: #16213e;
                color: #ecf0f1;
                border: 1px solid #34495e;
                border-radius: 4px;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 11px;
            }
        """)

        # Header
        header = QLabel("POKER GTO ASSISTANT")
        header.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.setStyleSheet("color: #1abc9c; padding: 10px;")
        layout.addWidget(header)

        subtitle = QLabel("Real-time GTO decisions for PokerStars tournaments")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet("color: #7f8c8d; font-size: 11px;")
        layout.addWidget(subtitle)

        # Tab widget
        tabs = QTabWidget()
        layout.addWidget(tabs)

        # Tab 1: Auto Scan
        scan_tab = QWidget()
        scan_layout = QVBoxLayout(scan_tab)
        tabs.addTab(scan_tab, "Auto Scan")

        # Scan controls
        scan_group = QGroupBox("Screen Scanner")
        scan_form = QFormLayout(scan_group)

        self.window_title_input = QLineEdit(self.config.pokerstars_window_title)
        scan_form.addRow("Window Title:", self.window_title_input)

        self.scan_interval = QSpinBox()
        self.scan_interval.setRange(200, 5000)
        self.scan_interval.setValue(self.config.capture_interval_ms)
        self.scan_interval.setSuffix(" ms")
        scan_form.addRow("Scan Interval:", self.scan_interval)

        self.table_type_combo = QComboBox()
        self.table_type_combo.addItems(["6-Max", "9-Max", "Heads Up"])
        scan_form.addRow("Table Type:", self.table_type_combo)

        self.tournament_check = QCheckBox("Tournament Mode (ICM)")
        self.tournament_check.setChecked(self.config.tournament_mode)
        scan_form.addRow(self.tournament_check)

        scan_layout.addWidget(scan_group)

        # Start/Stop buttons
        btn_layout = QHBoxLayout()

        self.start_btn = QPushButton("START SCANNING")
        self.start_btn.setObjectName("startBtn")
        self.start_btn.clicked.connect(self._start_scanning)
        btn_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("STOP")
        self.stop_btn.setObjectName("stopBtn")
        self.stop_btn.clicked.connect(self._stop_scanning)
        self.stop_btn.setEnabled(False)
        btn_layout.addWidget(self.stop_btn)

        scan_layout.addLayout(btn_layout)

        # Status
        self.scan_status = QLabel("Status: Idle")
        self.scan_status.setStyleSheet("color: #7f8c8d; padding: 5px;")
        scan_layout.addWidget(self.scan_status)

        scan_layout.addStretch()

        # Tab 2: Manual Input
        manual_tab = QWidget()
        manual_layout = QVBoxLayout(manual_tab)
        tabs.addTab(manual_tab, "Manual Input")

        manual_group = QGroupBox("Enter Hand Details")
        manual_form = QFormLayout(manual_group)

        self.hand_input = QLineEdit()
        self.hand_input.setPlaceholderText("e.g., AhKs, TdTc, 9s8s")
        manual_form.addRow("Your Hand:", self.hand_input)

        self.position_combo = QComboBox()
        self.position_combo.addItems(["UTG", "MP", "CO", "BTN", "SB", "BB"])
        manual_form.addRow("Position:", self.position_combo)

        self.stack_input = QDoubleSpinBox()
        self.stack_input.setRange(0.5, 500)
        self.stack_input.setValue(25)
        self.stack_input.setSuffix(" BB")
        self.stack_input.setDecimals(1)
        manual_form.addRow("Stack Size:", self.stack_input)

        self.bb_input = QDoubleSpinBox()
        self.bb_input.setRange(1, 1000000)
        self.bb_input.setValue(100)
        self.bb_input.setPrefix("$")
        manual_form.addRow("Big Blind:", self.bb_input)

        self.pot_input = QDoubleSpinBox()
        self.pot_input.setRange(0, 1000000)
        self.pot_input.setValue(0)
        self.pot_input.setPrefix("$")
        manual_form.addRow("Pot Size:", self.pot_input)

        self.board_input = QLineEdit()
        self.board_input.setPlaceholderText("e.g., Th9h2c Kd (leave blank for preflop)")
        manual_form.addRow("Board:", self.board_input)

        self.players_input = QSpinBox()
        self.players_input.setRange(2, 9)
        self.players_input.setValue(6)
        manual_form.addRow("Players:", self.players_input)

        manual_layout.addWidget(manual_group)

        self.analyze_btn = QPushButton("ANALYZE HAND")
        self.analyze_btn.setObjectName("startBtn")
        self.analyze_btn.clicked.connect(self._analyze_manual)
        manual_layout.addWidget(self.analyze_btn)

        # Results display
        self.result_display = QTextEdit()
        self.result_display.setReadOnly(True)
        self.result_display.setMinimumHeight(200)
        manual_layout.addWidget(self.result_display)

        # Tab 3: Settings
        settings_tab = QWidget()
        settings_layout = QVBoxLayout(settings_tab)
        tabs.addTab(settings_tab, "Settings")

        overlay_group = QGroupBox("Overlay Settings")
        overlay_form = QFormLayout(overlay_group)

        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(30, 100)
        self.opacity_slider.setValue(int(self.config.overlay_opacity * 100))
        overlay_form.addRow("Opacity:", self.opacity_slider)

        self.font_size_spin = QSpinBox()
        self.font_size_spin.setRange(8, 24)
        self.font_size_spin.setValue(self.config.font_size)
        overlay_form.addRow("Font Size:", self.font_size_spin)

        self.show_ranges_check = QCheckBox("Show Range %")
        self.show_ranges_check.setChecked(self.config.show_ranges)
        overlay_form.addRow(self.show_ranges_check)

        self.show_equity_check = QCheckBox("Show Equity")
        self.show_equity_check.setChecked(self.config.show_equity)
        overlay_form.addRow(self.show_equity_check)

        settings_layout.addWidget(overlay_group)

        save_btn = QPushButton("Save Settings")
        save_btn.clicked.connect(self._save_settings)
        settings_layout.addWidget(save_btn)

        settings_layout.addStretch()

        # Log area
        log_group = QGroupBox("Activity Log")
        log_layout = QVBoxLayout(log_group)
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setMaximumHeight(120)
        log_layout.addWidget(self.log_display)
        layout.addWidget(log_group)

    def _log(self, message: str):
        """Add a message to the log display."""
        self.log_display.append(message)
        logger.info(message)

    # === Auto Scan ===

    def _start_scanning(self):
        """Start automatic screen scanning."""
        self._is_scanning = True
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        # Update config
        self.table_reader.capture.window_title = self.window_title_input.text()

        # Create/show overlay
        if self.overlay is None:
            opacity = self.opacity_slider.value() / 100.0
            self.overlay = OverlayWidget(opacity=opacity)
            self.overlay.closed.connect(self._stop_scanning)

        self.overlay.show()
        self.overlay.set_scanning()

        # Start timer
        interval = self.scan_interval.value()
        self.scan_timer.start(interval)

        self.scan_status.setText("Status: Scanning...")
        self.scan_status.setStyleSheet("color: #2ecc71; padding: 5px;")
        self._log("Started scanning for PokerStars table...")

        # Try to find the window immediately
        rect = self.table_reader.capture.find_window()
        if rect:
            self._log(f"Found table window at {rect}")
        else:
            self._log("Table window not found - will retry on each scan.")

    def _stop_scanning(self):
        """Stop automatic screen scanning."""
        self._is_scanning = False
        self.scan_timer.stop()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

        if self.overlay:
            self.overlay.hide()

        self.scan_status.setText("Status: Stopped")
        self.scan_status.setStyleSheet("color: #e74c3c; padding: 5px;")
        self._log("Scanning stopped.")

    @pyqtSlot()
    def _on_scan_tick(self):
        """Called on each scan interval - capture and analyze."""
        if not self._is_scanning:
            return

        try:
            # Try to find window if not already found
            if not self.table_reader.capture.is_window_found():
                self.table_reader.capture.find_window()
                if not self.table_reader.capture.is_window_found():
                    if self.overlay:
                        self.overlay.set_no_window()
                    return

            # Read table state
            state = self.table_reader.read_state()
            if state is None:
                self._consecutive_failures += 1
                # After several failures, the window may have closed or moved;
                # invalidate and re-detect on the next tick
                if self._consecutive_failures >= 10:
                    self.table_reader.capture.invalidate_window()
                    self._consecutive_failures = 0
                    self._log("Lost table window - attempting to re-detect...")
                if self.overlay:
                    self.overlay.set_scanning()
                return

            self._consecutive_failures = 0

            # Get decision
            decision = self.advisor.analyze(state)

            # Update overlay
            if self.overlay and self.overlay.isVisible():
                self.overlay.update_decision(decision)

            # Update status
            hand = state.hand_notation or "..."
            pos = state.hero_position or "?"
            self.scan_status.setText(
                f"Status: Active | {hand} @ {pos} | "
                f"{state.hero_stack_bb:.1f}BB"
            )

        except Exception as e:
            logger.error("Scan error: %s", e)
            self._log(f"Error: {e}")

    # === Manual Input ===

    def _analyze_manual(self):
        """Analyze a manually entered hand."""
        hand_str = self.hand_input.text().strip()
        if not hand_str:
            self.result_display.setText("Please enter your hand (e.g., AhKs)")
            return

        position = self.position_combo.currentText()
        stack_bb = self.stack_input.value()
        bb = self.bb_input.value()
        pot = self.pot_input.value()
        board_str = self.board_input.text().strip()
        num_players = self.players_input.value()

        try:
            # Create manual game state
            state = self.table_reader.set_manual_state(
                hero_cards=hand_str,
                position=position,
                stack_bb=stack_bb,
                big_blind=bb,
                pot=pot,
                board=board_str,
                num_players=num_players,
            )

            # Get decision
            decision = self.advisor.analyze(state)

            # Format result
            result = self._format_decision(decision)
            self.result_display.setHtml(result)

            # Update overlay if visible
            if self.overlay and self.overlay.isVisible():
                self.overlay.update_decision(decision)

            self._log(f"Analyzed: {hand_str} @ {position} ({stack_bb}BB)")

        except Exception as e:
            self.result_display.setText(f"Error: {e}")
            self._log(f"Analysis error: {e}")

    @staticmethod
    def _format_decision(d: Decision) -> str:
        """Format a Decision into HTML for display."""
        color = d.color

        html = f"""
        <div style="font-family: Segoe UI, sans-serif;">
            <h2 style="color: {color}; text-align: center; margin: 5px 0;">
                {d.action}
            </h2>
            <p style="text-align: center; color: #bdc3c7;">
                Confidence: <b>{d.confidence_str}</b> ({d.confidence*100:.0f}%)
            </p>
        """

        if d.sizing_bb > 0:
            html += f"""
            <p style="text-align: center; color: #f39c12; font-size: 14px;">
                Size: <b>{d.sizing_bb:.1f}BB</b>
                {f'({d.sizing_pct*100:.0f}% pot)' if d.sizing_pct else ''}
            </p>
            """

        html += f"""
            <hr style="border-color: #34495e;">
            <p style="color: #ecf0f1;">{d.reasoning}</p>
        """

        if d.details:
            html += "<ul style='color: #bdc3c7;'>"
            for detail in d.details:
                html += f"<li>{detail}</li>"
            html += "</ul>"

        if d.icm_note:
            html += f"""
            <p style="color: #e67e22; font-weight: bold;">{d.icm_note}</p>
            """

        if d.alternative:
            html += f"""
            <p style="color: #7f8c8d; font-style: italic;">Alt: {d.alternative}</p>
            """

        html += "</div>"
        return html

    # === Settings ===

    def _save_settings(self):
        """Save current settings to config file."""
        self.config.overlay_opacity = self.opacity_slider.value() / 100.0
        self.config.font_size = self.font_size_spin.value()
        self.config.show_ranges = self.show_ranges_check.isChecked()
        self.config.show_equity = self.show_equity_check.isChecked()
        self.config.tournament_mode = self.tournament_check.isChecked()
        self.config.capture_interval_ms = self.scan_interval.value()
        self.config.pokerstars_window_title = self.window_title_input.text()

        self.config.save()
        self._log("Settings saved.")

    def closeEvent(self, event):
        """Clean up when closing."""
        self._stop_scanning()
        if self.overlay:
            self.overlay.close()
        event.accept()
