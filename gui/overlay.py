"""
Transparent overlay widget that displays GTO recommendations
on top of the PokerStars window.
"""

from typing import Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QGraphicsDropShadowEffect, QSizePolicy
)
from PyQt6.QtCore import Qt, QPoint, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPainter, QBrush, QPen, QMouseEvent

from decision.advisor import Decision


class ActionBadge(QLabel):
    """Colored badge showing the recommended action."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumHeight(50)
        self.setFont(QFont("Segoe UI", 20, QFont.Weight.Bold))
        self._color = "#95a5a6"
        self.update_style()

    def set_action(self, action: str, color: str):
        self.setText(action)
        self._color = color
        self.update_style()

    def update_style(self):
        self.setStyleSheet(f"""
            QLabel {{
                background-color: {self._color};
                color: white;
                border-radius: 8px;
                padding: 8px 16px;
                font-weight: bold;
            }}
        """)


class ConfidenceBar(QWidget):
    """Visual confidence indicator bar."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(8)
        self.setMaximumHeight(8)
        self._confidence = 0.0
        self._color = "#2ecc71"

    def set_confidence(self, value: float, color: str = "#2ecc71"):
        self._confidence = max(0.0, min(1.0, value))
        self._color = color
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Background
        painter.setBrush(QBrush(QColor("#2c3e50")))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(0, 0, self.width(), self.height(), 4, 4)

        # Filled portion
        fill_width = int(self.width() * self._confidence)
        if fill_width > 0:
            painter.setBrush(QBrush(QColor(self._color)))
            painter.drawRoundedRect(0, 0, fill_width, self.height(), 4, 4)

        painter.end()


class DetailLine(QLabel):
    """Styled label for detail information."""

    def __init__(self, text: str = "", parent=None):
        super().__init__(text, parent)
        self.setFont(QFont("Segoe UI", 9))
        self.setWordWrap(True)
        self.setStyleSheet("""
            QLabel {
                color: #bdc3c7;
                padding: 1px 0;
            }
        """)


class OverlayWidget(QWidget):
    """
    Semi-transparent overlay that shows GTO recommendations.
    Can be dragged and positioned anywhere on screen.
    """

    closed = pyqtSignal()

    def __init__(self, opacity: float = 0.90, parent=None):
        super().__init__(parent)

        # Window setup - stays on top, frameless, translucent
        self.setWindowFlags(
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setMinimumSize(320, 300)
        self.resize(350, 420)

        self._opacity = opacity
        self._drag_position: Optional[QPoint] = None
        self._is_dragging = False

        self._setup_ui()
        self._apply_shadow()

    def _setup_ui(self):
        """Build the overlay UI."""
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(12, 12, 12, 12)
        self.main_layout.setSpacing(8)

        # Title bar with drag handle and close button
        title_bar = QHBoxLayout()
        self.title_label = QLabel("GTO ASSISTANT")
        self.title_label.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        self.title_label.setStyleSheet("color: #ecf0f1;")
        title_bar.addWidget(self.title_label)

        title_bar.addStretch()

        self.status_label = QLabel("Scanning...")
        self.status_label.setFont(QFont("Segoe UI", 8))
        self.status_label.setStyleSheet("color: #7f8c8d;")
        title_bar.addWidget(self.status_label)

        self.close_btn = QLabel(" X ")
        self.close_btn.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        self.close_btn.setStyleSheet("""
            QLabel {
                color: #e74c3c;
                padding: 2px 6px;
                border-radius: 3px;
            }
            QLabel:hover {
                background-color: #e74c3c;
                color: white;
            }
        """)
        self.close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.close_btn.mousePressEvent = lambda e: self._on_close()
        title_bar.addWidget(self.close_btn)

        self.main_layout.addLayout(title_bar)

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("background-color: #34495e;")
        sep.setMaximumHeight(1)
        self.main_layout.addWidget(sep)

        # Action badge (the main recommendation)
        self.action_badge = ActionBadge()
        self.main_layout.addWidget(self.action_badge)

        # Confidence bar
        conf_layout = QHBoxLayout()
        conf_label = QLabel("Confidence:")
        conf_label.setFont(QFont("Segoe UI", 8))
        conf_label.setStyleSheet("color: #7f8c8d;")
        conf_layout.addWidget(conf_label)

        self.confidence_label = QLabel("--")
        self.confidence_label.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        self.confidence_label.setStyleSheet("color: #ecf0f1;")
        conf_layout.addWidget(self.confidence_label)

        conf_layout.addStretch()
        self.main_layout.addLayout(conf_layout)

        self.confidence_bar = ConfidenceBar()
        self.main_layout.addWidget(self.confidence_bar)

        # Sizing recommendation
        self.sizing_label = QLabel("")
        self.sizing_label.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        self.sizing_label.setStyleSheet("color: #f39c12;")
        self.sizing_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.main_layout.addWidget(self.sizing_label)

        # Reasoning
        self.reasoning_label = QLabel("")
        self.reasoning_label.setFont(QFont("Segoe UI", 10))
        self.reasoning_label.setStyleSheet("color: #ecf0f1;")
        self.reasoning_label.setWordWrap(True)
        self.main_layout.addWidget(self.reasoning_label)

        # Details section
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.HLine)
        sep2.setStyleSheet("background-color: #34495e;")
        sep2.setMaximumHeight(1)
        self.main_layout.addWidget(sep2)

        self.details_layout = QVBoxLayout()
        self.details_layout.setSpacing(2)
        self.main_layout.addLayout(self.details_layout)
        self._detail_labels: list = []

        # ICM note
        self.icm_label = QLabel("")
        self.icm_label.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        self.icm_label.setStyleSheet("color: #e67e22;")
        self.icm_label.setWordWrap(True)
        self.main_layout.addWidget(self.icm_label)

        # Alternative play
        self.alt_label = QLabel("")
        self.alt_label.setFont(QFont("Segoe UI", 8))
        self.alt_label.setStyleSheet("color: #7f8c8d; font-style: italic;")
        self.alt_label.setWordWrap(True)
        self.main_layout.addWidget(self.alt_label)

        self.main_layout.addStretch()

    def _apply_shadow(self):
        """Add drop shadow effect."""
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setColor(QColor(0, 0, 0, 160))
        shadow.setOffset(0, 4)
        self.setGraphicsEffect(shadow)

    def update_decision(self, decision: Decision):
        """Update the overlay with a new decision."""
        # Action
        self.action_badge.set_action(decision.action, decision.color)

        # Confidence
        self.confidence_label.setText(
            f"{decision.confidence_str} ({decision.confidence*100:.0f}%)"
        )
        self.confidence_bar.set_confidence(decision.confidence, decision.color)

        # Sizing
        if decision.sizing_bb > 0:
            if decision.sizing_pct > 0:
                self.sizing_label.setText(
                    f"Size: {decision.sizing_bb:.1f}BB "
                    f"({decision.sizing_pct*100:.0f}% pot)"
                )
            else:
                self.sizing_label.setText(f"Size: {decision.sizing_bb:.1f}BB")
        else:
            self.sizing_label.setText("")

        # Reasoning
        self.reasoning_label.setText(decision.reasoning)

        # Details
        self._clear_details()
        for detail in decision.details:
            label = DetailLine(detail)
            self.details_layout.addWidget(label)
            self._detail_labels.append(label)

        # ICM note
        self.icm_label.setText(decision.icm_note if decision.icm_note else "")
        self.icm_label.setVisible(bool(decision.icm_note))

        # Alternative
        if decision.alternative:
            self.alt_label.setText(f"Alt: {decision.alternative}")
            self.alt_label.setVisible(True)
        else:
            self.alt_label.setVisible(False)

        # Status
        self.status_label.setText(f"{decision.street.upper()} | {decision.position}")

    def set_scanning(self):
        """Show scanning state."""
        self.action_badge.set_action("SCANNING", "#95a5a6")
        self.confidence_label.setText("--")
        self.confidence_bar.set_confidence(0.0)
        self.sizing_label.setText("")
        self.reasoning_label.setText("Scanning PokerStars table for game state...")
        self._clear_details()
        self.icm_label.setText("")
        self.alt_label.setText("")
        self.status_label.setText("Scanning...")

    def set_no_window(self):
        """Show that PokerStars window was not found."""
        self.action_badge.set_action("NO TABLE", "#e74c3c")
        self.reasoning_label.setText(
            "PokerStars table not detected. "
            "Make sure a table is open and visible."
        )
        self.status_label.setText("No table found")

    def _clear_details(self):
        """Remove all detail labels."""
        for label in self._detail_labels:
            self.details_layout.removeWidget(label)
            label.deleteLater()
        self._detail_labels.clear()

    def _on_close(self):
        self.closed.emit()
        self.hide()

    # === Drag support ===

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            self._is_dragging = True
            event.accept()

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._is_dragging and self._drag_position:
            self.move(event.globalPosition().toPoint() - self._drag_position)
            event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent):
        self._is_dragging = False
        self._drag_position = None

    # === Paint background ===

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Draw rounded rect background
        painter.setBrush(QBrush(QColor(44, 62, 80, int(255 * self._opacity))))
        painter.setPen(QPen(QColor(52, 73, 94), 1))
        painter.drawRoundedRect(self.rect().adjusted(2, 2, -2, -2), 12, 12)

        painter.end()
