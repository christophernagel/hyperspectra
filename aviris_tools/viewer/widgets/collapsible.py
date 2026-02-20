"""
Collapsible section widget for the control panel.
"""

from qtpy.QtWidgets import QWidget, QVBoxLayout, QToolButton, QSizePolicy
from qtpy.QtCore import Qt


class CollapsibleSection(QWidget):
    """A collapsible container widget with a toggle button header."""

    def __init__(self, title, parent=None, collapsed=False):
        super().__init__(parent)
        self.is_collapsed = collapsed
        self.title = title

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Toggle button header
        self.toggle_button = QToolButton()
        arrow = "▶" if collapsed else "▼"
        self.toggle_button.setText(f"{arrow} {title}")
        self.toggle_button.setStyleSheet("""
            QToolButton {
                font-weight: bold;
                font-size: 12px;
                border: none;
                padding: 8px;
                background-color: #3d3d3d;
                color: white;
                text-align: left;
            }
            QToolButton:hover { background-color: #4d4d4d; }
        """)
        self.toggle_button.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.toggle_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.toggle_button.clicked.connect(self.toggle)
        main_layout.addWidget(self.toggle_button)

        # Content container
        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(8, 8, 8, 8)
        self.content_layout.setSpacing(8)
        main_layout.addWidget(self.content_widget)

        # Apply initial state
        self.content_widget.setVisible(not collapsed)

    def toggle(self):
        """Toggle the collapsed state."""
        self.is_collapsed = not self.is_collapsed
        self.content_widget.setVisible(not self.is_collapsed)
        arrow = "▶" if self.is_collapsed else "▼"
        self.toggle_button.setText(f"{arrow} {self.title}")

    def set_collapsed(self, collapsed):
        """Programmatically set the collapsed state."""
        if collapsed != self.is_collapsed:
            self.toggle()

    def add_widget(self, widget):
        """Add a widget to the content area."""
        self.content_layout.addWidget(widget)

    def add_layout(self, layout):
        """Add a layout to the content area."""
        self.content_layout.addLayout(layout)
