# style.py
from PyQt5 import QtGui
import pyqtgraph as pg

STYLE_SHEET = """
* {
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 11pt;
    color: #e6e6e6;
}
QWidget {
    background-color: #202124;
}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background: #2b2b2f;
    border: 1px solid #444;
    padding: 4px 6px;
    border-radius: 4px;
}
QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: right center;
    width: 18px;
    border-left: 1px solid #444;
}
QComboBox::down-arrow {
    image: none;
}
QPushButton {
    border: none;
    padding: 10px 16px;
    border-radius: 6px;
    font-weight: 600;
}
QPushButton#Play { background: #00c853; }
QPushButton#Stop { background: #d50000; }
QPushButton#Save { background: #00b8ff; }
QPushButton#File { background: #424242; }
QPushButton#Refresh { background: #424242; }
QTabBar::tab {
    background: #2b2b2f;
    color: #e6e6e6;
    padding: 6px 12px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
}
QTabBar::tab:selected {
    background: #424242;
    color: #ffffff;
}
QTabWidget::pane {
    border: none;
    top: -1px;
}
"""

def apply_app_style(app):
    app.setStyleSheet(STYLE_SHEET)
    pg.setConfigOption("background", "#202124")
    pg.setConfigOption("foreground", "#e6e6e6")
