# utils.py
from PyQt5.QtWidgets import QDoubleSpinBox, QSpinBox, QFrame
from PyQt5.QtCore import Qt

def labeled_spinbox(min_val, max_val, *, step=1.0, decimals=2):
    sb = QDoubleSpinBox()
    sb.setDecimals(decimals)
    sb.setRange(min_val, max_val)
    sb.setSingleStep(step)
    sb.setButtonSymbols(QSpinBox.UpDownArrows)
    sb.setFixedWidth(140)
    sb.setAlignment(Qt.AlignCenter)
    return sb

def thin_separator():
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setFrameShadow(QFrame.Sunken)
    line.setStyleSheet("color:#444;")
    return line
