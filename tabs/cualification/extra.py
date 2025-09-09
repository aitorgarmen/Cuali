# tabs/qualification/extra.py
from __future__ import annotations

from PyQt5.QtWidgets import QLabel, QVBoxLayout, QWidget


class ExtraTab(QWidget):
    """Subpestaña **Extra** de la pestaña *Cualificación*."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Contenido de la cualificación: Extra"))
