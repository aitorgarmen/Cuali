import os
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTabWidget
from .batch import BatchTab
from pgdb import BatchDB
from .pre import PreTab
from .one_four import OneFourTab
from .temp import TemperatureTab
from .one_ten import OneTenTab
from .bending import BendingTab
from .extra import ExtraTab


class QualificationTab(QWidget):
    def __init__(self, com_selector=None, db: BatchDB | None = None, parent=None):
        super().__init__(parent)
        self.com_selector = com_selector
        # Shared database connection for qualification tabs
        # If no db provided, initialize from QUALIF_PG_DSN env var
        dsn = os.getenv("QUALIF_PG_DSN")
        self.db = db or BatchDB(dsn)
        root = QVBoxLayout(self)
        tabs = QTabWidget()
        # Batch subtab with shared DB
        tabs.addTab(BatchTab(self.com_selector, db=self.db), "Batch")
        # Prequalification subtab with shared DB and COM selector
        tabs.addTab(PreTab(com_selector=self.com_selector, db=self.db), "Pre")
        tabs.addTab(OneFourTab(com_selector=self.com_selector, db=self.db), "1-4")
        tabs.addTab(TemperatureTab(com_selector=self.com_selector, db=self.db), "Temp")
        tabs.addTab(OneTenTab(com_selector=self.com_selector, db=self.db), "1-10")
        tabs.addTab(BendingTab(com_selector=self.com_selector, db=self.db), "Bending")
        # tabs.addTab(ExtraTab(), "Extra")
        # ...añade las demás subtabs...
        root.addWidget(tabs)
