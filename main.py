"""
Root app entry: login + role-based tabs.
Only shows Qualification and Baseline for roles 'admin' and 'cualificacion'.
Role 'baseline' sees only Baseline. Starts maximized, resizable.
"""
import sys
from pathlib import Path

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QHBoxLayout,
    QVBoxLayout, QLabel, QPushButton, QComboBox, QMessageBox, QLayout, QStyle
)
from PyQt5.QtGui import QIcon
from serial.tools import list_ports

from pgdb import BatchDB
from style import apply_app_style
from auth_dialog import AuthDialog

from tabs.baseline import BaselineTab
from tabs.cualification import QualificationTab


ASSETS = Path(__file__).parent / "assets"


class MainWindow(QMainWindow):
    def __init__(self, db, user: dict):
        super().__init__()
        self.db = db
        self.user = user

        # Title and icon
        self.setWindowTitle(self._title())
        try:
            self.setWindowIcon(QIcon(str(ASSETS / "erreka_logo.jpg")))
        except Exception:
            pass
        self.setMinimumSize(0, 0)

        # Top COM bar
        self.com_combo = QComboBox()
        self.refresh_btn = QPushButton(self.style().standardIcon(QStyle.SP_BrowserReload), "")
        self.refresh_btn.setFixedWidth(60)
        top = QHBoxLayout()
        top.setContentsMargins(10, 8, 10, 2)
        top.addWidget(QLabel("COM"))
        top.addWidget(self.com_combo, 1)
        top.addWidget(self.refresh_btn)
        top.addStretch()
        top_widget = QWidget(); top_widget.setLayout(top)

        # Tabs
        self.tabs = QTabWidget(); self.tabs.setDocumentMode(True)
        self._build_tabs_for_role(self.user.get('role', 'baseline'))

        # Central layout
        central = QWidget()
        v = QVBoxLayout(central)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSizeConstraint(QLayout.SetNoConstraint)
        central.setMinimumSize(0, 0)
        v.addWidget(top_widget)
        v.addWidget(self.tabs, 1)
        self.setCentralWidget(central)

        # Signals
        self.refresh_btn.clicked.connect(self.populate_ports)
        self.populate_ports(initial=True)

        # Session menu (readable colors)
        m = self.menuBar().addMenu("Session")
        act_logout = m.addAction("Log out")
        act_logout.triggered.connect(self._logout)
        try:
            self.menuBar().setStyleSheet(
                "QMenuBar{background:#2b2b2b;color:#ffffff;}"
                " QMenuBar::item:selected{background:#3c3f41;}"
                " QMenu{background:#2b2b2b;color:#ffffff;}"
                " QMenu::item:selected{background:#3c3f41;}"
            )
        except Exception:
            pass

    def _title(self) -> str:
        u = self.user or {}
        return f"App cualificacion — {u.get('username','?')} ({u.get('role','baseline')})"

    def _build_tabs_for_role(self, role: str) -> None:
        while self.tabs.count():
            self.tabs.removeTab(0)
        r = (role or 'baseline').lower()
        if r == 'baseline':
            self.tabs.addTab(BaselineTab(self.com_combo, db=self.db), "Baseline")
            return
        if r in ('cualificacion', 'admin'):
            self.tabs.addTab(QualificationTab(self.com_combo, db=self.db), "Cualificación")
            self.tabs.addTab(BaselineTab(self.com_combo, db=self.db), "Baseline")
            return
        # Fallback
        self.tabs.addTab(BaselineTab(self.com_combo, db=self.db), "Baseline")

    def _logout(self) -> None:
        # Hide main window and show login dialog (maximized)
        self.hide()
        dlg = AuthDialog(self.db)
        if dlg.exec_() == dlg.Accepted and dlg.user:
            self.user = dlg.user
            self._build_tabs_for_role(self.user.get('role', 'baseline'))
            self.setWindowTitle(self._title())
            try:
                self.showMaximized()
            except Exception:
                self.show()
        else:
            self.close()

    def populate_ports(self, *, initial=False):
        ports = []
        for p in list_ports.comports():
            try:
                if p.manufacturer and 'FTDI' in p.manufacturer:
                    ports.append(p.device)
            except Exception:
                pass
        self.com_combo.clear()
        if ports:
            self.com_combo.addItems(ports)
            self.com_combo.setEnabled(True)
        else:
            self.com_combo.addItem("No COM detected")
            self.com_combo.setEnabled(False)
            if not initial:
                QMessageBox.information(self, "Serial Ports", "No COM port detected.")


def main():
    # Connect DB
    print("Iniciando conexion a la BBDD...")
    db = BatchDB(
        host="192.168.6.168", port="5432", dbname="cualificacion",
        user="postgres", password="Erreka*88",
        ssh_host="192.168.6.168", ssh_port=22, ssh_user="edb", ssh_password="Erreka*88"
    )
    print("OK. Conectado a PostgreSQL" if db.is_connected() else "Aviso: No se pudo conectar, usando mock")

    app = QApplication(sys.argv)
    try:
        app.setWindowIcon(QIcon(str(ASSETS / "erreka_logo.jpg")))
    except Exception:
        pass
    apply_app_style(app)

    # Login first
    dlg = AuthDialog(db)
    if dlg.exec_() != dlg.Accepted or not dlg.user:
        return
    window = MainWindow(db, dlg.user)
    try:
        window.showMaximized()
    except Exception:
        window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        from PyQt5.QtWidgets import QApplication, QMessageBox
        app = QApplication(sys.argv)
        QMessageBox.critical(None, "Fatal Error", traceback.format_exc())
        sys.exit(1)

