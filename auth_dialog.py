from __future__ import annotations

from typing import Optional, Dict, Any

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon, QFont
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QTabWidget, QWidget, QLabel, QLineEdit,
    QPushButton, QFormLayout, QHBoxLayout, QMessageBox, QComboBox, QToolButton
)

from auth import ensure_auth_tables, authenticate, create_user, count_users
from pathlib import Path


class AuthDialog(QDialog):
    """Login / Signup dialog (English UI).

    Shows maximized and uses app logo. Creating users requires admin
    authorization, except when bootstrapping the very first admin.
    """

    def __init__(self, db) -> None:
        super().__init__()
        self.setWindowTitle("Sign In - EDB Qualification")
        self.setModal(True)
        self._db = db
        self.user: Optional[Dict[str, Any]] = None

        # Ensure table exists (no-op when mocked)
        try:
            ensure_auth_tables(self._db)
        except Exception as e:
            QMessageBox.warning(self, "Authentication", f"Could not prepare user table.\n{e}")

        # Window icon and maximized
        try:
            self.setWindowIcon(QIcon(str(Path(__file__).parent / "assets" / "erreka_logo.jpg")))
        except Exception:
            pass
        self.setWindowFlag(Qt.Window, True)
        self.setWindowState(self.windowState() | Qt.WindowMaximized)

        self._tabs = QTabWidget()
        self._login_tab = self._build_login_tab()
        self._signup_tab = self._build_signup_tab()
        self._tabs.addTab(self._login_tab, "Sign In")
        self._tabs.addTab(self._signup_tab, "Create User")

        root = QVBoxLayout(self)
        root.addWidget(self._tabs)

        # Footer buttons
        btns = QHBoxLayout()
        btns.addStretch(1)
        self._cancel = QPushButton("Cancel")
        self._cancel.clicked.connect(self.reject)
        btns.addWidget(self._cancel)
        root.addLayout(btns)

        self._update_signup_role_options()

    # --------------------------- UI builders ---------------------------
    def _build_login_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        self._li_user = QLineEdit()
        self._li_pass = QLineEdit(); self._li_pass.setEchoMode(QLineEdit.Password)
        self._li_pass.returnPressed.connect(self._do_login)
        self._li_user.setPlaceholderText("username")
        self._li_pass.setPlaceholderText("password")
        form.addRow("Username", self._li_user)
        # Password row with eye
        pw_row = QWidget()
        pw_layout = QHBoxLayout(pw_row)
        pw_layout.setContentsMargins(0, 0, 0, 0)
        pw_layout.addWidget(self._li_pass, 1)
        # Eye toggle: QToolButton avoids global QPushButton padding from style.py
        self._btn_eye_login = QToolButton(); self._btn_eye_login.setCheckable(True); self._btn_eye_login.setFixedWidth(28)
        self._btn_eye_login.setAutoRaise(True)
        self._btn_eye_login.setFocusPolicy(Qt.NoFocus)
        self._btn_eye_login.setFont(QFont("Segoe UI Emoji", 12))
        self._btn_eye_login.setText(chr(0x1F441))  # eye symbol
        self._btn_eye_login.setToolTip("Show/Hide password")
        self._btn_eye_login.setStyleSheet("QToolButton{padding:0;margin:0;background:transparent;}")
        self._btn_eye_login.clicked.connect(lambda: self._li_pass.setEchoMode(QLineEdit.Normal if self._btn_eye_login.isChecked() else QLineEdit.Password))
        pw_layout.addWidget(self._btn_eye_login)
        form.addRow("Password", pw_row)

        self._login_msg = QLabel("")
        self._login_msg.setStyleSheet("color: #d9534f")
        form.addRow(self._login_msg)

        row = QHBoxLayout()
        row.addStretch(1)
        self._login_btn = QPushButton("Sign In")
        self._login_btn.setDefault(True)
        self._login_btn.clicked.connect(self._do_login)
        row.addWidget(self._login_btn)
        form.addRow(row)
        return w

    def _build_signup_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        # Admin authorization block
        self._admin_authorized = False
        self._admin_user = QLineEdit(); self._admin_user.setPlaceholderText("admin username")
        self._admin_pass = QLineEdit(); self._admin_pass.setPlaceholderText("admin password"); self._admin_pass.setEchoMode(QLineEdit.Password)
        self._admin_status = QLabel("Administrator authorization required")
        self._admin_status.setStyleSheet("color: #d9534f")
        auth_row = QHBoxLayout()
        auth_row.addWidget(QLabel("Admin"))
        auth_row.addWidget(self._admin_user, 1)
        auth_row.addWidget(self._admin_pass, 1)
        self._btn_auth = QPushButton("Authorize"); self._btn_auth.clicked.connect(self._do_admin_auth)
        auth_row.addWidget(self._btn_auth)
        # add eye for admin password
        eye_admin = QToolButton(); eye_admin.setCheckable(True); eye_admin.setFixedWidth(28)
        eye_admin.setAutoRaise(True)
        eye_admin.setFont(QFont("Segoe UI Emoji", 12))
        eye_admin.setText(chr(0x1F441))
        eye_admin.setToolTip("Show/Hide admin password")
        eye_admin.setStyleSheet("QToolButton{padding:0;margin:0;background:transparent;}")
        eye_admin.clicked.connect(lambda: self._admin_pass.setEchoMode(QLineEdit.Normal if eye_admin.isChecked() else QLineEdit.Password))
        auth_row.addWidget(eye_admin)
        form.addRow(auth_row)
        form.addRow(self._admin_status)

        # New user fields
        self._su_user = QLineEdit(); self._su_user.setPlaceholderText("new username")
        self._su_pass1 = QLineEdit(); self._su_pass1.setEchoMode(QLineEdit.Password); self._su_pass1.setPlaceholderText("new password")
        self._su_pass2 = QLineEdit(); self._su_pass2.setEchoMode(QLineEdit.Password); self._su_pass2.setPlaceholderText("repeat password")
        self._su_role = QComboBox(); self._su_role.addItems(["cualificacion", "baseline", "admin"])  # filtered later
        self._su_admin_hint = QLabel("")
        self._su_admin_hint.setWordWrap(True)
        form.addRow("Username", self._su_user)
        # Password with eyes
        su1_row = QWidget(); su1 = QHBoxLayout(su1_row); su1.setContentsMargins(0,0,0,0); su1.addWidget(self._su_pass1,1)
        eye1 = QToolButton(); eye1.setCheckable(True); eye1.setFixedWidth(28)
        eye1.setAutoRaise(True)
        eye1.setFont(QFont("Segoe UI Emoji", 12))
        eye1.setText(chr(0x1F441))
        eye1.setToolTip("Show/Hide password")
        eye1.setStyleSheet("QToolButton{padding:0;margin:0;background:transparent;}")
        eye1.clicked.connect(lambda: self._su_pass1.setEchoMode(QLineEdit.Normal if eye1.isChecked() else QLineEdit.Password))
        su1.addWidget(eye1)
        su2_row = QWidget(); su2 = QHBoxLayout(su2_row); su2.setContentsMargins(0,0,0,0); su2.addWidget(self._su_pass2,1)
        eye2 = QToolButton(); eye2.setCheckable(True); eye2.setFixedWidth(28)
        eye2.setAutoRaise(True)
        eye2.setFont(QFont("Segoe UI Emoji", 12))
        eye2.setText(chr(0x1F441))
        eye2.setToolTip("Show/Hide password")
        eye2.setStyleSheet("QToolButton{padding:0;margin:0;background:transparent;}")
        eye2.clicked.connect(lambda: self._su_pass2.setEchoMode(QLineEdit.Normal if eye2.isChecked() else QLineEdit.Password))
        su2.addWidget(eye2)
        form.addRow("Password", su1_row)
        form.addRow("Repeat", su2_row)
        form.addRow("Role", self._su_role)
        form.addRow(self._su_admin_hint)

        row = QHBoxLayout()
        row.addStretch(1)
        self._signup_btn = QPushButton("Create User")
        self._signup_btn.clicked.connect(self._do_signup)
        row.addWidget(self._signup_btn)
        form.addRow(row)
        return w

    def _update_signup_role_options(self) -> None:
        try:
            total = count_users(self._db)
        except Exception:
            total = 0
        self._bootstrap = (total == 0)
        # Bootstrap: allow creating first admin without prior authorization
        if self._bootstrap:
            self._admin_status.setText("No users exist. Create the first Administrator.")
            self._admin_status.setStyleSheet("color: #5cb85c")
            self._admin_user.setEnabled(False)
            self._admin_pass.setEnabled(False)
            self._btn_auth.setEnabled(False)
            self._su_role.clear(); self._su_role.addItems(["admin"])  # force admin
            self._su_admin_hint.setText("Bootstrap mode: first user must be 'admin'.")
        else:
            self._su_role.clear(); self._su_role.addItems(["cualificacion", "baseline", "admin"])  # admin allowed by admins
            self._su_admin_hint.setText("Administrator authorization is required to create users.")

    # --------------------------- Actions ------------------------------
    def _do_login(self) -> None:
        uname = self._li_user.text().strip()
        pwd = self._li_pass.text()
        if not uname or not pwd:
            self._login_msg.setText("Enter username and password")
            return
        try:
            u = authenticate(self._db, uname, pwd)
        except Exception as e:
            QMessageBox.critical(self, "Sign In", f"Authentication error: {e}")
            return
        if not u:
            self._login_msg.setText("Invalid username or password")
            return
        self.user = u
        self.accept()

    def _do_signup(self) -> None:
        uname = self._su_user.text().strip()
        p1 = self._su_pass1.text()
        p2 = self._su_pass2.text()
        role = self._su_role.currentText()
        if not self._bootstrap and not self._admin_authorized:
            QMessageBox.warning(self, "Create User", "Administrator authorization required.")
            return
        if not uname or not p1 or not p2:
            QMessageBox.warning(self, "Create User", "Please fill all fields.")
            return
        if p1 != p2:
            QMessageBox.warning(self, "Create User", "Passwords do not match.")
            return
        try:
            u = create_user(self._db, uname, p1, role)
        except Exception as e:
            QMessageBox.critical(self, "Create User", f"Could not create user: {e}")
            return
        QMessageBox.information(self, "Create User", f"User '{u['username']}' created with role '{u['role']}'.")
        # Switch to login and prefill username
        self._li_user.setText(uname)
        self._tabs.setCurrentIndex(0)
        self._su_user.clear(); self._su_pass1.clear(); self._su_pass2.clear()
        self._update_signup_role_options()

    def _do_admin_auth(self) -> None:
        uname = self._admin_user.text().strip()
        pwd = self._admin_pass.text()
        if not uname or not pwd:
            QMessageBox.warning(self, "Authorize", "Enter admin credentials.")
            return
        try:
            u = authenticate(self._db, uname, pwd)
        except Exception as e:
            QMessageBox.critical(self, "Authorize", f"Authentication error: {e}")
            return
        if not u or u.get('role') != 'admin':
            QMessageBox.warning(self, "Authorize", "Administrator credentials required.")
            self._admin_status.setText("Administrator authorization required")
            self._admin_status.setStyleSheet("color: #d9534f")
            self._admin_authorized = False
            return
        self._admin_authorized = True
        self._admin_status.setText(f"Authorized as admin '{u['username']}'")
        self._admin_status.setStyleSheet("color: #5cb85c")
