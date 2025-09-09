# tabs/batch.py
from __future__ import annotations

from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QComboBox,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QFormLayout,
    QGroupBox,
    QScrollArea,
    QInputDialog,
    QLabel,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIntValidator
from PyQt5.QtGui import QFont

from utils import thin_separator
from pgdb import BatchDB


from PyQt5.QtWidgets import QComboBox
class RefreshComboBox(QComboBox):
    def __init__(self, refresher, parent=None):
        super().__init__(parent)
        self._refresher = refresher
    def showPopup(self) -> None:
        # refresh items before displaying
        self._refresher()
        super().showPopup()
class BatchTab(QWidget):
    """UI y lógica de la pestaña **Batch** con backend opcional."""

    def __init__(
        self, com_selector=None, *, db: BatchDB | None = None, parent=None
    ) -> None:
        super().__init__(parent)
        self.db = db or BatchDB()
        self._current_batch_id: str | None = None
        self.attr_edits: dict[str, QLineEdit] = {}
        # Track last suggested bolt number for defaulting in scanner dialog
        self._last_bolt_num: int = 0
        self._build_ui()
        self._populate_batches()
        self._populate_customers()
        # auto-select first batch to enable fields and load its data
        if self.cmb_batches.count() > 0:
            self.cmb_batches.setCurrentIndex(0)
            self._on_batch_selected()

    def _populate_batches(self) -> None:
        # Populate both small and large batch selectors
        items = self.db.list_batches()
        # small combo
        self.cmb_batches.blockSignals(True)
        self.cmb_batches.clear()
        self.cmb_batches.addItems(items)
        self.cmb_batches.blockSignals(False)
        # large display combo
        self.cmb_display_batch.blockSignals(True)
        self.cmb_display_batch.clear()
        self.cmb_display_batch.addItems(items)
        self.cmb_display_batch.blockSignals(False)

    def _populate_customers(self) -> None:
        items = self.db.list_customers()
        self.cmb_customer.blockSignals(True)
        self.cmb_customer.clear()
        self.cmb_customer.addItems(items)
        self.cmb_customer.blockSignals(False)

    # ------------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        # Scrollable area wrapper
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        content = QWidget()
        root = QVBoxLayout(content)
        root.setContentsMargins(20, 16, 20, 16)
        root.setSpacing(14)

        # ── Batch Creator ─────────────────────────────────────────
        creator_gb = QGroupBox("Batch Creator")
        creator_layout = QHBoxLayout(creator_gb)
        # campo para introducir nuevo Batch Num antes de crear
        self.ed_new_batch = QLineEdit()
        self.ed_new_batch.setPlaceholderText("Batch Num")
        self.ed_new_batch.setFixedWidth(120)
        self.ed_new_batch.setValidator(QIntValidator(0, 1000000, self))
        btn_new = QPushButton("Crear")
        btn_new.clicked.connect(self._create_batch)
        creator_layout.addWidget(self.ed_new_batch)
        creator_layout.addWidget(btn_new)
        creator_layout.addStretch()
        root.addWidget(creator_gb)

        # ── Preparo dropdown de batches existentes ─────────────────
        # usar ComboBox que refresca antes de mostrar popup
        self.cmb_batches = RefreshComboBox(self._refresh_batches, parent=self)
        self.cmb_batches.setEditable(False)
        # Al activar una opción (clic o teclado), cargar atributos del batch seleccionado
        self.cmb_batches.activated[str].connect(self._on_batch_selected)
        
        # ── Batch Attributes ─────────────────────────────────────
        attr_gb = QGroupBox("Batch Attributes")
        attr_layout = QVBoxLayout(attr_gb)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        form.setFormAlignment(Qt.AlignTop)
        form.addRow("Batch Num:", self.cmb_batches)
        self.cmb_customer = RefreshComboBox(self._refresh_customers, parent=self)
        self.cmb_customer.setEditable(True)
        form.addRow("Customer:", self.cmb_customer)
        attr_layout.addLayout(form)

        # Completed status + toggle
        status_row = QHBoxLayout()
        self.lbl_completed = QLabel("Completed: ?")
        self.btn_completed = QPushButton("Toggle")
        self.btn_completed.clicked.connect(self._toggle_completed)
        status_row.addWidget(self.lbl_completed)
        status_row.addStretch()
        status_row.addWidget(self.btn_completed)
        attr_layout.addLayout(status_row)

        fastener_gb = QGroupBox("Fastener information")
        fastener_form = QFormLayout(fastener_gb)
        for key, label in [
            ("metric", "Metric:"),
            ("length", "Length:"),
            ("ultrasonic_length", "Ultrasonic length:"),
            ("grade", "Grade:"),
            ("manufacturer", "Manufacturer:"),
            ("customer_part_number", "Customer part number:"),
            ("additional_comment", "Additional comment:"),
        ]:
            self.attr_edits[key] = self._attr_line(fastener_form, label)
        attr_layout.addWidget(fastener_gb)

        joint_gb = QGroupBox("Joint information")
        joint_form = QFormLayout(joint_gb)
        for key, label in [
            ("application_description", "Application description:"),
            ("nut_or_tapped_hole", "Nut or tapped hole:"),
            ("joint_length", "Joint length:"),
            ("max_load", "Maximum load:"),
            ("target_load", "Target load:"),
            ("min_load", "Minimum load:"),
            ("min_temp", "Minimum temp:"),
            ("max_temp", "Maximum temp:"),
        ]:
            self.attr_edits[key] = self._attr_line(joint_form, label)
        attr_layout.addWidget(joint_gb)

        us_gb = QGroupBox("Ultrasonic parameters")
        us_form = QFormLayout(us_gb)
        for key, label in [
            ("frequency", "Frequency:"),
            ("gain", "Gain:"),
            ("cycles_coarse", "Cylces in course:"),
            ("cycles_fine", "Cycles in fine:"),
            ("temperature", "Temperatue:"),
            ("reference_tof", "Reference ToF:"),
            ("temp_gradient", "Temperature gradient:"),
            ("short_temporal_window", "Short temporal window:"),
            ("short_signal_power_first_window", "Short signal power first window:"),
            ("long_temporal_window", "Long temporal window:"),
            ("long_signal_power_first_window", "Long signal power first window:"),
            ("short_correlation_window", "Short correlation window:"),
            ("long_correlation_window", "Long correlation window:"),
            ("temporal_signal_power", "Temporal signal power:"),
            ("correlation_signal_power", "Correlation signal power:"),
            ("xi", "Xi:"),
            ("alpha1", "Alpha 1:"),
            ("alpha2", "Alpha 2:"),
            ("alpha3", "Alpha 3:"),
        ]:
            self.attr_edits[key] = self._attr_line(us_form, label)
        attr_layout.addWidget(us_gb)

        btn_save = QPushButton("Guardar/Actualizar")
        btn_save.clicked.connect(self._save_batch)
        btn_refresh = QPushButton("Recargar")
        btn_refresh.clicked.connect(self._refresh_batches)
        row_widget = QWidget()
        hlayout = QHBoxLayout(row_widget)
        hlayout.addStretch()
        hlayout.addWidget(btn_refresh)
        hlayout.addWidget(btn_save)
        attr_layout.addWidget(row_widget)

        root.addWidget(attr_gb)
        # large batch selector dropdown with label
        self.cmb_display_batch = RefreshComboBox(self._refresh_batches, parent=self)
        self.cmb_display_batch.setEditable(False)
        # make dropdown larger font
        font = self.cmb_display_batch.font()
        font.setPointSize(14)
        font.setBold(True)
        self.cmb_display_batch.setFont(font)
        # when display dropdown changes, trigger batch selection
        self.cmb_display_batch.activated[str].connect(self._on_batch_selected)
        # wrap label and dropdown horizontally
        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        lbl = QLabel("Batch Num:", self)
        row_layout.addWidget(lbl)
        row_layout.addWidget(self.cmb_display_batch, 1)
        row_layout.addStretch()
        root.addWidget(row_widget)
        root.addWidget(thin_separator())

        # ── Bolts in Batch ────────────────────────────────────────
        gb_batch = QGroupBox("Bolts in Batch")
        batch_layout = QVBoxLayout(gb_batch)

        # Scanned Bolts section
        gb_scanned = QGroupBox("Scanned Bolts")
        scanned_layout = QVBoxLayout(gb_scanned)
        # scanner input
        self.ed_scanner = QLineEdit()
        self.ed_scanner.setPlaceholderText("Escanear/introducir código y pulsar Enter...")
        self.ed_scanner.returnPressed.connect(self._add_scanned_bolt)
        scanned_layout.addWidget(self.ed_scanner)
        # list of scanned codes pending assignment
        self.lst_to_add = QListWidget()
        scanned_layout.addWidget(self.lst_to_add, 1)
        # action buttons: assign to batch or erase pending scanned
        hb = QHBoxLayout()
        btn_assign = QPushButton("➕ Bolts to Batch")
        btn_assign.clicked.connect(self._commit_add)
        btn_erase = QPushButton("🗑️ Erase Bolts")
        btn_erase.clicked.connect(self._erase_scanned)
        hb.addWidget(btn_assign)
        hb.addWidget(btn_erase)
        scanned_layout.addLayout(hb)
        batch_layout.addWidget(gb_scanned)

        # Current Bolts in this batch
        # Current Bolts group with dynamic count
        self.gb_current = QGroupBox("Current Bolts")
        current_layout = QVBoxLayout(self.gb_current)
        self.lst_in_batch = QListWidget()
        current_layout.addWidget(self.lst_in_batch)
        batch_layout.addWidget(self.gb_current)
        root.addWidget(gb_batch, 1)

        root.addWidget(thin_separator())

        # ── Barra de acciones ──────────────────────────────────────
        actions = QHBoxLayout()
        btn_add = QPushButton("➕ Añadir seleccionados")
        btn_remove = QPushButton("🗑️ Quitar seleccionados")
        btn_save = QPushButton("💾 Guardar lote")
        btn_add.clicked.connect(self._commit_add)
        btn_remove.clicked.connect(self._remove_selected)
        btn_save.clicked.connect(self._save_batch)

        actions.addStretch()
        for btn in (btn_add, btn_remove, btn_save):
            actions.addWidget(btn)
        root.addLayout(actions)

        self._toggle_inputs(enabled=False)
        # Embed into scroll area
        scroll.setWidget(content)
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(scroll)

    def _attr_line(self, layout: QFormLayout, label: str) -> QLineEdit:
        edit = QLineEdit()
        edit.setFixedWidth(160)
        layout.addRow(label, edit)
        return edit

    def _toggle_inputs(self, *, enabled: bool) -> None:
        widgets = list(self.attr_edits.values()) + [
            self.cmb_customer,
            self.ed_scanner,
            self.lst_in_batch,
            self.lst_to_add,
        ]
        for widget in widgets:
            widget.setEnabled(enabled)

    # ------------------------------------------------------------------
    # Slots y lógica
    # ------------------------------------------------------------------
    def _on_batch_selected(self, *args) -> None:
        # Determine selected batch from args or combos
        if args and isinstance(args[0], str) and args[0]:
            batch_id = args[0]
        else:
            # fallback to display dropdown if used, else small dropdown
            batch_id = self.cmb_display_batch.currentText() or self.cmb_batches.currentText()
        # sync both dropdowns
        self.cmb_batches.blockSignals(True)
        self.cmb_batches.setCurrentText(batch_id)
        self.cmb_batches.blockSignals(False)
        self.cmb_display_batch.blockSignals(True)
        self.cmb_display_batch.setCurrentText(batch_id)
        self.cmb_display_batch.blockSignals(False)
        if not batch_id:
            self._current_batch_id = None
            self._toggle_inputs(enabled=False)
            return

        self._current_batch_id = batch_id
        data = self.db.get_batch(batch_id)
        self._load_attributes(data)
        self._reload_lists()
        self._toggle_inputs(enabled=True)
        # Update completed status
        try:
            completed = self.db.get_batch_completed(self._current_batch_id)
        except Exception:
            completed = False
        self._set_completed_ui(completed)

    def _create_batch(self) -> None:
        # usar texto del campo nuevo batch para crear
        new_id = self.ed_new_batch.text().strip()
        if not new_id:
            QMessageBox.warning(self, "Batch", "Introduzca un identificador")
            return
        try:
            self.db.create_batch(new_id)
        except Exception:
            QMessageBox.warning(self, "Batch", "El identificador ya existe")
            return
        # Repoblar y seleccionar el nuevo batch en ambos combos
        self._populate_batches()
        self.cmb_batches.setCurrentText(new_id)
        self.cmb_display_batch.setCurrentText(new_id)
        # Sincronizar datos y habilitar entradas
        self._on_batch_selected(new_id)
        self.ed_new_batch.clear()

    def _save_batch(self) -> None:
        if not self._current_batch_id:
            return
        attrs = {k: edit.text() for k, edit in self.attr_edits.items()}
        attrs["customer"] = self.cmb_customer.currentText()
        self.db.set_batch_attrs(self._current_batch_id, attrs)
        self._refresh_customers()
        # reload list to reflect any aliases
        self._reload_lists()
        QMessageBox.information(self, "Batch", "Datos guardados")

    def _set_completed_ui(self, completed: bool) -> None:
        text = "Yes" if completed else "No"
        self.lbl_completed.setText(f"Completed: {text}")
        self.btn_completed.setText("Mark as Not Completed" if completed else "Mark as Completed")

    def _toggle_completed(self) -> None:
        if not self._current_batch_id:
            return
        try:
            current = self.db.get_batch_completed(self._current_batch_id)
            self.db.set_batch_completed(self._current_batch_id, not current)
            self._set_completed_ui(not current)
        except Exception as e:
            QMessageBox.warning(self, "Batch", f"Unable to update status: {e}")

    # -------- bolts -----------------------------------------------------
    def _add_scanned_bolt(self) -> None:
        bolt_id = self.ed_scanner.text().strip()
        if not bolt_id:
            return
        # ask user for alias number
        default_num = int(self._last_bolt_num) + 1 if isinstance(self._last_bolt_num, int) else 1
        if default_num < 1:
            default_num = 1
        num, ok = QInputDialog.getInt(
            self,
            "Bolt Num",
            f"Ingrese Bolt Num para {bolt_id}",
            default_num,
            1,
            255,
        )
        if not ok:
            return
        # remember last chosen number to suggest next = previous + 1
        try:
            self._last_bolt_num = int(num)
        except Exception:
            self._last_bolt_num = default_num
        item = QListWidgetItem(f"{bolt_id} ({num})")
        # store tuple in item data
        item.setData(Qt.UserRole, (bolt_id, num))
        self.lst_to_add.addItem(item)
        self.ed_scanner.clear()

    def _commit_add(self) -> None:
        if not self._current_batch_id:
            return
        # add each scanned bolt and alias
        for i in range(self.lst_to_add.count()):
            item = self.lst_to_add.item(i)
            bolt_id, num = item.data(Qt.UserRole)
            # ensure bolt exists
            self.db.add_bolts(self._current_batch_id, [bolt_id])
            # add alias
            self.db.add_bolt_alias(self._current_batch_id, bolt_id, num)
        self.lst_to_add.clear()
        # reload current bolts with aliases
        self._reload_lists()

    def _erase_scanned(self) -> None:
        """Remove selected items from the pending scanned bolts list."""
        for item in self.lst_to_add.selectedItems():
            self.lst_to_add.takeItem(self.lst_to_add.row(item))

    def _remove_selected(self) -> None:
        if not self._current_batch_id:
            return
        # extract bolt_ids without alias suffix
        bolt_ids = []
        for item in self.lst_in_batch.selectedItems():
            text = item.text()
            # bolt_id is before ' (' if alias present
            bolt_id = text.split(' (', 1)[0]
            bolt_ids.append(bolt_id)
        # remove bolts (alias will cascade)
        self.db.remove_bolts(self._current_batch_id, bolt_ids)
        # reload list to reflect removal
        self._reload_lists()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _load_attributes(self, data: dict) -> None:
        attrs = data.get("attrs", {})
        for key, edit in self.attr_edits.items():
            edit.setText(attrs.get(key, ""))
        self.cmb_customer.setCurrentText(attrs.get("customer", ""))

    def _reload_lists(self) -> None:
        # list bolts with alias numbers
        self.lst_in_batch.clear()
        if not self._current_batch_id:
            return
        aliases = self.db.list_bolt_aliases(self._current_batch_id)
        # update group title with count
        count = len(aliases)
        self.gb_current.setTitle(f"Current Bolts ({count})")
        for bolt_id, num in aliases:
            text = f"{bolt_id} ({num})" if num is not None else bolt_id
            self.lst_in_batch.addItem(text)
        # update last bolt number suggestion with current max alias (if any)
        try:
            max_num = max([n for _, n in aliases if n is not None] or [0])
        except Exception:
            max_num = 0
        self._last_bolt_num = int(max_num)

    def _refresh_batches(self) -> None:
        """Repopulate batch list and reload current selection or fallback."""
        prev = self.cmb_batches.currentText()
        self._populate_batches()
        # restore previous if still present, else select first
        items = [self.cmb_batches.itemText(i) for i in range(self.cmb_batches.count())]
        if prev in items:
            self.cmb_batches.setCurrentText(prev)
        elif items:
            self.cmb_batches.setCurrentIndex(0)
        # load attributes for new selection
        self._on_batch_selected()

    def _refresh_customers(self) -> None:
        prev = self.cmb_customer.currentText()
        self._populate_customers()
        if prev:
            self.cmb_customer.setCurrentText(prev)



# Si se ejecuta este archivo directamente, mostrar la pestaña suelta
if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication
    import sys

    app = QApplication(sys.argv)
    win = BatchTab()
    win.setWindowTitle("Batch Tab - Demo")
    win.resize(600, 400)
    win.show()
    sys.exit(app.exec_())
