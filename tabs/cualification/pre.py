# tabs/qualification/pre.py
from __future__ import annotations

from PyQt5.QtWidgets import (
    QLabel,
    QVBoxLayout,
    QWidget,
    QTabWidget,
    QComboBox,
    QMessageBox,
    QScrollArea,
    QHBoxLayout,
    QDoubleSpinBox,
    QPushButton,
    QDialog,
    QLineEdit,
    QFormLayout,
)
from PyQt5.QtGui import QBrush, QColor, QStandardItemModel, QStandardItem
from PyQt5.QtCore import Qt
import pyqtgraph as pg
from tabs.frequency_scan import FrequencyScanTab
from tabs.cualification.one_ten import BoltLineEdit

from tabs.dj import DJTab
from ..filter import FilterTab, FilterWorker
from tabs.csv_viewer import CSVViewerTab
from serial.tools import list_ports
from pgdb import BatchDB


class PreTab(QWidget):
    """Subpestaña **Pre** de la pestaña *Cualificación*."""

    def __init__(self, com_selector=None, db: BatchDB | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        # Base de datos compartida
        self.db = db or BatchDB()
        self.com_selector = com_selector
        layout = QVBoxLayout(self)
        # Selección de Batch Num
        batch_layout = QHBoxLayout()
        batch_layout.addWidget(QLabel("Batch Num:"))
        self.cmb_batch = QComboBox()
        self.cmb_batch.addItems(self.db.list_batches())
        batch_layout.addWidget(self.cmb_batch)
        layout.addLayout(batch_layout)
        # Sub-tabs: Frequency Scan, DJ and Valid Combinations
        tabs = QTabWidget()
        # Pasa selector de batch a la pestaña de frecuencia
        tabs.addTab(PreFrequencyScanTab(com_selector=self.com_selector, db=self.db, batch_selector=self.cmb_batch), "Frequency Scan")
        # DJ sub-tab wrapped in scroll area for proper sizing
        dj_widget = PreDJTab(com_selector=self.com_selector, db=self.db, batch_selector=self.cmb_batch)
        dj_scroll = QScrollArea()
        dj_scroll.setWidgetResizable(True)
        dj_scroll.setWidget(dj_widget)
        tabs.addTab(dj_scroll, "DJ")
        from ..filter import FilterTab  # ensure import
        # Valid combinations with custom threshold filter
        scroll_pre = QScrollArea()
        scroll_pre.setWidgetResizable(True)
        scroll_pre.setWidget(PreFilterTab(com_selector=self.com_selector, db=self.db, batch_selector=self.cmb_batch))
        tabs.addTab(scroll_pre, "Valid Combinations")
        # CSV Viewer for pre-measurements
        tabs.addTab(PreCSVViewerTab(com_selector=self.com_selector, db=self.db, batch_selector=self.cmb_batch), "CSV Viewer")
        layout.addWidget(tabs)


# Custom Frequency Scan for PreTab: add bolt scanner and hide CSV controls
class PreFrequencyScanTab(QWidget):
    def __init__(self, com_selector=None, db: BatchDB | None = None, batch_selector: QComboBox | None = None, parent=None):
        super().__init__(parent)
        self.db = db
        self.batch_selector = batch_selector
        self.com_selector = com_selector
        layout = QVBoxLayout(self)
        # Bolt ID scanner
        self.bolt_edit = BoltLineEdit()
        self.bolt_edit.setPlaceholderText("Escanea o escribe bolt num y pulsa ↩")
        self.bolt_edit.returnPressed.connect(self._on_bolt_scanned)
        layout.addWidget(self.bolt_edit)
        # display scanned Bolt ID above Bolt Num
        self.lbl_bolt_id = QLabel("Bolt ID:")
        self.ed_bolt_id_display = QLineEdit()
        self.ed_bolt_id_display.setReadOnly(True)
        layout.addWidget(self.lbl_bolt_id)
        layout.addWidget(self.ed_bolt_id_display)
        # Store current scan params
        self._cur_freq = None
        self._cur_gain = None
        self._cur_pulse = None
        # Underlying frequency scan widget (hide CSV inputs)
        self.freq_tab = FrequencyScanTab(com_selector=com_selector)
        # Propagate main com_selector to child
        self.freq_tab.com_selector = self.com_selector
        try:
            # hide CSV file selection controls
            self.freq_tab.file_edit.hide()
            self.freq_tab.format_selector.hide()
            self.freq_tab.file_btn.hide()
        except AttributeError:
            pass
        # Bypass file selection: fake a non-empty path and disable save
        try:
            self.freq_tab.file_edit.text = lambda: 'dummy.xlsx'
            self.freq_tab._save_to_excel = lambda rows: None
        except Exception:
            pass
        # Add the frequency scan widget directly
        layout.addWidget(self.freq_tab)
        # Conectar cambio de batch para actualizar ultrasonic_length
        if self.batch_selector:
            self.batch_selector.currentIndexChanged.connect(
                self._load_ul_to_bolt_length
            )
            self._load_ul_to_bolt_length()

        # Override play to inject COM selector, track params and save each frame
        from PyQt5 import QtCore
        def _pre_on_play():
            # inject a dummy combo for COM validation bypass
            port = self.com_selector.currentText() if self.com_selector else ''
            class DummyCombo:
                def __init__(self, p): self._p = p
                def currentText(self): return self._p
                def __bool__(self): return True
            self.freq_tab.com_selector = DummyCombo(port)
            # call original play logic
            # start scan
            FrequencyScanTab._on_play(self.freq_tab)
            # connect param_update to track freq/gain/pulse
            try:
                worker = self.freq_tab._worker
                worker.param_update.connect(self._on_param_update, QtCore.Qt.QueuedConnection)
                # connect saving of each frame to DB
                worker.frame_ready.connect(self._save_frame, QtCore.Qt.QueuedConnection)
            except Exception:
                pass
        try:
            self.freq_tab.btn_play.clicked.disconnect()
            self.freq_tab.btn_play.clicked.connect(_pre_on_play)
        except Exception:
            pass
        # If no com_selector passed, create one from available ports
        if not self.com_selector:
            ports = [p.device for p in list_ports.comports()]
            cb = QComboBox()
            if ports:
                cb.addItems(ports)
                cb.setCurrentIndex(0)
            self.com_selector = cb
        # propagate to child
        self.freq_tab.com_selector = self.com_selector
        # Require bolt id before enabling PLAY: enable when Bolt ID display has text
        self.freq_tab.btn_play.setEnabled(False)
        self.ed_bolt_id_display.textChanged.connect(
            lambda text: self.freq_tab.btn_play.setEnabled(bool(text.strip()))
        )

    def _load_ul_to_bolt_length(self) -> None:
        """Load ultrasonic_length or reference_tof for the selected batch."""
        if not self.batch_selector or not self.db:
            return
        batch_id = self.batch_selector.currentText()
        if not batch_id:
            return
        data = self.db.get_batch(batch_id)
        attrs = data.get("attrs", {})
        ul_val = attrs.get("ultrasonic_length")
        ref_tof = attrs.get("reference_tof")
        if ul_val is not None:
            try:
                ul_val = float(ul_val)
            except Exception:
                ul_val = 0.0
            self.freq_tab.tof_mode.setCurrentIndex(1)  # Calculate
            self.freq_tab.bolt_length_label.show()
            self.freq_tab.bolt_length.show()
            self.freq_tab.bolt_length.setValue(ul_val)
        elif ref_tof is not None:
            try:
                self.freq_tab.tof_mode.setCurrentIndex(0)  # Manual
                self.freq_tab.tof.setValue(float(ref_tof))
            except Exception:
                pass
            self.freq_tab.bolt_length_label.hide()
            self.freq_tab.bolt_length.hide()
        else:
            QMessageBox.warning(
                self,
                "Batch",
                "Ni ultrasonic_length ni reference_tof definidos para el batch",
            )

    def _on_bolt_scanned(self) -> None:
        """On scanning a Bolt ID, auto-select its batch and load params.

        - Finds the associated batch_id for the scanned bolt.
        - Updates the shared batch selector combo above sub-tabs.
        - Sets FrequencyScanTab's db/batch_id so device params come from DB.
        - Sets the bolt alias spinner and ToF/length according to batch.
        """
        bolt_id = self.bolt_edit.text().strip()
        if not bolt_id or not self.batch_selector or not self.db:
            return
        # Show scanned Bolt ID immediately
        self.ed_bolt_id_display.setText(bolt_id)
        # Resolve batch from DB
        batch_id = self.db.find_batch_by_bolt(bolt_id)
        if not batch_id:
            QMessageBox.warning(self, "Bolt", f"Bolt ID {bolt_id} no asociado a ningun batch.")
            return
        # Ensure selector contains it and select
        idx = self.batch_selector.findText(batch_id)
        if idx < 0:
            self.batch_selector.addItem(batch_id)
            idx = self.batch_selector.findText(batch_id)
        if idx >= 0:
            self.batch_selector.setCurrentIndex(idx)
        # Provide DB context to child FrequencyScanTab so it can pull device params
        try:
            self.freq_tab.db = self.db
            self.freq_tab.batch_id = batch_id
        except Exception:
            pass
        # Set alias number in frequency scan (if exists)
        num = self.db.get_bolt_alias(batch_id, bolt_id)
        if num is not None:
            self.freq_tab.bolt.setValue(num)
        else:
            QMessageBox.information(self, "Bolt Alias", f"Bolt {bolt_id} sin alias en batch {batch_id}.")
        # Update ToF/length from batch parameters
        self._load_ul_to_bolt_length()
        # clear input for next scan
        self.bolt_edit.clear()
    
    def _save_frame(self, frame: dict) -> None:
        """Save each measurement frame into pre_measurement table with correct params."""
        if not self.batch_selector or not self.db:
            return
        batch_id = self.batch_selector.currentText()
        # use the displayed Bolt ID (not the cleared input field)
        bolt_id = self.ed_bolt_id_display.text().strip()
        if not batch_id or not bolt_id:
            return
        # Include current freq/gain/pulse into frame
        data = frame.copy()
        # Use tracked parameters if available
        data['freq'] = getattr(self, '_cur_freq', None)
        data['gain'] = getattr(self, '_cur_gain', None)
        data['pulse'] = getattr(self, '_cur_pulse', None)
        try:
            self.db.add_pre_measurement(batch_id, bolt_id, data)
        except Exception as e:
            QMessageBox.warning(self, "DB Error", f"Error guardando medición: {e}")

    def _on_param_update(self, freq: float, gain: float, pulse: float) -> None:
        """Track current frequency, gain and pulse for saving."""
        self._cur_freq = freq
        self._cur_gain = gain
        self._cur_pulse = pulse

    # ------------------------------------------------------------------

# Custom DJ for PreTab: add bolt scanner and hide CSV controls
class PreDJTab(QWidget):
    def __init__(self, com_selector=None, db: BatchDB | None = None, batch_selector: QComboBox | None = None, parent=None):
        super().__init__(parent)
        # Shared DB and batch selector
        self.db = db or BatchDB()
        self.batch_selector = batch_selector
        # Use or create COM selector like FrequencyScan
        self.com_selector = com_selector
        if not self.com_selector:
            ports = [p.device for p in list_ports.comports()]
            cb = QComboBox()
            if ports:
                cb.addItems(ports)
                cb.setCurrentIndex(0)
            self.com_selector = cb
        # Bolt ID scanner
        self.bolt_edit = BoltLineEdit()
        self.bolt_edit.setPlaceholderText("Escanea o escribe bolt num y pulsa ↩")
        # After scanning, set bolt number and update ultrasonic_length
        self.bolt_edit.returnPressed.connect(self._on_bolt_scanned)
        layout = QVBoxLayout(self)
        layout.addWidget(self.bolt_edit)
        # display selected Bolt ID (read-only) above alias spinner
        self.lbl_bolt_id = QLabel("Bolt ID:")
        self.ed_bolt_id_display = QLineEdit()
        self.ed_bolt_id_display.setReadOnly(True)
        layout.addWidget(self.lbl_bolt_id)
        layout.addWidget(self.ed_bolt_id_display)
        # Underlying DJ widget (hide file controls)
        self.dj_tab = DJTab(com_selector=self.com_selector)
        try:
            self.dj_tab.file_edit.hide()
            self.dj_tab.format_selector.hide()
            self.dj_tab.file_btn.hide()
        except AttributeError:
            pass
        # Propagate COM selector
        self.dj_tab.com_selector = self.com_selector
        # Override play: inject dummy combo for COM validation bypass
        def _pre_on_play():
            port = self.com_selector.currentText()
            class DummyCombo:
                def __init__(self, p): self._p = p
                def currentText(self): return self._p
                def __bool__(self): return True
            self.dj_tab.com_selector = DummyCombo(port)
            DJTab._on_play(self.dj_tab)
        try:
            self.dj_tab.btn_play.clicked.disconnect()
        except Exception:
            pass
        self.dj_tab.btn_play.clicked.connect(_pre_on_play)
        # Disable save until a Bolt ID has been scanned and displayed
        self.dj_tab.btn_save.setEnabled(False)
        # Enable save when the displayed Bolt ID changes
        self.ed_bolt_id_display.textChanged.connect(
            lambda text: self.dj_tab.btn_save.setEnabled(bool(text.strip()))
        )
        # Override save: stub DB save
        try:
            self.dj_tab.btn_save.clicked.disconnect()
        except Exception:
            pass
        self.dj_tab.btn_save.clicked.connect(self._on_save)
        # Connect batch change to load ultrasonic_length
        if self.batch_selector:
            self.batch_selector.currentIndexChanged.connect(
                self._load_ul_to_bolt_length
            )
            self._load_ul_to_bolt_length()
        # Wrap the DJ widget in a scroll area for proper sizing
        scroll_dj = QScrollArea()
        scroll_dj.setWidgetResizable(True)
        scroll_dj.setWidget(self.dj_tab)
        layout.addWidget(scroll_dj)

    def _load_ul_to_bolt_length(self):
        """Load ultrasonic_length or reference_tof for the selected batch."""
        if not self.batch_selector or not self.db:
            return
        batch_id = self.batch_selector.currentText()
        if not batch_id:
            return
        data = self.db.get_batch(batch_id)
        attrs = data.get("attrs", {})
        ul_val = attrs.get("ultrasonic_length")
        ref_tof = attrs.get("reference_tof")
        if ul_val is not None:
            try:
                ul_val = float(ul_val)
            except Exception:
                ul_val = 0.0
            self.dj_tab.tof_mode.setCurrentIndex(1)  # Calculate
            self.dj_tab.bolt_length_label.show()
            self.dj_tab.bolt_length.show()
            self.dj_tab.bolt_length.setValue(ul_val)
        elif ref_tof is not None:
            try:
                self.dj_tab.tof_mode.setCurrentIndex(0)  # Manual
                self.dj_tab.tof.setValue(float(ref_tof))
            except Exception:
                pass
            self.dj_tab.bolt_length_label.hide()
            self.dj_tab.bolt_length.hide()
        else:
            QMessageBox.warning(
                self,
                "Batch",
                "Ni ultrasonic_length ni reference_tof definidos para el batch",
            )

    def _on_bolt_scanned(self):
        """On scanning a Bolt ID, auto-select its batch and load params.

        - Finds the associated batch_id for the scanned bolt.
        - Updates the shared batch selector combo above sub-tabs.
        - Sets DJTab's db/batch_id so device params come from DB.
        - Sets the bolt alias spinner and ToF/length according to batch.
        """
        bolt_id = self.bolt_edit.text().strip()
        if not bolt_id or not self.batch_selector or not self.db:
            return
        # Show scanned Bolt ID
        self.ed_bolt_id_display.setText(bolt_id)
        # Resolve batch from DB
        batch_id = self.db.find_batch_by_bolt(bolt_id)
        if not batch_id:
            QMessageBox.warning(self, "Bolt", f"Bolt ID {bolt_id} no asociado a ningun batch.")
            return
        # Ensure selector contains it and select
        idx = self.batch_selector.findText(batch_id)
        if idx < 0:
            self.batch_selector.addItem(batch_id)
            idx = self.batch_selector.findText(batch_id)
        if idx >= 0:
            self.batch_selector.setCurrentIndex(idx)
        # Provide DB context to child DJTab so it can pull device params
        try:
            self.dj_tab.db = self.db
            self.dj_tab.batch_id = batch_id
        except Exception:
            pass
        # Set alias number in DJTab (if exists)
        num = self.db.get_bolt_alias(batch_id, bolt_id)
        if num is not None:
            self.dj_tab.bolt.setValue(num)
        else:
            QMessageBox.information(self, "Bolt Alias", f"Bolt {bolt_id} sin alias en batch {batch_id}.")
        # Update ToF/length from batch parameters
        self._load_ul_to_bolt_length()
        # clear input for next scan
        self.bolt_edit.clear()

    def _on_save(self):
        """Save the last DJ frame measurement into pre_measurement table using the displayed Bolt ID."""
        batch_id = self.batch_selector.currentText() if self.batch_selector else ''
        # Use the displayed Bolt ID, not the input field which is cleared after scanning
        bolt_id = self.ed_bolt_id_display.text().strip()
        last_frame = getattr(self.dj_tab, 'last_frame', None)
        if not batch_id:
            QMessageBox.warning(self, "Batch ID", "Selecciona un Batch antes de guardar.")
            return
        if not bolt_id:
            QMessageBox.warning(self, "Bolt ID", "Selecciona o escanea un Bolt ID antes de guardar.")
            return
        if last_frame is None:
            QMessageBox.warning(self, "Sin datos", "Aún no se ha recibido ningún frame para guardar.")
            return
        # Prepare data dict including scan params
        data = last_frame.copy()
        data['freq'] = float(self.dj_tab.freq.value())
        data['gain'] = float(self.dj_tab.gain.value())
        data['pulse'] = float(self.dj_tab.pulse.value())
        try:
            self.db.add_pre_measurement(batch_id, bolt_id, data)
            QMessageBox.information(self, "Guardado", "Medición DJ guardada en la base de datos.")
        except Exception as e:
            QMessageBox.warning(self, "DB Error", f"Error guardando medición DJ: {e}")

    # ------------------------------------------------------------------

# Custom Filter Tab with threshold input for pre qualification
class PreFilterTab(FilterTab):
    def __init__(self, com_selector=None, db: BatchDB | None = None, batch_selector: QComboBox | None = None, parent=None):
        super().__init__(com_selector=com_selector, parent=parent)
        # Shared DB and batch selector
        self.db = db or BatchDB()
        self.batch_selector = batch_selector
        # Data source selector
        src_layout = QHBoxLayout()
        src_layout.addWidget(QLabel("Source:"))
        self.src_selector = QComboBox()
        self.src_selector.addItems(["File", "Database"])
        # default to Database
        self.src_selector.setCurrentIndex(1)
        src_layout.addWidget(self.src_selector)
        self.layout().insertLayout(0, src_layout)
        # Load from DB button
        self.btn_load_db = QPushButton("Load from DB")
        self.btn_load_db.clicked.connect(self._filter_from_db)
        # Style Load from DB button
        self.btn_load_db.setStyleSheet("background-color: #3498db; border: 1px solid #2980b9; color: white;")
        self.layout().insertWidget(1, self.btn_load_db)
        # React to source change
        self.src_selector.currentIndexChanged.connect(lambda i: self._on_source_changed())
        # initialize view
        self._on_source_changed()
        # Ensure Start button triggers correct action per source
        # initial wiring done in _on_source_changed

        # Threshold control for valid combos
        thresh_layout = QHBoxLayout()
        thresh_layout.addWidget(QLabel("Min Valid % (pct & pico):"))
        self.valid_spin = QDoubleSpinBox()
        self.valid_spin.setRange(0.0, 100.0)
        self.valid_spin.setDecimals(0)
        self.valid_spin.setValue(100.0)
        thresh_layout.addWidget(self.valid_spin)
        # initial toggle
        self._on_source_changed()

        # Insert threshold control above peak checkbox
        self.layout().insertLayout(3, thresh_layout)
        # Connect threshold changes to re-colors table
        self.valid_spin.valueChanged.connect(self._apply_threshold)

        # Hide controls not required in Pre tab
        # (start filter button should remain visible)
        # self.btn_start.hide()
        self._hide_layout_with_widget(self.bolt_spin)  # hides bolt num & scan
        self._hide_layout_with_widget(self.temp_spin)  # hides device cfg row

        # Add button to store selections in DB
        self.btn_add_db = QPushButton("Add to DB")
        self.btn_add_db.clicked.connect(self._on_add_db)
        # Style Add to DB button
        self.btn_add_db.setStyleSheet("background-color: #e67e22; border: 1px solid #d35400; color: white;")
        self.layout().addWidget(self.btn_add_db)

        # Automatically launch filtering when file selected
        try:
            self.btn_open.clicked.disconnect()
        except Exception:
            pass
        self.btn_open.clicked.connect(self._choose_and_filter)
        # Style Open button
        self.btn_open.setStyleSheet("background-color: #95a5a6; border: 1px solid #7f8c8d; color: white;")
        # Track currently selected 'best' combination
        self.best_combo: tuple[float, float, float] | None = None
        self.best_combo_index: int | None = None
        self._table_df = None  # store table DataFrame for lookup
        # Store dat2/dat3 arrays per combo and bolt
        self.bolt_signals: dict[
            tuple[int, int, int], dict[int, dict[str, list]]
        ] = {}
        # Selector for which data type to plot
        data_layout = QHBoxLayout()
        data_layout.addWidget(QLabel("Plot data:"))
        self.data_selector = QComboBox()
        self.data_selector.addItems(["dat2", "dat3"])
        self.data_selector.setCurrentIndex(1)
        data_layout.addWidget(self.data_selector)
        idx_table = self.layout().indexOf(self.table)
        if idx_table < 0:
            self.layout().addLayout(data_layout)
        else:
            self.layout().insertLayout(idx_table, data_layout)
        # Open signal plot on table cell click
        self.table.clicked.connect(self._on_table_clicked)

    def _apply_threshold(self):
        threshold = self.valid_spin.value() / 100.0
        model = self._table_model
        if not model:
            return
        # Get header names
        headers = [model.headerData(c, Qt.Horizontal) for c in range(model.columnCount())]
        # Determine columns
        cols = [i for i, col in enumerate(headers) if col in ('pct_ok_frac', 'pico1_ok_frac')]
        for c in cols:
            for r in range(model.rowCount()):
                item = model.item(r, c)
                try:
                    val = float(item.text().strip('%'))/100.0
                except Exception:
                    continue
                # Color green if meets threshold, else clear
                if val >= threshold:
                    item.setBackground(QBrush(QColor('green')))
                else:
                    item.setBackground(QBrush())

    def _on_source_changed(self):
        """Toggle UI elements depending on source selection."""
        file_mode = (self.src_selector.currentText() == "File")
        # show/hide file controls
        self.btn_open.setVisible(file_mode)
        # show/hide DB load button
        self.btn_load_db.setVisible(not file_mode)
        # Configure Start button
        try:
            self.btn_start.clicked.disconnect()
        except Exception:
            pass
        if file_mode:
            # file mode: Start triggers file filter
            self.btn_start.setEnabled(bool(self.selected_path))
            self.btn_start.clicked.connect(self._start_filter)
        else:
            # DB mode: Start triggers DB filter
            self.btn_start.setEnabled(True)
            self.btn_start.clicked.connect(self._filter_from_db)

    def _filter_from_db(self):
        """Load measurements from DB and apply valid-combo filtering."""
        import pandas as pd, numpy as np
        if not self.batch_selector:
            return
        batch_id = self.batch_selector.currentText()
        if not batch_id:
            QMessageBox.warning(self, "Batch ID", "Selecciona un Batch antes de cargar.")
            return
        # Fetch data from pre_measurement (latest per bolt+combo)
        try:
            # Raw query: keep only most recent measurement by measured_at
            try:
                self.db.cur.execute(
                    """
                    SELECT DISTINCT ON (bolt_id, freq, gain, pulse)
                           freq, gain, pulse, bolt_id, pico1, pct_diff, dat2, dat3
                    FROM pre_measurement
                    WHERE batch_id=%s
                    ORDER BY bolt_id, freq, gain, pulse, measured_at DESC
                    """,
                    (batch_id,)
                )
            except Exception:
                self.db.cur.execute(
                    "SELECT freq, gain, pulse, bolt_id, pico1, pct_diff, dat2, dat3 FROM pre_measurement WHERE batch_id=%s",
                    (batch_id,)
                )
            rows = self.db.cur.fetchall()
            cols = [desc[0] for desc in self.db.cur.description]
            df = pd.DataFrame(rows, columns=cols)
            # Preserve original Bolt ID and compute numeric alias Bolt Num
            df.rename(columns={'bolt_id': 'Bolt ID'}, inplace=True)
            batch_id = self.batch_selector.currentText()
            df['Bolt Num'] = df['Bolt ID'].apply(lambda bid: self.db.get_bolt_alias(batch_id, bid))
            # Expand raw byte arrays to signal columns dat3_*, dat2_*
            try:
                # Convert BYTEA to numpy arrays with correct int dtype
                def _bytes_to_int16_array(b):
                    if b is None:
                        return np.array([], dtype=np.int16)
                    raw = b.tobytes() if hasattr(b, 'tobytes') else bytes(b)
                    return np.frombuffer(raw, dtype='<i2')
                def _bytes_to_int32_array(b):
                    if b is None:
                        return np.array([], dtype=np.int32)
                    raw = b.tobytes() if hasattr(b, 'tobytes') else bytes(b)
                    return np.frombuffer(raw, dtype='<i4')
                # Apply conversion
                dat2_arrays = df['dat2'].apply(_bytes_to_int16_array)
                dat3_arrays = df['dat3'].apply(_bytes_to_int32_array)
                # Determine signal lengths (expected 1024 samples)
                n2 = int(dat2_arrays.iloc[0].size) if not dat2_arrays.empty else 1024
                n3 = int(dat3_arrays.iloc[0].size) if not dat3_arrays.empty else 1024
                # Pad or truncate arrays to consistent length and convert to lists
                dat2_list = [(arr.tolist() + [0]*(n2 - arr.size))[:n2] for arr in dat2_arrays]
                dat3_list = [(arr.tolist() + [0]*(n3 - arr.size))[:n3] for arr in dat3_arrays]
                # Build DataFrames for signal columns
                dat2_df = pd.DataFrame(dat2_list, columns=[f'dat2_{i}' for i in range(n2)])
                dat3_df = pd.DataFrame(dat3_list, columns=[f'dat3_{i}' for i in range(n3)])
                # Combine with main DataFrame, dropping raw BYTEA columns
                df = pd.concat([df.drop(columns=['dat2', 'dat3']), dat3_df, dat2_df], axis=1)
            except Exception as e:
                QMessageBox.warning(self, "DB Error", f"Error procesando señales: {e}")
                return
            # Ensure pct_diff fraction if stored as fraction
            # store full data for _on_finished
            self._db_full_df = df.copy()
        except Exception as e:
            QMessageBox.critical(self, "DB Error", f"No se pudo cargar de DB: {e}")
            return
        # Now use existing filtering logic: emulate _on_finished
        # prepare debug_df
        total_bolts = df.groupby(['freq', 'gain', 'pulse'])['Bolt Num'].nunique().rename('total_bolts')
        df['pct_diff_num'] = df['pct_diff'] * 100 if df['pct_diff'].max() <= 1 else df['pct_diff']
        df_pct = df[df['pct_diff_num'] >= float(self.pct_spin.value())]
        if self.peak_chk.isChecked():
            df_core = df_pct[df_pct['pico1'] >= float(self.peak_spin.value())]
        else:
            df_core = df_pct
        valid_bolts = df_core.groupby(['freq', 'gain', 'pulse'])['Bolt Num'].nunique().rename('valid_bolts')
        debug_df = pd.concat([total_bolts, valid_bolts], axis=1).reset_index().fillna(0)
        combos_ok = debug_df.loc[debug_df['total_bolts'] == debug_df['valid_bolts'], ['freq', 'gain', 'pulse']]
        # filtered rows
        df_final = df_core.set_index(['freq', 'gain', 'pulse']).loc[list(combos_ok.itertuples(index=False, name=None))].reset_index()
        # adapt column names to expected
        df_final.rename(columns={'freq': 'Freq', 'gain': 'Gain', 'pulse': 'Pulse'}, inplace=True)
        # Rename debug_df columns to match keys used in _on_finished
        debug_df.rename(columns={'freq': 'Freq', 'gain': 'Gain', 'pulse': 'Pulse'}, inplace=True)
        # call existing finish handler
        self._on_finished(df_final, debug_df, "")

    # ------------------------------------------------------------------
    def _hide_layout_with_widget(self, widget):
        """Hide the layout that contains the given widget (labels included)."""
        lay = None
        main = self.layout()
        for i in range(main.count()):
            item = main.itemAt(i)
            l = item.layout()
            if l:
                for j in range(l.count()):
                    w = l.itemAt(j).widget()
                    if w is widget:
                        lay = l
                        break
            if lay:
                break
        if lay:
            for j in range(lay.count()):
                w = lay.itemAt(j).widget()
                if w:
                    w.hide()

    # ------------------------------------------------------------------
    def _choose_and_filter(self):
        """Open file dialog and automatically run filtering."""
        super()._choose_file()
        if self.selected_path:
            self._start_filter()

    # ------------------------------------------------------------------
    def _on_add_db(self):
        """Save the valid combinations into pre_valid_combo table."""
        if not self.batch_selector:
            return
        batch_id = self.batch_selector.currentText()
        if not batch_id:
            QMessageBox.warning(self, "Batch ID", "Selecciona un Batch antes de guardar.")
            return
        # Save only selected combos, marking the UI-chosen best
        model = self._table_model
        if model is None:
            QMessageBox.warning(self, "Sin datos", "No hay combinaciones para guardar.")
            return
        try:
            # Clear existing combos for this batch
            self.db.cur.execute(
                "DELETE FROM pre_valid_combo WHERE batch_id=%s",
                (batch_id,)
            )
            count = 0
            # Iterate table rows
            for row_idx in range(model.rowCount()):
                # Select checkbox is in column 0
                sel_item = model.item(row_idx, 0)
                if sel_item.checkState() != Qt.Checked:
                    continue
                # Read values from stored DataFrame
                combo = self._table_df.loc[row_idx]
                freq = int(combo['Freq'])
                gain = int(combo['Gain'])
                pulse = int(combo['Pulse'])
                # Best checkbox is in column 1
                best_state = model.item(row_idx, 1).checkState() == Qt.Checked
                # Insert
                self.db.cur.execute(
                    "INSERT INTO pre_valid_combo (batch_id, freq, gain, pulse, is_best) VALUES (%s, %s, %s, %s, %s)",
                    (batch_id, freq, gain, pulse, best_state)
                )

                # Update batch parameters only if this is the best combo
                if best_state:
                    self.db.cur.execute(
                        "UPDATE batch SET frequency=%s, gain=%s, cycles_coarse=%s, cycles_fine=%s WHERE batch_id=%s",
                        (freq, gain, pulse, pulse, batch_id)
                    )

                count += 1
            QMessageBox.information(self, "Guardado", f"{count} combinaciones guardadas en la base de datos.")
        except Exception as e:
            QMessageBox.warning(self, "DB Error", f"Error guardando combinaciones: {e}")

    # ------------------------------------------------------------------
    def _on_finished(self, df_filtered, df_debug, out_path):
        """Override to include valid combination flag and row preselection."""
        pct_thr = float(self.pct_spin.value())
        use_peak = self.peak_chk.isChecked()
        peak_thr = float(self.peak_spin.value())
        valid_thr = self.valid_spin.value() / 100.0
        self.current_pct_thr = pct_thr
        self.current_peak_used = use_peak
        self.current_peak_thr = peak_thr

        import pandas as __pd
        import numpy as __np
        from pathlib import Path
        # get full dataset: DB mode or file
        if hasattr(self, '_db_full_df'):
            df_full = self._db_full_df.copy()
            # ensure pct_diff numeric
            df_full['pct_diff_num'] = FilterWorker._pct_to_float(df_full['pct_diff'])
            df_full['pct_diff'] = df_full['pct_diff_num'] / 100
            # rename DB fields to match expected pivot keys
            df_full.rename(columns={'freq':'Freq', 'gain':'Gain', 'pulse':'Pulse'}, inplace=True)
            # drop duplicate measurements per combination to allow pivot
            df_full = df_full.drop_duplicates(subset=['Freq', 'Gain', 'Pulse', 'Bolt Num'], keep='first')
        else:
            # file mode: read from selected path
            ext = Path(self.selected_path or '').suffix.lower()
            if ext == '.csv':
                df_full = __pd.read_csv(self.selected_path)
            else:
                df_full = __pd.read_excel(self.selected_path)
            df_full['pct_diff_num'] = FilterWorker._pct_to_float(df_full['pct_diff'])
            df_full['pct_diff'] = df_full['pct_diff_num'] / 100

        pivot_pct = df_full.pivot(index=['Freq', 'Gain', 'Pulse'], columns='Bolt Num', values='pct_diff')
        bolts = list(pivot_pct.columns)
        frac_pct = (pivot_pct >= pct_thr/100).sum(axis=1) / len(bolts)
        if use_peak:
            pivot_peak = df_full.pivot(index=['Freq', 'Gain', 'Pulse'], columns='Bolt Num', values='pico1')
            frac_peak = (pivot_peak >= peak_thr).sum(axis=1) / len(bolts)

        df_res = __pd.DataFrame(index=pivot_pct.index)
        df_res['pct_ok_frac'] = frac_pct
        if use_peak:
            df_res['pico1_ok_frac'] = frac_peak
        for b in bolts:
            df_res[f'pct_{b}'] = pivot_pct[b]
            if use_peak:
                df_res[f'pico1_{b}'] = pivot_peak[b]

        table_df = df_res.reset_index()
        all_combos = df_debug[['Freq', 'Gain', 'Pulse']].drop_duplicates()
        table_df = all_combos.merge(table_df, on=['Freq', 'Gain', 'Pulse'], how='left')

        # Store dat2/dat3 arrays and pico1 for each combo and bolt
        self.bolt_signals.clear()
        dat2_cols = [c for c in df_full.columns if c.startswith('dat2_')]
        dat3_cols = [c for c in df_full.columns if c.startswith('dat3_')]
        for _, row in df_full.iterrows():
            combo_key = (int(row['Freq']), int(row['Gain']), int(row['Pulse']))
            bolt = int(row['Bolt Num'])
            sigs = self.bolt_signals.setdefault(combo_key, {})
            sigs[bolt] = {
                'dat2': row[dat2_cols].to_list(),
                'dat3': row[dat3_cols].to_list(),
                'pico1': float(row.get('pico1')) if row.get('pico1') is not None else None,
            }

        if use_peak:
            perfect = (table_df['pct_ok_frac'] == 1.0) & (table_df['pico1_ok_frac'] == 1.0)
            table_df['perfect'] = perfect
            table_df = table_df.sort_values(
                by=['perfect', 'pct_ok_frac', 'pico1_ok_frac'],
                ascending=[False, False, False]
            ).drop(columns=['perfect'])
        else:
            perfect = table_df['pct_ok_frac'] == 1.0
            table_df['perfect'] = perfect
            table_df = table_df.sort_values(
                by=['perfect', 'pct_ok_frac'],
                ascending=[False, False]
            ).drop(columns=['perfect'])

        # Add a valid percentage column to display combo success rates
        if use_peak:
            table_df['valid_pct'] = __np.minimum(table_df['pct_ok_frac'], table_df['pico1_ok_frac'])
        else:
            table_df['valid_pct'] = table_df['pct_ok_frac']
        # Mark rows meeting the threshold
        valid_cond = table_df['pct_ok_frac'] >= valid_thr
        if use_peak:
            valid_cond &= table_df['pico1_ok_frac'] >= valid_thr
        table_df['valid_combination'] = valid_cond

        self.valid_combos_df = table_df.loc[table_df['valid_combination'], ['Freq', 'Gain', 'Pulse']].reset_index(drop=True)
        self._load_dataframe_into_table(table_df)

        if out_path:
            QMessageBox.information(self, "Export", f"Archivo creado:\n{out_path}\n\nFilas exportadas: {len(df_filtered)}")

        self.btn_open.setEnabled(True)
        self.btn_start.setEnabled(True)

        self._apply_threshold()
        # initial checkbox states set in loading table

    # ------------------------------------------------------------------
    # Override to load table with selectable checkbox column
    def _load_dataframe_into_table(self, df):
        # First load all data and coloring using parent logic
        df = df.reset_index(drop=True)
        self._table_df = df
        display_df = df.drop(columns=['valid_pct', 'valid_combination'], errors='ignore')
        super()._load_dataframe_into_table(display_df)
        model = self._table_model
        # Insert 'Select' and 'Best' columns at the start
        model.insertColumn(0)
        model.insertColumn(1)
        model.setHeaderData(0, Qt.Horizontal, "Select")
        model.setHeaderData(1, Qt.Horizontal, "Best")
        # Determine best default row by highest valid_pct
        try:
            best_idx = df['valid_pct'].idxmax()
            self.best_combo_index = int(best_idx)
            self.best_combo = tuple(df.loc[best_idx, ['Freq', 'Gain', 'Pulse']])
        except Exception:
            best_idx = -1
            self.best_combo_index = None
            self.best_combo = None
        # Populate checkbox items
        for row in range(model.rowCount()):
            # Select column
            sel_item = QStandardItem()
            sel_item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            sel_state = Qt.Checked if df.iloc[row].get('valid_combination', False) else Qt.Unchecked
            sel_item.setCheckState(sel_state)
            model.setItem(row, 0, sel_item)
            # Best column (single checkbox)
            best_item = QStandardItem()
            best_item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            best_item.setCheckState(Qt.Checked if row == best_idx else Qt.Unchecked)
            model.setItem(row, 1, best_item)
        # Apply updated model and connect for exclusive Best selection
        self.table.setModel(model)
        model.itemChanged.connect(self._on_best_toggled)

    def _on_table_clicked(self, index):
        """Handle cell click to plot selected dat2/dat3 signal."""
        if not index.isValid() or self._table_df is None:
            return
        header = self._table_model.headerData(index.column(), Qt.Horizontal)
        if not isinstance(header, str):
            return
        if header.startswith('pct_') or header.startswith('pico1_'):
            if header.endswith('_frac'):
                return
            try:
                bolt = int(header.split('_')[1])
            except Exception:
                return
            row = index.row()
            if row >= len(self._table_df):
                return
            combo = tuple(int(self._table_df.loc[row, c]) for c in ['Freq', 'Gain', 'Pulse'])
            dtype = self.data_selector.currentText()
            self._open_plot(combo, bolt, dtype)

    def _open_plot(self, combo: tuple[int, int, int], bolt: int, dtype: str) -> None:
        """Show dat2 or dat3 signal for the given combo and bolt."""
        data = (self.bolt_signals.get(combo, {})
                .get(bolt, {})
                .get(dtype))
        if not data:
            QMessageBox.warning(self, "Data", "Signal not available for this selection")
            return
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Bolt {bolt} - {dtype} F{combo[0]} G{combo[1]} P{combo[2]}")
        lay = QVBoxLayout(dlg)
        plot = pg.PlotWidget()
        plot.showGrid(x=True, y=True, alpha=0.2)
        plot.setLabel('bottom', 'Samples')
        plot.setLabel('left', 'Amplitude')
        x_vals = list(range(len(data)))
        curve = plot.plot(x_vals, data, pen=pg.mkPen('#00c853', width=2), name='Signal')

        # Legend with readable text on dark theme
        try:
            legend = plot.addLegend(offset=(-10, 10), labelTextColor='#e6e6e6')
            legend.setBrush(pg.mkBrush(30, 30, 30, 200))
            legend.setPen(pg.mkPen('#9e9e9e'))
        except Exception:
            legend = None

        # Legend: Max amplitude (from DB pico1) and Cursor value rows
        max_label_item = None
        cursor_label_item = None
        try:
            # Use pico1 from the corresponding DB row for this combo+bolt
            pico1_val = (self.bolt_signals.get(combo, {})
                         .get(bolt, {})
                         .get('pico1'))
            max_amp = float(pico1_val) if pico1_val is not None else float('nan')
            sample_max = pg.PlotDataItem([], [])
            sample_max.setPen(pg.mkPen(0, 0, 0, 0))
            legend.addItem(sample_max, f"Max amplitude: {max_amp:.3f}")
            max_label_item = legend.items[-1][1] if getattr(legend, 'items', None) else None

            sample_cur = pg.PlotDataItem([], [])
            sample_cur.setPen(pg.mkPen(0, 0, 0, 0))
            legend.addItem(sample_cur, "Cursor: -")
            cursor_label_item = legend.items[-1][1] if getattr(legend, 'items', None) else None
        except Exception:
            pass

        # Hover helpers: vertical line and text label showing amplitude
        vline = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('#00c853', width=1))
        plot.addItem(vline); vline.setVisible(False)
        label = pg.TextItem(color='#e6e6e6', anchor=(0, 1))
        plot.addItem(label); label.setVisible(False)

        vb = plot.getPlotItem().getViewBox()
        scene = plot.getPlotItem().scene()

        def _on_move(pos):
            try:
                if not plot.sceneBoundingRect().contains(pos):
                    vline.setVisible(False)
                    label.setVisible(False)
                    return
                mouse_pt = vb.mapSceneToView(pos)
                x = int(round(mouse_pt.x()))
                if x < 0 or x >= len(x_vals):
                    vline.setVisible(False)
                    label.setVisible(False)
                    return
                amp = float(data[x])
                vline.setPos(x)
                label.setText(f"Amplitude: {amp:.3f}")
                label.setPos(x, amp)
                vline.setVisible(True)
                label.setVisible(True)
                # Update cursor legend row
                try:
                    if cursor_label_item is not None:
                        cursor_label_item.setText(f"Cursor: {amp:.3f}")
                except Exception:
                    pass
            except Exception:
                pass

        try:
            scene.sigMouseMoved.connect(_on_move)
        except Exception:
            pass
        # Disable mouse zoom/pan on dialog plot
        try:
            _vb = plot.getPlotItem().getViewBox()
            _vb.setMouseEnabled(False, False)
            _vb.setMenuEnabled(False)
            plot.wheelEvent = lambda ev: None
            _vb.wheelEvent = lambda ev: None
            _vb.mouseDoubleClickEvent = lambda ev: None
            _vb.mouseDragEvent = lambda ev: None
        except Exception:
            pass
        # --- Controls row (Fullscreen toggle) ---
        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 0, 0, 0)
        fs_btn = QPushButton("Full Screen")
        btn_row.addStretch(1)
        btn_row.addWidget(fs_btn)
        btn_row.addStretch(1)
        lay.addLayout(btn_row)
        lay.addWidget(plot)

        # Toggle fullscreen handler
        state = { 'fs': False }
        def _toggle_full():
            try:
                if not state['fs']:
                    dlg.showFullScreen()
                    fs_btn.setText("Exit Full Screen")
                    state['fs'] = True
                else:
                    dlg.showNormal()
                    fs_btn.setText("Full Screen")
                    state['fs'] = False
            except Exception:
                pass
        fs_btn.clicked.connect(_toggle_full)

        # ESC exits fullscreen
        old_keypress = getattr(dlg, 'keyPressEvent', None)
        def _kp(ev):
            try:
                if ev.key() == Qt.Key_Escape and state['fs']:
                    _toggle_full(); return
            except Exception:
                pass
            if callable(old_keypress):
                old_keypress(ev)
        try:
            dlg.keyPressEvent = _kp  # type: ignore[assignment]
        except Exception:
            pass
        # Make dialog resizable
        try:
            dlg.setSizeGripEnabled(True)
        except Exception:
            pass
        dlg.resize(600, 400)
        dlg.exec_()

    def _on_best_toggled(self, item: QStandardItem):
        # Enforce exclusive single choice for 'Best' column
        if item.column() != 1:
            return
        model = self._table_model
        # Prevent recursion
        model.blockSignals(True)
        row_count = model.rowCount()
        if item.checkState() == Qt.Checked:
            # Uncheck all others
            for r in range(row_count):
                if r != item.row():
                    other = model.item(r, 1)
                    if other.checkState() == Qt.Checked:
                        other.setCheckState(Qt.Unchecked)
            # Update best reference
            self.best_combo_index = item.row()
            if self._table_df is not None and 0 <= item.row() < len(self._table_df):
                row = self._table_df.loc[item.row()]
                self.best_combo = tuple(row[col] for col in ['Freq', 'Gain', 'Pulse'])
            else:
                self.best_combo = None
        else:
            # If unchecking leaves none checked, re-check this one
            any_checked = False
            for r in range(row_count):
                if model.item(r, 1).checkState() == Qt.Checked:
                    any_checked = True
                    # update best reference
                    if self._table_df is not None and 0 <= r < len(self._table_df):
                        row = self._table_df.loc[r]
                        self.best_combo = tuple(row[col] for col in ['Freq', 'Gain', 'Pulse'])
                        self.best_combo_index = r
                    break
            if not any_checked:
                item.setCheckState(Qt.Checked)
        model.blockSignals(False)

    # ------------------------------------------------------------------
    def get_best_combo(self):
        """Return the (Freq, Gain, Pulse) tuple of the selected best combo."""
        return self.best_combo


# New CSV Viewer sub-tab for Pre
class PreCSVViewerTab(QWidget):
    """Visor de mediciones CSV o desde pre_measurement en BBDD."""
    def __init__(self, com_selector=None, db: BatchDB=None, batch_selector=None, parent=None):
        super().__init__(parent)
        self.db = db
        self.batch_selector = batch_selector
        self.viewer = CSVViewerTab(db=db, batch_selector=batch_selector)
        # Hook table updates to strip file prefixes when showing DB data
        self._orig_update_table = self.viewer._update_table
        def _update_table_wrapper():
            self._orig_update_table()
            self._clean_headers()
        self.viewer._update_table = _update_table_wrapper
        layout = QVBoxLayout(self)
        # Source selector
        src_layout = QHBoxLayout()
        src_layout.addWidget(QLabel("Source:"))
        self.src_selector = QComboBox()
        self.src_selector.addItems(["File", "Database"])
        # default to Database mode
        self.src_selector.setCurrentIndex(1)
        src_layout.addWidget(self.src_selector)
        # Load from DB button
        self.load_db_btn = QPushButton("Load from DB")
        self.load_db_btn.clicked.connect(self._load_from_db)
        # Style Load from DB button
        self.load_db_btn.setStyleSheet("background-color: #3498db; border: 1px solid #2980b9; color: white;")
        src_layout.addWidget(self.load_db_btn)
        layout.addLayout(src_layout)
        # Add viewer
        layout.addWidget(self.viewer)
        self.src_selector.currentIndexChanged.connect(self._on_source_changed)
        self._on_source_changed()

    def _on_source_changed(self):
        db_mode = (self.src_selector.currentText() == "Database")
        # Toggle file controls
        for w in (self.viewer.file1_edit, self.viewer.browse1_btn, self.viewer.file2_edit, self.viewer.browse2_btn, self.viewer.load_btn):
            w.setVisible(not db_mode)
        self.load_db_btn.setVisible(db_mode)

    def _load_from_db(self):
        if not self.batch_selector or not self.db:
            QMessageBox.warning(self, "Batch ID", "Selecciona un Batch antes de cargar.")
            return
        batch_id = self.batch_selector.currentText()
        try:
            import pandas as pd, numpy as np
            # Select latest per bolt+combo by measured_at when available
            try:
                self.db.cur.execute(
                    """
                    SELECT DISTINCT ON (bolt_id, freq, gain, pulse)
                           freq, gain, pulse, bolt_id, temp, pico1, pct_diff, dat2, dat3
                    FROM pre_measurement
                    WHERE batch_id=%s
                    ORDER BY bolt_id, freq, gain, pulse, measured_at DESC
                    """,
                    (batch_id,)
                )
            except Exception:
                self.db.cur.execute(
                    "SELECT freq, gain, pulse, bolt_id, temp, pico1, pct_diff, dat2, dat3 FROM pre_measurement WHERE batch_id=%s",
                    (batch_id,)
                )
            rows = self.db.cur.fetchall()
            cols = [d[0] for d in self.db.cur.description]
            df = pd.DataFrame(rows, columns=cols)
            # Adapt column names to CSVViewerTab expectations
            df.rename(columns={'freq':'Freq', 'gain':'Gain', 'pulse':'Pulse', 'bolt_id':'Bolt ID'}, inplace=True)
            # Compute Bolt Num alias
            batch_id = self.batch_selector.currentText()
            df['Bolt Num'] = df['Bolt ID'].apply(lambda bid: self.db.get_bolt_alias(batch_id, bid))
            # Ensure temp column exists (DB schema fallback)
            if 'temp' not in df.columns:
                df['temp'] = 0.0
            # Convert raw dat2/dat3 BYTEA to int arrays and expand into columns
            def _bytes_to_int16_array(b):
                if b is None:
                    return np.array([], dtype=np.int16)
                raw = b.tobytes() if hasattr(b, 'tobytes') else bytes(b)
                return np.frombuffer(raw, dtype='<i2')
            def _bytes_to_int32_array(b):
                if b is None:
                    return np.array([], dtype=np.int32)
                raw = b.tobytes() if hasattr(b, 'tobytes') else bytes(b)
                return np.frombuffer(raw, dtype='<i4')
            # Apply conversion
            dat2_arrays = df['dat2'].apply(_bytes_to_int16_array)
            dat3_arrays = df['dat3'].apply(_bytes_to_int32_array)
            n2 = int(dat2_arrays.iloc[0].size) if not dat2_arrays.empty else 1024
            n3 = int(dat3_arrays.iloc[0].size) if not dat3_arrays.empty else 1024
            dat2_list = [(arr.tolist() + [0]*(n2 - arr.size))[:n2] for arr in dat2_arrays]
            dat3_list = [(arr.tolist() + [0]*(n3 - arr.size))[:n3] for arr in dat3_arrays]
            dat2_df = pd.DataFrame(dat2_list, columns=[f'dat2_{i}' for i in range(n2)])
            dat3_df = pd.DataFrame(dat3_list, columns=[f'dat3_{i}' for i in range(n3)])
            # Drop raw columns and merge expanded data
            df = pd.concat([df.drop(columns=['dat2', 'dat3']), dat2_df, dat3_df], axis=1)
        except Exception as e:
            QMessageBox.critical(self, "DB Error", f"Error cargando de DB: {e}")
            return
        # Drop duplicate measurements per Bolt to plot one signal each
        df = df.drop_duplicates(subset=['Freq','Gain','Pulse','Bolt Num'], keep='first')
        # Provide to viewer and populate selectors manually (DB mode)
        self.viewer._df_list = [df]
        # Populate freq, gain, pulse
        freqs = sorted(df['Freq'].astype(int).unique())
        self.viewer.freq_cb.clear(); self.viewer.freq_cb.addItems([str(f) for f in freqs]); self.viewer.freq_cb.setEnabled(True)
        gains = sorted(df['Gain'].astype(int).unique())
        self.viewer.gain_cb.clear(); self.viewer.gain_cb.addItems([str(g) for g in gains]); self.viewer.gain_cb.setEnabled(True)
        pulses = sorted(df['Pulse'].astype(int).unique())
        self.viewer.pulse_cb.clear(); self.viewer.pulse_cb.addItems([str(p) for p in pulses]); self.viewer.pulse_cb.setEnabled(True)
        # Populate temperature selector
        temps = sorted(df['temp'].unique())
        self.viewer.temp_cb.clear(); self.viewer.temp_cb.addItems([str(t) for t in temps]); self.viewer.temp_cb.setEnabled(True)
        # Populate bolt multiselect with 'All' and each bolt
        bolt_model = self.viewer.bolt_cb.model()
        bolt_model.clear()
        all_item = QStandardItem('All'); all_item.setFlags(Qt.ItemIsEnabled|Qt.ItemIsUserCheckable); all_item.setCheckState(Qt.Checked)
        bolt_model.appendRow(all_item)
        for b in sorted(df['Bolt Num'].astype(int).unique()):
            it = QStandardItem(str(b)); it.setFlags(Qt.ItemIsEnabled|Qt.ItemIsUserCheckable); it.setCheckState(Qt.Unchecked)
            bolt_model.appendRow(it)
        self.viewer.bolt_cb.setEnabled(True); self.viewer.bolt_cb.lineEdit().setText('All')
        # Data type options
        self.viewer.type_cb.clear(); self.viewer.type_cb.addItems(['dat2','dat3']); self.viewer.type_cb.setEnabled(True)
        # Set initial selection for controls
        if freqs:
            for cb in (self.viewer.freq_cb, self.viewer.gain_cb, self.viewer.pulse_cb, self.viewer.bolt_cb, self.viewer.type_cb, self.viewer.temp_cb):
                cb.setCurrentIndex(0)
        # Update plot and table
        self.viewer._update_plot(); self.viewer._update_table()
        # Adjust legend entries to only show Bolt number
        try:
            lg_items = self.viewer.plot.plotItem.legend.items
        except Exception:
            lg_items = []
        for sample, label in lg_items:
            try:
                txt = label.text if hasattr(label, 'text') else label.text()
                if 'B:' in txt:
                    num = txt.split('B:')[-1].strip()
                    label.setText(f'Bolt {num}')
            except Exception:
                pass
        # Clean table column headers to only show Bolt number
        self._clean_headers()


    def _clean_headers(self):
        """Remove file prefixes from table headers when in DB mode."""
        if self.src_selector.currentText() != "Database":
            return
        tbl = self.viewer.table
        for col in range(tbl.columnCount()):
            hdr = tbl.horizontalHeaderItem(col)
            if hdr:
                txt = hdr.text()
                if 'Bolt' in txt:
                    num = ''.join(ch for ch in txt.split('Bolt')[-1] if ch.isdigit())
                    hdr.setText(f'Bolt {num}')

