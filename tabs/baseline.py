# tabs/baseline.py
# Baseline tab for single signal acquisition and validation

from __future__ import annotations

from datetime import datetime, timedelta
from functools import partial
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
import pyqtgraph as pg
import logging

from typing import Any
import struct

from device import Device, _pack32
from pgdb import BatchDB

# Minimum amplitude required to consider a frame as a valid attempt
AUTO_MIN_AMP = 100000


class ClearOnClickLineEdit(QtWidgets.QLineEdit):
    """QLineEdit that clears its contents whenever it's clicked."""

    def mousePressEvent(self, event):  # type: ignore[override]
        self.clear()
        super().mousePressEvent(event)


class BaselineWorker(QThread):
    """Thread that configures the device and grabs one frame."""

    frame_ready = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, port: str, params: dict[str, Any], parent=None) -> None:
        super().__init__(parent)
        self.port = port
        self.params = params
        self.dev = None  # type: Device | None

    def run(self) -> None:
        dev: Device | None = None
        try:
            dev = Device(self.port, baudrate=115200, timeout=1)
            self.dev = dev

            p = self.params
            # Standby and configure device, then send parameters
            dev.modo_standby()
            dev.modo_configure()
            # Debug summary of configuration parameters
            logging.debug(f"BaselineWorker: configuring device with params: {p}")
            dev.enviar(_pack32(20.0), "10")
            dev.enviar(_pack32(p["diftemp"]), "11")
            dev.enviar(_pack32(p["tof"]), "12")
            dev.enviar(_pack32(p["freq"]), "14")
            dev.enviar(_pack32(p["pulse"]), "15")
            dev.enviar(_pack32(p["pulse"]), "16")
            dev.enviar(_pack32(p["gain"]), "17")
            dev.enviar(_pack32(p.get("xi", 0.0)), "18")
            dev.enviar(_pack32(p.get("alpha", 0.0)), "19")
            # Windows: short_corr->1A, long_corr->1B, short_temp->1C, long_temp->1D
            dev.enviar(_pack32(p.get("short_corr", 990)), "1A")
            dev.enviar(_pack32(p.get("long_corr", 10)), "1B")
            dev.enviar(_pack32(p.get("short_temp", 1000)), "1C")
            dev.enviar(_pack32(p.get("long_temp", 1000)), "1D")
            dev.enviar(_pack32(0), "2C")
            dev.enviar(_pack32(0), "2D")
            logging.debug("BaselineWorker: configuration sent, starting acquisition")
            dev.modo_save()
            dev.modo_standby()
            dev.modo_single()

            # --- acquisition ---
            dev.enviar_temp()
            dev.start_measure()
            frame = dev.lectura() or {}
            # Merge configuration parameters without overwriting measured ToF
            frame.update({k: v for k, v in p.items() if k != "tof"})
            frame["input_tof"] = p.get("tof")
            self.frame_ready.emit(frame)
        except Exception as e:
            self.error.emit(str(e))
        finally:
            try:
                if dev and hasattr(dev, "ser") and dev.ser.is_open:
                    dev.ser.close()
            except Exception:
                pass


class BaselineTab(QtWidgets.QWidget):
    """Simple tab to acquire one signal from the device and classify it."""

    def __init__(
        self,
        com_selector: QtWidgets.QComboBox | None = None,
        db: BatchDB | None = None,
    ) -> None:
        super().__init__()
        self.com_selector = com_selector
        # Allow using a shared DB connection while keeping a fallback to a new one
        self.db = db or BatchDB()
        # store pending measurements until user saves them
        self._pending_results: list[dict[str, Any]] = []
        self._pending_valid: list[dict[str, Any]] = []
        self._pending_invalid: list[dict[str, Any]] = []
        # track automatic acquisition when scanning
        self._auto_mode = False
        self._auto_start = datetime.now()
        self._build_ui()
        # react to batch changes and bolt scans
        self.batch_cb.currentIndexChanged.connect(self._on_batch_changed)
        self.bolt_edit.returnPressed.connect(self._on_bolt_scanned)
        self._load_saved_baselines()

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        self.batch_cb = QtWidgets.QComboBox()
        # Initial load of completed batches
        try:
            items = [str(b) for b in self.db.list_completed_batches()]
        except Exception:
            items = []
        self.batch_cb.addItems(items)

        self.pallet_spin = QtWidgets.QSpinBox()
        self.pallet_spin.setRange(0, 255)

        self.bolt_edit = ClearOnClickLineEdit()
        self.bolt_edit.setPlaceholderText("Scan Bolt ID and press ↩")

        self.data_cb = QtWidgets.QComboBox()
        self.data_cb.addItems(["dat2", "dat3"])
        self.data_cb.setCurrentIndex(1)

        self.diff_spin = QtWidgets.QDoubleSpinBox()
        self.diff_spin.setRange(0.0, 100.0)
        self.diff_spin.setDecimals(1)
        self.diff_spin.setValue(20.0)
        self.diff_spin.setSuffix(" %")

        self.amp_spin = QtWidgets.QDoubleSpinBox()
        self.amp_spin.setRange(0.0, 1e6)
        self.amp_spin.setDecimals(1)
        self.amp_spin.setValue(100000.0)

        self.play_btn = QtWidgets.QPushButton("PLAY")
        self.play_btn.clicked.connect(self._on_play)
        self.play_btn.setStyleSheet(
            "QPushButton{background:#28a745;color:white;font-weight:600;padding:6px 10px;border-radius:4px;}"
            "QPushButton:hover{background:#218838;}"
            "QPushButton:disabled{background:#6c757d;color:#e6e6e6;}"
        )

        # Buttons for DB persistence and SQL generation
        self.add_db_btn = QtWidgets.QPushButton("Add to DB")
        self.add_db_btn.clicked.connect(self._on_add_db)
        self.add_db_btn.setStyleSheet(
            "QPushButton{background:#007bff;color:white;font-weight:600;padding:6px 10px;border-radius:4px;}"
            "QPushButton:hover{background:#0069d9;}"
            "QPushButton:disabled{background:#6c757d;color:#e6e6e6;}"
        )
        self.sql_btn = QtWidgets.QPushButton("Create Client SQL")
        self.sql_btn.clicked.connect(self._on_create_sql)
        self.sql_btn.setStyleSheet(
            "QPushButton{background:#f0ad4e;color:#111;font-weight:600;padding:6px 10px;border-radius:4px;}"
            "QPushButton:hover{background:#ec971f;}"
            "QPushButton:disabled{background:#6c757d;color:#e6e6e6;}"
        )

        # --- layout rows
        row1 = QtWidgets.QHBoxLayout()
        row1.addWidget(QtWidgets.QLabel("Batch ID"))
        row1.addWidget(self.batch_cb)
        row1.addWidget(QtWidgets.QLabel("Pallet"))
        row1.addWidget(self.pallet_spin)
        row1.addWidget(QtWidgets.QLabel("Bolt ID"))
        row1.addWidget(self.bolt_edit)
        row1.addWidget(QtWidgets.QLabel("Data"))
        row1.addWidget(self.data_cb)
        row1.addWidget(self.play_btn)

        row2 = QtWidgets.QHBoxLayout()
        row2.addWidget(QtWidgets.QLabel("Min pct_diff"))
        row2.addWidget(self.diff_spin)
        row2.addWidget(QtWidgets.QLabel("Amp threshold"))
        row2.addWidget(self.amp_spin)
        row2.addStretch()

        # --- tables
        self.table_valid = QtWidgets.QTableWidget(0, 4)
        self.table_valid.setHorizontalHeaderLabels(["Timestamp", "Bolt ID", "ToF", "Temp"])
        self.table_valid.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.table_valid.setStyleSheet("QHeaderView::section { background-color: #444; color: white; }")
        self.table_valid.cellClicked.connect(partial(self._on_pending_row_clicked, 'valid'))

        self.table_invalid = QtWidgets.QTableWidget(0, 4)
        self.table_invalid.setHorizontalHeaderLabels(["Timestamp", "Bolt ID", "ToF", "Temp"])
        self.table_invalid.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.table_invalid.setStyleSheet("QHeaderView::section { background-color: #444; color: white; }")
        self.table_invalid.cellClicked.connect(partial(self._on_pending_row_clicked, 'invalid'))

        row3 = QtWidgets.QHBoxLayout()
        row3.addWidget(self.add_db_btn)
        row3.addWidget(self.sql_btn)

        # Table for saved valid baselines
        self.table_db_valid = QtWidgets.QTableWidget(0, 4)
        self.table_db_valid.setHorizontalHeaderLabels(["Timestamp", "Bolt ID", "ToF", "Temp"])
        self.table_db_valid.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.table_db_valid.setStyleSheet("QHeaderView::section { background-color: #444; color: white; }")
        self.table_db_valid.cellClicked.connect(self._on_saved_row_clicked)

        left_layout = QtWidgets.QVBoxLayout()
        left_layout.addLayout(row1)
        left_layout.addLayout(row2)
        self.label_valid = QtWidgets.QLabel("Valid results (0)")
        left_layout.addWidget(self.label_valid)
        left_layout.addWidget(self.table_valid)
        self.label_invalid = QtWidgets.QLabel("Invalid results (0)")
        left_layout.addWidget(self.label_invalid)
        left_layout.addWidget(self.table_invalid)
        left_layout.addLayout(row3)
        self.label_db_valid = QtWidgets.QLabel("Saved valid baselines (0)")
        left_layout.addWidget(self.label_db_valid)
        left_layout.addWidget(self.table_db_valid)
        left_layout.addStretch()
        left_widget = QtWidgets.QWidget(); left_widget.setLayout(left_layout)

        # --- plot
        self.plot = pg.PlotWidget(title="Signal")
        self.plot.showGrid(x=True, y=True, alpha=0.2)
        self.plot.setLabel("bottom", "Samples")
        self.plot.setLabel("left", "Amplitude")
        self.curve = self.plot.plot([], [], pen=pg.mkPen("#00c853", width=2))
        # Disable mouse zoom/pan to avoid unintended changes
        try:
            _vb = self.plot.getPlotItem().getViewBox()
            _vb.setMouseEnabled(False, False)
            _vb.setMenuEnabled(False)
            self.plot.wheelEvent = lambda ev: None
            _vb.wheelEvent = lambda ev: None
            _vb.mouseDoubleClickEvent = lambda ev: None
            _vb.mouseDragEvent = lambda ev: None
        except Exception:
            pass

        right_layout = QtWidgets.QVBoxLayout(); right_layout.addWidget(self.plot)
        right_widget = QtWidgets.QWidget(); right_widget.setLayout(right_layout)

        root = QtWidgets.QHBoxLayout(self)
        root.addWidget(left_widget, 3)
        root.addWidget(right_widget, 2)

        # If no completed batches, disable controls until a batch gets selected
        if self.batch_cb.count() == 0:
            try:
                QtWidgets.QMessageBox.information(self, "Baseline", "No hay batches completados.")
            except Exception:
                pass
            self._set_controls_enabled(False)
        else:
            # Enable controls when there are items (selection may enable them again on change)
            self._set_controls_enabled(True)

        # Refresh the batch list when user opens the dropdown (so it updates on click)
        try:
            self._orig_show_popup = self.batch_cb.showPopup
            def _show_and_refresh():
                self._refresh_batches()
                self._orig_show_popup()
            self.batch_cb.showPopup = _show_and_refresh  # type: ignore[assignment]
        except Exception:
            # Fallback: refresh when combo gains focus
            self.batch_cb.focusInEvent = lambda ev: (self._refresh_batches(), QtWidgets.QComboBox.focusInEvent(self.batch_cb, ev))  # type: ignore[assignment]

    def _set_controls_enabled(self, enabled: bool) -> None:
        """Enable/disable all controls except the batch selector."""
        widgets = [
            getattr(self, 'pallet_spin', None),
            getattr(self, 'bolt_edit', None),
            getattr(self, 'data_cb', None),
            getattr(self, 'diff_spin', None),
            getattr(self, 'amp_spin', None),
            getattr(self, 'play_btn', None),
            getattr(self, 'add_db_btn', None),
            getattr(self, 'sql_btn', None),
            getattr(self, 'table_valid', None),
            getattr(self, 'table_invalid', None),
            getattr(self, 'table_db_valid', None),
        ]
        for w in widgets:
            try:
                if w is not None:
                    w.setEnabled(enabled)
            except Exception:
                pass

    def _refresh_batches(self) -> None:
        """Reload completed batches into the combo and keep selection when possible."""
        try:
            current = self.batch_cb.currentText()
            had_items = self.batch_cb.count() > 0
            new_items = [str(b) for b in self.db.list_completed_batches()]
        except Exception:
            new_items = []
            had_items = self.batch_cb.count() > 0
            current = self.batch_cb.currentText()
        self.batch_cb.blockSignals(True)
        self.batch_cb.clear()
        self.batch_cb.addItems(new_items)
        # Restore previous selection if there were items; otherwise force user selection
        if had_items and current and current in new_items:
            idx = new_items.index(current)
            self.batch_cb.setCurrentIndex(idx)
        else:
            # No previous items or selection changed; do not preselect to ensure user action
            try:
                self.batch_cb.setCurrentIndex(-1)
            except Exception:
                # Fallback: keep at 0 but rely on user to choose a different one to enable controls
                pass
        self.batch_cb.blockSignals(False)
        # Do not enable controls here; wait for explicit selection change

    # ------------------------------------------------------------------ logic
    def _on_play(self, auto: bool = False) -> None:
        """Trigger a measurement. When *auto* is True it is part of the
        automatic loop launched after scanning a Bolt ID."""
        # any manual play cancels auto mode
        if not auto:
            self._auto_mode = False
        # If a previous worker exists, stop it and close its port
        if hasattr(self, 'worker') and self.worker.isRunning():
            try:
                self.worker.terminate()
                self.worker.wait(100)
                if hasattr(self.worker, 'dev') and self.worker.dev and hasattr(self.worker.dev, 'ser'):
                    if self.worker.dev.ser.is_open:
                        self.worker.dev.ser.close()
            except Exception:
                pass
        # read signal in background thread
        port = self.com_selector.currentText() if self.com_selector else ''
        # reject placeholder or invalid ports
        if not port or port.startswith("No COM"):
            QtWidgets.QMessageBox.critical(self, "COM", "Selecciona un puerto COM válido.")
            self.play_btn.setEnabled(True)
            self._auto_mode = False
            return
        # fetch valid parameters from DB
        try:
            freq_int, gain_int, pulse_int = self.db.get_baseline_params(self.batch_cb.currentText())
            freq, gain, pulse = float(freq_int), float(gain_int), float(pulse_int)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "DB", f"Parámetros no disponibles: {e}")
            self.play_btn.setEnabled(True)
            self._auto_mode = False
            return
        bolt_id = self.bolt_edit.text().strip()
        if not bolt_id:
            QtWidgets.QMessageBox.critical(self, "Bolt", "Introduce un Bolt ID válido.")
            self.play_btn.setEnabled(True)
            self._auto_mode = False
            return
        # Retrieve ToF from pre_measurement; fallback to batch reference_tof
        tof = self.db.get_pre_tof(self.batch_cb.currentText(), bolt_id, freq, gain, pulse)
        if tof is None:
            batch_data = self.db.get_batch(self.batch_cb.currentText())
            attrs = batch_data.get("attrs", {})
            ref_tof = attrs.get("reference_tof")
            if ref_tof is not None:
                try:
                    tof = float(ref_tof)
                except Exception:
                    tof = None
        if tof is None:
            QtWidgets.QMessageBox.critical(
                self,
                "ToF",
                "No se encontró ToF en pre_measurement ni reference_tof en batch.",
            )
            self.play_btn.setEnabled(True)
            self._auto_mode = False
            return
        # Additional device parameters (temperature gradient & long_corr)
        diftemp, long_corr, xi, alpha = self.db.get_device_params(self.batch_cb.currentText())
        # Resolve window parameters from DB (fallback to current defaults)
        stw = ltw = scw = None
        try:
            attrs = (self.db.get_batch(self.batch_cb.currentText()) or {}).get("attrs", {})
            stw = int(float(attrs.get("short_temporal_window"))) if attrs.get("short_temporal_window") not in (None, "") else None
            ltw = int(float(attrs.get("long_temporal_window"))) if attrs.get("long_temporal_window") not in (None, "") else None
            scw = int(float(attrs.get("short_correlation_window"))) if attrs.get("short_correlation_window") not in (None, "") else None
        except Exception:
            stw = ltw = scw = None
        params = {
            "freq": freq,
            "gain": gain,
            "pulse": pulse,
            "tof": tof,
            "diftemp": diftemp,
            "long_corr": long_corr,
            "xi": xi,
            "alpha": alpha,
            # default fallbacks match previous constants
            "short_temp": stw if stw is not None else 1000,
            "long_temp": ltw if ltw is not None else 1000,
            "short_corr": scw if scw is not None else 990,
        }
        # disable play button while reading
        self.play_btn.setEnabled(False)
        self.worker = BaselineWorker(port, params)
        # connect to frame_ready and error signals
        self.worker.frame_ready.connect(self._on_play_finished)
        self.worker.error.connect(self._on_play_error)
        if not auto:
            # start a timeout to avoid infinite wait only for manual measurements
            self._timeout_timer = QTimer(self)
            self._timeout_timer.setSingleShot(True)
            self._timeout_timer.timeout.connect(self._on_play_timeout)
            self._timeout_timer.start(3000)  # 3 seconds timeout
        self.worker.start()

    def _process_frame(self, frame: dict[str, Any]) -> None:
        # Process incoming frame dict
        # frame may be empty but still plot and show warning
        pct = frame.get("porcentaje_diferencia")
        amp = frame.get("maxcorry")
        data = frame.get(self.data_cb.currentText())
        # Warn if missing data
        if data is None:
            QtWidgets.QMessageBox.warning(self, "Sin señal", "No se encontró la señal seleccionada en el frame.")
            data = []
        # Plot signal
        self.curve.setData(data, autoDownsample=True)
        # Ignore frames whose amplitude is below the mandatory threshold
        if amp is None or amp < AUTO_MIN_AMP:
            QtWidgets.QMessageBox.warning(
                self,
                "Amplitud",
                f"La amplitud máxima ({amp or 0:.0f}) es inferior a {AUTO_MIN_AMP}.",
            )
            return
        # Only add to tables if pct and amp are present
        if pct is not None and amp is not None:
            bolt_id = self.bolt_edit.text().strip()
            ts = datetime.now()
            row = [ts.strftime("%H:%M:%S"), bolt_id, f"{frame.get('tof', 0):.1f}", f"{frame.get('temp', 0):.1f}"]
            frame["measured_at"] = ts
            is_valid = pct >= self.diff_spin.value() and amp >= self.amp_spin.value()
            if is_valid:
                self._add_row(self.table_valid, row)
                self._pending_valid.append(frame)
            else:
                self._add_row(self.table_invalid, row)
                self._pending_invalid.append(frame)
            self._pending_results.append({
                "batch_id": self.batch_cb.currentText(),
                "bolt_id": bolt_id,
                "pallet": self.pallet_spin.value(),
                "frame": frame,
                "is_valid": is_valid,
                "measured_at": ts,
            })
            self._update_counts()


    def _schedule_auto_retry(self, error_msg: str | None = None) -> None:
        """Retry automatic acquisition until timeout or amplitude threshold."""
        if (datetime.now() - self._auto_start) > timedelta(seconds=10):
            self._auto_mode = False
            self.play_btn.setEnabled(True)
            if error_msg:
                QtWidgets.QMessageBox.critical(self, "Error Lectura", error_msg)
            else:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Auto",
                    (
                        f"No se alcanzó amplitud > {AUTO_MIN_AMP} en 10s.\n"
                        "Coloca bien el sensor y pulsa PLAY para medir manualmente."
                    ),
                )
            self.bolt_edit.setFocus()
            self.bolt_edit.selectAll()
        else:
            QTimer.singleShot(100, lambda: self._on_play(auto=True))

    def _on_play_finished(self, frame: dict[str, Any]) -> None:
        """Handle successful frame read from worker."""
        if hasattr(self, '_timeout_timer'):
            self._timeout_timer.stop()
        amp = frame.get("maxcorry") or 0
        if self._auto_mode:
            if amp >= AUTO_MIN_AMP:
                self.play_btn.setEnabled(True)
                self._auto_mode = False
                self._process_frame(frame)
                self.bolt_edit.setFocus()
                self.bolt_edit.selectAll()
            else:
                self._schedule_auto_retry()
            return
        self.play_btn.setEnabled(True)
        self._process_frame(frame)
        self.bolt_edit.setFocus()
        self.bolt_edit.selectAll()

    def _on_play_error(self, msg: str) -> None:
        """Handle errors during device read."""
        if hasattr(self, '_timeout_timer'):
            self._timeout_timer.stop()
        try:
            self.worker.terminate()
        except Exception:
            pass
        if self._auto_mode:
            self._schedule_auto_retry(msg)
        else:
            QtWidgets.QMessageBox.critical(self, "Error Lectura", msg)
            self.play_btn.setEnabled(True)
            self._auto_mode = False
            self.bolt_edit.setFocus()
            self.bolt_edit.selectAll()

    def _on_play_timeout(self) -> None:
        """Called when reading signal times out."""
        QtWidgets.QMessageBox.warning(self, "Timeout", "Tiempo de lectura agotado sin recibir señal.")
        # close serial port if open and stop worker thread
        try:
            if hasattr(self.worker, 'dev') and self.worker.dev and hasattr(self.worker.dev, 'ser'):
                self.worker.dev.ser.close()
        except Exception:
            pass
        try:
            self.worker.terminate()
            self.worker.wait(100)
        except Exception:
            pass
        # re-enable play button
        self.play_btn.setEnabled(True)
        self._auto_mode = False
        self.bolt_edit.setFocus()
        self.bolt_edit.selectAll()

    def _add_row(self, table: QtWidgets.QTableWidget, values: list[Any]) -> None:
        row_idx = table.rowCount()
        table.insertRow(row_idx)
        for col, val in enumerate(values):
            item = QtWidgets.QTableWidgetItem(str(val))
            item.setTextAlignment(Qt.AlignCenter)
            table.setItem(row_idx, col, item)

    def _on_pending_row_clicked(self, table_name: str, row: int, _col: int) -> None:
        """Plot pending signal when a row in valid/invalid tables is clicked."""
        frames = self._pending_valid if table_name == 'valid' else self._pending_invalid
        if not (0 <= row < len(frames)):
            return
        frame = frames[row]
        data = frame.get(self.data_cb.currentText())
        if data is None:
            data = []
        self.curve.setData(data, autoDownsample=True)

    def _on_add_db(self) -> None:
        """Persist pending measurements with their validity flag."""
        if not self._pending_results:
            QtWidgets.QMessageBox.information(self, "DB", "No hay medidas para guardar.")
            return
        errors = 0
        for item in self._pending_results:
            try:
                self.db.add_baseline(
                    batch_id=item["batch_id"],
                    bolt_id=item["bolt_id"],
                    pallet_num=item["pallet"],
                    frame=item["frame"],
                    is_valid=item["is_valid"],
                    measured_at=item.get("measured_at"),
                )
            except Exception as e:
                errors += 1
                # Debug detailed error for each failed save
                logging.error(f"BaselineTab: failed to save measurement {item} - exception: {e}", exc_info=True)
        saved = len(self._pending_results) - errors
        self._pending_results.clear()
        self._pending_valid.clear()
        self._pending_invalid.clear()
        self.table_valid.setRowCount(0)
        self.table_invalid.setRowCount(0)
        self._update_counts()
        if errors:
            QtWidgets.QMessageBox.warning(self, "DB", f"{errors} medidas no se guardaron")
        else:
            QtWidgets.QMessageBox.information(self, "DB", f"{saved} medidas guardadas")
        self._load_saved_baselines()

    def _on_create_sql(self) -> None:
        """Generate SQL file for client DB with batch and bolt data."""
        batch = self.batch_cb.currentText()
        if not batch:
            QtWidgets.QMessageBox.warning(self, "SQL", "Selecciona un batch primero")
            return
        # Obtain batch attributes
        try:
            data = self.db.get_batch(batch)
        except Exception as e:  # pragma: no cover - DB errors
            QtWidgets.QMessageBox.critical(self, "SQL", f"No se pudieron obtener datos del batch:\n{e}")
            return
        attrs = data.get("attrs", {})
        # Helper to escape single quotes
        esc = lambda s: (s or "").replace("'", "''")

        # Helpers to format numbers avoiding unnecessary decimals
        def fmt_int(val: Any) -> str:
            try:
                return str(int(float(val)))
            except Exception:
                return "0"

        def fmt_float(val: Any) -> str:
            try:
                f = float(val)
            except Exception:
                return "0"
            s = f"{f}"
            return s.rstrip("0").rstrip(".")

        metric = fmt_int(attrs.get("metric"))
        length = fmt_int(attrs.get("length"))
        desc = f"M{metric}x{length}"
        xi = fmt_float(attrs.get("xi"))
        alpha1 = fmt_float(attrs.get("alpha1"))

        batch_sql = (
            "INSERT INTO `batch` VALUES\n"
            f"({batch},0,{xi},{alpha1},0,0,"
            f"'{esc(attrs.get('customer', ''))}','{esc(desc)}',"
            f"{fmt_int(attrs.get('customer_part_number'))},{fmt_int(attrs.get('joint_length'))},"
            f"{fmt_int(attrs.get('target_load'))},{fmt_int(attrs.get('max_load'))},{fmt_int(attrs.get('min_load'))},"
            f"'{esc(attrs.get('application_description', ''))}',"
            f"{fmt_int(attrs.get('frequency'))},{fmt_int(attrs.get('gain'))},"
            f"{fmt_int(attrs.get('cycles_coarse'))},{fmt_int(attrs.get('cycles_fine'))},"
            f"{fmt_int(attrs.get('temp_gradient'))},{fmt_int(attrs.get('short_correlation_window'))},"
            f"{fmt_int(attrs.get('long_correlation_window'))},{fmt_int(attrs.get('short_temporal_window'))},"
            f"{fmt_int(attrs.get('long_temporal_window'))},0,0,506,0,196);\n"
        )

        # Collect bolt data
        try:
            bolts = self.db.baseline_export_rows(batch)
        except Exception as e:  # pragma: no cover - DB errors
            QtWidgets.QMessageBox.critical(self, "SQL", f"No se pudieron obtener las baselines:\n{e}")
            return
        if not bolts:
            QtWidgets.QMessageBox.information(self, "SQL", "No hay baselines válidos para exportar")
            return
        # Select output file
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Guardar SQL del cliente",
            f"batch_{batch}.sql",
            "SQL (*.sql)",
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(batch_sql)
                fh.write(f"# Batch {batch}\n")
                for b in bolts:
                    first = b["first_ts"].strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                    last = b["last_ts"].strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                    tof = int(round(float(b.get("tof", 0) or 0)))
                    temp = float(b.get("temp", 0) or 0.0)
                    fh.write(
                        "INSERT INTO `bolt` (`DB_ADD_DATE`,`BOLT_PR_TOFREF`,`BOLT_PR_TEMPREF`,`BOLT_ID`,`BATCH_ID`,`LAST_MODIFIED`) VALUES "
                        f"('{first}',{tof},{temp:.1f},{b['bolt_id']},{batch},'{last}');\n"
                    )
        except Exception as e:  # pragma: no cover - file errors
            QtWidgets.QMessageBox.critical(self, "SQL", f"Error guardando fichero:\n{e}")
            return
        QtWidgets.QMessageBox.information(self, "SQL", "Fichero SQL creado correctamente")

    # ------------------------------------------------------------------ helpers
    def _update_counts(self) -> None:
        """Update labels with unique counts of valid/invalid measurements."""
        valid_ids = {self.table_valid.item(r, 1).text() for r in range(self.table_valid.rowCount())}
        invalid_ids = {self.table_invalid.item(r, 1).text() for r in range(self.table_invalid.rowCount())}
        invalid_unique = invalid_ids - valid_ids
        self.label_valid.setText(f"Valid results ({len(valid_ids)})")
        self.label_invalid.setText(f"Invalid results ({len(invalid_unique)})")

    def _load_saved_baselines(self) -> None:
        """Load latest valid baselines for selected batch into the table."""
        batch = self.batch_cb.currentText()
        try:
            rows = self.db.latest_valid_baselines(batch)
        except Exception:
            rows = []
        self._latest_valid = rows
        self.table_db_valid.setRowCount(0)
        for r in rows:
            row_idx = self.table_db_valid.rowCount()
            self.table_db_valid.insertRow(row_idx)
            ts = r["measured_at"].strftime("%Y-%m-%d %H:%M:%S") if r.get("measured_at") else ""
            vals = [ts, r.get("bolt_id", ""), f"{r.get('tof', 0):.1f}", f"{r.get('temp', 0):.1f}"]
            for c, val in enumerate(vals):
                item = QtWidgets.QTableWidgetItem(str(val))
                item.setTextAlignment(Qt.AlignCenter)
                self.table_db_valid.setItem(row_idx, c, item)
        unique_ids = {r.get("bolt_id") for r in rows if r.get("bolt_id")}
        self.label_db_valid.setText(f"Saved valid baselines ({len(unique_ids)})")

    def _on_saved_row_clicked(self, row: int, _col: int) -> None:
        """Plot signal for selected saved baseline row."""
        if not (0 <= row < len(getattr(self, "_latest_valid", []))):
            return
        rec = self._latest_valid[row]
        data_type = self.data_cb.currentText()
        if data_type == "dat2":
            data = self._bytes_to_int16_array(rec.get("dat2"))
        else:
            data = self._bytes_to_int32_array(rec.get("dat3"))
        self.curve.setData(data, autoDownsample=True)

    def _on_batch_changed(self) -> None:
        """Refresh saved baselines and clear pending tables when batch changes."""
        # Enable/disable controls depending on whether a batch is selected
        self._set_controls_enabled(bool(self.batch_cb.currentText()))
        self.table_valid.setRowCount(0)
        self.table_invalid.setRowCount(0)
        self._pending_results.clear()
        self._pending_valid.clear()
        self._pending_invalid.clear()
        self._update_counts()
        self._load_saved_baselines()

    def _on_bolt_scanned(self) -> None:
        """Clear current plot and start automatic acquisition when scanning."""
        self.curve.setData([], [])
        self.bolt_edit.setFocus()
        self.bolt_edit.selectAll()
        # kick off automatic measurements until amplitude threshold met
        self._auto_mode = True
        self._auto_start = datetime.now()
        self._on_play(auto=True)

    # ------------------------------------------------------------------ conversion helpers
    @staticmethod
    def _bytes_to_int16_array(b: Any) -> list[int]:
        if b is None:
            return []
        count = len(b) // 2
        return list(struct.unpack('<' + str(count) + 'h', b))

    @staticmethod
    def _bytes_to_int32_array(b: Any) -> list[int]:
        if b is None:
            return []
        count = len(b) // 4
        return list(struct.unpack('<' + str(count) + 'i', b))


