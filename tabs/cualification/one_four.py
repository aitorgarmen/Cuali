"""tabs/cualification/one_four.py

Implementation of the *1-4* subtab for the Qualification module.

The full hardware behaviour described in the project is complex and
not completely implemented yet.  This module provides the basic UI
structure required: six subtabs are created, one for each bolt of a
batch plus two helper subtabs.  Each bolt tab is based on the
Each bolt tab is a standalone widget with its own real-time acquisition,
independent from the 1-10 tab implementation.

The first four subtabs are identical and will handle the 4 bolts of the
current batch.  The fifth subtab uses :class:`OneFourFilterTab` to display
and store the valid combinations obtained in this stage.  Finally a
CSV viewer is provided for quick inspection of generated files.
"""
from __future__ import annotations

from PyQt5.QtWidgets import (
    QTabWidget, QVBoxLayout, QWidget, QPushButton, QHBoxLayout,
    QLabel, QComboBox, QDoubleSpinBox, QMessageBox, QDialog, QScrollArea,
    QLineEdit
)
from PyQt5 import QtCore
from PyQt5.QtCore import Qt

from ..dj import DeviceWorker
from pgdb import BatchDB
from device import Device
from ..filter import FilterTab, FilterWorker
from PyQt5.QtGui import QBrush, QColor, QStandardItem, QStandardItemModel
from ..csv_viewer import CSVViewerTab
import pandas as pd
import numpy as np
import pyqtgraph as pg
import logging

logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.DEBUG)
logger.setLevel(logging.DEBUG)

class ComboScanWorker(QtCore.QThread):
    """Scan over a list of (freq, gain, pulse) combos and emit frames.

    ``ComboScanWorker`` normally closes the device when finished but
    callers can keep the port open by passing ``close_device=False``.
    """

    frame_ready = QtCore.pyqtSignal(dict)
    progress = QtCore.pyqtSignal(int)
    finished = QtCore.pyqtSignal(list)
    error = QtCore.pyqtSignal(str)

    def __init__(self, device: Device, combos: list[tuple[float, float, float]],
                 bolt_num: float, config: dict[str, float], *,
                 close_device: bool = True):
        super().__init__()
        self._device = device
        self._combos = combos
        self._config = config
        self._bolt = bolt_num
        self._rows: list = []
        self._stop = False
        self._close = close_device

    def stop(self):
        self._stop = True

    def run(self):
        try:
            total = len(self._combos)
            # helper for sending full config before each scan iteration
            cfg_worker = DeviceWorker(self._device, lambda: {})
            for idx, (freq, gain, pulse) in enumerate(self._combos, start=1):
                if self._stop:
                    break
                # Log parameters sent to microcontroller
                # Debug: print parameters sent to microcontroller
                tof_override = self._config.get('tof_map', {}).get((freq, gain, pulse)) if isinstance(self._config.get('tof_map'), dict) else None
                print(f"[ComboScanWorker] Iter {idx}/{total}: freq={freq}, gain={gain}, pulse={pulse}, tof_override={tof_override}")
                # Determine ToF override if available
                tof_val = None
                if isinstance(self._config.get('tof_map'), dict):
                    tof_val = self._config['tof_map'].get((freq, gain, pulse))
                # Configure device for this combo
                try:
                    # determine ToF override or default
                    tof_to_use = tof_val if tof_val is not None else self._config.get('tof')
                    cfg_params = {
                        'algo': 0,
                        'threshold': 0,
                        'tof': tof_to_use,
                        'freq': freq,
                        'pulse': pulse,
                        'gain': gain
                    }
                    cfg_worker._send_full_config(cfg_params)
                except Exception as e:
                    self.error.emit(f"Config error: {e}")
                    break
                # Perform measurement
                try:
                    self._device.enviar_temp()
                    self._device.start_measure()
                    frame = self._device.lectura()
                    # Ensure emitted frame carries the parameters used
                    frame["freq"] = freq
                    frame["gain"] = gain
                    frame["pulse"] = pulse
                    print(f"[ComboScanWorker] Received frame: {frame}")
                except Exception as e:
                    self.error.emit(str(e))
                    break
                # Store and emit frame
                self._rows.append(frame)
                self.frame_ready.emit(frame)
                # Emit progress
                pct = int(idx / total * 100)
                self.progress.emit(pct)
        except Exception as e:
            self.error.emit(str(e))
        finally:
            # Close device serial port if requested
            if getattr(self, '_close', False):
                try:
                    self._device.ser.close()
                except Exception:
                    pass
            # Emit the collected rows when done
            self.finished.emit(self._rows)


class OneFourBoltTab(QWidget):
    """Bolt tab used inside :class:`OneFourTab`.

    Standalone widget that performs acquisition for a single Bolt ID.
    The UI provides a Bolt ID input, PLAY/STOP controls, data selector,
    and signal/ToF plots. It does not depend on the 1-10 tab.
    """
    def __init__(self, bolt_index: int, *, db: BatchDB | None = None,
                 com_selector=None, batch_selector: QComboBox | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.db = db or BatchDB()
        self.com_selector = com_selector
        self.batch_selector = batch_selector
        self.bolt_index = bolt_index

        # State
        self._active_params: dict | None = None
        self._device: Device | None = None
        self._worker: DeviceWorker | None = None
        self._tof_buffer: list[float] = []
        self._seq: int = 0
        self._last_frame: dict | None = None
        self.manual_tof: float | None = None
        self._final_force_prompting: bool = False
        # Scan/acquisition state
        self._scan_worker: ComboScanWorker | None = None
        self._scan_combos: list[tuple[float, float, float]] = []
        self._frames: list[dict] = []
        self._initial_frame: dict | None = None
        self._final_frame: dict | None = None
        self._scan_rows: list | None = None

        # UI
        main_layout = QVBoxLayout(self)

        controls = QHBoxLayout()
        self.data_cb = QComboBox(); self.data_cb.addItems(["dat2", "dat3"]); self.data_cb.setCurrentIndex(1)
        controls.addWidget(QLabel("Data"))
        controls.addWidget(self.data_cb)
        controls.addSpacing(12)
        controls.addWidget(QLabel("ToF:"))
        self.tof_label = QLabel("---"); controls.addWidget(self.tof_label)
        controls.addStretch(1)
        controls.addWidget(QLabel("Bolt ID:"))
        self.bolt_edit = QLineEdit(); self.bolt_edit.setPlaceholderText('Escanea/introduce Bolt ID y Enter')
        controls.addWidget(self.bolt_edit)
        self.alias_label = QLabel("Bolt Num: -"); controls.addWidget(self.alias_label)
        controls.addStretch(1)
        self.btn_play = QPushButton("PLAY", objectName="Play"); self.btn_play.setEnabled(False)
        self.btn_stop = QPushButton("STOP", objectName="Stop"); self.btn_stop.setEnabled(False)
        self.btn_play.setStyleSheet("background-color: #28a745; color: white;")
        self.btn_stop.setStyleSheet("background-color: #dc3545; color: white;")
        controls.addWidget(self.btn_play); controls.addWidget(self.btn_stop)
        main_layout.addLayout(controls)

        plots = QHBoxLayout()
        self.signal_plot = pg.PlotWidget(title="Señal")
        self.signal_plot.showGrid(x=True, y=True, alpha=0.2)
        self.signal_plot.setLabel("bottom", "Muestras")
        self.signal_plot.setLabel("left", "Amplitud")
        self.signal_curve = self.signal_plot.plot([], [], pen=pg.mkPen("#00c853", width=2))
        self.marker = pg.ScatterPlotItem(size=8, pen=pg.mkPen("#ff5722", width=2), brush=pg.mkBrush("#ff5722"))
        self.signal_plot.addItem(self.marker); self.marker.setVisible(False)
        self.threshold_line = pg.InfiniteLine(angle=0, pen=pg.mkPen('#FFEA00', width=1, style=Qt.DashLine), movable=False)
        self.signal_plot.addItem(self.threshold_line); self.threshold_line.setVisible(False)
        try:
            _vb = self.signal_plot.getPlotItem().getViewBox()
            _vb.setMouseEnabled(False, False)
            _vb.setMenuEnabled(False)
            self.signal_plot.wheelEvent = lambda ev: None
            _vb.wheelEvent = lambda ev: None
            _vb.mouseDoubleClickEvent = lambda ev: None
            _vb.mouseDragEvent = lambda ev: None
        except Exception:
            pass
        plots.addWidget(self.signal_plot, 1)

        self.tof_plot = pg.PlotWidget(title="ToF (µs)")
        self.tof_plot.showGrid(x=True, y=True, alpha=0.2)
        self.tof_plot.setLabel("bottom", "Frame #")
        self.tof_plot.setLabel("left", "ToF (µs)")
        self.tof_curve = self.tof_plot.plot([], [], pen=pg.mkPen("#ff9800", width=1))
        try:
            _vb2 = self.tof_plot.getPlotItem().getViewBox()
            _vb2.setMouseEnabled(False, False)
            _vb2.setMenuEnabled(False)
            self.tof_plot.wheelEvent = lambda ev: None
            _vb2.wheelEvent = lambda ev: None
            _vb2.mouseDoubleClickEvent = lambda ev: None
            _vb2.mouseDragEvent = lambda ev: None
        except Exception:
            pass
        plots.addWidget(self.tof_plot, 1)
        main_layout.addLayout(plots)

        # Extra controls row (Frequency Scan)
        extra = QHBoxLayout()
        self.btn_freq_scan = QPushButton("Frequency Scan")
        self.btn_freq_scan.setEnabled(False)
        self.btn_freq_scan.setStyleSheet("background-color: #007ACC; color: white;")
        extra.addStretch(1); extra.addWidget(self.btn_freq_scan)
        main_layout.addLayout(extra)

        # Hidden line edit to pass final force value into saver
        self.force_edit = QLineEdit(); self.force_edit.hide(); self.force_edit.setEnabled(False)

        # Wiring
        self.btn_play.clicked.connect(self._on_play)
        self.btn_stop.clicked.connect(self._on_stop)
        self.btn_freq_scan.clicked.connect(self._on_freq_scan)
        self.bolt_edit.returnPressed.connect(self._on_bolt_scanned)

        # External batch selector
        if self.batch_selector:
            self.batch_selector.currentIndexChanged.connect(self._on_batch_changed)
            self._on_batch_changed(0)

        # Window hint
        self.setWindowTitle(f"Bolt {bolt_index}")

        # (scan state already initialized above)

    def _collect_params(self):
        """Override ToF parameter to use manual tof from pre_measurement if available."""
        params = {
            'algo': int(self._active_params.get('algo', 0) if self._active_params else 0),
            'threshold': float(self._active_params.get('threshold', 0) if self._active_params else 0),
            'freq': float(self._active_params.get('freq', 0) if self._active_params else 0),
            'gain': float(self._active_params.get('gain', 0) if self._active_params else 0),
            'pulse': float(self._active_params.get('pulse', 0) if self._active_params else 0),
            'tof': float(self._active_params.get('tof', 0) if self._active_params else 0),
            'temp': 20.0,
        }
        if hasattr(self, 'manual_tof') and getattr(self, 'manual_tof', None) is not None:
            params['tof'] = self.manual_tof
        return params

    def _on_bolt_scanned(self) -> None:
        """Scan bolt ID, auto-select its batch and ask initial Real force."""
        bolt_id = self.bolt_edit.text().strip()
        # Reset alias display
        if hasattr(self, 'alias_label'):
            self.alias_label.setText("Bolt Num: -")
        if not bolt_id:
            QMessageBox.warning(self, "Bolt", "Introduce un Bolt ID valido.")
            return
        # Resolve batch id from DB for this bolt
        try:
            batch_id = self.db.find_batch_by_bolt(bolt_id) if hasattr(self.db, 'find_batch_by_bolt') else None
        except Exception:
            batch_id = None
        if not batch_id:
            QMessageBox.warning(self, "BBDD", f"No se ha encontrado batch para el Bolt ID {bolt_id}.")
            return
        # Update shared batch combo without clearing current bolt field
        if self.batch_selector:
            try:
                self.batch_selector.blockSignals(True)
                idx = self.batch_selector.findText(str(batch_id))
                if idx >= 0:
                    self.batch_selector.setCurrentIndex(idx)
                else:
                    self.batch_selector.addItem(str(batch_id))
                    # setCurrentIndex to newly added item (last)
                    self.batch_selector.setCurrentIndex(self.batch_selector.count() - 1)
            finally:
                self.batch_selector.blockSignals(False)
        # Validate bolt exists for that batch
        try:
            if not self.db.bolt_exists(batch_id, bolt_id):
                QMessageBox.warning(self, "BBDD", f"Bolt {bolt_id} no encontrado en batch {batch_id}.")
                return
        except Exception as e:
            QMessageBox.warning(self, "BBDD", f"Error validando bolt: {e}")
            return
        # Fetch and display alias number (use bolt_alias table if available)
        if hasattr(self, 'alias_label'):
            try:
                if hasattr(self.db, 'get_bolt_alias'):
                    alias = self.db.get_bolt_alias(batch_id, bolt_id)
                else:
                    alias = self.db.get_bolt_num(batch_id, bolt_id)
            except Exception:
                alias = None
            if alias is not None:
                self.alias_label.setText(f"Bolt Num: {alias}")
            else:
                self.alias_label.setText("Bolt Num: ?")
        # Get active parameters for this bolt
        try:
            params = self.db.params_for(batch_id, bolt_id)
            self._active_params = params
        except Exception as e:
            QMessageBox.warning(self, "DB Error", str(e))
            return
        # Ask for initial Real force (default 0) and enable PLAY
        force = self._ask_force_dialog(title="Fuerza inicial", prompt="Introduce la fuerza real inicial")
        if force is None:
            return
        try:
            self.force_edit.setText(f"{force:.1f}")
        except Exception:
            pass
        self._update_play_button()
        # Clear any previous scan frames
        self._frames.clear(); self._initial_frame = None; self._final_frame = None

    def _ask_force_dialog(self, *, title: str, prompt: str) -> float | None:
        """Small modal dialog to capture Real force (default 0).

        Prevents double opening using an internal guard.
        """
        # Guard against double-open (e.g., Return + editingFinished)
        if getattr(self, "_force_dialog_open", False):
            return None
        self._force_dialog_open = True
        dlg = QDialog(self)
        dlg.setWindowTitle(title)
        try:
            dlg.setWindowFlags(dlg.windowFlags() & ~Qt.WindowCloseButtonHint)
        except Exception:
            pass
        lay = QVBoxLayout(dlg)
        lay.addWidget(QLabel(prompt))
        spin = QDoubleSpinBox(dlg)
        spin.setDecimals(1)
        spin.setRange(-1e6, 1e6)
        spin.setSingleStep(0.1)
        spin.setValue(0.0)
        lay.addWidget(spin)
        btn = QPushButton("Aceptar", dlg)
        btn.setDefault(True)
        btn.clicked.connect(dlg.accept)
        lay.addWidget(btn)
        dlg.setModal(True)
        try:
            if dlg.exec_() == QDialog.Accepted:
                return float(spin.value())
            return None
        finally:
            self._force_dialog_open = False

        self._scan_worker: ComboScanWorker | None = None
        self._scan_combos: list[tuple[float,float,float]] = []
        self._frames: list[dict] = []
        self._initial_frame: dict | None = None
        self._final_frame: dict | None = None
        self._scan_rows: list | None = None

    def _on_play(self) -> None:
        # Clear previous frames
        self._frames.clear()
        self._initial_frame = None
        self._final_frame = None
        # Disable frequency scan until stop
        self.btn_freq_scan.setEnabled(False)
        # Determine selected batch and bolt
        batch_id = self.batch_selector.currentText() if self.batch_selector else None
        bolt_id = self.bolt_edit.text().strip()
        if not batch_id or not bolt_id:
            QMessageBox.warning(self, "Batch/Bolt", "Selecciona batch y bolt antes de iniciar.")
            return
        # Fetch pre-valid combos from DB using only batch_id
        self.db.cur.execute(
            "SELECT freq, gain, pulse, is_best FROM pre_valid_combo WHERE batch_id=%s",
            (batch_id,)
        )
        combos = []
        best_params = None
        for f, g, p, is_best in self.db.cur.fetchall():
            combos.append((f, g, p))
            if is_best:
                best_params = (f, g, p)
        if not combos:
            QMessageBox.warning(self, "Combos", "No hay combinaciones pre-validas.")
            return
        # Load bolt length from batch ultrasonic_length; fallback to reference_tof
        self.db.cur.execute(
            "SELECT ultrasonic_length, reference_tof FROM batch WHERE batch_id=%s",
            (batch_id,),
        )
        row = self.db.cur.fetchone()
        ul = 0.0
        if row:
            ul_val, ref_tof = row
            if ul_val is not None:
                ul = float(ul_val)
            elif ref_tof is not None:
                self.manual_tof = float(ref_tof)
            else:
                QMessageBox.warning(
                    self,
                    "Batch",
                    "Ni ultrasonic_length ni reference_tof definidos para el batch",
                )
        else:
            QMessageBox.warning(
                self,
                "Batch",
                "Ni ultrasonic_length ni reference_tof definidos para el batch",
            )
        # Set active params to best combo
        if best_params:
            f, g, p = best_params
        else:
            f, g, p = combos[0]
        self._active_params = {"freq": f, "gain": g, "pulse": p, "ul": ul}
        # Retrieve manual ToF from pre_measurement if available
        try:
            self.db.cur.execute(
                "SELECT tof FROM pre_measurement WHERE batch_id=%s AND bolt_id=%s AND freq=%s AND gain=%s AND pulse=%s LIMIT 1",
                (batch_id, bolt_id, f, g, p),
            )
            row = self.db.cur.fetchone()
            if row and row[0] is not None:
                self.manual_tof = float(row[0])
        except Exception:
            pass
        # Start real-time acquisition with DeviceWorker (will use manual_tof)
        if not self.com_selector or not self.com_selector.currentText():
            QMessageBox.critical(self, "COM", "Selecciona un puerto COM válido.")
            return
        try:
            self._device = Device(self.com_selector.currentText(), baudrate=115200, timeout=1)
        except Exception as e:
            QMessageBox.critical(self, "UART", f"No se pudo abrir el puerto:\n{e}")
            return
        # Reset state
        self._tof_buffer.clear(); self.tof_curve.setData([], []); self.tof_label.setText("---")
        self._seq = 0
        self._last_frame = None
        # Worker
        self._worker = DeviceWorker(self._device, self._collect_params, self.db, batch_id)
        self._worker.data_ready.connect(self._update_ui, Qt.QueuedConnection)
        self._worker.error.connect(self._on_worker_error, Qt.QueuedConnection)
        self._worker.start()
        QMessageBox.information(self, "PLAY", "Medida en tiempo real iniciada.")
        # UI
        self.btn_play.setEnabled(False)
        self.btn_stop.setEnabled(True)

    def _on_stop(self) -> None:
        """Stop and request final Real force via dialog; then save."""
        # Stop acquisition
        try:
            self._stop_worker_if_running()
        except Exception:
            pass
        self.btn_stop.setEnabled(False)
        # Await final force and prompt
        if getattr(self, "_final_force_prompting", False):
            return
        self._final_force_prompting = True
        self._awaiting_final_force = True
        force = self._ask_force_dialog(title="Fuerza final", prompt="Introduce la fuerza real final")
        if force is None:
            self._final_force_prompting = False
            return
        try:
            self.force_edit.setText(f"{force:.1f}")
        except Exception:
            pass
        # Nota: en 1-4 no se guarda la medida final con fuerza manual
        # (se omite el guardado de una medición "final" aquí).
        # Enable Frequency Scan button after fully stopping real-time acquisition
        self.btn_freq_scan.setEnabled(True)
        self._final_force_prompting = False

    def _on_batch_changed(self, *_args) -> None:
        """Reset the per-bolt state when external batch selector changes."""
        # Stop running worker
        self._stop_worker_if_running()
        # Reset state
        self._active_params = None
        self._last_frame = None
        self._initial_frame = None
        self._final_frame = None
        self._frames.clear()
        self._tof_buffer.clear(); self.tof_curve.setData([], [])
        self.tof_label.setText("---")
        # Reset UI controls
        self.btn_play.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.btn_freq_scan.setEnabled(False)
        try:
            self.bolt_edit.clear()
        except Exception:
            pass
        try:
            self.alias_label.setText("Bolt Num: -")
        except Exception:
            pass

    def _update_play_button(self) -> None:
        """Enable Play when active params are available for current Bolt ID."""
        self.btn_play.setEnabled(self._active_params is not None)

    def _on_freq_scan(self) -> None:
        # Fetch valid pre-combinations from DB for this batch and bolt
        batch_id = self.batch_selector.currentText() if self.batch_selector else ''
        bolt_id = self.bolt_edit.text().strip()
        if not batch_id or not bolt_id:
            QMessageBox.warning(self, "Batch/Bolt", "Selecciona batch y bolt antes de escanear.")
            return
        # Read valid pre-combinations from DB for this batch
        self.db.cur.execute(
            "SELECT freq, gain, pulse FROM pre_valid_combo WHERE batch_id=%s",
            (batch_id,)
        )
        combos = [(float(f), float(g), float(p)) for f, g, p in self.db.cur.fetchall()]
        if not combos:
            QMessageBox.warning(self, "Combos", "No hay combinaciones pre-validas para escanear.")
            return
        # Fetch manual ToF overrides from pre_measurement
        try:
            self.db.cur.execute(
                "SELECT freq, gain, pulse, tof FROM pre_measurement WHERE batch_id=%s AND bolt_id=%s",
                (batch_id, bolt_id)
            )
            rows_tof = self.db.cur.fetchall()
            tof_map = {(float(f), float(g), float(p)): float(t) for f, g, p, t in rows_tof if t is not None}
        except Exception:
            tof_map = {}
        if not self.com_selector or not self.com_selector.currentText():
            QMessageBox.warning(self, "COM", "Selecciona un puerto COM válido")
            return
        try:
            import logging
            logging.debug(f"Valid combinations for batch {batch_id}: {combos}")
            device = Device(self.com_selector.currentText(), baudrate=115200, timeout=1)
        except Exception as e:
            QMessageBox.critical(self, "UART", str(e))
            return
        # Compute default ToF: use manual override if available, else calculate
        if getattr(self, 'manual_tof', None) is not None:
            default_tof = self.manual_tof
        else:
            bolt_length = float(self._active_params.get('ul', 370.0))
            default_tof = (2 * bolt_length) / 5900.0 * 1e6
        # Build scan config
        config = {'temp': 20.0, 'diftemp': -103.0, 'tof': default_tof, 'tof_map': tof_map}
        # Start frequency scan
        self.btn_freq_scan.setEnabled(False)
        bolt = float(self.bolt_edit.text() or 0)
        # Instantiate scan worker: close COM port when done (default close_device=True)
        self._scan_worker = ComboScanWorker(device, combos, bolt, config)
        self._scan_worker.frame_ready.connect(self._update_scan_plot)
        self._scan_worker.finished.connect(self._on_scan_finished)
        self._scan_worker.error.connect(lambda msg: QMessageBox.critical(self, "Scan", msg))
        self._scan_worker.start()
        self._scan_combos = combos

    def _update_scan_plot(self, frame: dict) -> None:
        dtype = self.data_cb.currentText()
        data = frame.get(dtype, frame.get("dat3", []))
        # Plot received signal (dat2/dat3)
        self.signal_curve.setData(data, autoDownsample=True)
        # Update red marker for dat3 as in DJ/frequency scan
        try:
            if dtype == "dat3":
                x = frame.get("maxcorrx", None)
                y = frame.get("maxcorry", None)
                if x is not None and y is not None:
                    self.marker.setData(x=[x], y=[y])
                    self.marker.setVisible(True)
                else:
                    self.marker.setVisible(False)
            else:
                self.marker.setVisible(False)
        except Exception:
            pass
        # Refresh plot display
        try:
            self.signal_plot.repaint()
        except Exception:
            pass

    def _update_ui(self, frame: dict) -> None:
        if frame:
            # Plot received data
            dtype = self.data_cb.currentText()
            data = frame.get(dtype, frame.get("dat3", []))
            self.signal_curve.setData(data, autoDownsample=True)
            # Marker for dat3
            if dtype == "dat3":
                try:
                    x = frame.get("maxcorrx", None); y = frame.get("maxcorry", None)
                    if x is not None and y is not None:
                        self.marker.setData(x=[x], y=[y]); self.marker.setVisible(True)
                    else:
                        self.marker.setVisible(False)
                except Exception:
                    self.marker.setVisible(False)
            else:
                self.marker.setVisible(False)
            # Threshold line
            if self._active_params and int(self._active_params.get("algo", 0)) in (1, 2):
                try:
                    y0 = float(self._active_params.get("threshold", 0))
                    self.threshold_line.setValue(y0); self.threshold_line.setVisible(True)
                except Exception:
                    self.threshold_line.setVisible(False)
            else:
                self.threshold_line.setVisible(False)
            # ToF buffer and label
            tof_val = frame.get("tof")
            if tof_val is not None:
                try:
                    self._tof_buffer.append(float(tof_val))
                    self.tof_curve.setData(list(range(len(self._tof_buffer))), self._tof_buffer)
                    self.tof_label.setText(f"{float(tof_val):.2f}")
                except Exception:
                    self.tof_label.setText(str(tof_val))

            # Bookkeeping and persistence
            self._last_frame = frame.copy()
            self._frames.append(frame)
            if self._initial_frame is None:
                self._initial_frame = frame
            self._final_frame = frame
            # Persist loading data for each measurement
            try:
                batch_id = self.batch_selector.currentText() if self.batch_selector else None
                bolt_id = self.bolt_edit.text().strip()
                if batch_id and bolt_id:
                    seq = len(self._frames)
                    data = {
                        'tof': frame.get('tof'),
                        'force': frame.get('force'),  # currently None
                        'dat2': frame.get('dat2'),
                        'dat3': frame.get('dat3'),
                    }
                    self.db.add_one4_loading(batch_id, bolt_id, seq, data)
            except Exception as e:
                logger.warning(f"Error saving one4_loading: {e}")

    def _on_worker_error(self, msg: str) -> None:
        QMessageBox.critical(self, "Error", msg)
        self._on_stop()

    def _stop_worker_if_running(self) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.stop(); self._worker.wait()
        self._worker = None
        if self._device:
            try:
                self._device.ser.close()
            except Exception:
                pass
            self._device = None

    # Store final one-4 measurement after STOP using the active params
    def _on_force_edit_finished(self) -> None:
        try:
            batch_id = self.batch_selector.currentText() if self.batch_selector else None
            bolt_id = self.bolt_edit.text().strip()
            if not (batch_id and bolt_id and self._final_frame and getattr(self, '_active_params', None)):
                return
            # Compose measurement frame with active parameters and final force
            data = dict(self._final_frame)
            data.update({
                'freq': self._active_params.get('freq'),
                'gain': self._active_params.get('gain'),
                'pulse': self._active_params.get('pulse'),
            })
            # Read force from hidden field (set by dialog)
            try:
                data['force'] = float(self.force_edit.text())
            except Exception:
                data['force'] = None
            try:
                self.db.add_one4_measurement(batch_id, bolt_id, data)
            except Exception as e:
                QMessageBox.warning(self, 'DB Error', f'Error guardando medida final 1-4: {e}')
        except Exception:
            # Defensive: do not crash UI on persistence issues
            pass

    def _on_scan_finished(self, rows: list) -> None:
        """Handle completion of frequency scan: save measurements and notify user."""
        # Disable frequency scan button
        self.btn_freq_scan.setEnabled(False)
        # Get identifiers
        batch_id = self.batch_selector.currentText() if self.batch_selector else ''
        bolt_id = self.bolt_edit.text().strip()
        if not batch_id or not bolt_id:
            QMessageBox.warning(self, "Batch/Bolt", "Batch o Bolt no definido.")
            return
        # Save measurements to database
        try:
            for combo, frame in zip(self._scan_combos, rows):
                freq, gain, pulse = combo
                data = {
                    'freq': freq,
                    'gain': gain,
                    'pulse': pulse,
                    'pico1': frame.get('pico1'),
                    'porcentaje_diferencia': frame.get('porcentaje_diferencia'),
                    'tof': frame.get('tof'),
                    'temp': frame.get('temp'),
                    'force': frame.get('force'),
                    'maxcorrx': frame.get('maxcorrx'),
                    'maxcorry': frame.get('maxcorry'),
                    'dat2': frame.get('dat2'),
                    'dat3': frame.get('dat3'),
                }
                self.db.add_one4_measurement(batch_id, bolt_id, data)
            QMessageBox.information(self, "DB", f"{len(rows)} mediciones guardadas en 1-4.")
        except Exception as e:
            QMessageBox.warning(self, "DB Error", f"Error guardando 1-4: {e}")
            return
        # Store scan rows and notify scan finished
        self._scan_rows = rows
        QMessageBox.information(self, "Scan", "Frequency scan finished")
        # Clean up scan worker reference to release port
        self._scan_worker = None


class OneFourFilterTab(FilterTab):
    def __init__(self, com_selector=None, db: BatchDB | None = None,
                 batch_selector: QComboBox | None = None, parent=None):
        super().__init__(com_selector=com_selector, parent=parent)
        self.db = db or BatchDB()
        self.batch_selector = batch_selector

        # Source selector identical to PreFilterTab
        src_layout = QHBoxLayout()
        src_layout.addWidget(QLabel("Source:"))
        self.src_selector = QComboBox()
        self.src_selector.addItems(["File", "Database"])
        self.src_selector.setCurrentIndex(1)  # default to DB
        src_layout.addWidget(self.src_selector)
        self.layout().insertLayout(0, src_layout)

        # Load from DB button
        self.btn_load_db = QPushButton("Load from DB")
        self.btn_load_db.setStyleSheet("background-color: #3498db; border: 1px solid #2980b9; color: white;")
        self.btn_load_db.clicked.connect(self._filter_from_db)
        self.layout().insertWidget(1, self.btn_load_db)

        # React to source change
        self.src_selector.currentIndexChanged.connect(lambda i: self._on_source_changed())
        self._on_source_changed()

        # Threshold for valid combos
        thresh_layout = QHBoxLayout()
        thresh_layout.addWidget(QLabel("Min Valid % (pct & pico):"))
        self.valid_spin = QDoubleSpinBox()
        self.valid_spin.setRange(0.0, 100.0)
        self.valid_spin.setDecimals(0)
        self.valid_spin.setValue(100.0)
        thresh_layout.addWidget(self.valid_spin)
        self.layout().insertLayout(3, thresh_layout)
        self.valid_spin.valueChanged.connect(self._apply_threshold)

        # Hide bolt/temp controls just like in Pre tab
        self._hide_layout_with_widget(self.bolt_spin)
        self._hide_layout_with_widget(self.temp_spin)

        # Add button to store selections in DB
        self.btn_add_db = QPushButton("Add to DB")
        self.btn_add_db.setStyleSheet("background-color: #e67e22; border: 1px solid #d35400; color: white;")
        self.btn_add_db.clicked.connect(self._on_add_db)
        self.layout().addWidget(self.btn_add_db)

        # Auto launch filtering when file chosen
        try:
            self.btn_open.clicked.disconnect()
        except Exception:
            pass
        self.btn_open.clicked.connect(self._choose_and_filter)
        self.btn_open.setStyleSheet("background-color: #95a5a6; border: 1px solid #7f8c8d; color: white;")

        # Track current best combo and stored signals
        self.best_combo: tuple[float, float, float] | None = None
        self.best_combo_index: int | None = None
        self._table_df = None
        self.bolt_signals: dict[tuple[int, int, int], dict[int, dict[str, list]]] = {}

        # Selector for dat2/dat3 when plotting
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

        # Connect table click for plotting
        self.table.clicked.connect(self._on_table_clicked)

    def _apply_threshold(self):
        threshold = self.valid_spin.value() / 100.0
        model = self._table_model
        if not model:
            return
        headers = [model.headerData(c, Qt.Horizontal) for c in range(model.columnCount())]
        cols = [i for i, col in enumerate(headers) if col in ('pct_ok_frac', 'pico1_ok_frac')]
        for c in cols:
            for r in range(model.rowCount()):
                item = model.item(r, c)
                try:
                    val = float(item.text().strip('%'))/100.0
                except Exception:
                    continue
                if val >= threshold:
                    item.setBackground(QBrush(QColor('green')))
                else:
                    item.setBackground(QBrush())

    def _on_source_changed(self):
        file_mode = (self.src_selector.currentText() == "File")
        self.btn_open.setVisible(file_mode)
        self.btn_load_db.setVisible(not file_mode)
        try:
            self.btn_start.clicked.disconnect()
        except Exception:
            pass
        if file_mode:
            self.btn_start.setEnabled(bool(self.selected_path))
            self.btn_start.clicked.connect(self._start_filter)
        else:
            self.btn_start.setEnabled(True)
            self.btn_start.clicked.connect(self._filter_from_db)

    def _filter_from_db(self):
        import pandas as pd, numpy as np
        if not self.batch_selector:
            return
        batch_id = self.batch_selector.currentText()
        if not batch_id:
            QMessageBox.warning(self, "Batch ID", "Selecciona un Batch antes de cargar.")
            return
        try:
            # Pick latest measurement per bolt+combo by measured_at
            try:
                self.db.cur.execute(
                    """
                    SELECT DISTINCT ON (bolt_id, freq, gain, pulse)
                           freq, gain, pulse, bolt_id, pico1, pct_diff, dat2, dat3
                    FROM one4_measurement
                    WHERE batch_id=%s
                    ORDER BY bolt_id, freq, gain, pulse, measured_at DESC
                    """,
                    (batch_id,),
                )
            except Exception:
                # Fallback without DISTINCT ON ordering if measured_at missing
                self.db.cur.execute(
                    "SELECT freq, gain, pulse, bolt_id, pico1, pct_diff, dat2, dat3 FROM one4_measurement WHERE batch_id=%s",
                    (batch_id,),
                )
            rows = self.db.cur.fetchall()
            cols = [desc[0] for desc in self.db.cur.description]
            df = pd.DataFrame(rows, columns=cols)
            # Preserve Bolt ID column and compute numeric alias
            df.rename(columns={'bolt_id': 'Bolt ID'}, inplace=True)
            df['Bolt Num'] = df['Bolt ID'].apply(lambda bid: self.db.get_bolt_alias(batch_id, bid))
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
            dat2_arrays = df['dat2'].apply(_bytes_to_int16_array)
            dat3_arrays = df['dat3'].apply(_bytes_to_int32_array)
            n2 = int(dat2_arrays.iloc[0].size) if not dat2_arrays.empty else 1024
            n3 = int(dat3_arrays.iloc[0].size) if not dat3_arrays.empty else 1024
            dat2_list = [(arr.tolist() + [0]*(n2 - arr.size))[:n2] for arr in dat2_arrays]
            dat3_list = [(arr.tolist() + [0]*(n3 - arr.size))[:n3] for arr in dat3_arrays]
            dat2_df = pd.DataFrame(dat2_list, columns=[f'dat2_{i}' for i in range(n2)])
            dat3_df = pd.DataFrame(dat3_list, columns=[f'dat3_{i}' for i in range(n3)])
            df = pd.concat([df.drop(columns=['dat2', 'dat3']), dat3_df, dat2_df], axis=1)
            self._db_full_df = df.copy()
        except Exception as e:
            QMessageBox.critical(self, "DB Error", f"No se pudo cargar de DB: {e}")
            return
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
        df_final = df_core.set_index(['freq', 'gain', 'pulse']).loc[list(combos_ok.itertuples(index=False, name=None))].reset_index()
        df_final.rename(columns={'freq': 'Freq', 'gain': 'Gain', 'pulse': 'Pulse'}, inplace=True)
        debug_df.rename(columns={'freq': 'Freq', 'gain': 'Gain', 'pulse': 'Pulse'}, inplace=True)
        self._on_finished(df_final, debug_df, "")

    def _hide_layout_with_widget(self, widget):
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

    def _choose_and_filter(self):
        super()._choose_file()
        if self.selected_path:
            self._start_filter()

    def _on_add_db(self):
        if not self.batch_selector:
            return
        batch_id = self.batch_selector.currentText()
        if not batch_id:
            QMessageBox.warning(self, "Batch ID", "Selecciona un Batch antes de guardar.")
            return
        model = self._table_model
        if model is None:
            QMessageBox.warning(self, "Sin datos", "No hay combinaciones para guardar.")
            return
        try:
            # Clear existing combos for this batch
            self.db.cur.execute(
                "DELETE FROM one4_valid_combo WHERE batch_id=%s",
                (batch_id,),
            )
            # Track best combo to update batch parameters later
            best_combo: tuple[int, int, float] | None = None
            count = 0
            for row_idx in range(model.rowCount()):
                # Column 0 = Select, Column 1 = Best
                sel_item = model.item(row_idx, 0)
                if sel_item.checkState() != Qt.Checked:
                    continue
                combo = self._table_df.loc[row_idx]
                freq = int(combo['Freq'])
                gain = int(combo['Gain'])
                pulse = int(combo['Pulse'])
                # Determine if this combo is marked as best
                best_state = model.item(row_idx, 1).checkState() == Qt.Checked
                self.db.cur.execute(
                    "INSERT INTO one4_valid_combo (batch_id, freq, gain, pulse, is_best) VALUES (%s, %s, %s, %s, %s)",
                    (batch_id, freq, gain, pulse, best_state),
                )
                # Capture best combo for batch update
                if best_state:
                    best_combo = (freq, gain, pulse)
                count += 1
            # Update batch record with the best combo parameters
            if best_combo:
                freq_b, gain_b, pulse_b = best_combo
                self.db.cur.execute(
                    "UPDATE batch SET frequency=%s, gain=%s, cycles_coarse=%s, cycles_fine=%s WHERE batch_id=%s",
                    (freq_b, gain_b, pulse_b, pulse_b, batch_id),
                )
            QMessageBox.information(self, "Guardado", f"{count} combinaciones guardadas en la base de datos.")
        except Exception as e:
            QMessageBox.warning(self, "DB Error", f"Error guardando combinaciones: {e}")

    def _on_finished(self, df_filtered, df_debug, out_path):
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

        if hasattr(self, '_db_full_df'):
            df_full = self._db_full_df.copy()
            df_full['pct_diff_num'] = FilterWorker._pct_to_float(df_full['pct_diff'])
            df_full['pct_diff'] = df_full['pct_diff_num'] / 100
            df_full.rename(columns={'freq': 'Freq', 'gain': 'Gain', 'pulse': 'Pulse'}, inplace=True)
            df_full = df_full.drop_duplicates(subset=['Freq', 'Gain', 'Pulse', 'Bolt Num'], keep='first')
        else:
            ext = Path(self.selected_path or '').suffix.lower()
            if ext == '.csv':
                df_full = __pd.read_csv(self.selected_path)
            else:
                df_full = __pd.read_excel(self.selected_path)
            df_full['pct_diff_num'] = FilterWorker._pct_to_float(df_full['pct_diff'])
            df_full['pct_diff'] = df_full['pct_diff_num'] / 100

        pivot_pct = df_full.pivot(index=['Freq', 'Gain', 'Pulse'], columns='Bolt Num', values='pct_diff')
        bolts = list(pivot_pct.columns)
        frac_pct = (pivot_pct >= pct_thr / 100).sum(axis=1) / len(bolts)
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

        if use_peak:
            table_df['valid_pct'] = __np.minimum(table_df['pct_ok_frac'], table_df['pico1_ok_frac'])
        else:
            table_df['valid_pct'] = table_df['pct_ok_frac']

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

    def _load_dataframe_into_table(self, df):
        df = df.reset_index(drop=True)
        self._table_df = df
        display_df = df.drop(columns=['valid_pct', 'valid_combination'], errors='ignore')
        super()._load_dataframe_into_table(display_df)
        model = self._table_model
        model.insertColumn(0)
        model.insertColumn(1)
        model.setHeaderData(0, Qt.Horizontal, "Select")
        model.setHeaderData(1, Qt.Horizontal, "Best")
        try:
            best_idx = df['valid_pct'].idxmax()
            self.best_combo_index = int(best_idx)
            self.best_combo = tuple(df.loc[best_idx, ['Freq', 'Gain', 'Pulse']])
        except Exception:
            best_idx = -1
            self.best_combo_index = None
            self.best_combo = None
        for row in range(model.rowCount()):
            sel_item = QStandardItem()
            sel_item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            sel_state = Qt.Checked if df.iloc[row].get('valid_combination', False) else Qt.Unchecked
            sel_item.setCheckState(sel_state)
            model.setItem(row, 0, sel_item)
            best_item = QStandardItem()
            best_item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            best_item.setCheckState(Qt.Checked if row == best_idx else Qt.Unchecked)
            model.setItem(row, 1, best_item)
        self.table.setModel(model)
        model.itemChanged.connect(self._on_best_toggled)

    def _on_table_clicked(self, index):
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
            # Use pico1 captured from DB for this combo+bolt
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
        if item.column() != 1:
            return
        model = self._table_model
        model.blockSignals(True)
        row_count = model.rowCount()
        if item.checkState() == Qt.Checked:
            for r in range(row_count):
                if r != item.row():
                    other = model.item(r, 1)
                    if other.checkState() == Qt.Checked:
                        other.setCheckState(Qt.Unchecked)
            self.best_combo_index = item.row()
            if self._table_df is not None and 0 <= item.row() < len(self._table_df):
                row = self._table_df.loc[item.row()]
                self.best_combo = tuple(row[col] for col in ['Freq', 'Gain', 'Pulse'])
            else:
                self.best_combo = None
        else:
            any_checked = False
            for r in range(row_count):
                if model.item(r, 1).checkState() == Qt.Checked:
                    any_checked = True
                    if self._table_df is not None and 0 <= r < len(self._table_df):
                        row = self._table_df.loc[r]
                        self.best_combo = tuple(row[col] for col in ['Freq', 'Gain', 'Pulse'])
                        self.best_combo_index = r
                    break
            if not any_checked:
                item.setCheckState(Qt.Checked)
        model.blockSignals(False)

    def get_best_combo(self):
        return self.best_combo




class OneFourTab(QWidget):
    """Sub-pestaña **1-4** de *Cualificación*.

    It hosts six subtabs:
        1-4 - one for each bolt in the batch,
        *Valid combinations* - provided by :class:`OneFourFilterTab`,
        *CSV Viewer* - reused from :class:`CSVViewerTab`.
    """

    def __init__(self, com_selector=None, db: BatchDB | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        # Shared DB and COM selector
        self.db = db or BatchDB()
        self.com_selector = com_selector
        layout = QVBoxLayout(self)
        # Batch selector dropdown
        batch_layout = QHBoxLayout()
        batch_layout.addWidget(QLabel("Batch Num:"))
        self.batch_cb = QComboBox()
        self.batch_cb.addItems(self.db.list_batches())
        batch_layout.addWidget(self.batch_cb)
        layout.addLayout(batch_layout)
        # Sub-tabs for bolts and valid combos
        tabs = QTabWidget()
        for i in range(4):
            bolt_tab = OneFourBoltTab(
                i + 1,
                db=db,
                com_selector=self.com_selector,
                batch_selector=self.batch_cb
            )
            tabs.addTab(bolt_tab, f"Bolt {i+1}")

        # Valid combinations tab showing results from one4_measurement (scrollable)
        filter_tab = OneFourFilterTab(com_selector=self.com_selector, db=db, batch_selector=self.batch_cb)
        scroll_filter = QScrollArea()
        scroll_filter.setWidgetResizable(True)
        scroll_filter.setWidget(filter_tab)
        tabs.addTab(scroll_filter, "Valid combinations")
        tabs.addTab(OneFourCSVViewerTab(db=self.db, batch_selector=self.batch_cb), "CSV Viewer")

        layout.addWidget(tabs)


class OneFourCSVViewerTab(QWidget):
    """CSV viewer that overlays pre and post load measurements."""

    def __init__(self, *, db: BatchDB | None = None, batch_selector: QComboBox | None = None, parent=None):
        super().__init__(parent)
        self.db = db
        self.batch_selector = batch_selector

        self.viewer = CSVViewerTab(db=db, batch_selector=batch_selector)
        # Debug pulse changes
        try:
            self.viewer.pulse_cb.currentTextChanged.connect(
                lambda text: logger.debug("Pulse combo changed to %s", text)
            )
        except Exception:
            pass
        # Map of (Freq, Gain, Pulse) -> temperatures available
        self._temp_map: dict[tuple[int, int, float], set[float]] = {}
        # Keep temperature selector in sync with freq/gain/pulse
        for cb in (self.viewer.freq_cb, self.viewer.gain_cb, self.viewer.pulse_cb):
            try:
                cb.currentIndexChanged.connect(self._sync_temp_options)
            except Exception:
                pass
        # Wrap viewer update methods so we can relabel legend and headers after
        # every refresh
        self._orig_update_plot = self.viewer._update_plot
        self._orig_update_table = self.viewer._update_table
        self.viewer._update_plot = lambda: (self._orig_update_plot(), self._apply_labels())
        self.viewer._update_table = lambda: (self._orig_update_table(), self._apply_labels())
        layout = QVBoxLayout(self)

        src_layout = QHBoxLayout()
        src_layout.addWidget(QLabel("Source:"))
        self.src_selector = QComboBox()
        self.src_selector.addItems(["File", "Database"])
        self.src_selector.setCurrentIndex(1)
        src_layout.addWidget(self.src_selector)

        self.load_db_btn = QPushButton("Load from DB")
        self.load_db_btn.clicked.connect(self._load_from_db)
        self.load_db_btn.setStyleSheet("background-color: #3498db; border: 1px solid #2980b9; color: white;")
        src_layout.addWidget(self.load_db_btn)
        layout.addLayout(src_layout)

        layout.addWidget(self.viewer)

        self.src_selector.currentIndexChanged.connect(self._on_source_changed)
        self._on_source_changed()

    def _apply_labels(self):
        """Update legend and table headers with load state labels."""
        try:
            subs = self.viewer._filter_df_current()
            after_rows = subs[0].shape[0] if len(subs) > 0 else 0
            pre_rows = subs[1].shape[0] if len(subs) > 1 else 0
        except Exception:
            after_rows = pre_rows = 0

        try:
            lg_items = self.viewer.plot.plotItem.legend.items
        except Exception:
            lg_items = []
        for idx, (_, lbl) in enumerate(lg_items):
            try:
                txt = lbl.text if hasattr(lbl, 'text') else lbl.text()
                if isinstance(txt, bytes):
                    txt = txt.decode(errors='ignore')
                # remove any existing suffix
                for suffix in (" con carga", " sin carga"):
                    if txt.endswith(suffix):
                        txt = txt[: -len(suffix)]
                if idx < after_rows:
                    lbl.setText(f"{txt} con carga")
                else:
                    lbl.setText(f"{txt} sin carga")
            except Exception:
                pass

        tbl = self.viewer.table
        col = 1
        for b in sorted(subs[0]['Bolt Num'].astype(int).unique()) if after_rows else []:
            hdr = tbl.horizontalHeaderItem(col)
            if hdr:
                txt = hdr.text()
                for suffix in (" con carga", " sin carga"):
                    if txt.endswith(suffix):
                        txt = txt[: -len(suffix)]
                if self.src_selector.currentText() == "Database":
                    for pref in ("Fichero 1-", "Fichero 2-"):
                        if txt.startswith(pref):
                            txt = txt[len(pref):]
                hdr.setText(f"{txt} con carga")
            col += 1
        for b in sorted(subs[1]['Bolt Num'].astype(int).unique()) if pre_rows else []:
            hdr = tbl.horizontalHeaderItem(col)
            if hdr:
                txt = hdr.text()
                for suffix in (" con carga", " sin carga"):
                    if txt.endswith(suffix):
                        txt = txt[: -len(suffix)]
                if self.src_selector.currentText() == "Database":
                    for pref in ("Fichero 1-", "Fichero 2-"):
                        if txt.startswith(pref):
                            txt = txt[len(pref):]
                hdr.setText(f"{txt} sin carga")
            col += 1

    def _sync_temp_options(self):
        """Update temperature dropdown to match current freq/gain/pulse."""
        if not self._temp_map:
            return
        try:
            f = int(float(self.viewer.freq_cb.currentText()))
            g = int(float(self.viewer.gain_cb.currentText()))
            p = float(self.viewer.pulse_cb.currentText())
        except Exception:
            return
        temps = sorted(self._temp_map.get((f, g, p), []))
        logger.debug("Temp options for F=%s G=%s P=%s -> %s", f, g, p, temps)
        cb = self.viewer.temp_cb
        cb.blockSignals(True)
        cb.clear()
        cb.addItems([str(t) for t in temps])
        cb.blockSignals(False)
        if temps:
            cb.setCurrentIndex(0)
        else:
            # No temp available; clear display and refresh
            cb.setCurrentText("")
            self.viewer._update_plot()
            self.viewer._update_table()

    def _on_source_changed(self):
        db_mode = self.src_selector.currentText() == "Database"
        for w in (self.viewer.file1_edit, self.viewer.browse1_btn,
                  self.viewer.file2_edit, self.viewer.browse2_btn,
                  self.viewer.load_btn):
            w.setVisible(not db_mode)
        self.load_db_btn.setVisible(db_mode)

    def _load_from_db(self):
        if not self.batch_selector or not self.db:
            QMessageBox.warning(self, "Batch ID", "Selecciona un Batch antes de cargar.")
            return
        batch_id = self.batch_selector.currentText()

        try:
            import pandas as pd, numpy as np

            # After load measurements
            # After-load (1-4) measurements: latest per bolt+combo
            try:
                self.db.cur.execute(
                    """
                    SELECT DISTINCT ON (bolt_id, freq, gain, pulse)
                           freq, gain, pulse, bolt_id, temp, pico1, pct_diff, tof, dat2, dat3
                    FROM one4_measurement
                    WHERE batch_id=%s
                    ORDER BY bolt_id, freq, gain, pulse, measured_at DESC
                    """,
                    (batch_id,),
                )
            except Exception:
                self.db.cur.execute(
                    "SELECT freq, gain, pulse, bolt_id, temp, pico1, pct_diff, tof, dat2, dat3 FROM one4_measurement WHERE batch_id=%s",
                    (batch_id,),
                )
            rows_after = self.db.cur.fetchall()
            cols_after = [d[0] for d in self.db.cur.description]
            df_after = pd.DataFrame(rows_after, columns=cols_after)
            df_after.rename(columns={'freq': 'Freq', 'gain': 'Gain', 'pulse': 'Pulse', 'bolt_id': 'Bolt ID'}, inplace=True)
            df_after['Bolt Num'] = df_after['Bolt ID'].apply(lambda bid: self.db.get_bolt_alias(batch_id, bid))
            # Keep pulse with full precision, cast numeric columns explicitly
            df_after[['Freq', 'Gain', 'Bolt Num']] = df_after[['Freq', 'Gain', 'Bolt Num']].astype(int)
            df_after['Pulse'] = pd.to_numeric(df_after['Pulse'], errors='coerce')
            if 'temp' not in df_after.columns:
                df_after['temp'] = 0.0
            logger.debug("Loaded 'after' pulses: %s", df_after['Pulse'].unique())

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

            dat2_after = df_after['dat2'].apply(_bytes_to_int16_array)
            dat3_after = df_after['dat3'].apply(_bytes_to_int32_array)
            n2_after = int(dat2_after.iloc[0].size) if not dat2_after.empty else 1024
            n3_after = int(dat3_after.iloc[0].size) if not dat3_after.empty else 1024
            dat2_df_after = pd.DataFrame([(arr.tolist() + [0]*(n2_after - arr.size))[:n2_after] for arr in dat2_after],
                                        columns=[f'dat2_{i}' for i in range(n2_after)])
            dat3_df_after = pd.DataFrame([(arr.tolist() + [0]*(n3_after - arr.size))[:n3_after] for arr in dat3_after],
                                        columns=[f'dat3_{i}' for i in range(n3_after)])
            df_after = pd.concat([df_after.drop(columns=['dat2', 'dat3']), dat2_df_after, dat3_df_after], axis=1)

            # Pre-load measurements
            # Pre-load measurements: latest per bolt+combo
            try:
                self.db.cur.execute(
                    """
                    SELECT DISTINCT ON (bolt_id, freq, gain, pulse)
                           freq, gain, pulse, bolt_id, temp, pico1, pct_diff, tof, dat2, dat3
                    FROM pre_measurement
                    WHERE batch_id=%s
                    ORDER BY bolt_id, freq, gain, pulse, measured_at DESC
                    """,
                    (batch_id,),
                )
            except Exception:
                self.db.cur.execute(
                    "SELECT freq, gain, pulse, bolt_id, temp, pico1, pct_diff, tof, dat2, dat3 FROM pre_measurement WHERE batch_id=%s",
                    (batch_id,),
                )
            rows_pre = self.db.cur.fetchall()
            cols_pre = [d[0] for d in self.db.cur.description]
            df_pre = pd.DataFrame(rows_pre, columns=cols_pre)
            df_pre.rename(columns={'freq': 'Freq', 'gain': 'Gain', 'pulse': 'Pulse', 'bolt_id': 'Bolt ID'}, inplace=True)
            df_pre['Bolt Num'] = df_pre['Bolt ID'].apply(lambda bid: self.db.get_bolt_alias(batch_id, bid))
            df_pre[['Freq', 'Gain', 'Bolt Num']] = df_pre[['Freq', 'Gain', 'Bolt Num']].astype(int)
            df_pre['Pulse'] = pd.to_numeric(df_pre['Pulse'], errors='coerce')
            if 'temp' not in df_pre.columns:
                df_pre['temp'] = 0.0
            logger.debug("Loaded 'pre' pulses: %s", df_pre['Pulse'].unique())

            dat2_pre = df_pre['dat2'].apply(_bytes_to_int16_array)
            dat3_pre = df_pre['dat3'].apply(_bytes_to_int32_array)
            n2_pre = int(dat2_pre.iloc[0].size) if not dat2_pre.empty else 1024
            n3_pre = int(dat3_pre.iloc[0].size) if not dat3_pre.empty else 1024

            # Calculate theoretical ToF (ns) from ultrasonic_length or use reference_tof directly
            self.db.cur.execute(
                "SELECT ultrasonic_length, reference_tof FROM batch WHERE batch_id=%s",
                (batch_id,),
            )
            row_len = self.db.cur.fetchone()
            bolt_len = 0.0
            if row_len:
                ul_val, ref_tof = row_len
                if ul_val is not None:
                    bolt_len = float(ul_val)
                    tof_calc_ns = (2 * (bolt_len / 1000.0)) / 5900.0 * 1e9
                elif ref_tof is not None:
                    tof_calc_ns = float(ref_tof)
                else:
                    tof_calc_ns = 0.0
                    QMessageBox.warning(
                        self,
                        "Batch",
                        "Ni ultrasonic_length ni reference_tof definidos para el batch",
                    )
            else:
                tof_calc_ns = 0.0
                QMessageBox.warning(
                    self,
                    "Batch",
                    "Ni ultrasonic_length ni reference_tof definidos para el batch",
                )
            sample_ns = 8.0

            def _shift_array(arr: np.ndarray, shift: int) -> np.ndarray:
                arr = np.roll(arr, shift)
                if shift > 0:
                    arr[:shift] = 0
                elif shift < 0:
                    arr[shift:] = 0
                return arr

            shifted_dat2: list[np.ndarray] = []
            shifted_dat3: list[np.ndarray] = []
            shift_samples: list[int] = []
            shift_ns_list: list[float] = []
            for arr2, arr3, tof_pre, freq, gain, pulse, bolt_num in zip(
                dat2_pre, dat3_pre, df_pre['tof'], df_pre['Freq'],
                df_pre['Gain'], df_pre['Pulse'], df_pre['Bolt Num']
            ):
                if tof_pre is None:
                    shifted_dat2.append(arr2)
                    shifted_dat3.append(arr3)
                    shift_samples.append(0)
                    shift_ns_list.append(0.0)
                    print(f"[pre-shift] Bolt {bolt_num} F{freq} G{gain} P{pulse}: no ToF; shift=0")
                    continue
                tof_pre_ns = float(tof_pre)
                if tof_pre_ns < 1e3:  # stored in microseconds → convert to ns
                    tof_pre_ns *= 1e3
                diff_ns = tof_calc_ns - tof_pre_ns
                s = int(round(diff_ns / sample_ns))
                print(
                    f"[pre-shift] Bolt {bolt_num} F{freq} G{gain} P{pulse}: "
                    f"shift {s} samples ({diff_ns:.1f} ns)"
                )
                shifted_dat2.append(_shift_array(arr2, s))
                shifted_dat3.append(_shift_array(arr3, s))
                shift_samples.append(s)
                shift_ns_list.append(diff_ns)

            dat2_df_pre = pd.DataFrame(
                [(arr.tolist() + [0] * (n2_pre - arr.size))[:n2_pre] for arr in shifted_dat2],
                columns=[f'dat2_{i}' for i in range(n2_pre)]
            )
            dat3_df_pre = pd.DataFrame(
                [(arr.tolist() + [0] * (n3_pre - arr.size))[:n3_pre] for arr in shifted_dat3],
                columns=[f'dat3_{i}' for i in range(n3_pre)]
            )
            df_pre = pd.concat(
                [
                    df_pre.drop(columns=['dat2', 'dat3']).assign(
                        shift_ns=shift_ns_list, shift_samples=shift_samples
                    ),
                    dat2_df_pre,
                    dat3_df_pre,
                ],
                axis=1,
            )

        except Exception as e:
            QMessageBox.critical(self, "DB Error", f"Error cargando de DB: {e}")
            return

        df_after = df_after.drop_duplicates(subset=['Freq', 'Gain', 'Pulse', 'Bolt Num'], keep='first')
        df_pre = df_pre.drop_duplicates(subset=['Freq', 'Gain', 'Pulse', 'Bolt Num'], keep='first')

        combos_after = set(df_after[['Freq', 'Gain', 'Pulse', 'Bolt Num']].apply(tuple, axis=1))
        df_pre = df_pre[df_pre[['Freq', 'Gain', 'Pulse', 'Bolt Num']].apply(tuple, axis=1).isin(combos_after)]

        self.viewer._df_list = [df_after, df_pre]
        # Build temperature map for quick selector updates
        temp_map: dict[tuple[int, int, float], set[float]] = {}
        df_all = pd.concat([df_after, df_pre], ignore_index=True)
        for _, row in df_all.iterrows():
            key = (int(row['Freq']), int(row['Gain']), float(row['Pulse']))
            temp_map.setdefault(key, set()).add(float(row['temp']))
        self._temp_map = temp_map
        logger.debug("Built temp map with %d entries", len(temp_map))

        freqs = sorted(df_after['Freq'].astype(int).unique())
        self.viewer.freq_cb.clear(); self.viewer.freq_cb.addItems([str(f) for f in freqs]); self.viewer.freq_cb.setEnabled(True)
        gains = sorted(df_after['Gain'].astype(int).unique())
        self.viewer.gain_cb.clear(); self.viewer.gain_cb.addItems([str(g) for g in gains]); self.viewer.gain_cb.setEnabled(True)
        pulses = sorted(pd.to_numeric(df_after['Pulse'], errors='coerce').unique())
        self.viewer.pulse_cb.clear()
        self.viewer.pulse_cb.addItems([f"{p:g}" for p in pulses])
        self.viewer.pulse_cb.setEnabled(True)
        # Debug: show pulses populated
        logger.debug("Pulse selector populated with: %s", pulses)
        temps = sorted(df_after['temp'].unique())
        self.viewer.temp_cb.clear(); self.viewer.temp_cb.addItems([str(t) for t in temps]); self.viewer.temp_cb.setEnabled(True)

        bolt_model: QStandardItemModel = self.viewer.bolt_cb.model()
        bolt_model.clear()
        all_item = QStandardItem('All')
        all_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
        all_item.setCheckState(Qt.Checked)
        bolt_model.appendRow(all_item)
        for b in sorted(df_after['Bolt Num'].astype(int).unique()):
            it = QStandardItem(str(b))
            it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Unchecked)
            bolt_model.appendRow(it)
        self.viewer.bolt_cb.setEnabled(True)
        self.viewer.bolt_cb.lineEdit().setText('All')

        self.viewer.type_cb.clear(); self.viewer.type_cb.addItems(['dat2', 'dat3']); self.viewer.type_cb.setEnabled(True)

        if freqs:
            for cb in (self.viewer.freq_cb, self.viewer.gain_cb, self.viewer.pulse_cb,
                       self.viewer.bolt_cb, self.viewer.type_cb, self.viewer.temp_cb):
                cb.setCurrentIndex(0)

        # Sync temperature options for initial selection
        self._sync_temp_options()

        self.viewer._update_plot()
        self.viewer._update_table()
        self._apply_labels()

