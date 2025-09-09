# tabs/qualification/one_ten.py
"""Pestaña 1-10 con adquisición en tiempo real y 10 botones de bolt.

Características:
- 10 botones con colores como en Temp: azul (escaneado), amarillo (en curso), verde (terminado).
- Al pulsar cada botón: se pide Bolt ID, se resuelve su batch y se solicita Real force inicial en subventana (por defecto 0).
- Play solo cuando los 10 bolts están escaneados. Se procesan uno a uno.
- Sin dependencia de selector de batch: se usa el batch asociado al Bolt ID escaneado.
- En Stop, subventana para introducir Real force final.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Iterable

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
import pyqtgraph as pg
from serial.tools import list_ports

from tabs.dj import DeviceWorker  # type: ignore
from device import Device
from pgdb import BatchDB
import numpy as np


class OneTenTab(QWidget):
    """Pestaña 1-10 con lógica de cualificación per-bolt."""

    def __init__(
        self,
        db: BatchDB | None = None,
        com_selector: Optional[QComboBox] = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.db = db or BatchDB()
        self.com_selector = com_selector

        # Estado de adquisición
        self._device: Device | None = None
        self._worker: DeviceWorker | None = None
        self._active_params: Dict[str, Any] | None = None
        self._tof_buffer: List[float] = []
        self._current_batch: str | None = None
        self._current_bolt: str | None = None
        self._seq: int = 0
        self._initial_saved: bool = False
        self._last_frame: Dict[str, Any] | None = None
        self._initial_force: float = 0.0

        # Estado de 10 bolts
        self._num_bolts: int = 10
        self._current_idx: Optional[int] = None
        self._bolt_buttons: List[QPushButton] = []
        self._bolt_codes: List[Optional[str]] = [None] * self._num_bolts
        self._bolt_batches: List[Optional[str]] = [None] * self._num_bolts
        self._bolt_params: List[Dict[str, Any]] = [{} for _ in range(self._num_bolts)]
        self._bolt_initial_force: List[Optional[float]] = [None] * self._num_bolts
        self._bolt_processed: List[bool] = [False] * self._num_bolts

        # Compatibilidad con OneFourTab: campos no visibles
        self.batch_cb = QComboBox()          # no se añade al layout
        self.bolt_edit = BoltLineEdit()      # no se añade al layout
        self.force_edit = QLineEdit()        # no se añade al layout

        self._build_ui()

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:  # noqa: C901
        # Fila de 10 botones
        bolts_row = QHBoxLayout()
        bolts_row.addWidget(QLabel("Bolts:"))
        for i in range(self._num_bolts):
            btn = QPushButton(str(i + 1), objectName="File")
            btn.setCheckable(True)
            # Asegurar que el texto (p.ej. "10") quepa con el padding global
            btn.setMinimumWidth(72)
            btn.clicked.connect(lambda _=False, idx=i: self._on_bolt_button_clicked(idx))
            self._bolt_buttons.append(btn)
            bolts_row.addWidget(btn)

        # Controles de data + info + acciones
        controls_row = QHBoxLayout()
        self.data_cb = QComboBox(); self.data_cb.addItems(["dat2", "dat3"]); self.data_cb.setCurrentIndex(1)
        controls_row.addWidget(QLabel("Data"))
        controls_row.addWidget(self.data_cb)
        controls_row.addSpacing(12)
        controls_row.addWidget(QLabel("ToF:"))
        self.tof_label = QLabel("---"); controls_row.addWidget(self.tof_label)
        controls_row.addSpacing(24)
        controls_row.addWidget(QLabel("Bolt ID:"))
        self.bolt_id_display = QLabel(""); controls_row.addWidget(self.bolt_id_display)
        controls_row.addStretch()
        self.btn_play = QPushButton("PLAY", objectName="Play"); self.btn_play.setEnabled(False); self.btn_play.clicked.connect(self._on_play)
        self.btn_stop = QPushButton("STOP", objectName="Stop"); self.btn_stop.setEnabled(False); self.btn_stop.clicked.connect(self._on_stop)
        self.btn_delete = QPushButton("DELETE", objectName="Stop"); self.btn_delete.setEnabled(False); self.btn_delete.clicked.connect(self._on_delete_bolt)
        controls_row.addWidget(self.btn_play)
        controls_row.addWidget(self.btn_stop)
        controls_row.addWidget(self.btn_delete)

        # Gráficos
        plots_layout = QHBoxLayout()
        self.signal_plot = pg.PlotWidget(title="Señal")
        self.signal_plot.showGrid(x=True, y=True, alpha=0.2)
        self.signal_plot.setLabel("bottom", "Muestras")
        self.signal_plot.setLabel("left", "Amplitud")
        self.signal_curve = self.signal_plot.plot([], [], pen=pg.mkPen("#00c853", width=2))
        self.marker = pg.ScatterPlotItem(size=8, pen=pg.mkPen("#ff5722", width=2), brush=pg.mkBrush("#ff5722"))
        self.signal_plot.addItem(self.marker)
        self.marker.setVisible(False)
        self.threshold_line = pg.InfiniteLine(angle=0, pen=pg.mkPen('#FFEA00', width=1, style=Qt.DashLine), movable=False)
        self.signal_plot.addItem(self.threshold_line)
        self.threshold_line.setVisible(False)
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
        plots_layout.addWidget(self.signal_plot, 1)

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
        plots_layout.addWidget(self.tof_plot, 1)

        main_layout = QVBoxLayout(self)
        # Centrar la fila de botones
        bolts_wrap = QHBoxLayout()
        bolts_wrap.addStretch(1)
        bolts_wrap.addLayout(bolts_row)
        bolts_wrap.addStretch(1)
        main_layout.addLayout(bolts_wrap)
        main_layout.addLayout(controls_row)
        main_layout.addLayout(plots_layout)

        # -- Manual xi calculation controls (below bottom graphs) --------
        xi_row = QtWidgets.QHBoxLayout()
        xi_row.setContentsMargins(0, 8, 0, 0)
        xi_row.setSpacing(8)
        xi_row.setAlignment(Qt.AlignCenter)
        xi_row.addStretch()
        xi_row.addWidget(QtWidgets.QLabel("Batch num:"))
        self.xi_batch_cb = QtWidgets.QComboBox(self)
        self.xi_batch_cb.setMinimumWidth(200)
        xi_row.addWidget(self.xi_batch_cb)
        self.xi_btn = QtWidgets.QPushButton("Calculate xi")
        self.xi_btn.setCursor(Qt.PointingHandCursor)
        self.xi_btn.setStyleSheet(
            "QPushButton{background:#1976D2;color:#FFFFFF;border:none;border-radius:6px;padding:6px 14px;font-weight:600;}"
            "QPushButton:hover{background:#1E88E5;}"
            "QPushButton:pressed{background:#1565C0;}"
            "QPushButton:disabled{background:#90A4AE;color:#ECEFF1;}"
        )
        self.xi_btn.clicked.connect(self._on_manual_calc_xi)
        xi_row.addWidget(self.xi_btn)
        xi_row.addStretch()
        main_layout.addLayout(xi_row)
        # Populate eligible batches
        try:
            self._refresh_xi_batches()
        except Exception:
            pass

    # -------------------------------------------------------------- helpers
    def _all_bolts_scanned(self) -> bool:
        return all(code is not None for code in self._bolt_codes)

    def _set_btn_style(self, idx: int, color: Optional[str]) -> None:
        if not (0 <= idx < len(self._bolt_buttons)):
            return
        btn = self._bolt_buttons[idx]
        if color == "blue":
            btn.setStyleSheet("QPushButton{background:#BBDEFB;color:#202124;}")
        elif color == "yellow":
            btn.setStyleSheet("QPushButton{background:#FFF59D;color:#202124;}")
        elif color == "green":
            btn.setStyleSheet("QPushButton{background:#A5D6A7;color:#202124;}")
        else:
            btn.setStyleSheet("")

    def _refresh_button_styles(self, selected: Optional[int]) -> None:
        """Actualizar estilos de los 10 botones según su estado.

        - Verde: procesado
        - Amarillo: seleccionado (si no está procesado)
        - Azul: escaneado pero no seleccionado ni procesado
        - Default: sin escanear
        """
        for i in range(self._num_bolts):
            if self._bolt_processed[i]:
                self._set_btn_style(i, "green")
            elif selected is not None and i == selected:
                self._set_btn_style(i, "yellow")
            elif self._bolt_codes[i] is not None:
                self._set_btn_style(i, "blue")
            else:
                self._set_btn_style(i, None)

    # -------------------------------------------------------------- buttons
    def _on_bolt_button_clicked(self, idx: int) -> None:
        # Permitir seleccionar bolts ya finalizados para poder repetirlos/borrarlos

        self._current_idx = idx
        # Si no está escaneado, pedir Bolt ID
        if self._bolt_codes[idx] is None:
            bolt_id, ok = QtWidgets.QInputDialog.getText(self, "Scan bolt", f"Bolt {idx+1} ID:")
            bolt_id = bolt_id.strip() if isinstance(bolt_id, str) else ""
            if not ok or not bolt_id:
                self._bolt_buttons[idx].setChecked(False)
                self._current_idx = None
                return
            batch_id = self.db.find_batch_by_bolt(bolt_id)
            if not batch_id:
                QtWidgets.QMessageBox.warning(self, "Bolt", f"Bolt '{bolt_id}' no encontrado en la BBDD")
                self._bolt_buttons[idx].setChecked(False)
                self._current_idx = None
                return
            self._bolt_codes[idx] = bolt_id
            self._bolt_batches[idx] = batch_id
            try:
                self._bolt_params[idx] = self.db.get_batch_params(batch_id)
            except Exception:
                self._bolt_params[idx] = {}
            # Pedir fuerza inicial
            val, ok2 = QtWidgets.QInputDialog.getDouble(
                self, "Real force", f"Fuerza inicial para bolt {idx+1}:", 0.0, -1e9, 1e9, 2
            )
            if not ok2:
                self._bolt_codes[idx] = None
                self._bolt_batches[idx] = None
                self._bolt_params[idx] = {}
                self._bolt_buttons[idx].setChecked(False)
                self._current_idx = None
                return
            self._bolt_initial_force[idx] = float(val)

        # Seleccionar este bolt para medición
        self._active_params = self._bolt_params[idx] if self._bolt_params[idx] else None
        self._current_batch = self._bolt_batches[idx]
        self._current_bolt = self._bolt_codes[idx]
        # Poner amarillo el botón seleccionado inmediatamente
        self._refresh_button_styles(idx)
        # Play disponible si los 10 están escaneados y este bolt no ha sido procesado
        self.btn_play.setEnabled(self._all_bolts_scanned() and self._active_params is not None and not self._bolt_processed[idx])
        # Habilitar Play (permitir repetir)
        self.btn_play.setEnabled(self._all_bolts_scanned() and self._active_params is not None)
        # Info y acciones adicionales
        if self._all_bolts_scanned():
            self.bolt_id_display.setText(self._current_bolt or "")
        else:
            self.bolt_id_display.setText("")
        self.btn_delete.setEnabled(bool(self._current_batch and self._current_bolt))

    # ------------------------------------------------------------------ batch change (compat 1-4)
    def _on_batch_changed(self, *_args) -> None:
        """Compatibilidad con OneFour: al cambiar batch se reinicia estado.

        No se usa en el flujo 1-10 (sin selector de batch), pero OneFour
        conecta su combo de batch a este método.
        """
        # Parar cualquier adquisición en curso
        self._stop_worker_if_running()
        # Limpiar selección y deshabilitar Play/Stop
        self._active_params = None
        self._current_batch = None
        self._current_bolt = None
        self._last_frame = None
        self._seq = 0
        self._initial_saved = False
        self.btn_play.setEnabled(False)
        self.btn_stop.setEnabled(False)
        # Limpiar campo bolt_edit si existe
        try:
            if isinstance(getattr(self, 'bolt_edit', None), QLineEdit):
                self.bolt_edit.clear()
        except Exception:
            pass

    # ------------------------------------------------------------------ play
    def _on_play(self) -> None:
        require_all = not hasattr(self, 'batch_selector')
        if require_all and not self._all_bolts_scanned():
            QtWidgets.QMessageBox.warning(self, "Bolts", "Escanea primero los 10 Bolt IDs")
            return
        if self._current_idx is None and require_all:
            QtWidgets.QMessageBox.warning(self, "Bolt", "Selecciona un bolt para procesar")
            return
        if False and self._current_idx is not None:
            if self._bolt_processed[self._current_idx]:
                QtWidgets.QMessageBox.information(self, "Bolt", "Este bolt ya está procesado")
                return
        # Refrescar COMs
        if self.com_selector:
            ports = [p.device for p in list_ports.comports()]
            self.com_selector.clear()
            if ports:
                self.com_selector.addItems(ports)
        if not self._active_params:
            QtWidgets.QMessageBox.warning(self, "PLAY", "Primero valida un bolt y fuerza antes de PLAY.")
            return
        # Fuerza inicial
        if hasattr(self, 'batch_selector') and isinstance(getattr(self, 'force_edit', None), QLineEdit):
            try:
                txt = self.force_edit.text().strip()
                self._initial_force = float(txt) if txt else 0.0
            except Exception:
                self._initial_force = 0.0
        else:
            try:
                idx = int(self._current_idx)
                self._initial_force = float(self._bolt_initial_force[idx] or 0.0)
            except Exception:
                self._initial_force = 0.0
        if not self.com_selector or not self.com_selector.currentText():
            QtWidgets.QMessageBox.critical(self, "COM", "Selecciona un puerto COM válido.")
            return
        # Abrir dispositivo
        try:
            self._device = Device(self.com_selector.currentText(), baudrate=115200, timeout=1)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "UART", f"No se pudo abrir el puerto:\n{e}")
            return
        # Restaurar selección si procede
        previous_port = self.com_selector.currentText() if self.com_selector else ''
        ports = [p.device for p in list_ports.comports()]
        self.com_selector.clear(); self.com_selector.addItems(ports)
        idx_sel = self.com_selector.findText(previous_port)
        if idx_sel != -1:
            self.com_selector.setCurrentIndex(idx_sel)

        # Reset buffers y estado
        self._tof_buffer.clear(); self.tof_curve.setData([], []); self.tof_label.setText("---")
        self._seq = 0
        self._initial_saved = False
        self._last_frame = None

        # Hilo de adquisición
        self._worker = DeviceWorker(self._device, self._collect_params, self)
        self._worker.data_ready.connect(self._update_ui, Qt.QueuedConnection)
        self._worker.error.connect(self._on_worker_error, Qt.QueuedConnection)
        self._worker.start()
        QtWidgets.QMessageBox.information(self, "PLAY", "Medida en tiempo real iniciada.")

        # UI
        self.btn_play.setEnabled(False)
        self.btn_stop.setEnabled(True)
        # Permitir cancelar con DELETE durante la medida
        self.btn_delete.setEnabled(True)
        # Deshabilitar el resto de botones mientras se procesa
        if self._current_idx is not None:
            for i, btn in enumerate(self._bolt_buttons):
                btn.setEnabled(i == self._current_idx)
            # asegurar color amarillo del seleccionado
            self._refresh_button_styles(self._current_idx)

    # ------------------------------------------------------------------ stop
    def _on_stop(self) -> None:
        # Parar adquisición
        self._stop_worker_if_running()
        self.btn_stop.setEnabled(False)

        # Pedir fuerza final
        if self._current_batch and self._current_bolt and self._last_frame and self._active_params and self._current_idx is not None:
            default_val = float(self._initial_force)
            val, ok = QtWidgets.QInputDialog.getDouble(
                self, "Real force", f"Fuerza final para bolt {self._current_idx+1}:", default_val, -1e9, 1e9, 2
            )
            if ok:
                data = self._last_frame.copy()
                data.update({
                    "freq": self._active_params.get("freq"),
                    "gain": self._active_params.get("gain"),
                    "pulse": self._active_params.get("pulse"),
                    "force": float(val),
                })
                try:
                    self.db.add_one10_measurement(self._current_batch, self._current_bolt, "final", data)
                except Exception as e:
                    print(f"[OneTenTab] DB final save error: {e}")

        # Marcar bolt terminado
        if self._current_idx is not None:
            self._bolt_processed[self._current_idx] = True
            self._set_btn_style(self._current_idx, "green")
            self._bolt_buttons[self._current_idx].setChecked(False)

        # Reset estado activo
        self.btn_play.setEnabled(False)
        self.btn_delete.setEnabled(False)
        self._active_params = None
        self._current_bolt = None
        self._current_batch = None
        # Rehabilitar botones para poder repetir cualquier bolt
        for i, btn in enumerate(self._bolt_buttons):
            btn.setEnabled(True)
        self._current_idx = None

        # If all bolts are processed, compute xi automatically per batch
        try:
            if all(self._bolt_processed) and any(self._bolt_codes):
                self._auto_compute_and_save_xi()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Xi", f"No se pudo calcular xi: {e}")

    # ---------------------------------------------------------- compat 1-4
    def _on_force_edit_finished(self) -> None:
        """Compatibilidad para OneFour: guardar fuerza final desde force_edit.

        OneFour oculta la fuerza inline y pide el valor por diálogo propio,
        asignándolo en self.force_edit antes de llamar a este método.
        """
        if not (self._current_batch and self._current_bolt and self._last_frame and self._active_params):
            return
        try:
            force_val = float(self.force_edit.text())
        except Exception:
            QtWidgets.QMessageBox.warning(self, "Force", "Valor de fuerza inválido.")
            return
        data = self._last_frame.copy()
        data.update({
            "freq": self._active_params.get("freq"),
            "gain": self._active_params.get("gain"),
            "pulse": self._active_params.get("pulse"),
            "force": force_val,
        })
        try:
            self.db.add_one10_measurement(self._current_batch, self._current_bolt, "final", data)
        except Exception as e:
            print(f"[OneTenTab] DB final save error (compat 1-4): {e}")
        # No resets de UI aquí; OneFour seguirá con su flujo

    def _update_play_button(self) -> None:
        """Compatibilidad: habilitar Play según contexto (1-10 vs 1-4)."""
        if hasattr(self, 'batch_selector'):
            self.btn_play.setEnabled(self._active_params is not None)
        else:
            self.btn_play.setEnabled(self._all_bolts_scanned() and self._active_params is not None)

    # -------------------------------------------------------------- worker ctl
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

    # -------------------------------------------------------------- params for HW
    def _collect_params(self) -> Dict[str, Any]:
        assert self._active_params is not None, "_collect_params llamado sin params"
        freq = float(self._active_params.get("freq", 0))
        gain = float(self._active_params.get("gain", 0))
        pulse = float(self._active_params.get("pulse", 0))
        tof_val = self._active_params.get('tof')
        if tof_val is None:
            bolt_length = float(self._active_params.get("ul", 0.0))
            velocity = 5900.0  # m/s
            tof_val = (2 * bolt_length) / velocity * 1e6
        else:
            tof_val = float(tof_val)
        temp = 20.0
        algo = int(self._active_params.get("algo", 0))
        threshold = float(self._active_params.get("threshold", 0))
        params = {
            "freq": freq,
            "gain": gain,
            "pulse": pulse,
            "tof": tof_val,
            "temp": temp,
            "algo": algo,
            "threshold": threshold,
        }
        print(f"[OneTenTab] _collect_params returns: {params}")
        return params

    # -------------------------------------------------------------- update UI
    def _update_ui(self, frame: Dict[str, Any]) -> None:
        if not frame:
            return
        print(f"[OneTenTab] _update_ui frame: {frame}")
        # Señal
        dtype = self.data_cb.currentText()
        data = frame.get(dtype, frame.get("dat3", []))
        self.signal_curve.setData(data, autoDownsample=True)
        # Marcador para dat3
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
        # Línea de umbral
        if self._active_params and self._active_params.get("algo", 0) in (1, 2):
            y0 = self._active_params.get("threshold", 0)
            self.threshold_line.setValue(y0); self.threshold_line.setVisible(True)
        else:
            self.threshold_line.setVisible(False)
        # ToF acumulado
        tof_val = frame.get("tof")
        if tof_val is not None:
            self._tof_buffer.append(tof_val)
            self.tof_curve.setData(list(range(len(self._tof_buffer))), self._tof_buffer)
            try:
                self.tof_label.setText(f"{float(tof_val):.2f}")
            except Exception:
                self.tof_label.setText(str(tof_val))
        # Guardar DB
        if self._current_batch and self._current_bolt:
            # 1) Guardar medida inicial antes de comenzar la carga
            if not self._initial_saved and self._active_params:
                data_ini = frame.copy()
                data_ini.update({
                    "freq": self._active_params.get("freq"),
                    "gain": self._active_params.get("gain"),
                    "pulse": self._active_params.get("pulse"),
                    "force": self._initial_force,
                })
                try:
                    self.db.add_one10_measurement(self._current_batch, self._current_bolt, "initial", data_ini)
                    self._initial_saved = True
                except Exception as e:
                    print(f"[OneTenTab] DB initial save error: {e}")
            # 2) Guardar loading para cada frame
            try:
                self.db.add_one10_loading(self._current_batch, self._current_bolt, self._seq, frame)
            except Exception as e:
                print(f"[OneTenTab] DB loading save error: {e}")
            self._seq += 1
            self._last_frame = frame.copy()

    # -------------------------------------------------------------- errors/close
    def _on_worker_error(self, msg: str) -> None:
        QtWidgets.QMessageBox.critical(self, "Error", msg)
        self._on_stop()

    # -------------------------------------------------------------- repeat/delete
    def _on_delete_bolt(self) -> None:
        idx = self._current_idx
        if idx is None or not self._bolt_codes[idx] or not self._bolt_batches[idx]:
            QtWidgets.QMessageBox.information(self, "Delete", "Selecciona primero un bolt.")
            return
        # Si está ejecutando, parar
        if self._worker and self._worker.isRunning():
            self._stop_worker_if_running()
        # Borrar en BBDD
        try:
            self.db.delete_one10_data(self._bolt_batches[idx], self._bolt_codes[idx])
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Delete", f"No se pudo borrar en BBDD: {e}")
        # Reset estados (manteniendo ID escaneado)
        self._bolt_processed[idx] = False
        self._bolt_initial_force[idx] = None
        self._active_params = None
        self._current_batch = None
        self._current_bolt = None
        self._last_frame = None
        self._seq = 0
        self._initial_saved = False
        self._tof_buffer.clear(); self.tof_curve.setData([], [])
        self.tof_label.setText("---")
        self.btn_play.setEnabled(False); self.btn_stop.setEnabled(False)
        for i, btn in enumerate(self._bolt_buttons):
            btn.setEnabled(True)
        self._refresh_button_styles(None)

    # --------------------------- Xi helpers -------------------------------
    def _refresh_xi_batches(self) -> None:
        items: List[str] = []
        try:
            items = self.db.list_batches_with_one10_process()
        except Exception:
            items = []
        self.xi_batch_cb.blockSignals(True)
        self.xi_batch_cb.clear()
        self.xi_batch_cb.addItems(items)
        self.xi_batch_cb.blockSignals(False)

    def _on_manual_calc_xi(self) -> None:
        batch_id = self.xi_batch_cb.currentText().strip()
        if not batch_id:
            QtWidgets.QMessageBox.information(self, "Xi", "Selecciona un Batch num valido.")
            return
        try:
            xi, failures = self._compute_xi_for_batch(batch_id)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Xi", f"Error calculando xi: {e}")
            return
        if xi is None:
            QtWidgets.QMessageBox.information(self, "Xi", "No hay datos suficientes (inicial y final) para calcular xi.")
            return
        # Validar 3%
        failures = [f for f in failures if f.get('fail', False)]
        if not failures:
            try:
                self.db.update_xi(batch_id, float(xi))
            except Exception:
                pass
            QtWidgets.QMessageBox.information(self, "Xi", f"Batch {batch_id}: xi = {float(xi):.6g}\nGuardado en tabla batch.")
        else:
            # Construir mensaje de fallos con Bolt ID y Bolt Num
            msgs = []
            for f in failures:
                bid = f.get('bolt_id')
                bnum = self.db.get_bolt_num(batch_id, bid)
                dev = f.get('deviation')
                msgs.append(f"Bolt {bid} (Num {bnum}): desv = {dev*100:.2f}%")
            QtWidgets.QMessageBox.warning(
                self,
                "Xi",
                "No cumple el 3% en los siguientes bolts:\n" + "\n".join(msgs),
            )

    def _auto_compute_and_save_xi(self) -> None:
        # Agrupar bolts procesados por batch
        batch_bolts: Dict[str, List[str]] = {}
        for i, processed in enumerate(self._bolt_processed):
            if not processed:
                continue
            bid = self._bolt_batches[i]
            bolt_id = self._bolt_codes[i]
            if bid and bolt_id:
                batch_bolts.setdefault(bid, []).append(bolt_id)
        if not batch_bolts:
            return
        # Calcular por cada batch
        for batch_id, bolts in batch_bolts.items():
            try:
                xi, failures = self._compute_xi_for_batch(batch_id, restrict_bolts=bolts)
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Xi", f"Error calculando xi para batch {batch_id}: {e}")
                continue
            if xi is None:
                continue
            failures = [f for f in failures if f.get('fail', False)]
            if not failures:
                try:
                    self.db.update_xi(batch_id, float(xi))
                except Exception:
                    pass
                # Mostrar dialogo resumen
                dlg = QtWidgets.QDialog(self)
                dlg.setWindowTitle("Xi por batch")
                lay = QtWidgets.QVBoxLayout(dlg)
                lay.addWidget(QtWidgets.QLabel(f"Batch {batch_id}: xi = {float(xi):.6g}"))
                btn = QtWidgets.QPushButton("OK"); btn.clicked.connect(dlg.accept)
                lay.addWidget(btn)
                dlg.resize(360, 100)
                dlg.exec_()
            else:
                msgs = []
                for f in failures:
                    bid = f.get('bolt_id')
                    bnum = self.db.get_bolt_num(batch_id, bid)
                    dev = f.get('deviation')
                    msgs.append(f"Bolt {bid} (Num {bnum}): desv = {dev*100:.2f}%")
                QtWidgets.QMessageBox.warning(
                    self,
                    "Xi",
                    f"Batch {batch_id}: no cumple el 3% en:\n" + "\n".join(msgs),
                )

    def _compute_xi_for_batch(self, batch_id: str, restrict_bolts: Optional[Iterable[str]] = None) -> tuple[Optional[float], List[Dict[str, Any]]]:
        """Compute xi using initial/final frames per bolt of a batch.

        Returns (xi_average or None, details per bolt with deviation and fail flag).
        If restrict_bolts is provided, only consider those bolt IDs.
        """
        # Obtener mediciones inicial/final
        bolt_map = self.db.fetch_one10_initial_final(batch_id, bolt_ids=restrict_bolts)
        xi_vals: List[float] = []
        details: List[Dict[str, Any]] = []
        for bolt_id, stages in bolt_map.items():
            ini = stages.get('initial'); fin = stages.get('final')
            if not ini or not fin:
                continue
            try:
                tof_ini = float(ini.get('tof')) if ini.get('tof') is not None else None
                tof_fin = float(fin.get('tof')) if fin.get('tof') is not None else None
                lc_ini = float(ini.get('force_load_cell')) if ini.get('force_load_cell') is not None else None
                lc_fin = float(fin.get('force_load_cell')) if fin.get('force_load_cell') is not None else None
            except Exception:
                continue
            if None in (tof_ini, tof_fin, lc_ini, lc_fin):
                continue
            delta_tof = tof_fin - tof_ini
            delta_lc = lc_fin - lc_ini
            if not np.isfinite(delta_tof) or delta_tof == 0.0 or not np.isfinite(delta_lc):
                continue
            xi_bolt = delta_lc / delta_tof
            if np.isfinite(xi_bolt):
                xi_vals.append(float(xi_bolt))
            details.append({
                'bolt_id': bolt_id,
                'delta_tof': float(delta_tof),
                'delta_load': float(delta_lc),
            })
        if not xi_vals:
            return None, []
        xi_avg = float(np.mean(xi_vals))
        # Validacion 3%
        out_details: List[Dict[str, Any]] = []
        for d in details:
            delta_tof = d['delta_tof']; delta_lc = d['delta_load']
            if delta_lc == 0:
                deviation = float('inf')
            else:
                gurea_load = xi_avg * delta_tof
                deviation = (gurea_load / delta_lc) - 1.0
            fail = not (np.isfinite(deviation) and abs(deviation) < 0.03)
            out_details.append({**d, 'xi': xi_avg, 'gurea_load': xi_avg * delta_tof if np.isfinite(delta_tof) else None, 'deviation': deviation, 'fail': fail})
        return xi_avg, out_details

    def closeEvent(self, event):  # type: ignore[override]
        self._stop_worker_if_running()
        super().closeEvent(event)

class BoltLineEdit(QLineEdit):
    """QLineEdit que emite returnPressed con Enter/Return y limpia al click."""
    def keyPressEvent(self, ev):  # type: ignore[override]
        if ev.key() in (Qt.Key_Return, Qt.Key_Enter):
            self.returnPressed.emit()
        else:
            super().keyPressEvent(ev)

    def mousePressEvent(self, ev):  # type: ignore[override]
        self.clear()
        super().mousePressEvent(ev)

