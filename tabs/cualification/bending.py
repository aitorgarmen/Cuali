# tabs/qualification/bending.py
from __future__ import annotations

from typing import Any, Dict, List, Optional

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QComboBox,
    QLabel,
    QHBoxLayout,
    QVBoxLayout,
    QGridLayout,
    QPushButton,
    QWidget,
)
import pyqtgraph as pg
from serial.tools import list_ports

from pgdb import BatchDB
from tabs.dj import DeviceWorker
from device import Device


class BendingTab(QWidget):
    """Bending con 5 bolts x 3 posiciones.

    - Escaneo por bolt (resuelve batch automáticamente a partir del Bolt ID).
    - 3 botones de posición por bolt: azul (escaneado), amarillo (en curso), verde (terminado).
    - Play/Stop por selección; se deshabilita el resto mientras procesa.
    - Guarda streaming en bending_loading, y medidas initial/final en bending_measurement.
    """

    def __init__(
        self,
        db: BatchDB | None = None,
        com_selector: Optional[QComboBox] = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.db = db or BatchDB()
        self.com_selector = com_selector

        self._device: Device | None = None
        self._worker: DeviceWorker | None = None
        self._active_params: Dict[str, Any] | None = None

        # Estado 5x3
        self._num_bolts = 5
        self._scan_buttons: List[QPushButton] = []
        self._pos_buttons: List[QPushButton] = []
        self._bolt_codes: List[Optional[str]] = [None] * self._num_bolts
        self._bolt_batches: List[Optional[str]] = [None] * self._num_bolts
        self._bolt_params: List[Dict[str, Any]] = [{} for _ in range(self._num_bolts)]
        self._pos_processed: List[List[bool]] = [[False]*3 for _ in range(self._num_bolts)]
        self._initial_force: List[List[Optional[float]]] = [[None]*3 for _ in range(self._num_bolts)]
        self._last_frames: List[List[Dict[str, Any] | None]] = [[None]*3 for _ in range(self._num_bolts)]
        self._initial_saved: List[List[bool]] = [[False]*3 for _ in range(self._num_bolts)]
        self._seq: List[List[int]] = [[0]*3 for _ in range(self._num_bolts)]
        self._current_bolt_idx: Optional[int] = None
        self._current_pos_idx: Optional[int] = None

        # Buffers y plots por posición (3 filas de gráficos)
        self._tof_buffers: List[List[float]] = [[] for _ in range(3)]
        self.signal_plots: List[pg.PlotWidget] = []
        self.signal_curves: List[Any] = []
        self.markers: List[pg.ScatterPlotItem] = []
        self.threshold_lines: List[pg.InfiniteLine] = []
        self.tof_plots: List[pg.PlotWidget] = []
        self.tof_curves: List[Any] = []

        self._build_ui()

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        # Matriz de botones 5x(Scan+3pos)
        grid_btns = QGridLayout()
        # Cabeceras centradas y con etiquetas completas
        bolt_hdr = QLabel("Bolt"); scan_hdr = QLabel("Scan"); pos1_hdr = QLabel("Position 1"); pos2_hdr = QLabel("Position 2"); pos3_hdr = QLabel("Position 3")
        grid_btns.addWidget(bolt_hdr, 0, 0, alignment=Qt.AlignCenter)
        grid_btns.addWidget(scan_hdr, 0, 1, alignment=Qt.AlignCenter)
        grid_btns.addWidget(pos1_hdr, 0, 2, alignment=Qt.AlignCenter)
        grid_btns.addWidget(pos2_hdr, 0, 3, alignment=Qt.AlignCenter)
        grid_btns.addWidget(pos3_hdr, 0, 4, alignment=Qt.AlignCenter)

        # NUEVO: fila de 5 botones para escanear/seleccionar bolts
        bolts_row = QHBoxLayout()
        bolts_row.addWidget(QLabel("Bolts:"))
        for i in range(self._num_bolts):
            b = QPushButton(f"Bolt {i+1}", objectName="File")
            b.setCheckable(True)
            b.setMinimumWidth(120)
            b.clicked.connect(lambda _=False, bi=i: self._on_bolt_button_clicked(bi))
            self._scan_buttons.append(b)
            bolts_row.addWidget(b, alignment=Qt.AlignCenter)

        # NUEVO: fila global de 3 posiciones
        pos_row = QHBoxLayout()
        pos_row.addWidget(QLabel("Positions:"))
        for j in range(3):
            p = QPushButton(f"Position {j+1}", objectName="File")
            p.setCheckable(True)
            p.setMinimumWidth(120)
            p.setEnabled(False)
            p.clicked.connect(lambda _=False, pj=j: self._on_click_pos(pj))
            self._pos_buttons.append(p)
            pos_row.addWidget(p, alignment=Qt.AlignCenter)
        # Guardar referencias a los botones nuevos para evitar duplicados del grid antiguo
        new_scan_buttons = list(self._scan_buttons)
        new_pos_buttons = list(self._pos_buttons)
        # Matriz antigua (no usada en layout principal)
        for i in range(self._num_bolts):
            grid_btns.addWidget(QLabel(f"{i+1}"), i+1, 0, alignment=Qt.AlignCenter)
            scan = QPushButton("Scan", objectName="File")
            # Asegurar que el texto quepa y esté centrado
            scan.setMinimumWidth(120)
            scan.clicked.connect(lambda _=False, bi=i: None)
            self._scan_buttons.append(scan)
            grid_btns.addWidget(scan, i+1, 1, alignment=Qt.AlignCenter)
            row: List[QPushButton] = []
            for j in range(3):
                p = QPushButton(f"Position {j+1}", objectName="File")
                p.setCheckable(True)
                # Asegurar que el texto "Position N" quepa
                p.setMinimumWidth(120)
                p.setEnabled(False)
                p.clicked.connect(lambda _=False, bi=i, pj=j: self._on_click_pos(bi, pj))
                row.append(p)
                grid_btns.addWidget(p, i+1, 2 + j, alignment=Qt.AlignCenter)
            self._pos_buttons.append(row)
        # Reasignar listas para quedarnos solo con los nuevos botones
        self._scan_buttons = new_scan_buttons
        self._pos_buttons = new_pos_buttons

        # Controles: Data (por defecto dat3) + info + acciones Play/Stop/Repeat/Delete
        self.data_cb = QComboBox(); self.data_cb.addItems(["dat2", "dat3"]); self.data_cb.setCurrentText("dat3")
        self.btn_play = QPushButton("PLAY", objectName="Play"); self.btn_play.setEnabled(False); self.btn_play.clicked.connect(self._on_play)
        self.btn_stop = QPushButton("STOP", objectName="Stop"); self.btn_stop.setEnabled(False); self.btn_stop.clicked.connect(self._on_stop)
        self.btn_delete = QPushButton("DELETE", objectName="Stop"); self.btn_delete.setEnabled(False); self.btn_delete.clicked.connect(self._on_delete_pos)
        ctrl_layout = QHBoxLayout()
        ctrl_layout.addWidget(QLabel("Data"))
        ctrl_layout.addWidget(self.data_cb)
        ctrl_layout.addSpacing(12)
        ctrl_layout.addWidget(QLabel("ToF:"))
        self.tof_label = QLabel("---"); ctrl_layout.addWidget(self.tof_label)
        ctrl_layout.addSpacing(24)
        ctrl_layout.addWidget(QLabel("Bolt ID:"))
        self.bolt_id_display = QLabel(""); ctrl_layout.addWidget(self.bolt_id_display)
        ctrl_layout.addStretch()
        ctrl_layout.addWidget(self.btn_play)
        ctrl_layout.addWidget(self.btn_stop)
        ctrl_layout.addWidget(self.btn_delete)

        # Plots 3 filas x 2 columnas (se muestran las 3 posiciones)
        grid = QGridLayout()
        for i in range(3):
            sp = pg.PlotWidget(title=f"Señal {i+1}")
            sp.setMinimumHeight(320)
            sp.showGrid(x=True, y=True, alpha=0.2)
            sp.setLabel("bottom", "Muestras")
            sp.setLabel("left", "Amplitud")
            curve = sp.plot([], [], pen=pg.mkPen("#00c853", width=2))
            marker = pg.ScatterPlotItem(size=8, pen=pg.mkPen("#ff5722", width=2), brush=pg.mkBrush("#ff5722"))
            sp.addItem(marker); marker.setVisible(False)
            thresh = pg.InfiniteLine(angle=0, pen=pg.mkPen('#FFEA00', width=1, style=Qt.DashLine), movable=False)
            sp.addItem(thresh); thresh.setVisible(False)
            # disable mouse
            try:
                _vb = sp.getPlotItem().getViewBox()
                _vb.setMouseEnabled(False, False); _vb.setMenuEnabled(False)
                sp.wheelEvent = lambda ev: None; _vb.wheelEvent = lambda ev: None
                _vb.mouseDoubleClickEvent = lambda ev: None; _vb.mouseDragEvent = lambda ev: None
            except Exception:
                pass
            self.signal_plots.append(sp); self.signal_curves.append(curve); self.markers.append(marker); self.threshold_lines.append(thresh)
            grid.addWidget(sp, i, 0)

            tp = pg.PlotWidget(title=f"ToF (µs) {i+1}")
            tp.setMinimumHeight(320)
            tp.showGrid(x=True, y=True, alpha=0.2)
            tp.setLabel("bottom", "Frame #")
            tp.setLabel("left", "ToF (µs)")
            tcurve = tp.plot([], [], pen=pg.mkPen("#ff9800", width=1))
            try:
                _vb2 = tp.getPlotItem().getViewBox()
                _vb2.setMouseEnabled(False, False); _vb2.setMenuEnabled(False)
                tp.wheelEvent = lambda ev: None; _vb2.wheelEvent = lambda ev: None
                _vb2.mouseDoubleClickEvent = lambda ev: None; _vb2.mouseDragEvent = lambda ev: None
            except Exception:
                pass
            self.tof_plots.append(tp); self.tof_curves.append(tcurve)
            grid.addWidget(tp, i, 1)

        # Layout principal
        main = QVBoxLayout(self)
        # Centrar nuevas filas de botones
        bolts_wrap = QHBoxLayout(); bolts_wrap.addStretch(1); bolts_wrap.addLayout(bolts_row); bolts_wrap.addStretch(1)
        pos_wrap = QHBoxLayout(); pos_wrap.addStretch(1); pos_wrap.addLayout(pos_row); pos_wrap.addStretch(1)
        main.addLayout(bolts_wrap)
        main.addLayout(pos_wrap)
        main.addLayout(ctrl_layout)
        plots_container = QWidget(); plots_container.setLayout(grid)
        scroll = QtWidgets.QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(plots_container)
        main.addWidget(scroll)

        self._refresh_button_styles()

    # ------------------------------------------------------------------ botón Scan
    def _on_bolt_button_clicked(self, bolt_idx: int) -> None:
        # Si no está escaneado aún, pedir ID y resolver batch/params
        if self._bolt_codes[bolt_idx] is None:
            code, ok = QtWidgets.QInputDialog.getText(self, "Scan bolt", f"Bolt {bolt_idx+1} ID:")
            code = code.strip() if isinstance(code, str) else ""
            if not ok or not code:
                try:
                    self._scan_buttons[bolt_idx].setChecked(False)
                except Exception:
                    pass
                return
            try:
                batch_id = self.db.find_batch_by_bolt(code)
            except Exception:
                batch_id = None
            if not batch_id:
                QtWidgets.QMessageBox.warning(self, "Bolt", f"Bolt '{code}' no encontrado en la BBDD")
                try:
                    self._scan_buttons[bolt_idx].setChecked(False)
                except Exception:
                    pass
                return
            self._bolt_codes[bolt_idx] = code
            self._bolt_batches[bolt_idx] = batch_id
            try:
                params = self.db.get_batch_params(batch_id)
            except Exception:
                params = {}
            # Intentar ToF de PRE
            try:
                if params:
                    pre_tof = self.db.get_pre_tof(
                        batch_id,
                        code,
                        int(params.get('freq', 0)),
                        int(params.get('gain', 0)),
                        int(params.get('pulse', 0))
                    )
                    if pre_tof is not None:
                        params['tof'] = pre_tof
            except Exception:
                pass
            self._bolt_params[bolt_idx] = params
            # Si todavía no están escaneados los 5, no seleccionar aún
            if not self._all_bolts_scanned():
                try:
                    self._scan_buttons[bolt_idx].setChecked(False)
                except Exception:
                    pass
                self._refresh_button_styles()
                return

        # Si ya están escaneados los 5, permitir seleccionar un bolt no terminado
        if self._bolt_done(bolt_idx):
            QtWidgets.QMessageBox.information(self, "Bolt", f"Bolt {bolt_idx+1} ya está completamente procesado")
            try:
                self._scan_buttons[bolt_idx].setChecked(False)
            except Exception:
                pass
            return

        # Seleccionar este bolt para procesar
        self._current_bolt_idx = bolt_idx
        self._active_params = self._bolt_params[bolt_idx] if self._bolt_params[bolt_idx] else None
        # Selección única
        for i, b in enumerate(self._scan_buttons):
            if i != bolt_idx:
                b.setChecked(False)
        self._scan_buttons[bolt_idx].setChecked(True)
        # habilitar posiciones disponibles
        for j, pbtn in enumerate(self._pos_buttons):
            # Permitir repetir: habilitar incluso si ya está procesada
            pbtn.setEnabled(self._all_bolts_scanned())
            pbtn.setChecked(False)
        self._current_pos_idx = None
        self.btn_play.setEnabled(False)
        # Mostrar Bolt ID si ya están los 5 escaneados
        if self._all_bolts_scanned():
            self.bolt_id_display.setText(self._bolt_codes[bolt_idx] or "")
        else:
            self.bolt_id_display.setText("")
        # Botones repeat/delete dependen de selección de posición
        self.btn_delete.setEnabled(bool(self._all_bolts_scanned()))
        self._refresh_button_styles()

    # ------------------------------------------------------------------ click posición
    def _on_click_pos(self, pos_idx: int) -> None:
        bi = self._current_bolt_idx
        if bi is None:
            try:
                self._pos_buttons[pos_idx].setChecked(False)
            except Exception:
                pass
            return
        if not self._all_bolts_scanned():
            QtWidgets.QMessageBox.information(self, "Bolts", "Escanea primero los 5 bolts")
            self._pos_buttons[pos_idx].setChecked(False)
            return
        # Permitir repetir: continuar aunque ya estuviera procesada
        if self._bolt_codes[bi] is None or not self._bolt_params[bi]:
            QtWidgets.QMessageBox.warning(self, "Bolt", "Escanea primero el Bolt")
            self._pos_buttons[pos_idx].setChecked(False)
            return
        self._current_pos_idx = pos_idx
        val, ok = QtWidgets.QInputDialog.getDouble(
            self, "Real force", f"Fuerza inicial Bolt {bi+1} Position {pos_idx+1}:", 0.0, -1e9, 1e9, 2
        )
        if not ok:
            self._pos_buttons[pos_idx].setChecked(False)
            self._current_pos_idx = None
            return
        self._initial_force[bi][pos_idx] = float(val)
        self._active_params = self._bolt_params[bi]
        self.btn_play.setEnabled(True)
        # Habilitar repetir si esta posición ya estaba procesada (permite repetir)
        self.btn_delete.setEnabled(True)
        self._refresh_button_styles()

    # ------------------------------------------------------------------ play/stop
    def _on_play(self) -> None:
        if self.com_selector:
            ports = [p.device for p in list_ports.comports()]
            self.com_selector.clear();
            if ports:
                self.com_selector.addItems(ports)
        if not self._active_params:
            QtWidgets.QMessageBox.warning(self, "PLAY", "Primero selecciona un bolt y posición.")
            return
        if self._current_bolt_idx is None or self._current_pos_idx is None:
            QtWidgets.QMessageBox.warning(self, "Selección", "Selecciona Bolt y Posición.")
            return
        if not self.com_selector or not self.com_selector.currentText():
            QtWidgets.QMessageBox.critical(self, "COM", "Selecciona un puerto COM válido.")
            return
        # Abrir dispositivo
        try:
            self._device = Device(self.com_selector.currentText(), baudrate=115200, timeout=1)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "UART", f"No se pudo abrir el puerto:\n{e}")
            return
        # Restaurar selección de COM si sigue existiendo
        prev = self.com_selector.currentText() if self.com_selector else ''
        ports = [p.device for p in list_ports.comports()]
        self.com_selector.clear(); self.com_selector.addItems(ports)
        idx = self.com_selector.findText(prev)
        if idx != -1:
            self.com_selector.setCurrentIndex(idx)

        # Reset plots y estados para la posición
        bi = self._current_bolt_idx; pi = self._current_pos_idx
        try:
            self._tof_buffers[pi] = []
        except Exception:
            pass
        for k, (curve, plot) in enumerate(zip(self.tof_curves, self.tof_plots)):
            if k == pi:
                curve.setData([], [])
                plot.enableAutoRange(x=True, y=True)
        for k, (sig, plot) in enumerate(zip(self.signal_curves, self.signal_plots)):
            if k == pi:
                sig.setData([], [])
                plot.enableAutoRange(x=True, y=True)
                try:
                    self.markers[k].setVisible(False)
                    self.threshold_lines[k].setVisible(False)
                except Exception:
                    pass
        bi = self._current_bolt_idx; pi = self._current_pos_idx
        self._initial_saved[bi][pi] = False
        self._last_frames[bi][pi] = None
        self._seq[bi][pi] = 0

        self._worker = DeviceWorker(self._device, self._collect_params, self)
        self._worker.data_ready.connect(self._update_ui, QtCore.Qt.QueuedConnection)
        self._worker.error.connect(self._on_worker_error, QtCore.Qt.QueuedConnection)
        self._worker.start()
        QtWidgets.QMessageBox.information(self, "PLAY", "Medida en tiempo real iniciada.")

        # Deshabilitar resto de botones mientras procesa
        self.btn_play.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_delete.setEnabled(False)
        for i, b in enumerate(self._scan_buttons):
            b.setEnabled(i == bi)
        for j, p in enumerate(self._pos_buttons):
            p.setEnabled(j == pi)
        # Habilitar DELETE durante la medida para cancelar sin guardar
        self.btn_delete.setEnabled(True)
        self._refresh_button_styles()

    def _on_stop(self) -> None:
        self._stop_worker_if_running()
        self.btn_stop.setEnabled(False)
        self.btn_play.setEnabled(False)
        bi = self._current_bolt_idx; pi = self._current_pos_idx
        if bi is not None and pi is not None:
            frame = self._last_frames[bi][pi]
            if frame is not None and (self._bolt_batches[bi] and self._bolt_codes[bi] and self._active_params):
                default_force = float(self._initial_force[bi][pi] or 0.0)
                val, ok = QtWidgets.QInputDialog.getDouble(
                    self, "Real force", f"Fuerza final Bolt {bi+1} Position {pi+1}:", default_force, -1e9, 1e9, 2
                )
                if ok:
                    data = frame.copy(); data.update({
                        "freq": self._active_params.get("freq"),
                        "gain": self._active_params.get("gain"),
                        "pulse": self._active_params.get("pulse"),
                        "force_load_cell": float(val),
                    })
                    try:
                        self.db.add_bending_measurement(self._bolt_batches[bi], self._bolt_codes[bi], pi+1, "final", data)
                    except Exception as e:
                        print(f"[BendingTab] DB final save error: {e}")
            # Marcar posición procesada y actualizar disponibilidad
            self._pos_processed[bi][pi] = True
            if self._bolt_done(bi):
                # Bolt completado: poner en verde y resetear posiciones
                self._scan_buttons[bi].setChecked(False)
                for p in self._pos_buttons:
                    p.setChecked(False)
                    p.setEnabled(False)
                self._current_bolt_idx = None
                self._current_pos_idx = None
            else:
                # Permitir seleccionar otras posiciones del mismo bolt
                for j, p in enumerate(self._pos_buttons):
                    # Permitir repetir: mantener habilitadas todas
                    p.setEnabled(True)
                self._current_pos_idx = None
            # Rehabilitar botones de bolt aún pendientes
            for i, b in enumerate(self._scan_buttons):
                b.setEnabled(self._bolt_codes[i] is not None)
            # Habilitar borrar esta posición
            self.btn_delete.setEnabled(True)
            self._refresh_button_styles()

    # ------------------------------------------------------------------ worker/params
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

    def _collect_params(self) -> Dict[str, Any]:
        assert self._active_params is not None, "_collect without params"
        freq = float(self._active_params.get("freq", 0))
        gain = float(self._active_params.get("gain", 0))
        pulse = float(self._active_params.get("pulse", 0))
        tof_val = self._active_params.get("tof")
        if tof_val is None:
            bolt_length = float(self._active_params.get("ul", 0.0))
            velocity = 5900.0
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
        return params

    def _update_ui(self, frame: Dict[str, Any]) -> None:
        if not frame:
            return
        # Posición activa
        if self._current_pos_idx is None:
            return
        idx = self._current_pos_idx
        dtype = self.data_cb.currentText()
        data = frame.get(dtype, frame.get("dat3", []))
        self.signal_curves[idx].setData(data, autoDownsample=True)
        self.signal_plots[idx].autoRange()
        if dtype == "dat3":
            mx = frame.get("maxcorrx", 0); my = frame.get("maxcorry", 0)
            self.markers[idx].setData([mx], [my]); self.markers[idx].setVisible(True)
        else:
            self.markers[idx].setVisible(False)
        if self._active_params and self._active_params.get("algo", 0) in (1, 2):
            y0 = self._active_params.get("threshold", 0)
            self.threshold_lines[idx].setValue(y0); self.threshold_lines[idx].setVisible(True)
        else:
            self.threshold_lines[idx].setVisible(False)
        tof_val = frame.get("tof")
        if tof_val is not None:
            self._tof_buffers[idx].append(tof_val)
            x = list(range(len(self._tof_buffers[idx])))
            self.tof_curves[idx].setData(x, self._tof_buffers[idx])
            self.tof_plots[idx].autoRange()
            try:
                self.tof_label.setText(f"{float(tof_val):.2f}")
            except Exception:
                self.tof_label.setText(str(tof_val))

        # Guardado DB
        bi = self._current_bolt_idx; pi = self._current_pos_idx
        if bi is None or pi is None:
            return
        self._last_frames[bi][pi] = frame.copy()
        if not self._initial_saved[bi][pi] and self._active_params:
            data0 = frame.copy(); data0.update({
                "freq": self._active_params.get("freq"),
                "gain": self._active_params.get("gain"),
                "pulse": self._active_params.get("pulse"),
                "force_load_cell": float(self._initial_force[bi][pi] or 0.0),
            })
            try:
                self.db.add_bending_measurement(self._bolt_batches[bi], self._bolt_codes[bi], pi+1, "initial", data0)
                self._initial_saved[bi][pi] = True
            except Exception as e:
                print(f"[BendingTab] DB initial save error: {e}")
        # loading streaming
        try:
            self.db.add_bending_loading(self._bolt_batches[bi], self._bolt_codes[bi], pi+1, int(self._seq[bi][pi]), frame)
            self._seq[bi][pi] += 1
        except Exception as e:
            print(f"[BendingTab] DB loading save error: {e}")

    def _on_worker_error(self, msg: str) -> None:
        QtWidgets.QMessageBox.critical(self, "Error", msg)
        self._on_stop()

    # ------------------------------------------------------------------ delete pos
    def _on_delete_pos(self) -> None:
        bi = self._current_bolt_idx; pi = self._current_pos_idx
        if bi is None or pi is None:
            QtWidgets.QMessageBox.information(self, "Delete", "Selecciona un bolt y una posicion.")
            return
        # Si esta ejecutando, parar
        if self._worker and self._worker.isRunning():
            self._stop_worker_if_running()
        # Borrar en BBDD
        try:
            self.db.delete_bending_data(self._bolt_batches[bi], self._bolt_codes[bi], pi+1)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Delete", f"No se pudo borrar en BBDD: {e}")
        # Reset de estado de esa posicion (manteniendo el ID del bolt)
        self._pos_processed[bi][pi] = False
        self._initial_saved[bi][pi] = False
        self._last_frames[bi][pi] = None
        self._seq[bi][pi] = 0
        self._initial_force[bi][pi] = None
        self.btn_play.setEnabled(False); self.btn_stop.setEnabled(False)
        self.tof_label.setText("---")
        # habilitar posiciones no procesadas
        for j, p in enumerate(self._pos_buttons):
            p.setEnabled(not self._pos_processed[bi][j])
        self._refresh_button_styles()

    # ------------------------------------------------------------------ estilos
    def _set_btn_style(self, btn: QPushButton, color: Optional[str]) -> None:
        if color == "blue":
            btn.setStyleSheet("QPushButton{background:#BBDEFB;color:#202124;}")
        elif color == "yellow":
            btn.setStyleSheet("QPushButton{background:#FFF59D;color:#202124;}")
        elif color == "green":
            btn.setStyleSheet("QPushButton{background:#A5D6A7;color:#202124;}")
        else:
            btn.setStyleSheet("")

    def _refresh_button_styles(self) -> None:
        # Bolts: verde si completo; amarillo si seleccionado; azul si escaneado; default si no
        for i in range(self._num_bolts):
            if self._bolt_done(i):
                self._set_btn_style(self._scan_buttons[i], "green")
            elif self._current_bolt_idx is not None and self._current_bolt_idx == i:
                self._set_btn_style(self._scan_buttons[i], "yellow")
            elif self._bolt_codes[i] is not None:
                self._set_btn_style(self._scan_buttons[i], "blue")
            else:
                self._set_btn_style(self._scan_buttons[i], None)

        # Posiciones: reflejan el estado del bolt seleccionado
        bi = self._current_bolt_idx
        for j, p in enumerate(self._pos_buttons):
            if bi is None:
                self._set_btn_style(p, None)
            else:
                if self._pos_processed[bi][j]:
                    self._set_btn_style(p, "green")
                elif self._current_pos_idx is not None and self._current_pos_idx == j:
                    self._set_btn_style(p, "yellow")
                else:
                    self._set_btn_style(p, "blue")

    def _all_bolts_scanned(self) -> bool:
        return all(code is not None for code in self._bolt_codes)

    def _bolt_done(self, idx: int) -> bool:
        try:
            return all(self._pos_processed[idx])
        except Exception:
            return False
