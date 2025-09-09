# Minimal DJ widget with only graph and Start/Stop buttons
# -----------------------------------------------------------------------------
from __future__ import annotations

from typing import Any, Callable, Dict

from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg

from tabs.dj import DeviceWorker
from device import Device


class MiniDJ(QtWidgets.QWidget):
    """Simplified DJ measurement widget.

    Only shows the correlated signal graph and provides Start/Stop buttons.
    Parameters for the measurement are supplied programmatically via
    :meth:`set_params` and sent unchanged to :class:`DeviceWorker`.
    """

    started = QtCore.pyqtSignal()
    stopped = QtCore.pyqtSignal()

    def __init__(self, com_selector: QtWidgets.QComboBox | None = None,
                 parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.com_selector = com_selector
        self._device: Device | None = None
        self._worker: DeviceWorker | None = None
        self._params: Dict[str, Any] | None = None
        self.last_frame: Dict[str, Any] | None = None
        self._build_ui()

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        self.btn_start = QtWidgets.QPushButton("Start", objectName="Play")
        self.btn_stop = QtWidgets.QPushButton("Stop", objectName="Stop")
        self.btn_stop.setEnabled(False)
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addWidget(self.btn_start)
        btn_row.addWidget(self.btn_stop)

        self.plot = pg.PlotWidget(title="Correlated signal")
        self.plot.showGrid(x=True, y=True, alpha=0.2)
        self.plot.setLabel("bottom", "Samples")
        self.plot.setLabel("left", "Amplitude")
        self.plot.setMinimumHeight(320)
        self.curve = self.plot.plot([], [], pen=pg.mkPen("#00c853", width=2))
        self.marker = pg.ScatterPlotItem(
            size=8,
            pen=pg.mkPen("#ff5722", width=2),
            brush=pg.mkBrush("#ff5722"),
            symbol="o",
        )
        self.plot.addItem(self.marker)
        self.marker.setVisible(False)
        self.threshold_line = pg.InfiniteLine(
            angle=0,
            pen=pg.mkPen("#FFEA00", width=1, style=QtCore.Qt.DashLine),
            movable=False,
        )
        self.plot.addItem(self.threshold_line)
        self.threshold_line.setVisible(False)
        # Disable mouse zoom/pan on mini plot
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

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(btn_row)
        layout.addWidget(self.plot, 1)

        self.btn_start.clicked.connect(self._on_start)
        self.btn_stop.clicked.connect(self._on_stop)

    # ---------------------------------------------------------------- params
    def set_params(self, params: Dict[str, Any]) -> None:
        """Set measurement parameters sent to the device worker."""
        self._params = params

    # ---------------------------------------------------------------- start
    def _on_start(self) -> None:
        if not self.com_selector or not self.com_selector.currentText():
            QtWidgets.QMessageBox.critical(self, "COM", "Selecciona un puerto COM válido.")
            return
        if not self._params:
            QtWidgets.QMessageBox.warning(self, "Params", "Selecciona un bolt antes de empezar.")
            return
        try:
            self._device = Device(self.com_selector.currentText(), baudrate=115200, timeout=1)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "UART", f"No se pudo abrir el puerto:\n{e}")
            return

        self.curve.setData([])
        self.marker.setVisible(False)
        self.threshold_line.setVisible(False)

        self._worker = DeviceWorker(self._device, lambda: self._params or {}, self)
        self._worker.data_ready.connect(self._update_ui, QtCore.Qt.QueuedConnection)
        self._worker.error.connect(self._on_worker_error, QtCore.Qt.QueuedConnection)
        self._worker.start()
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.started.emit()

    # ---------------------------------------------------------------- stop
    def _on_stop(self) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.stop()
            self._worker.wait()
        self._worker = None
        if self._device:
            try:
                self._device.ser.close()
            except Exception:
                pass
            self._device = None
        self.btn_stop.setEnabled(False)
        self.btn_start.setEnabled(True)
        self.stopped.emit()

    # ---------------------------------------------------------------- update
    def _update_ui(self, frame: Dict[str, Any]) -> None:
        if not frame:
            return
        self.last_frame = frame
        data = frame.get("dat3", [])
        self.curve.setData(data, autoDownsample=True)
        self.marker.setData([frame.get("maxcorrx", 0)], [frame.get("maxcorry", 0)])
        self.marker.setVisible(True)
        if self._params and self._params.get("algo", 0) in (1, 2):
            y0 = float(self._params.get("threshold", 0))
            self.threshold_line.setValue(y0)
            self.threshold_line.setVisible(True)
        else:
            self.threshold_line.setVisible(False)

    # ---------------------------------------------------------------- error
    def _on_worker_error(self, msg: str) -> None:
        QtWidgets.QMessageBox.critical(self, "Error", msg)
        self._on_stop()

    # ---------------------------------------------------------------- close
    def closeEvent(self, event: QtCore.QEvent) -> None:  # type: ignore[override]
        self._on_stop()
        super().closeEvent(event)

