from __future__ import annotations
"""Temperature‑Sequence tab

Adds a new tab that automatically drives the Weiss‑Technik ClimeEvent chamber
through the set‑points ‑20 → 0 → 20 → 40 → 50 °C.  When the temperature is
stable (\u00b1tol \u00baC for *hold_s* s) it triggers a Frequency Scan using the
exact same parameters that exist in *Frequency Scan*.

The implementation re‑uses **ScanWorker** from ``frequency_scan.py`` and the
shared utility functions from ``utils.py``.  Chamber control is handled via the
S!MPAC\u00ae SimServ protocol (TCP 2049) with just three commands:

* 11001 \u00b6 1 \u00b6 1 \u00b6 <setpoint>  → SET SETPOINT
* 11004 \u00b6 1 \u00b6 1                → GET ACTUAL VALUE
* 14001 \u00b6 1 \u00b6 1 \u00b6 1        → START MAN MODE

A small *SimServClient* wraps the low‑level socket I/O, and *ChamberController*
provides a more pythonic façade.

Add one line to *main.py*:
```python
from climate_sequence_tab import TemperatureSequenceTab
...
tabs.addTab(TemperatureSequenceTab(self.com_combo), "Temp Sequence")
```
"""

import socket
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import Qt

import numpy as np
import pyqtgraph as pg

# Local imports from the existing code‑base
from utils import labeled_spinbox, thin_separator
from tabs.frequency_scan import ScanWorker as FreqScanWorker
from device import Device, _pack32      # Re‑use serial Scan worker

################################################################################
# ─── Low‑level SimServ helper ────────────────────────────────────────────────
################################################################################

DELIM = b"\xB6"      # \u00b6 ¶
CR    = b"\r"
BUFFER = 512

class SimServClient:
    """Very small wrapper around the TCP socket used by SimServ."""

    def __init__(self, host: str, port: int = 2049, *, timeout: float = 5.0):
        self._sock = socket.create_connection((host, port), timeout=timeout)
        self._sock.settimeout(timeout)

    # .....................................................................
    def _make_cmd(self, cmd: str, *args: str) -> bytes:
        parts = [cmd, "1", *args]                     # chamber index always 1
        return DELIM.join(p.encode("ascii") for p in parts) + CR

    # .....................................................................
    def query(self, cmd: str, *args: str) -> str:
        """Send command and return raw ASCII payload of the response."""
        buf = self._make_cmd(cmd, *args)
        self._sock.sendall(buf)
        data = self._sock.recv(BUFFER).rstrip()        # keep it simple
        # Response format: "1¶value<CR><LF>"  → split by DELIM, ignore index
        parts = data.split(DELIM)
        if not parts:
            raise RuntimeError("Empty response from chamber")
        status = parts[0].decode()
        if status != "1":
            raise RuntimeError(f"SimServ error status {status!r}")
        return parts[1].decode() if len(parts) > 1 else ""

    # .....................................................................
    def close(self):
        try:
            self._sock.close()
        except Exception:
            pass

################################################################################
# ─── High‑level chamber façade ───────────────────────────────────────────────
################################################################################

class ChamberController:
    """Convenience wrapper around the SimServClient for temperature control."""

    def __init__(self, host: str):
        self._cli = SimServClient(host)
        # make sure manual mode is running
        self._cli.query("14001", "1", "1")   # START MAN MODE

    # .....................................................................
    def set_setpoint(self, deg_c: float):
        self._cli.query("11001", "1", f"{deg_c:.2f}")

    # .....................................................................
    def get_actual(self) -> float:
        val = self._cli.query("11004", "1")  # ← returns ASCII float
        return float(val)

    # .....................................................................
    def wait_stable(self, target: float, *, tol: float = 0.3,
                    hold_s: int = 60, poll_s: float = 5.0,
                    callback=None) -> None:
        """Block until |T‑target|<tol for *hold_s* seconds."""
        okay_since = None
        while True:
            t = self.get_actual()
            if callback:
                callback(t)
            if abs(t - target) < tol:
                if okay_since is None:
                    okay_since = time.monotonic()
                if time.monotonic() - okay_since >= hold_s:
                    return
            else:
                okay_since = None
            time.sleep(poll_s)

    # .....................................................................
    def close(self):
        self._cli.close()

################################################################################
# ─── Worker that drives the whole sequence ───────────────────────────────────
################################################################################

class TempSequenceWorker(QtCore.QThread):
    progress      = QtCore.pyqtSignal(str)          # status text
    temperature   = QtCore.pyqtSignal(float)        # live temperature
    step_done     = QtCore.pyqtSignal(float)        # temp reached & scan finished
    finished      = QtCore.pyqtSignal()
    error         = QtCore.pyqtSignal(str)

    def __init__(self, chamber_host: str,
                 setpoints: List[float],
                 scan_params: Dict[str, Any],
                 com_port: str,
                 tol: float = 0.3, hold_s: int = 60):
        super().__init__()
        self._host   = chamber_host
        self._pts    = setpoints
        self._scan_p = scan_params
        self._com    = com_port
        self._tol    = tol
        self._hold   = hold_s
        self._abort  = False

    # .....................................................................
    def stop(self):
        self._abort = True

    # .....................................................................
    def run(self):
        try:
            chamber = ChamberController(self._host)
            for idx, sp in enumerate(self._pts, 1):
                if self._abort:
                    break
                self.progress.emit(f"[{idx}/{len(self._pts)}] Set‑point {sp:.0f} °C ...")
                chamber.set_setpoint(sp)
                chamber.wait_stable(sp, tol=self._tol, hold_s=self._hold,
                                      callback=self.temperature.emit)
                if self._abort:
                    break
                self.progress.emit(f"Temp {sp:.0f} °C stable → starting scan ...")
                self._run_scan()
                self.step_done.emit(sp)
            chamber.close()
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))

    # .....................................................................
    def _run_scan(self):
        """Run one Frequency Scan with exactly the same logic as the original tab."""
        # Build the serial Device just like FrequencyScanTab does
        dev  = Device(self._com, baudrate=115200, timeout=1)
        # Build the worker (ranges + const extracted from UI dict)
        f_list = np.arange(self._scan_p["freq_start"],
                           self._scan_p["freq_end"] + 0.1,
                           self._scan_p["freq_step"], dtype=np.float32)
        g_list = np.arange(self._scan_p["gain_start"],
                           self._scan_p["gain_end"] + 0.1,
                           self._scan_p["gain_step"], dtype=np.float32)
        p_list = np.arange(self._scan_p["pulse_start"],
                           self._scan_p["pulse_end"] + 0.1,
                           self._scan_p["pulse_step"], dtype=np.float32)
        const = self._scan_p["const"]                 # exactly as FrequencyScanTab
        algo  = self._scan_p["algo"]
        worker = FreqScanWorker(dev, ranges=(f_list, g_list, p_list),
                                const_params=const, algo_params=algo,
                                bolt_num=self._scan_p.get("bolt", 0))
        worker.start()
        worker.wait()          # ← block sequence thread until scan ends

################################################################################
# ─── GUI Tab ─────────────────────────────────────────────────────────────────
################################################################################

class TemperatureSequenceTab(QtWidgets.QWidget):
    """User interface glue‑code.

    *   Lets the user specify the chamber IP, tolerance, hold‑time and all the
        Frequency Scan parameters in a condensed form.
    *   Shows a live plot of the chamber temperature and a log area.
    """

    _DEFAULT_PTS       = [-20.0, 0.0, 20.0, 40.0, 50.0]
    _CHAMBER_IP        = "192.168.121.100"   # sensible Weiss default

    def __init__(self, com_selector: QtWidgets.QComboBox):
        super().__init__()
        self._com_selector = com_selector
        self._worker: TempSequenceWorker | None = None
        self._build_ui()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        left  = QtWidgets.QFormLayout(); left.setFormAlignment(Qt.AlignTop)
        right = QtWidgets.QVBoxLayout(); right.setSpacing(8)

        # ── Chamber connection ─────────────────────────────────────────
        self.host_edit   = QtWidgets.QLineEdit(self._CHAMBER_IP)
        self.tol_spin    = labeled_spinbox(0.05, 5.0, step=0.05, decimals=2)
        self.tol_spin.setValue(0.3)
        self.hold_spin   = labeled_spinbox(5, 600, step=5, decimals=0) ; self.hold_spin.setValue(60)
        left.addRow("Chamber IP",       self.host_edit)
        left.addRow("Tolerance (°C)",   self.tol_spin)
        left.addRow("Hold time (s)",     self.hold_spin)
        left.addRow(thin_separator())

        # ── Frequency‑Scan parameters (condensed) ──────────────────────
        self.freq_start = labeled_spinbox(2, 100, step=1,  decimals=0)
        self.freq_end   = labeled_spinbox(2, 100, step=1,  decimals=0)
        self.freq_step  = labeled_spinbox(1, 10,  step=1,  decimals=0)
        self.gain_start = labeled_spinbox(5, 80,  step=1,  decimals=0)
        self.gain_end   = labeled_spinbox(5, 80,  step=1,  decimals=0)
        self.gain_step  = labeled_spinbox(1, 10,  step=1,  decimals=0)
        self.pulse_start= labeled_spinbox(1, 16,  step=1,  decimals=0)
        self.pulse_end  = labeled_spinbox(1, 16,  step=1,  decimals=0)
        self.pulse_step = labeled_spinbox(1, 6,   step=1,  decimals=0)
        self.freq_start.setValue(24); self.freq_end.setValue(36); self.freq_step.setValue(1)
        self.gain_start.setValue(25); self.gain_end.setValue(25); self.gain_step.setValue(5)
        self.pulse_start.setValue(10); self.pulse_end.setValue(16); self.pulse_step.setValue(1)
        left.addRow("Freq start",  self.freq_start)
        left.addRow("Freq end",    self.freq_end)
        left.addRow("Freq step",   self.freq_step)
        left.addRow("Gain start",  self.gain_start)
        left.addRow("Gain end",    self.gain_end)
        left.addRow("Gain step",   self.gain_step)
        left.addRow("Pulse start", self.pulse_start)
        left.addRow("Pulse end",   self.pulse_end)
        left.addRow("Pulse step",  self.pulse_step)
        left.addRow(thin_separator())

        # ── Control buttons ────────────────────────────────────────────
        self.btn_start = QtWidgets.QPushButton("START", objectName="Play")
        self.btn_stop  = QtWidgets.QPushButton("STOP",  objectName="Stop")
        hb = QtWidgets.QHBoxLayout(); hb.addWidget(self.btn_start); hb.addWidget(self.btn_stop)
        left.addRow(hb)

        # ── Right‑hand side: live temperature plot + log ───────────────
        self.plot = pg.PlotWidget(title="Chamber temperature")
        self.plot.setMinimumHeight(240)
        self.plot.showGrid(x=True, y=True, alpha=0.2)
        self.plot.setLabel("bottom", "Time (s)")
        self.plot.setLabel("left",   "°C")
        self._t_curve = self.plot.plot([], [], pen=pg.mkPen("#00c853", width=2))
        # Disable mouse zoom/pan on temperature plot
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
        right.addWidget(self.plot, 1)

        self.log = QtWidgets.QTextEdit(readOnly=True)
        right.addWidget(self.log, 1)

        # ── Main layout ────────────────────────────────────────────────
        h = QtWidgets.QHBoxLayout(self)
        leftw = QtWidgets.QWidget(); leftw.setLayout(left)
        h.addWidget(leftw)
        h.addLayout(right, 1)

        # ── Signals ────────────────────────────────────────────────────
        self.btn_start.clicked.connect(self._on_start)
        self.btn_stop.clicked.connect(self._on_stop)
        self.btn_stop.setEnabled(False)

    # ------------------------------------------------------------------ helpers
    def _build_scan_param_dict(self) -> Dict[str, Any]:
        const = dict(
            temp=20.0, diftemp=np.float32(-103), tof=244500.0,
            xi=np.float32(0.8468), alpha=np.float32(10.417),
            short_corr=np.uint32(1000), long_corr=np.uint32(10),
            short_temp=np.uint32(1000), long_temp=np.uint32(990),
        )
        return dict(
            freq_start=float(self.freq_start.value()),
            freq_end  =float(self.freq_end.value()),
            freq_step =float(self.freq_step.value()),
            gain_start=float(self.gain_start.value()),
            gain_end  =float(self.gain_end.value()),
            gain_step =float(self.gain_step.value()),
            pulse_start=float(self.pulse_start.value()),
            pulse_end =float(self.pulse_end.value()),
            pulse_step=float(self.pulse_step.value()),
            const=const,
            algo=(np.uint32(0), np.uint32(0)),   # Absolute maximum, no threshold
        )

    # ------------------------------------------------------------------ slots
    def _on_start(self):
        if not self._com_selector.currentText():
            QtWidgets.QMessageBox.warning(self, "COM", "Selecciona un puerto COM válido")
            return
        host = self.host_edit.text().strip()
        if not host:
            QtWidgets.QMessageBox.warning(self, "Host", "Introduce la IP del ClimeEvent")
            return
        pts = self._DEFAULT_PTS
        scan_p = self._build_scan_param_dict()
        self._worker = TempSequenceWorker(host, pts, scan_p,
                                          self._com_selector.currentText(),
                                          tol=float(self.tol_spin.value()),
                                          hold_s=int(self.hold_spin.value()))
        self._worker.progress.connect(self._log)
        self._worker.temperature.connect(self._update_plot)
        self._worker.step_done.connect(lambda sp: self._log(f"Scan at {sp:.0f} °C done"))
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)

    def _on_stop(self):
        if self._worker and self._worker.isRunning():
            self._worker.stop()
            self._worker.wait()
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

    def _on_finished(self):
        self._log("Sequence finished ✅")
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

    def _on_error(self, msg: str):
        QtWidgets.QMessageBox.critical(self, "Sequence", msg)
        self._on_stop()

    # ------------------------------------------------------------------ UI helpers
    def _log(self, txt: str):
        self.log.append(f"<b>{time.strftime('%H:%M:%S')}</b>  {txt}")
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())

    def _update_plot(self, t: float):
        curve = self._t_curve
        x, y = curve.getData()
        if x is None:
            x, y = [], []
        now = time.time() if not x else x[-1] + 1
        x = np.append(x, now)
        y = np.append(y, t)
        curve.setData(x, y, autoDownsample=True)
