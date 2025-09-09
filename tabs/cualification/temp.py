# tabs/qualification/temp.py
from __future__ import annotations

from PyQt5.QtWidgets import (
    QLabel,
    QVBoxLayout,
    QWidget,
    QHBoxLayout,
    QSpinBox,
    QPushButton,
    QScrollArea,
    QDoubleSpinBox,
    QButtonGroup,
    QComboBox,
    QInputDialog,
    QTabWidget,
    QMessageBox,
)
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import Qt

import pyqtgraph as pg
import time
import pandas as pd
import numpy as np
import logging

from pgdb import BatchDB
from tabs.minidj import MiniDJ
from device import Device
from .simserv_oven import SimServOven, SimServError
from ..dj import DeviceWorker
from ..filter import FilterTab, FilterWorker
from ..csv_viewer import CSVViewerTab
import types

logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO)
logger.setLevel(logging.INFO)
logging.getLogger("paramiko").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)


def _compute_batch_setpoints(min_temp: float | None, max_temp: float | None) -> list[float]:
    """Return 5 setpoint temperatures [min, mid(min,20), 20, mid(20,max), max].

    Falls back to [-20, 0, 20, 40, 50] when min/max are missing or invalid.
    Values are rounded to one decimal and returned ascending.
    """
    try:
        base = 20.0
        if min_temp is None or max_temp is None:
            raise ValueError
        tmin = float(min_temp)
        tmax = float(max_temp)
        if not np.isfinite(tmin) or not np.isfinite(tmax):
            raise ValueError
        t2 = (tmin + base) / 2.0
        t4 = (tmax + base) / 2.0
        temps = [tmin, t2, base, t4, tmax]
        temps = sorted(round(float(t), 1) for t in temps)
        # Ensure uniqueness while preserving order
        out: list[float] = []
        for t in temps:
            if not out or abs(out[-1] - t) > 1e-6:
                out.append(t)
        # If collapsing reduced count, pad with defaults to keep 5 entries
        defaults = [-20.0, 0.0, 20.0, 40.0, 50.0]
        i = 0
        while len(out) < 5 and i < len(defaults):
            dt = defaults[i]
            if all(abs(x - dt) > 1e-6 for x in out):
                out.append(dt)
            i += 1
        return sorted(out)[:5]
    except Exception:
        return [-20.0, 0.0, 20.0, 40.0, 50.0]


class ComboScanWorker(QtCore.QThread):
    """Scan over a list of (freq, gain, pulse) combos and emit frames."""

    frame_ready = QtCore.pyqtSignal(dict)
    progress = QtCore.pyqtSignal(int)
    finished = QtCore.pyqtSignal(list)
    error = QtCore.pyqtSignal(str)

    def __init__(self, device: Device, combos: list[tuple[float, float, float]],
                 bolt_num: float, config: dict[str, float], *,
                 close_device: bool = True,
                 db: BatchDB | None = None,
                 batch_id: str | None = None):
        super().__init__()
        self._device = device
        self._combos = combos
        self._config = config
        self._bolt = bolt_num
        self._rows: list = []
        self._stop = False
        self._close = close_device
        self._db = db
        self._batch_id = batch_id

    def stop(self):
        self._stop = True

    def run(self):
        try:
            total = len(self._combos)
            cfg_worker = DeviceWorker(self._device, lambda: {}, db=self._db, batch_id=self._batch_id)
            for idx, (freq, gain, pulse) in enumerate(self._combos, start=1):
                if self._stop:
                    break
                tof_val = None
                if isinstance(self._config.get('tof_map'), dict):
                    tof_val = self._config['tof_map'].get((freq, gain, pulse))
                try:
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
                try:
                    self._device.enviar_temp()
                    self._device.start_measure()
                    frame = self._device.lectura()
                    frame["freq"] = freq
                    frame["gain"] = gain
                    frame["pulse"] = pulse
                except Exception as e:
                    self.error.emit(str(e))
                    break
                self._rows.append(frame)
                self.frame_ready.emit(frame)
                pct = int(idx / total * 100)
                self.progress.emit(pct)
        except Exception as e:
            self.error.emit(str(e))
        finally:
            if getattr(self, '_close', False):
                try:
                    self._device.ser.close()
                except Exception:
                    pass
            self.finished.emit(self._rows)

class _NoWheelPlot(pg.PlotWidget):
    """PlotWidget that ignores wheel events so the parent scroll area moves."""

    def wheelEvent(self, ev: QtGui.QWheelEvent) -> None:  # type: ignore[override]
        ev.ignore()


class RefreshComboBox(QComboBox):
    """QComboBox that refreshes its items whenever the popup is shown."""

    def __init__(self, refresher, parent=None):
        super().__init__(parent)
        self._refresher = refresher

    def showPopup(self) -> None:  # type: ignore[override]
        self._refresher()
        super().showPopup()


def _summary_header_labels(by_bolt: dict, rows: list[dict], targets: list[int], bolts: list[str]) -> list[str]:
    """Compute mean temperature labels per target across included bolts.

    Falls back to all rows near target if per-bolt latest map lacks values.
    """
    labels: list[str] = []
    for t in targets:
        vals: list[float] = []
        for b in bolts:
            rec = (by_bolt or {}).get(b, {}).get(t)
            if rec and rec.get("temp") is not None:
                try:
                    vals.append(float(rec.get("temp")))
                except Exception:
                    pass
        if not vals:
            try:
                vals = [float(r.get("temp")) for r in rows if r.get("temp") is not None and abs(float(r["temp"]) - t) <= 0.5]
            except Exception:
                vals = []
        mean = sum(vals)/len(vals) if vals else float(t)
        labels.append(f"{mean:.1f} \u00B0C")
    return labels

class TemperatureSummaryTab(QtWidgets.QWidget):
    """Summary view: ToF matrix by temperature per bolt + delta-ToF plot.

    - Top: Completed batches dropdown + data selector (dat2/dat3)
    - Table: rows per bolt (ordered by earliest timestamp), columns per 5 temp groups.
    - Click cell: opens signal viewer window with selected dat.
    - Bottom: plot lines per bolt of delta_tof vs temperature.
    """

    def __init__(self, *, db: BatchDB | None = None, batch_selector: QComboBox | None = None, parent=None) -> None:
        super().__init__(parent)
        self.db = db or BatchDB()
        self._batch_selector = batch_selector
        self._targets: list[int] = []
        self._cell_records: dict[tuple[int,int], dict] = {}

        root = QVBoxLayout(self)
        top = QHBoxLayout()
        top.addWidget(QLabel("Batches (temp process):"))
        self.cb_batches = RefreshComboBox(self._refresh_batches, self)
        top.addWidget(self.cb_batches, 1)
        # Signal selector moved into popup dialog
        top.addStretch()
        root.addLayout(top)

        # Table
        self.table = QtWidgets.QTableWidget(0, 0)
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(True)
        self.table.horizontalHeader().setStyleSheet(
            "QHeaderView::section{background:#263238;color:#ECEFF1;padding:4px;border:none;}"
        )
        try:
            self.table.verticalHeader().setStyleSheet(
                "QHeaderView::section{background:#263238;color:#ECEFF1;padding:4px;border:none;}"
            )
        except Exception:
            pass
        # Plot will be created below; aspect/labels set after creation
        self.table.setAlternatingRowColors(True)
        self.table.setStyleSheet(
            "QTableWidget{gridline-color:#CFD8DC;}"
            "QTableWidget::item{padding:2px;color:#263238;}"
            "QTableWidget::item:!selected{background:#FFFFFF;}"
            "QTableWidget::item:!selected:alternate{background:#F5F5F5;}"
        )
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        # Show all rows without internal scrollbars; scroll on the tab, not inside the table
        self.table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.table.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.table.cellClicked.connect(self._on_cell_clicked)
        root.addWidget(self.table, 3)

        # Plot delta-ToF
        self.plot = pg.PlotWidget(title="\u0394ToF vs Temperature")
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.plot.setMaximumWidth(900)
        self.plot.setMaximumHeight(900)
        # Square geometry enforced by resizeEvent
        try:
            self.plot.setLabel("left", "\u0394ToF (ns)")
            self.plot.setLabel("bottom", "Temperature (\u00B0C)")
        except Exception:
            # Fallback without unicode if needed
            self.plot.setLabel("left", "Delta ToF (ns)")
            self.plot.setLabel("bottom", "Temperature (C)")
        plot_row = QtWidgets.QHBoxLayout()
        plot_row.addStretch(1)
        plot_row.addWidget(self.plot)
        plot_row.addStretch(1)
        root.addLayout(plot_row)

        # Signals
        self.cb_batches.activated[str].connect(self._rebuild)

        # Initial load
        self._refresh_batches()
        if self.cb_batches.count() > 0:
            self.cb_batches.setCurrentIndex(0)
            self._rebuild()

    def _refresh_batches(self) -> None:
        items: list[str] = []
        try:
            items = self.db.list_batches_with_temp_process(min_temps=5)
        except Exception:
            items = []
        self.cb_batches.blockSignals(True)
        self.cb_batches.clear()
        self.cb_batches.addItems(items)
        # Try sync with outer selector if it's in completed set
        if self._batch_selector is not None:
            current = self._batch_selector.currentText().strip()
            if current and current in items:
                self.cb_batches.setCurrentText(current)
        self.cb_batches.blockSignals(False)

    def _latest_map(self, rows: list[dict]) -> dict[str, dict[int, dict]]:
        """Latest row per bolt and integer-rounded temp from temp_loading."""
        from collections import defaultdict
        def ts(x):
            return x.get("measured_at") or x.get("seq") or 0
        by_bolt: dict[str, dict[int, dict]] = defaultdict(dict)
        for r in rows:
            try:
                bid = str(r.get("bolt_id"))
                t = r.get("temp")
                if t is None:
                    continue
                tgt = int(round(float(t)))
            except Exception:
                continue
            prev = by_bolt[bid].get(tgt)
            if prev is None or ts(r) > ts(prev):
                by_bolt[bid][tgt] = r
        return by_bolt

    def _batch_targets(self, batch_id: str) -> list[int]:
        """Compute target integer temperatures from batch min/max.

        Returns 5 integer targets rounding the 5 setpoints: [min, mid(min,20), 20, mid(20,max), max].
        """
        if not self.db or not batch_id:
            return [-20, 0, 20, 40, 50]
        try:
            b = self.db.get_batch(batch_id) or {"attrs": {}}
            attrs = b.get("attrs", {}) or {}
            def _to_float(v):
                try:
                    return float(v)
                except Exception:
                    return None
            tmin = _to_float(attrs.get("min_temp"))
            tmax = _to_float(attrs.get("max_temp"))
            floats = _compute_batch_setpoints(tmin, tmax)
            return [int(round(t)) for t in floats]
        except Exception:
            return [-20, 0, 20, 40, 50]

    def _group_targets(self, rows: list[dict]) -> list[int]:
        # Use batch-defined setpoints rather than data-driven grouping
        batch_id = self.cb_batches.currentText().strip()
        return self._batch_targets(batch_id)

    def _header_labels(self, rows: list[dict], targets: list[int]) -> list[str]:
        labels = []
        for t in targets:
            vals = [float(r.get("temp")) for r in rows if r.get("temp") is not None and abs(float(r["temp"]) - t) <= 0.5]
            mean = sum(vals)/len(vals) if vals else float(t)
            labels.append(f"{mean:.1f} \u00B0C")
        return labels

    def _build_complete_bolts(self, rows: list[dict], targets: list[int]) -> list[str]:
        from collections import defaultdict
        def ts(x):
            return x.get("measured_at") or x.get("seq") or 0
        by_bolt: dict[str, dict[int, dict]] = defaultdict(dict)
        for r in rows:
            try:
                bid = str(r.get("bolt_id"))
                t = r.get("temp")
                if t is None:
                    continue
                tgt = int(round(float(t)))
            except Exception:
                continue
            if tgt not in targets:
                continue
            prev = by_bolt[bid].get(tgt)
            if prev is None or ts(r) > ts(prev):
                by_bolt[bid][tgt] = r
        complete = {b: m for b, m in by_bolt.items() if all(t in m for t in targets)}
        def last_ts(b):
            try:
                return max(ts(m) for m in complete[b].values())
            except Exception:
                return 0
        self._by_bolt_latest = complete
        return sorted(complete.keys(), key=lambda b: last_ts(b), reverse=True)

    def _order_bolts(self, rows: list[dict]) -> list[str]:
        order: dict[str, float] = {}
        for r in rows:
            bid = str(r.get("bolt_id"))
            ts = r.get("measured_at")
            if bid not in order:
                order[bid] = ts or 0
            else:
                if ts is not None and ts < order[bid]:
                    order[bid] = ts
        return [b for b, _ in sorted(order.items(), key=lambda kv: (kv[1], kv[0]))]

    def _latest_for(self, rows: list[dict], bolt_id: str, target: int) -> dict | None:
        cand = [r for r in rows if str(r.get("bolt_id")) == str(bolt_id) and r.get("temp") is not None and abs(float(r["temp"]) - target) <= 0.5]
        if not cand:
            return None
        cand.sort(key=lambda r: r.get("measured_at") or 0, reverse=True)
        return cand[0]

    def _rebuild(self) -> None:
        batch_id = self.cb_batches.currentText().strip()
        if not batch_id:
            self.table.setRowCount(0); self.table.setColumnCount(0); self.plot.clear(); return
        try:
            rows = self.db.fetch_temp_loading_batch(batch_id)
        except Exception:
            rows = []
        if not rows:
            self.table.setRowCount(0); self.table.setColumnCount(0); self.plot.clear(); return
        targets = self._group_targets(rows)
        bolts = self._build_complete_bolts(rows, targets)
        headers = _summary_header_labels(getattr(self, "_by_bolt_latest", {}), rows, targets, bolts)
        # rows = bolts, cols = temps
        self.table.setRowCount(len(bolts))
        self.table.setColumnCount(len(targets))
        for j, lab in enumerate(headers):
            self.table.setHorizontalHeaderItem(j, QtWidgets.QTableWidgetItem(lab))
        # Override with properly formatted °C labels using measured means
        try:
            fixed_headers: list[str] = []
            for t in targets:
                vals = [float(r.get("temp")) for r in rows if r.get("temp") is not None and abs(float(r["temp"]) - t) <= 0.5]
                mean = sum(vals)/len(vals) if vals else float(t)
                fixed_headers.append(f"{mean:.1f} \u00B0C")
            for j, lab in enumerate(fixed_headers):
                self.table.setHorizontalHeaderItem(j, QtWidgets.QTableWidgetItem(lab))
        except Exception:
            pass
        for i, bid in enumerate(bolts):
            self.table.setVerticalHeaderItem(i, QtWidgets.QTableWidgetItem(f"{i+1}: {bid}"))
        self._cell_records.clear()
        for i, bid in enumerate(bolts):
            for j, t in enumerate(targets):
                r = self._by_bolt_latest.get(bid, {}).get(t)
                it = QtWidgets.QTableWidgetItem("")
                it.setTextAlignment(Qt.AlignCenter)
                if r and r.get("tof") is not None:
                    try:
                        it.setText(f"{float(r['tof']):.3f}")
                        self._cell_records[(i, j)] = r
                    except Exception:
                        pass
                self.table.setItem(i, j, it)

        # Plot Î”ToF lines per bolt
        self._resize_table_to_rows()
        self.plot.clear()
        try:
            vb = self.plot.getPlotItem().getViewBox(); vb.setMouseEnabled(False, False); vb.setMenuEnabled(False)
            self.plot.wheelEvent = lambda ev: None
        except Exception:
            pass
        colors = ["#e41a1c","#377eb8","#4daf4a","#984ea3","#ff7f00","#a65628","#f781bf","#999999"]
        for idx, bid in enumerate(bolts):
            xs, ys = [], []
            base = self._latest_for(rows, bid, 20)
            tof20 = float(base["tof"]) if base and base.get("tof") is not None else None
            if tof20 is None:
                continue
            for t in targets:
                if abs(t - 20) <= 0:
                    continue
                r = self._latest_for(rows, bid, t)
                if not r or r.get("tof") is None:
                    continue
                try:
                    xs.append(float(r.get("temp") if r.get("temp") is not None else t))
                    ys.append(float(r["tof"]) - tof20)
                except Exception:
                    continue
            if xs and ys:
                pen = pg.mkPen(colors[idx % len(colors)], width=2)
                self.plot.plot(xs, ys, pen=pen, symbol='o', name=str(bid))

    def _on_cell_clicked(self, row: int, col: int) -> None:
        key = (row, col)
        r = self._cell_records.get(key)
        if not r:
            return

        def _decode(which_key: str):
            data = r.get(which_key)
            if not data:
                return []
            import struct
            try:
                ba = bytes(data)
            except Exception:
                ba = data
            if which_key == 'dat2':
                fmt = '<h'; size = 2
            else:
                fmt = '<i'; size = 4
            n = len(ba)//size if ba else 0
            vals = []
            try:
                for i in range(n):
                    vals.append(struct.unpack_from(fmt, ba, i*size)[0])
            except Exception:
                pass
            return vals

        dlg = QtWidgets.QDialog(self)
        # Proper title with degree symbol
        try:
            t_txt = f"{float(r.get('temp')):.1f} °C" if r.get('temp') is not None else ""
        except Exception:
            t_txt = ""
        try:
            dlg.setWindowTitle(f"Signal - bolt {r.get('bolt_id')} @ {t_txt}")
        except Exception:
            dlg.setWindowTitle("Signal")

        lay = QVBoxLayout(dlg)
        # Header with data selector
        hdr = QHBoxLayout()
        hdr.addWidget(QLabel("Signal:"))
        cb = QComboBox(); cb.addItems(["dat3","dat2"]) 
        hdr.addWidget(cb); hdr.addStretch(); lay.addLayout(hdr)

        # Plot styled like Valid Combinations
        plot = pg.PlotWidget(); plot.showGrid(x=True, y=True, alpha=0.2)
        plot.setLabel('bottom', 'Samples'); plot.setLabel('left', 'Amplitude')
        try:
            vb_plot = plot.getPlotItem().getViewBox(); vb_plot.setMouseEnabled(False, False); vb_plot.setMenuEnabled(False)
            plot.wheelEvent = lambda ev: None
        except Exception:
            pass
        plot.setMinimumHeight(600)
        x_vals: list[int] = []
        curve = plot.plot([], [], pen=pg.mkPen('#00c853', width=2), name='Signal')

        # Legend on the right with boxed background
        try:
            legend = plot.addLegend(offset=(-10, 10), labelTextColor='#e6e6e6')
            legend.setBrush(pg.mkBrush(30, 30, 30, 200))
            legend.setPen(pg.mkPen('#9e9e9e'))
        except Exception:
            legend = None

        # Legend entries: max and cursor
        max_label_item = None
        cursor_label_item = None
        try:
            sample_max = pg.PlotDataItem([], [])
            sample_max.setPen(pg.mkPen(0, 0, 0, 0))
            if legend is not None:
                legend.addItem(sample_max, "Max amplitude: -")
                max_label_item = legend.items[-1][1] if getattr(legend, 'items', None) else None
            sample_cur = pg.PlotDataItem([], [])
            sample_cur.setPen(pg.mkPen(0, 0, 0, 0))
            if legend is not None:
                legend.addItem(sample_cur, "Cursor: -")
                cursor_label_item = legend.items[-1][1] if getattr(legend, 'items', None) else None
        except Exception:
            pass

        # Hover helpers: green vline + label
        vline = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('#00c853', width=1))
        plot.addItem(vline); vline.setVisible(False)
        label = pg.TextItem(color='#e6e6e6', anchor=(0, 1))
        plot.addItem(label); label.setVisible(False)

        # Data + interactions
        cur_series = {"data": []}

        def _refresh_curve():
            series = _decode(cb.currentText())
            cur_series["data"] = series
            nonlocal x_vals
            x_vals = list(range(len(series)))
            curve.setData(x_vals, series)
            try:
                import numpy as _np
                if legend is not None and max_label_item is not None:
                    max_amp = float(_np.nanmax(_np.asarray(series, dtype=float))) if series else float('nan')
                    max_label_item.setText(f"Max amplitude: {max_amp:.3f}")
            except Exception:
                pass
            if x_vals:
                try:
                    vline.setPos(x_vals[0])
                except Exception:
                    pass
            if cursor_label_item is not None:
                cursor_label_item.setText("Cursor: -")

        def _on_mouse_move(pos):
            try:
                vb = plot.getPlotItem().getViewBox()
                if vb is None or not plot.sceneBoundingRect().contains(pos):
                    vline.setVisible(False); label.setVisible(False)
                    return
                mouse_point = vb.mapSceneToView(pos)
                x = int(round(mouse_point.x()))
                data = cur_series["data"]
                if not data or x < 0 or x >= len(data):
                    vline.setVisible(False); label.setVisible(False)
                    return
                amp = float(data[x])
                vline.setPos(x)
                label.setText(f"Amplitude: {amp:.3f}")
                label.setPos(x, amp)
                vline.setVisible(True); label.setVisible(True)
                if cursor_label_item is not None:
                    cursor_label_item.setText(f"Cursor: {amp:.3f}")
            except Exception:
                pass

        try:
            plot.getPlotItem().scene().sigMouseMoved.connect(_on_mouse_move)
        except Exception:
            pass

        # Fullscreen toggle row (no scroll area)
        btn_row = QHBoxLayout(); btn_row.setContentsMargins(0, 0, 0, 0)
        fs_btn = QPushButton("Full Screen"); btn_row.addStretch(1); btn_row.addWidget(fs_btn); btn_row.addStretch(1)
        lay.addLayout(btn_row)
        lay.addWidget(plot)

        state = {'fs': False}

        def _toggle_full():
            try:
                if not state['fs']:
                    dlg.showFullScreen(); fs_btn.setText("Exit Full Screen"); state['fs'] = True
                else:
                    dlg.showNormal(); fs_btn.setText("Full Screen"); state['fs'] = False
            except Exception:
                pass
        fs_btn.clicked.connect(_toggle_full)

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

        cb.currentIndexChanged.connect(_refresh_curve)
        _refresh_curve()
        try:
            dlg.setSizeGripEnabled(True)
        except Exception:
            pass
        dlg.resize(900, 700)
        dlg.exec_()

    def _resize_table_to_rows(self) -> None:
        try:
            header_h = self.table.horizontalHeader().height()
            rows_h = sum(self.table.rowHeight(r) for r in range(self.table.rowCount()))
            margins = (self.table.frameWidth() or 0) * 2
            total = header_h + rows_h + margins
            if total > 0:
                self.table.setFixedHeight(int(total))
        except Exception:
            pass


    def resizeEvent(self, ev):
        try:
            super().resizeEvent(ev)
        except Exception:
            pass
        try:
            if getattr(self, "plot", None) is not None:
                w = self.plot.width()
                if w > 0 and self.plot.height() != w:
                    self.plot.setFixedHeight(w)
        except Exception:
            pass

class TemperatureProcessTab(QWidget):
    """Process subtab for temperature qualification."""

    def __init__(self, db: BatchDB | None = None,
                 com_selector=None,
                 batch_selector: QComboBox | None = None,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.db = db
        self.com_selector = com_selector
        self.batch_cb: QComboBox | None = batch_selector

        self._current_idx: int | None = None
        self._done: list[bool] = []
        self._prog_done: list[bool] = []
        self._temp_done: list[bool] = []
        self._current_temp: float | None = None
        self._worker: TempSequenceWorker | None = None
        self._phase: str | None = None  # "scan" or "stab" or None
        self._time_counter: int = 0
        self._scan_curve: pg.PlotDataItem | None = None
        self._temps: list[float] = []
        self._batch_id: str | None = None
        self._bolt_codes: list[str | None] = []
        # Per-bolt resolved batch IDs (from scanned bolt code)
        self._bolt_batches: list[str | None] = []
        # Optional cached params per-bolt batch
        self._bolt_params: list[dict] = []
        self._batch_params: dict[str, float] = {}
        self._alpha_computed: bool = False

        self._build_ui()
        if batch_selector is None:
            self._refresh_batches()

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        # --- Scrollable container ---------------------------------------
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # -- Selector de batch -------------------------------------------------
        if self.batch_cb is None:
            row = QHBoxLayout()
            row.addWidget(QLabel("Batch num:"))
            self.batch_cb = RefreshComboBox(self._refresh_batches, self)
            self.batch_cb.activated[str].connect(self._on_batch_changed)
            row.addWidget(self.batch_cb)
            row.addStretch()
            layout.addLayout(row)
        else:
            self.batch_cb.activated[str].connect(self._on_batch_changed)

        # -- Selector nÃƒÂºmero de bolts
        row = QHBoxLayout()
        row.addWidget(QLabel("Num bolts:"))
        self.spin_bolts = QSpinBox(); self.spin_bolts.setRange(1, 10); self.spin_bolts.setValue(1)
        self.spin_bolts.valueChanged.connect(self._on_num_changed)
        row.addWidget(self.spin_bolts)
        row.addStretch()
        layout.addLayout(row)

        # -- Selector de bolts mediante botones ---------------------------
        self.bolt_group = QButtonGroup(self)
        self.bolt_group.setExclusive(True)
        self.bolt_widget = QWidget()
        self.bolt_layout = QHBoxLayout(self.bolt_widget)
        self.bolt_layout.setContentsMargins(0, 0, 0, 0)
        self.bolt_layout.setSpacing(4)
        self.bolt_group.buttonClicked[int].connect(self._on_bolt_selected)
        layout.addWidget(self.bolt_widget)

        # -- Mini DJ widget (sÃƒÂ³lo grÃƒÂ¡fica + Start/Stop)
        self.dj = MiniDJ(self.com_selector)
        self.dj.started.connect(
            lambda: self._update_bolt_highlight(
                self._current_idx, highlight_progress=False))
        self.dj.stopped.connect(
            lambda: self._update_bolt_highlight(None, highlight_progress=False))
        layout.addWidget(self.dj)

        # -- BotÃƒÂ³n Bolt OK
        self.btn_ok = QPushButton("Bolt OK", objectName="Play")
        self.btn_ok.clicked.connect(self._on_bolt_ok)
        layout.addWidget(self.btn_ok)

        # -- Add to DB button (appears when all bolts done)
        self.btn_add_db = QPushButton("Add to DB", objectName="Save")
        self.btn_add_db.setEnabled(False)
        self.btn_add_db.clicked.connect(self._on_add_db)
        layout.addWidget(self.btn_add_db)

        # -- Temperature inputs row (hidden initially)
        self.temp_widget = QWidget()
        temp_row = QHBoxLayout(self.temp_widget)
        temp_row.addWidget(QLabel("Temperatures:"))
        self.temp_spins: list[QDoubleSpinBox] = []
        for val in (-20.0, 0.0, 20.0, 40.0, 50.0):
            sb = QDoubleSpinBox()
            sb.setDecimals(1)
            sb.setRange(-100.0, 200.0)
            sb.setSingleStep(1.0)
            sb.setAlignment(Qt.AlignCenter)
            sb.setFixedWidth(70)
            sb.setValue(val)
            temp_row.addWidget(sb)
            self.temp_spins.append(sb)
        self.btn_save_t = QPushButton("Save", objectName="Save")
        self.btn_save_t.clicked.connect(self._on_save_temps)
        temp_row.addWidget(self.btn_save_t)
        self.temp_widget.setEnabled(False)
        layout.addWidget(self.temp_widget)

        # -- Stabilization controls + start/stop ------------------------
        self.stab_widget = QWidget()
        stab_row = QHBoxLayout(self.stab_widget)
        stab_row.addWidget(QLabel("Stab time (min):"))
        # Tiempo de estabilizaciÃ³n en minutos
        self.stab_time = QDoubleSpinBox(); self.stab_time.setDecimals(1); self.stab_time.setSingleStep(0.1); self.stab_time.setRange(0.1, 600.0); self.stab_time.setValue(10.0)
        stab_row.addWidget(self.stab_time)
        # Intervalo de muestreo durante estabilizaciÃ³n (minutos)
        stab_row.addWidget(QLabel("Sample every (min):"))
        self.stab_sample = QDoubleSpinBox(); self.stab_sample.setDecimals(1); self.stab_sample.setSingleStep(0.1); self.stab_sample.setRange(0.1, 120.0); self.stab_sample.setValue(1.0)
        stab_row.addWidget(self.stab_sample)
        stab_row.addWidget(QLabel("Max \u0394ToF (ns):"))
        self.stab_tol = QDoubleSpinBox(); self.stab_tol.setDecimals(1); self.stab_tol.setRange(0.1, 1000.0); self.stab_tol.setValue(50.0)
        stab_row.addWidget(self.stab_tol)

        stab_row.addWidget(QLabel("Data:"))
        self.data_cb = QtWidgets.QComboBox(); self.data_cb.addItems(["dat2", "dat3"])
        stab_row.addWidget(self.data_cb)

        # Bolt selector for stabilization ToF
        stab_row.addWidget(QLabel("Bolt:"))
        self.stab_bolt_cb = QtWidgets.QComboBox()
        self.stab_bolt_cb.setMinimumWidth(60)
        self.stab_bolt_cb.addItems(["1"])  # default, will sync on _on_num_changed
        stab_row.addWidget(self.stab_bolt_cb)
        # Al cambiar el bolt de estabilizaciÃ³n, actualizar si Start se puede habilitar
        self.stab_bolt_cb.currentIndexChanged.connect(lambda _=None: self._update_start_enabled())

        # --- valor temperatura horno -----------------------------------------
        stab_row.addWidget(QLabel("Oven T (°C):"))
        self.oven_temp = QLabel("--.-")
        self.oven_temp.setFixedWidth(60)
        self.oven_temp.setAlignment(Qt.AlignCenter)
        stab_row.addWidget(self.oven_temp)


        self.btn_start_seq = QPushButton("Start", objectName="Play")
        self.btn_stop_seq = QPushButton("Stop", objectName="Stop")
        self.btn_start_seq.setEnabled(False)
        self.btn_stop_seq.setEnabled(False)
        stab_row.addStretch()
        stab_row.addWidget(self.btn_start_seq)
        stab_row.addWidget(self.btn_stop_seq)

        self.stab_widget.setEnabled(False)
        layout.addWidget(self.stab_widget)

        # -- Progress buttons for each bolt during the sequence -----------
        self.prog_widget = QWidget()
        self.prog_layout = QHBoxLayout(self.prog_widget)
        self.prog_layout.setContentsMargins(0, 0, 0, 0)
        self.prog_layout.setSpacing(4)
        self.prog_layout.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.prog_widget)
        self._prog_buttons: list[QPushButton] = []

        self.signals_plot = _NoWheelPlot(title="Signal")
        self.signals_plot.setMinimumHeight(320)
        self.signals_plot.showGrid(x=True, y=True, alpha=0.3)
        self.signals_plot.setLabel("left", "Amplitude")
        self.signals_plot.setLabel("bottom", "Samples")
        self.tof_plot = _NoWheelPlot(title="ToF")
        self.tof_plot.setMinimumHeight(320)
        self.tof_plot.showGrid(x=True, y=True, alpha=0.3)
        self.tof_plot.setLabel("left", "ToF (ns)")
        self.tof_plot.setLabel("bottom", "Samples")
        plot_row = QHBoxLayout()
        plot_row.addWidget(self.signals_plot, 1)
        plot_row.addWidget(self.tof_plot, 1)
        layout.addLayout(plot_row)

        # Tabla: Ãºltimos 10 valores de ToF (estabilizaciÃ³n)
        self.last10_table = QtWidgets.QTableWidget(0, 3)
        self.last10_table.setHorizontalHeaderLabels(["Timestamp", "ToF (ns)", "Temp (°C)"])
        # Estilo de cabeceras para mejor legibilidad
        self.last10_table.horizontalHeader().setStyleSheet(
            "QHeaderView::section{background:#263238;color:#ECEFF1;padding:4px;border:none;}"
        )
        # Estilo de celdas y alternado de filas
        self.last10_table.setAlternatingRowColors(True)
        self.last10_table.setStyleSheet(
            "QTableWidget{gridline-color:#CFD8DC;}"
            "QTableWidget::item{padding:2px;color:#263238;}"
            "QTableWidget::item:!selected{background:#FFFFFF;}"
            "QTableWidget::item:!selected:alternate{background:#F5F5F5;}"
        )
        # Columnas de igual anchura
        self.last10_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.last10_table.verticalHeader().setVisible(False)
        # Sin barras de scroll internas (se usa el scroll de la pestaÃ±a)
        self.last10_table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.last10_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        # Ajustes de interacciÃ³n
        self.last10_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.last10_table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        # Altura se ajustarÃ¡ dinÃ¡micamente para mostrar hasta 10 filas
        layout.addWidget(self.last10_table)
        # Primer ajuste de altura
        try:
            self._resize_last10_table()
        except Exception:
            pass
        self._last10_rows = []

        # -- Manual AlphaTemp calculation controls (below bottom graphs) --
        alpha_row = QHBoxLayout()
        alpha_row.setContentsMargins(0, 4, 0, 0)
        alpha_row.setSpacing(8)
        alpha_row.setAlignment(Qt.AlignCenter)
        alpha_row.addStretch()
        alpha_row.addWidget(QLabel("Batch Num:"))
        self.alpha_batch_cb = RefreshComboBox(self._refresh_alpha_batches, self)
        self.alpha_batch_cb.setMinimumWidth(200)
        alpha_row.addWidget(self.alpha_batch_cb)
        self.alpha_btn = QPushButton("Calculate Alfatemp")
        self.alpha_btn.setCursor(Qt.PointingHandCursor)
        self.alpha_btn.setStyleSheet(
            "QPushButton{background:#1976D2;color:#FFFFFF;border:none;border-radius:6px;padding:6px 14px;font-weight:600;}"
            "QPushButton:hover{background:#1E88E5;}"
            "QPushButton:pressed{background:#1565C0;}"
            "QPushButton:disabled{background:#90A4AE;color:#ECEFF1;}"
        )
        self.alpha_btn.clicked.connect(self._on_manual_calc_alpha)
        alpha_row.addWidget(self.alpha_btn)
        alpha_row.addStretch()
        layout.addLayout(alpha_row)
        # Pre-populate eligible batches once
        try:
            self._refresh_alpha_batches()
        except Exception:
            pass

        layout.addStretch()

        self._on_num_changed(1)
        self.btn_start_seq.clicked.connect(self._on_start_sequence)
        self.btn_stop_seq.clicked.connect(self._on_stop_sequence)

        # Set scroll widget
        scroll.setWidget(content)
        main = QVBoxLayout(self)
        main.setContentsMargins(0, 0, 0, 0)
        main.addWidget(scroll)

    def _resize_last10_table(self) -> None:
        """Resize the last-10 ToF table to show up to 10 rows without inner scroll."""
        table = self.last10_table
        if table is None:
            return
        rows = min(len(self._last10_rows), 10)
        # Garantiza al menos 1 fila de alto para ver cabecera bien
        rows = max(rows, 1)
        header_h = table.horizontalHeader().height()
        row_h = table.verticalHeader().defaultSectionSize()
        # Opcional: filas mÃ¡s compactas para que quepan 10 cÃ³modamente
        table.verticalHeader().setDefaultSectionSize(max(20, row_h))
        fw = 2 * getattr(table, 'frameWidth', lambda: 1)()
        table.setFixedHeight(int(header_h + rows * table.verticalHeader().defaultSectionSize() + fw))

    def _refresh_batches(self) -> None:
        if not self.batch_cb:
            return
        if self.db is None:
            items: list[str] = []
        else:
            try:
                items = self.db.batch_numbers()
            except Exception:
                items = []
        self.batch_cb.blockSignals(True)
        self.batch_cb.clear()
        self.batch_cb.addItems(items)
        self.batch_cb.blockSignals(False)
        if items:
            self._batch_id = items[0]
            self.batch_cb.setCurrentIndex(0)
            self._load_batch_params()
        else:
            self._batch_id = None
            self._batch_params = {}
        self._bolt_codes = [None] * len(self._done)
        self._bolt_batches = [None] * len(self._done)
        self._bolt_params = [{} for _ in range(len(self._done))]
        if hasattr(self.dj, "btn_start"):
            self.dj.btn_start.setEnabled(False)

    def _on_batch_changed(self, text: str) -> None:
        self._batch_id = text
        self._load_batch_params()
        self._bolt_codes = [None] * len(self._done)
        self._bolt_batches = [None] * len(self._done)
        self._bolt_params = [{} for _ in range(len(self._done))]
        if hasattr(self.dj, "btn_start"):
            self.dj.btn_start.setEnabled(False)

    def _load_batch_params(self) -> None:
        """Fetch freq/gain/pulse and ToF or UL for the current batch."""
        if self.db is None or self._batch_id is None:
            self._batch_params = {}
            return
        try:
            self._batch_params = self.db.get_batch_params(self._batch_id)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "DB", str(e))
            self._batch_params = {}
        # Also derive default temperature setpoints from batch min/max
        try:
            self._apply_batch_temps(self._batch_id)
        except Exception:
            pass

    def _apply_batch_temps(self, batch_id: str | None) -> None:
        """Fetch min/max temps for batch and update the 5 temperature spinboxes.

        Uses setpoints: [min, (min+20)/2, 20, (20+max)/2, max]. Spinboxes remain editable.
        """
        if not self.db or not batch_id:
            return
        if not getattr(self, 'temp_spins', None):
            return
        try:
            b = self.db.get_batch(batch_id) or {"attrs": {}}
            attrs = b.get("attrs", {}) or {}
            def _to_float(v):
                try:
                    return float(v)
                except Exception:
                    return None
            tmin = _to_float(attrs.get("min_temp"))
            tmax = _to_float(attrs.get("max_temp"))
            temps = _compute_batch_setpoints(tmin, tmax)
            for sb, val in zip(self.temp_spins, temps):
                sb.setValue(float(val))
        except Exception:
            pass

    # ----------------------------------------------------------------- num
    def _on_num_changed(self, val: int) -> None:
        # Remove old buttons
        for btn in self.bolt_group.buttons():
            self.bolt_group.removeButton(btn)
            btn.deleteLater()
        while self.bolt_layout.count():
            item = self.bolt_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        self._done = [False] * val
        for i in range(val):
            btn = QPushButton(str(i + 1), objectName="File")
            btn.setCheckable(True)
            btn.setFixedWidth(40)
            self.bolt_group.addButton(btn, i)
            self.bolt_layout.addWidget(btn)

        # Progress buttons
        while self.prog_layout.count():
            item = self.prog_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        self._prog_buttons = []
        self.prog_layout.addStretch()
        for i in range(val):
            btn = QPushButton(str(i + 1), objectName="File")
            btn.setEnabled(False)
            btn.setFixedWidth(40)
            self.prog_layout.addWidget(btn)
            self._prog_buttons.append(btn)
        self.prog_layout.addStretch()
        self._prog_done = [False] * val

        self._current_idx = None
        self.btn_add_db.setEnabled(False)
        self.temp_widget.setEnabled(False)
        self.stab_widget.setEnabled(False)
        self.btn_start_seq.setEnabled(False)
        self._bolt_codes = [None] * val
        self._bolt_batches = [None] * val
        self._bolt_params = [{} for _ in range(val)]
        # update stabilization bolt selector
        self.stab_bolt_cb.blockSignals(True)
        self.stab_bolt_cb.clear()
        self.stab_bolt_cb.addItems([str(i+1) for i in range(val)])
        self.stab_bolt_cb.setCurrentIndex(0)
        self.stab_bolt_cb.blockSignals(False)
        if hasattr(self.dj, "btn_start"):
            self.dj.btn_start.setEnabled(False)
        # Re-evaluar habilitaciÃ³n del Start
        try:
            self._update_start_enabled()
        except Exception:
            pass

    # ----------------------------------------------------------------- button
    def _on_bolt_selected(self, idx: int) -> None:
        self._current_idx = idx
        # Request bolt code if not scanned yet
        if self._bolt_codes[idx] is None:
            code, ok = QInputDialog.getText(self, "Scan bolt", f"Bolt {idx + 1} code:")
            code = code.strip() if isinstance(code, str) else ""
            if not ok or not code:
                btn = self.bolt_group.button(idx)
                if btn:
                    btn.setChecked(False)
                self._current_idx = None
                return
            # Resolve batch by scanned bolt ID (no dependency on selected batch)
            if self.db:
                try:
                    resolved_batch = self.db.find_batch_by_bolt(code)
                except Exception:
                    resolved_batch = None
                if not resolved_batch:
                    QtWidgets.QMessageBox.warning(
                        self, "Bolt", f"Bolt '{code}' no encontrado en la base de datos")
                    btn = self.bolt_group.button(idx)
                    if btn:
                        btn.setChecked(False)
                    self._current_idx = None
                    return
                self._bolt_batches[idx] = resolved_batch
                try:
                    self._bolt_params[idx] = self.db.get_batch_params(resolved_batch)
                except Exception:
                    self._bolt_params[idx] = {}
                # Update temperature setpoints from the resolved batch
                try:
                    self._apply_batch_temps(resolved_batch)
                except Exception:
                    pass
            self._bolt_codes[idx] = code
            btn = self.bolt_group.button(idx)
            if btn:
                btn.setStyleSheet("QPushButton{background:#BBDEFB;color:#202124;}")

        if all(code is not None for code in self._bolt_codes):
            if hasattr(self.dj, "btn_start"):
                self.dj.btn_start.setEnabled(True)
        else:
            if hasattr(self.dj, "btn_start"):
                self.dj.btn_start.setEnabled(False)

        try:
            # Use per-bolt resolved batch params if available, else fallback
            params = self._bolt_params[idx] if (0 <= idx < len(self._bolt_params) and self._bolt_params[idx]) else self._get_params(idx + 1)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "DB", str(e))
            return
        self._update_bolt_highlight(idx, highlight_progress=False)
        if params.get("tof") is not None:
            tof_val = float(params.get("tof"))
        else:
            bolt_len = float(params.get("ul", 80.0))
            tof_val = (2 * bolt_len) / 5900.0 * 1e6
        dj_params = {
            "freq": float(params.get("freq", 40.0)),
            "gain": float(params.get("gain", 20.0)),
            "pulse": float(params.get("pulse", 2.0)),
            "tof": tof_val,
            "temp": 20.0,
            "bolt": int(idx + 1),
            "algo": int(params.get("algo", 0)),
            "threshold": float(params.get("threshold", 0.0)),
        }
        self.dj.set_params(dj_params)
        # Enable Start if temperatures defined and stabilization bolt scanned
        if self._temps:
            try:
                self.btn_start_seq.setEnabled(self._stab_bolt_scanned())
            except Exception:
                pass

    # ---------------------------------------------------------------- params
    def _get_params(self, bolt: int) -> dict:
        if not self._batch_params:
            return {
                "freq": 40.0,
                "gain": 20.0,
                "pulse": 2.0,
                "ul": 80.0,
                "algo": 0,
                "threshold": 0.0,
                "tof": None,
            }
        params = self._batch_params.copy()
        params.setdefault("algo", 0)
        params.setdefault("threshold", 0.0)
        return params

    # ----------------------------------------------------------------- OK btn
    def _on_bolt_ok(self) -> None:
        if self._current_idx is None:
            return
        btn = self.bolt_group.button(self._current_idx)
        if btn:
            btn.setStyleSheet(
                "QPushButton{background:#A5D6A7;color:#202124;}")
            btn.setChecked(False)
        self._done[self._current_idx] = True
        self._current_idx = None
        self._update_bolt_highlight(None, highlight_progress=False)
        if all(self._done):
            self.btn_add_db.setEnabled(True)
            self.temp_widget.setEnabled(True)

    # ----------------------------------------------------------------- Add DB
    def _on_add_db(self) -> None:
        QtWidgets.QMessageBox.information(self, "Info",
            "Save to database not implemented yet.")

    # ----------------------------------------------------------------- Save tmp
    def _on_save_temps(self) -> None:
        temps = [sb.value() for sb in self.temp_spins]
        if temps != sorted(temps):
            QtWidgets.QMessageBox.warning(
                self, "Temps", "Temperatures must be in ascending order")
            return
        self._temps = temps
        self._temp_done = [False] * len(temps)
        self._current_temp = None
        self._update_temp_highlight(None)
        QtWidgets.QMessageBox.information(self, "Temps",
            f"Saved temperatures: {temps}")
        self.stab_widget.setEnabled(True)
        # Only allow start if all bolts are scanned
        try:
            self._update_start_enabled()
        except Exception:
            self.btn_start_seq.setEnabled(False)
        self.btn_stop_seq.setEnabled(False)

    def _ordered_temps(self, temps: list[float]) -> list[float]:
        """Return temperatures ordered starting at 20°C then below then above."""
        base = 20.0
        below = sorted([t for t in temps if t < base], reverse=True)
        above = sorted([t for t in temps if t > base])
        ordered: list[float] = []
        if base in temps:
            ordered.append(base)
        ordered.extend(below)
        ordered.extend(above)
        return ordered

    # -------------------------------------------------------------- sequence
    def _on_start_sequence(self) -> None:
        if not self.com_selector or not self.com_selector.currentText():
            QtWidgets.QMessageBox.warning(self, "COM", "Selecciona un puerto COM valido")
            return
        # Require all bolts to be scanned before starting
        # Ya no requerimos escanear todos los bolts; basta con el bolt de estabilizaciÃ³n
        # Determine stabilization bolt index (0-based) and ensure it is scanned
        try:
            stab_bolt_index = int(self.stab_bolt_cb.currentText()) - 1
        except Exception:
            stab_bolt_index = 0
        if not (0 <= stab_bolt_index < len(self._bolt_codes)):
            QtWidgets.QMessageBox.warning(self, "Bolt", "Selecciona un bolt de estabilizaciÃƒÂ³n vÃƒÂ¡lido")
            return
        if self._bolt_codes[stab_bolt_index] is None or self._bolt_batches[stab_bolt_index] is None:
            QtWidgets.QMessageBox.warning(self, "Bolt", "Escanea primero el bolt seleccionado para estabilizar")
            return
        # Build combos per bolt (each bolt may belong to a different batch)
        combos_list: list[list[tuple[float, float, float]]] = []
        try:
            for i in range(len(self._done)):
                bid = self._bolt_batches[i]
                combos_i, _ = self._get_valid_combos_for_batch(bid)
                combos_list.append(combos_i)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "DB", str(e))
            return
        # Convertir minutos (float) a segundos enteros
        hold_s = max(1, int(round(float(self.stab_time.value()) * 60)))
        sample_s = max(1, int(round(float(self.stab_sample.value()) * 60)))
        tol_ns = float(self.stab_tol.value())
        self.signals_plot.clear()
        self.tof_plot.clear()
        self._stab_curve = None
        self._scan_curve = None
        self._tof_curve = None
        self._phase = None
        self._current_temp = None
        self._update_temp_highlight(None)
        temps = self._ordered_temps(self._temps)
        # Batch params: take from the stabilization bolt's resolved batch
        bp = self._bolt_params[stab_bolt_index] if (0 <= stab_bolt_index < len(self._bolt_params)) else {}
        stab_combo = (
            float(bp.get("freq", 40.0)),
            float(bp.get("gain", 20.0)),
            float(bp.get("pulse", 2.0)),
        )
        num_bolts = len(self._done)
        init_tof = bp.get("tof")
        if init_tof is None:
            ul = float(bp.get("ul", 80.0))
            init_tof = (2 * ul) / 5900.0 * 1e6
        self._worker = TempSequenceWorker(
            self.com_selector.currentText(), combos_list, temps, hold_s, tol_ns,
            num_bolts, self.data_cb.currentText(), stab_combo, init_tof=init_tof,
            sample_every_s=sample_s,
            stab_bolt_index=stab_bolt_index,
            db=self.db, batch_id=self._batch_id
        )
        # Use queued connection to ensure slot runs in the main thread
        self._worker.frame_ready.connect(self._update_plots,
                                         QtCore.Qt.QueuedConnection)
        self._worker.frame_ready.connect(self._store_frame,
                                         QtCore.Qt.QueuedConnection)
        self._worker.finished.connect(self._on_seq_finished)
        self._worker.error.connect(lambda msg: QtWidgets.QMessageBox.critical(self, "Seq", msg))
        self._worker.temp_started.connect(self._on_temp_started)
        self._worker.temp_done.connect(self._on_temp_done)
        self._worker.bolt_started.connect(self._on_bolt_started)
        self._worker.bolt_done.connect(self._on_bolt_done)
        self._worker.start()
        self.btn_start_seq.setEnabled(False)
        self.btn_stop_seq.setEnabled(True)
        # Reset estado de cÃ¡lculo alpha para esta sesiÃ³n
        self._alpha_computed = False

        # Disable editing of upper controls while the sequence runs
        self.spin_bolts.setEnabled(False)
        self.bolt_widget.setEnabled(False)
        self.dj.setEnabled(False)
        self.btn_ok.setEnabled(False)
        self.btn_add_db.setEnabled(False)
        self.temp_widget.setEnabled(False)
        self.stab_time.setEnabled(False)
        self.stab_tol.setEnabled(False)
        self.stab_sample.setEnabled(False)
        # Bloquear cambios en selector de bolt de estabilizaciÃ³n durante el proceso
        try:
            self.stab_bolt_cb.blockSignals(True)
            self.stab_bolt_cb.setEnabled(False)
        except Exception:
            pass

    def _on_stop_sequence(self) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.stop()
            self._worker.wait()
        self._worker = None
        self._update_start_enabled()
        self.btn_stop_seq.setEnabled(False)
        # Re-enable previously disabled controls
        self.spin_bolts.setEnabled(True)
        self.bolt_widget.setEnabled(True)
        self.dj.setEnabled(True)
        self.btn_ok.setEnabled(True)
        self.btn_add_db.setEnabled(all(self._done))
        self.temp_widget.setEnabled(bool(self._temps))
        self.stab_time.setEnabled(True)
        self.stab_tol.setEnabled(True)
        self.stab_sample.setEnabled(True)
        self.data_cb.setEnabled(True)
        # Rehabilitar selector de bolt de estabilizaciÃ³n al terminar
        try:
            self.stab_bolt_cb.setEnabled(True)
            self.stab_bolt_cb.blockSignals(False)
        except Exception:
            pass
        for i, btn in enumerate(self._prog_buttons):
            btn.setStyleSheet("")
            self._prog_done[i] = False
        self._update_bolt_highlight(None)
        self._current_temp = None
        self._update_temp_highlight(None)

    def _all_bolts_scanned(self) -> bool:
        if not self._bolt_codes or not self._bolt_batches:
            return False
        return all(code is not None for code in self._bolt_codes) and all(bid is not None for bid in self._bolt_batches)

    def _stab_bolt_scanned(self) -> bool:
        """Return True if the currently selected stabilization bolt is scanned."""
        try:
            idx = int(self.stab_bolt_cb.currentText()) - 1
        except Exception:
            return False
        if not (0 <= idx < len(self._bolt_codes)):
            return False
        return (self._bolt_codes[idx] is not None) and (self._bolt_batches[idx] is not None)

    def _update_start_enabled(self) -> None:
        self.btn_start_seq.setEnabled(bool(self._temps) and self._stab_bolt_scanned())

    def _on_seq_finished(self) -> None:
        QtWidgets.QMessageBox.information(self, "Seq", "Sequence finished")
        self._on_stop_sequence()
        # Alpha_temp ya se calcula al terminar la Ãºltima temperatura; mantener manual opcional
        if not getattr(self, '_alpha_computed', False):
            try:
                results = self._compute_and_save_alpha_temp()
                if results:
                    dlg = QtWidgets.QDialog(self)
                    dlg.setWindowTitle("Alpha_temp per batch")
                    lay = QVBoxLayout(dlg)
                    lay.addWidget(QLabel("Resultados Alpha_temp (alpha1):"))
                    for b, val in results.items():
                        lay.addWidget(QLabel(f"Batch {b}: {val:.6g}"))
                    btn = QPushButton("OK"); btn.clicked.connect(dlg.accept); lay.addWidget(btn)
                    dlg.resize(380, 120); dlg.exec_()
                self._alpha_computed = True
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Alpha_temp", f"No se pudo calcular Alpha_temp: {e}")

    def _compute_and_save_alpha_temp(self) -> dict[str, float]:
        """Compute alpha_temp (alpha1) per batch from temp_measurement data.

        Steps per batch, using only the bolts processed in this session:
        - For each bolt, retrieve latest ToF at each target temperature for the
          batch's configured (freq, gain, pulse).
        - Compute deltas relative to the measured reference near 20°C:
            delta_ToF(T)   = ToF_T   - ToF_20
            delta_Temp(T)  = Temp_T  - Temp_20   (Temp_20 measured in DB, not literal 20)
        - Compute slope ratio = delta_ToF(T) / delta_Temp(T) for each valid case.
        - Average all ratios across bolts and temperatures to obtain alpha_temp.
        - Persist alpha_temp into batch.alpha1.
        Returns a mapping batch_id -> alpha_temp for all involved batches.
        """
        if self.db is None:
            return {}
        # Map batch_id -> list of bolt_ids processed in this run
        batch_bolts: dict[str, list[str]] = {}
        for i, code in enumerate(self._bolt_codes):
            if code is None:
                continue
            bid = self._bolt_batches[i] or self._batch_id
            if not bid:
                continue
            batch_bolts.setdefault(str(bid), []).append(str(code))

        if not batch_bolts:
            return {}

        # Temperature targets configured for the run
        targets = list(self._temps or [])
        # Ensure numeric values and stable ordering
        try:
            targets = [float(t) for t in targets]
        except Exception:
            targets = []
        if not targets:
            return {}
        tol = 0.6  # tolerance in °C to match measured temps to targets

        results: dict[str, float] = {}
        for batch_id, bolts in batch_bolts.items():
            # Fetch batch params (freq, gain, pulse)
            try:
                params = self.db.get_batch_params(batch_id)
                freq_i = int(params.get("freq")) if params.get("freq") is not None else None
                gain_i = int(params.get("gain")) if params.get("gain") is not None else None
                pulse_i = int(params.get("pulse")) if params.get("pulse") is not None else None
            except Exception:
                freq_i = gain_i = pulse_i = None

            ratios: list[float] = []

            for bolt_id in bolts:
                try:
                    rows = self.db.fetch_temp_measurements(batch_id, bolt_id, freq_i, gain_i, pulse_i)
                except Exception:
                    rows = []
                if not rows:
                    continue

                # Helper: latest row for a target temperature within tolerance
                def _latest_for(target: float):
                    cand = [r for r in rows if r.get("temp") is not None and abs(float(r["temp"]) - target) <= tol]
                    if not cand:
                        return None
                    cand.sort(key=lambda r: r.get("measured_at") or 0, reverse=True)
                    return cand[0]

                base = _latest_for(20.0)
                if not base or base.get("tof") is None:
                    continue
                tof20 = float(base["tof"])
                base_temp = float(base.get("temp")) if base.get("temp") is not None else 20.0

                for t in targets:
                    if abs(t - 20.0) <= 1e-6:
                        continue
                    r = _latest_for(t)
                    if not r or r.get("tof") is None or r.get("temp") is None:
                        continue
                    delta_tof = float(r["tof"]) - tof20
                    if not np.isfinite(delta_tof):
                        continue
                    temp_meas = float(r["temp"])  # measured temperature (real from DB)
                    delta_temp = temp_meas - base_temp
                    if delta_temp == 0.0 or not np.isfinite(delta_temp):
                        continue
                    ratio = delta_tof / delta_temp
                    if np.isfinite(ratio):
                        ratios.append(ratio)

            if ratios:
                alpha = float(np.mean(ratios))
                try:
                    self.db.update_alpha1(batch_id, alpha)
                except Exception:
                    pass
                results[batch_id] = alpha

        return results

    def _on_temp_started(self, temp: float) -> None:
        """Highlight the spin box of the temperature being processed."""
        self._current_temp = temp
        self._update_temp_highlight(temp)
        # Reinicia la tabla de Ãºltimos 10 para la nueva estabilizaciÃ³n
        try:
            self._last10_rows = []
            self.last10_table.setRowCount(0)
            self._resize_last10_table()
        except Exception:
            pass

    def _on_temp_done(self, temp: float) -> None:
        """Mark the spin box corresponding to ``temp`` in green."""
        for sb in self.temp_spins:
            if abs(sb.value() - temp) < 1e-6:
                sb.setStyleSheet(
                    "QDoubleSpinBox{background:#A5D6A7;color:#202124;}")
                idx = self.temp_spins.index(sb)
                if idx < len(self._temp_done):
                    self._temp_done[idx] = True
                break
        self._current_temp = None
        self._update_temp_highlight(None)
        # reset progress buttons for next temperature
        for i, btn in enumerate(self._prog_buttons):
            btn.setStyleSheet("")
            self._prog_done[i] = False
        # Si esta era la Ãºltima temperatura, calcular alpha_temp ahora
        try:
            if self._temp_done and all(self._temp_done) and not getattr(self, '_alpha_computed', False):
                results = self._compute_and_save_alpha_temp()
                if results:
                    dlg = QtWidgets.QDialog(self)
                    dlg.setWindowTitle("Alpha_temp per batch")
                    lay = QVBoxLayout(dlg)
                    lay.addWidget(QLabel("Resultados Alpha_temp (alpha1):"))
                    for b, val in results.items():
                        lay.addWidget(QLabel(f"Batch {b}: {val:.6g}"))
                    btn = QPushButton("OK"); btn.clicked.connect(dlg.accept); lay.addWidget(btn)
                    dlg.resize(380, 120); dlg.exec_()
                self._alpha_computed = True
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Alpha_temp", f"No se pudo calcular Alpha_temp: {e}")

    def _on_bolt_done(self, idx: int) -> None:
        """Mark progress button ``idx`` when a bolt scan finishes."""
        if 0 <= idx < len(self._prog_buttons):
            self._prog_buttons[idx].setStyleSheet(
                "QPushButton{background:#A5D6A7;color:#202124;}")
            self._prog_done[idx] = True
        self._update_bolt_highlight(None)

    def _on_bolt_started(self, idx: int) -> None:
        """Highlight the bolt currently being scanned."""
        self._update_bolt_highlight(idx)

    def _update_bolt_highlight(self, current: int | None,
                               highlight_progress: bool = True) -> None:
        """Update colors for bolt selector and progress buttons."""
        # Top selector buttons
        for i, btn in enumerate(self.bolt_group.buttons()):
            if self._done[i]:
                btn.setStyleSheet(
                    "QPushButton{background:#A5D6A7;color:#202124;}")
            elif current is not None and i == current:
                btn.setStyleSheet(
                    "QPushButton{background:#FFF59D;color:#202124;}")
            else:
                btn.setStyleSheet("")
        if not highlight_progress:
            return
        # Progress buttons
        for i, btn in enumerate(self._prog_buttons):
            if self._prog_done[i]:
                btn.setStyleSheet(
                    "QPushButton{background:#A5D6A7;color:#202124;}")
            elif current is not None and i == current:
                btn.setStyleSheet(
                    "QPushButton{background:#FFF59D;color:#202124;}")
            else:
                btn.setStyleSheet("")

    def _update_temp_highlight(self, current: float | None) -> None:
        """Update colors for temperature spin boxes."""
        for i, sb in enumerate(self.temp_spins):
            done = i < len(self._temp_done) and self._temp_done[i]
            if done:
                sb.setStyleSheet(
                    "QDoubleSpinBox{background:#A5D6A7;color:#202124;}")
            elif current is not None and abs(sb.value() - current) < 1e-6:
                sb.setStyleSheet(
                    "QDoubleSpinBox{background:#FFF59D;color:#202124;}")
            else:
                sb.setStyleSheet("")
    def _update_plots(self, frame: dict, temp: float, idx: int) -> None:
        """Update signal and ToF plots.

        ``idx`` designates the combo number during a frequency scan.
        ``-1`` means the oven is stabilizing.
        """
        data_key = self.data_cb.currentText()
        # refresca campo Ã‚"Oven TÃ‚"
        ovt = frame.get("oven_temp")
        if ovt is not None:
            self.oven_temp.setText(f"{ovt:5.1f}")

        if idx < 0:
            if self._phase != "stab":
                self._phase = "stab"
                self.signals_plot.clear()
                self._stab_curve = None
                self.tof_plot.clear()
                self._tof_curve = None
                self._time_counter = 0
            self._time_counter += 1
            curve = getattr(self, "_tof_curve", None)
            if curve is None:
                curve = self.tof_plot.plot([], [], pen=pg.mkPen("#FF5722", width=2))
                self._tof_curve = curve
            x, y = curve.getData()
            if x is None:
                x, y = [], []
            x = list(x)
            y = list(y)
            x.append(self._time_counter)
            y.append(frame.get("tof", 0.0))
            # curve.setData(x, y, symbol='o')
            curve.setData(x, y)
            # Actualizar tabla de Ãºltimos 10 puntos (timestamp, tof, temp)
            try:
                ts = frame.get("measured_at")
                if ts is None:
                    ts = time.strftime("%Y-%m-%d %H:%M:%S")
                tof_val = float(frame.get("tof") or 0.0)
                temp_val = float(frame.get("oven_temp") or 0.0)
                self._last10_rows.append((str(ts), tof_val, temp_val))
                self._last10_rows = self._last10_rows[-10:]
                self.last10_table.setRowCount(len(self._last10_rows))
                for i, (ts_i, tof_i, temp_i) in enumerate(self._last10_rows):
                    self.last10_table.setItem(i, 0, QtWidgets.QTableWidgetItem(str(ts_i)))
                    self.last10_table.setItem(i, 1, QtWidgets.QTableWidgetItem(f"{tof_i:.3f}"))
                    self.last10_table.setItem(i, 2, QtWidgets.QTableWidgetItem(f"{temp_i:.2f}"))
            except Exception:
                pass
            # Ajustar altura de tabla para mostrar hasta 10 filas sin scroll interno
            try:
                self._resize_last10_table()
            except Exception:
                pass
        else:
            if self._phase != "scan":
                self._phase = "scan"
                self.signals_plot.clear()
                self._scan_curve = None
                self.tof_plot.clear()
                self._tof_curve = None
            data = frame.get(data_key, [])
            x_vals = list(range(len(data)))
            if self._scan_curve is None:
                self._scan_curve = self.signals_plot.plot([], [], pen=pg.mkPen("#2196F3", width=1))
            self._scan_curve.setData(x_vals, data, autoDownsample=True)

    def _store_frame(self, frame: dict, temp: float, idx: int) -> None:
        """Persist frames to database depending on stage."""
        if self.db is None:
            return
        bolt = frame.get("bolt")
        if bolt is None:
            return
        bolt_idx = int(bolt) - 1
        bolt_id = self._bolt_codes[bolt_idx] if 0 <= bolt_idx < len(self._bolt_codes) and self._bolt_codes[bolt_idx] else str(bolt)
        # Resolve batch per bolt; fallback to top-level selection if not available
        batch_id = None
        if 0 <= bolt_idx < len(self._bolt_batches):
            batch_id = self._bolt_batches[bolt_idx]
        if not batch_id:
            batch_id = self._batch_id
        if not batch_id:
            return  # cannot persist without a batch id
        data = frame.copy()
        oven_t = frame.get("oven_temp")
        if oven_t is not None:
            data["temp"] = oven_t
        try:
            if idx < 0:
                # Guardar punto de estabilizaciÃ³n en temp_tof_loading
                seqs = self._ordered_temps(self._temps)
                seq = seqs.index(temp) + 1 if temp in seqs else 0
                self.db.add_temp_tof_loading(
                    batch_id,
                    bolt_id,
                    seq,
                    setpoint=float(temp) if temp is not None else None,
                    oven_temp=(frame.get("oven_temp")),
                    tof=frame.get("tof"),
                    freq=frame.get("freq"),
                    gain=frame.get("gain"),
                    pulse=frame.get("pulse"),
                )
            elif idx == 0:
                seqs = self._ordered_temps(self._temps)
                seq = seqs.index(temp) + 1 if temp in seqs else 0
                self.db.add_temp_loading(batch_id, bolt_id, seq, data)
            elif idx > 0:
                self.db.add_temp_measurement(batch_id, bolt_id, data)
        except Exception as e:
            print(f"DB error: {e}")

    def _get_valid_combos(self) -> tuple[list[tuple[float, float, float]], bool]:
        """Return (combos, from_db)."""
        if self.db is None or self._batch_id is None:
            freqs = range(22, 29)
            pulses = range(10, 17)
            return [(float(f), 25.0, float(p)) for f in freqs for p in pulses], False
        try:
            combos = self.db.one4_valid_combinations(self._batch_id)
            return combos, True
        except Exception:
            raise

    def _get_valid_combos_for_batch(self, batch_id: str | None) -> tuple[list[tuple[float, float, float]], bool]:
        """Return (combos, from_db) for a specific batch id or defaults."""
        if self.db is None or batch_id is None:
            freqs = range(22, 29)
            pulses = range(10, 17)
            return [(float(f), 25.0, float(p)) for f in freqs for p in pulses], False
        try:
            combos = self.db.one4_valid_combinations(batch_id)
            return combos, True
        except Exception:
            freqs = range(22, 29)
            pulses = range(10, 17)
            return [(float(f), 25.0, float(p)) for f in freqs for p in pulses], False

    # ------------------------ Manual AlphaTemp helpers -----------------------
    def _refresh_alpha_batches(self) -> None:
        """Populate the manual alpha_temp dropdown with eligible batches.

        Eligible batches are those having at least one bolt with measurements
        in five different temperatures (rounded to integer degrees) in the
        temp_measurement table.
        """
        if self.db is None:
            items: list[str] = []
        else:
            try:
                items = self.db.list_batches_with_temp_process(min_temps=5)
            except Exception:
                items = []
        self.alpha_batch_cb.blockSignals(True)
        self.alpha_batch_cb.clear()
        self.alpha_batch_cb.addItems(items)
        self.alpha_batch_cb.blockSignals(False)

    def _on_manual_calc_alpha(self) -> None:
        """Compute and save alpha_temp for the selected batch from DB data."""
        if self.db is None:
            QtWidgets.QMessageBox.warning(self, "Alpha_temp", "Base de datos no disponible.")
            return
        batch_id = self.alpha_batch_cb.currentText().strip()
        if not batch_id:
            QtWidgets.QMessageBox.information(self, "Alpha_temp", "Selecciona un Batch num vÃƒÂ¡lido.")
            return
        try:
            alpha = self._compute_and_save_alpha_temp_for_batch(batch_id)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Alpha_temp", f"Error calculando alfatemp: {e}")
            return
        if alpha is None:
            QtWidgets.QMessageBox.information(
                self, "Alpha_temp", "No hay suficientes datos para calcular alfatemp.")
            return
        QtWidgets.QMessageBox.information(
            self,
            "Alpha_temp",
            f"Batch {batch_id}: alfatemp (alpha1) = {alpha:.6g}\nGuardado en tabla batch.",
        )

    def _compute_and_save_alpha_temp_for_batch(self, batch_id: str) -> float | None:
        """Compute alfatemp (alpha1) for a batch from temp_measurement data.

        Logic mirrors the automatic computation after finishing a run, but uses
        existing DB data for the whole batch instead of the current session.

        Steps:
        - Fetch batch params (freq, gain, pulse) to filter measurements.
        - For each bolt in the batch, get latest ToF near 20°C and at other
          temperatures (grouped by integer degrees using a tolerance of Ã‚Â±0.6°C).
        - Compute ratio temp_measured / (ToF_T - ToF_20) for all valid cases.
        - Average ratios to obtain alpha1 and persist via update_alpha1.
        Returns the computed alpha or None if not enough data.
        """
        if self.db is None:
            return None
        # Fetch batch params
        try:
            params = self.db.get_batch_params(batch_id)
            freq_i = int(params.get("freq")) if params.get("freq") is not None else None
            gain_i = int(params.get("gain")) if params.get("gain") is not None else None
            pulse_i = int(params.get("pulse")) if params.get("pulse") is not None else None
        except Exception:
            freq_i = gain_i = pulse_i = None

        # Collect bolts in batch
        try:
            bolts = self.db.list_bolts(batch_id)
        except Exception:
            bolts = []
        if not bolts:
            return None

        tol = 0.6  # °C tolerance to match target temperatures
        ratios: list[float] = []

        for bolt_id in bolts:
            try:
                rows = self.db.fetch_temp_measurements(batch_id, bolt_id, freq_i, gain_i, pulse_i)
            except Exception:
                rows = []
            if not rows:
                continue

            # Helper to select latest row close to a target temperature
            def _latest_for(target: float):
                cand = [r for r in rows if r.get("temp") is not None and abs(float(r["temp"]) - target) <= tol]
                if not cand:
                    return None
                cand.sort(key=lambda r: r.get("measured_at") or 0, reverse=True)
                return cand[0]

            base = _latest_for(20.0)
            if not base or base.get("tof") is None:
                continue
            tof20 = float(base["tof"])
            base_temp = float(base.get("temp")) if base.get("temp") is not None else 20.0

            # Determine integer-rounded temperature groups present in rows
            try:
                present_targets = sorted({int(round(float(r.get("temp") or 0.0))) for r in rows if r.get("temp") is not None})
            except Exception:
                present_targets = []
            for t in present_targets:
                if abs(float(t) - 20.0) <= 1e-6:
                    continue
                r = _latest_for(float(t))
                if not r or r.get("tof") is None or r.get("temp") is None:
                    continue
                delta_tof = float(r["tof"]) - tof20
                if not np.isfinite(delta_tof):
                    continue
                temp_meas = float(r["temp"])  # use measured temp
                delta_temp = temp_meas - base_temp
                if delta_temp == 0.0 or not np.isfinite(delta_temp):
                    continue
                ratio = delta_tof / delta_temp
                if np.isfinite(ratio):
                    ratios.append(ratio)

        if not ratios:
            return None
        alpha = float(np.mean(ratios))
        try:
            self.db.update_alpha1(batch_id, alpha)
        except Exception:
            pass
        return alpha


class DummyOven:
    """Placeholder controller for the industrial oven."""

    def set_temperature(self, temp: float) -> None:
        print(f"[Oven] set temperature to {temp} °C")

    def wait_stable(self, hold_s: int) -> None:
        time.sleep(hold_s)


class TempSequenceWorker(QtCore.QThread):
    frame_ready = QtCore.pyqtSignal(dict, float, int)
    finished = QtCore.pyqtSignal()
    error = QtCore.pyqtSignal(str)
    temp_done = QtCore.pyqtSignal(float)
    temp_started = QtCore.pyqtSignal(float)
    bolt_done = QtCore.pyqtSignal(int)
    bolt_started = QtCore.pyqtSignal(int)

    def __init__(self, com_port: str, combos_list: list[list[tuple[float, float, float]]],
                 temps: list[float], hold_s: int, tol_ns: float,
                 num_bolts: int, data_key: str = "dat3",
                 stab_combo: tuple[float, float, float] | None = None,
                 temp_tol: float = 0.5,   # Ã‚Â±0,5 °C por defecto
                 init_tof: float | None = None,
                 *,
                 stab_bolt_index: int = 0,
                 db: BatchDB | None = None,
                 batch_id: str | None = None,
                 sample_every_s: int = 60
                 ) -> None:
        super().__init__()
        self._port = com_port
        self._combos_list = combos_list
        self._temps = temps
        self._hold = hold_s
        self._sample_s = max(1, int(sample_every_s))
        self._tol = tol_ns
        self._temp_tol = float(temp_tol)
        self._data_key = data_key
        self._stab_combo = stab_combo
        self._num_bolts = num_bolts
        self._stab_bolt_index = max(0, min(int(stab_bolt_index), max(0, num_bolts - 1)))
        if self._stab_combo is None:
            try:
                self._stab_combo = self._combos_list[self._stab_bolt_index][0]
            except Exception:
                self._stab_combo = (40.0, 20.0, 2.0)
        self._stop = False
        self._scan_worker: ComboScanWorker | None = None
        # Default ToF from provided value or from an 80 mm bolt if absent
        self._current_tof = init_tof if init_tof is not None else (2 * 80.0) / 5900.0 * 1e6
        self._oven = SimServOven("192.168.122.101")   # IP real de la cÃƒÂ¡mara

        # Optional DB to fetch xi/alpha once for the session, and window params
        self._db = db
        self._batch_id = batch_id
        xi, alpha = 0.0, 0.0
        long_corr = 10
        try:
            if self._db and self._batch_id:
                _dif, _lc, xi_v, alpha_v = self._db.get_device_params(self._batch_id)
                xi, alpha = xi_v, alpha_v
                long_corr = int(_lc) if _lc is not None else 10
        except Exception:
            pass
        self._xi = float(xi)
        self._alpha = float(alpha)
        self._long_corr = int(long_corr)
        # Window params from DB; fall back to current defaults used in this tab
        scw = stw = ltw = None
        try:
            if self._db and self._batch_id:
                attrs = (self._db.get_batch(self._batch_id) or {}).get("attrs", {})
                def _to_int(v):
                    try:
                        return int(float(v))
                    except Exception:
                        return None
                stw = _to_int(attrs.get("short_temporal_window"))
                ltw = _to_int(attrs.get("long_temporal_window"))
                scw = _to_int(attrs.get("short_correlation_window"))
        except Exception:
            scw = stw = ltw = None
        self._short_corr = int(scw if scw is not None else 900)
        self._short_temp = int(stw if stw is not None else 990)
        self._long_temp  = int(ltw if ltw is not None else 990)


    def stop(self) -> None:
        self._stop = True
        if self._scan_worker and self._scan_worker.isRunning():
            self._scan_worker.stop()

        # Detiene el programa manual del horno en caso de parada
        try:
            self._oven.enable_manual(False)
        except Exception:
            pass

    def run(self) -> None:
        oven = self._oven
        oven.enable_manual(True)        #  Ã¢â€ Â NECESARIO en SimServ
        # Initialize oven to first target temperature or ambient (20°C)
        initial_temp = self._temps[0] if self._temps else 20.0
        oven.set_temperature(initial_temp)
        try:
            device = Device(self._port, baudrate=115200, timeout=1)
            for idx, temp in enumerate(self._temps):
                if self._stop:
                    break
                self.temp_started.emit(temp)
                oven.set_temperature(temp)
                # Espera a que ToF y temperatura se estabilicen antes de escanear
                self._current_tof, last_frame = self._wait_until_stable(
                    device, temp, self._current_tof
                )
                self.frame_ready.emit(last_frame, temp, 0)
                # Medidas para el resto de bolts con los parÃƒÂ¡metros de estabilizaciÃƒÂ³n
                freq, gain, pulse = self._stab_combo
                for bolt in range(self._num_bolts):
                    if bolt == self._stab_bolt_index:
                        continue
                    if self._stop:
                        break
                    cfg = dict(temp=temp, diftemp=-103.0, tof=self._current_tof,
                              freq=freq, pulse=pulse, gain=gain, algo=0, threshold=0)
                    self._send_full_config(device, cfg)
                    frame = self._acquire_frame(device)
                    frame["oven_temp"] = self._oven.get_actual_temperature()
                    frame["bolt"] = bolt + 1
                    self.frame_ready.emit(frame, temp, 0)
                complete = True
                for bolt in range(self._num_bolts):
                    if self._stop:
                        complete = False
                        break
                    cfg = dict(temp=temp, diftemp=-103.0, tof=self._current_tof)
                    combos_for_bolt = []
                    try:
                        combos_for_bolt = self._combos_list[bolt]
                    except Exception:
                        combos_for_bolt = self._combos_list[0] if self._combos_list else []

                    worker = ComboScanWorker(
                        device, combos_for_bolt, bolt + 1, cfg, close_device=False,
                        db=self._db, batch_id=self._batch_id
                    )
                    self._scan_worker = worker
                    self.bolt_started.emit(bolt)
                    scan_idx = 0

                    def handle_frame(f, t=temp, b=bolt):
                        nonlocal scan_idx
                        f["oven_temp"] = self._oven.get_actual_temperature()
                        f["bolt"] = b + 1
                        scan_idx += 1
                        self.frame_ready.emit(f, t, scan_idx)

                    worker.frame_ready.connect(
                        handle_frame, QtCore.Qt.DirectConnection
                    )
                    worker.error.connect(self.error.emit)
                    worker.start()
                    worker.wait()
                    self._scan_worker = None
                    self.bolt_done.emit(bolt)

                if complete and not self._stop:
                    self.temp_done.emit(temp)
                if self._stop:
                    break

            # Al finalizar, regresar a 20Ã¢â‚¬Â¯°C sin esperar estabilizaciÃƒÂ³n completa
            if not self._stop:
                oven.set_temperature(20.0)
                try:
                    self._wait_until_reached(device, 20.0, self._current_tof)
                except Exception:
                    pass
                try:
                    oven.enable_manual(False)
                except Exception:
                    pass

            if device.ser.is_open:
                device.ser.close()

            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))
        finally:
            try:
                oven.enable_manual(False)
            except Exception:
                pass

    # -------------------------------------------------------------- helpers
    def _wait_until_stable(self, device: Device, temp: float, tof: float) -> tuple[float, dict]:
        """Bloquea hasta que el ToF y la temperatura del horno se estabilicen.

        Devuelve el ToF medio y el ÃƒÂºltimo frame cuando ambas condiciones se
        han mantenido durante ``self._hold`` segundos."""
        freq, gain, pulse = self._stab_combo
        p = dict(temp=temp, diftemp=-103.0, tof=tof, freq=freq,
                pulse=pulse, gain=gain, algo=0, threshold=0)
        self._send_full_config(device, p)

        frame = self._acquire_frame(device)
        last_tof = frame.get("tof", 0.0) if frame else 0.0
        oven_temp = self._oven.get_actual_temperature()
        if frame is None:
            frame = {}
        # Anotar combo utilizado para estabilizaciÃ³n
        frame["freq"], frame["gain"], frame["pulse"] = freq, gain, pulse
        frame["oven_temp"] = oven_temp
        frame["bolt"] = int(self._stab_bolt_index + 1)
        self.frame_ready.emit(frame, temp, -1)

        in_band_since: float | None = None
        tof_at_band_start: float | None = None
        last_sample = time.monotonic()
        while not self._stop:
            time.sleep(0.25)
            # Muestrea el micro segÃºn el intervalo configurado
            if (time.monotonic() - last_sample) < getattr(self, "_sample_s", 60):
                continue
            frame = self._acquire_frame(device)
            last_sample = time.monotonic()
            if not frame:
                continue
            try:
                oven_temp = self._oven.get_actual_temperature()
            except SimServError as e:
                self.error.emit(str(e))
                oven_temp = float("nan")

            # Anotar combo utilizado
            frame["freq"], frame["gain"], frame["pulse"] = freq, gain, pulse
            frame["oven_temp"] = oven_temp
            frame["bolt"] = int(self._stab_bolt_index + 1)
            self.frame_ready.emit(frame, temp, -1)

            current_tof = float(frame.get("tof", 0.0) or 0.0)
            temp_ok = abs(float(oven_temp) - float(temp)) <= float(self._temp_tol)

            now = time.monotonic()
            if temp_ok:
                if in_band_since is None:
                    # Entramos en banda â†’ arrancar ventana
                    in_band_since = now
                    tof_at_band_start = current_tof
                elapsed = now - in_band_since
                if elapsed >= float(self._hold) and tof_at_band_start is not None:
                    if abs(current_tof - float(tof_at_band_start)) <= float(self._tol):
                        return current_tof, frame
            else:
                # Salimos de banda â†’ reiniciar ventana
                in_band_since = None
                tof_at_band_start = None

        return last_tof, frame

    def _wait_until_reached(self, device: Device, temp: float, tof: float) -> float:
        """Espera a que el horno alcance ``temp`` mostrando la grÃƒÂ¡fica de ToF.

        No se comprueba estabilidad: en cuanto la temperatura estÃƒÂ¡ dentro de
        ``self._temp_tol`` la funciÃƒÂ³n devuelve el ÃƒÂºltimo ToF leÃƒÂ­do."""

        freq, gain, pulse = self._stab_combo
        p = dict(temp=temp, diftemp=-103.0, tof=tof, freq=freq,
                 pulse=pulse, gain=gain, algo=0, threshold=0)
        self._send_full_config(device, p)

        frame = self._acquire_frame(device)
        last_tof = frame.get("tof", 0.0) if frame else 0.0
        oven_temp = self._oven.get_actual_temperature()
        frame["oven_temp"] = oven_temp
        # Anotar combo utilizado
        frame["freq"], frame["gain"], frame["pulse"] = freq, gain, pulse
        if frame:
            self.frame_ready.emit(frame, temp, -1)

        last_sample = time.monotonic()
        while not self._stop:
            time.sleep(0.25)
            if (time.monotonic() - last_sample) < getattr(self, "_sample_s", 60):
                continue
            frame = self._acquire_frame(device)
            last_sample = time.monotonic()
            if not frame:
                continue
            try:
                oven_temp = self._oven.get_actual_temperature()
            except SimServError as e:
                self.error.emit(str(e))
                oven_temp = float("nan")

            frame["oven_temp"] = oven_temp
            # Anotar combo
            frame["freq"], frame["gain"], frame["pulse"] = freq, gain, pulse
            self.frame_ready.emit(frame, temp, -1)

            last_tof = frame.get("tof", last_tof)

            if abs(oven_temp - temp) <= self._temp_tol:
                return last_tof

        return last_tof


    def _send_full_config(self, device: Device, p: dict) -> None:
        from device import _pack32
        d = device
        d.modo_standby(); d.modo_configure()
        d.enviar(_pack32(p['temp']),    '10')
        d.enviar(_pack32(p['diftemp']), '11')
        d.enviar(_pack32(p['tof']),     '12')
        d.enviar(_pack32(p['freq']),    '14')
        d.enviar(_pack32(p['pulse']),   '15')
        d.enviar(_pack32(p['pulse']),   '16')
        d.enviar(_pack32(p['gain']),    '17')
        d.enviar(_pack32(self._xi), '18')
        d.enviar(_pack32(self._alpha), '19')
        d.enviar(_pack32(self._short_corr), '1A')
        d.enviar(_pack32(self._long_corr),  '1B')
        d.enviar(_pack32(self._short_temp), '1C')
        d.enviar(_pack32(self._long_temp),  '1D')
        d.enviar(_pack32(p.get('algo', 0)), '2C')
        d.enviar(_pack32(p.get('threshold', 0)), '2D')
        d.modo_save(); d.modo_standby(); d.modo_single()

    def _acquire_frame(self, device: Device) -> dict:
        # ProtecciÃ³n ante errores esporÃ¡dicos de escritura en puerto serie
        # (p.ej., "WriteFile failed" / "Access is denied").
        # Reintentamos una vez con pequeÃ±o backoff y garantizamos un
        # espaciamiento mÃ­nimo entre IOs consecutivos.
        if not hasattr(self, "_last_io_time"):
            self._last_io_time = 0.0
        now = time.monotonic()
        min_gap = 0.1  # s; adicional a sample_every_s
        if now - float(self._last_io_time) < min_gap:
            time.sleep(min_gap - (now - float(self._last_io_time)))
        for attempt in range(3):
            try:
                device.enviar_temp()
                device.start_measure()
                out = device.lectura()
                self._last_io_time = time.monotonic()
                return out
            except Exception as e:
                msg = str(e)
                low = msg.lower()
                transient = ("writefile" in low) or ("access is denied" in low) or ("denegado" in low)
                if transient and attempt == 0:
                    # Flush buffers y reintento breve
                    try:
                        if getattr(device, 'ser', None):
                            try:
                                device.ser.reset_output_buffer(); device.ser.reset_input_buffer()
                            except Exception:
                                pass
                    except Exception:
                        pass
                    time.sleep(0.6)
                    continue
                if transient and attempt == 1:
                    # Reabrir el puerto serie y reintentar
                    try:
                        if getattr(device, 'ser', None):
                            try:
                                device.ser.reset_output_buffer(); device.ser.reset_input_buffer()
                            except Exception:
                                pass
                            try:
                                if getattr(device.ser, 'is_open', False):
                                    device.ser.close()
                            except Exception:
                                pass
                            time.sleep(0.5)
                            try:
                                device.ser.open()
                            except Exception:
                                # Si no se puede abrir, abortamos el reintento
                                pass
                            time.sleep(0.3)
                    except Exception:
                        pass
                    continue
                self.error.emit(msg)
                break
        return {}


class TemperatureFilterTab(FilterTab):
    def __init__(self, *, com_selector=None, db: BatchDB | None = None,
                 batch_selector: QComboBox | None = None, parent=None):
        super().__init__(com_selector=com_selector, parent=parent)
        self.db = db or BatchDB()
        self.batch_selector = batch_selector

        src_layout = QHBoxLayout()
        src_layout.addWidget(QLabel("Source:"))
        self.src_selector = QComboBox()
        self.src_selector.addItems(["File", "Database"])
        self.src_selector.setCurrentIndex(1)
        src_layout.addWidget(self.src_selector)
        self.layout().insertLayout(0, src_layout)

        self.btn_load_db = QPushButton("Load from DB")
        self.btn_load_db.setStyleSheet("background-color: #3498db; border: 1px solid #2980b9; color: white;")
        self.btn_load_db.clicked.connect(self._filter_from_db)
        self.layout().insertWidget(1, self.btn_load_db)

        self.src_selector.currentIndexChanged.connect(lambda _: self._on_source_changed())
        self._on_source_changed()

        thresh_layout = QHBoxLayout()
        thresh_layout.addWidget(QLabel("Min Valid % (pct & pico):"))
        self.valid_spin = QDoubleSpinBox()
        self.valid_spin.setRange(0.0, 100.0)
        self.valid_spin.setDecimals(0)
        self.valid_spin.setValue(100.0)
        thresh_layout.addWidget(self.valid_spin)
        self.layout().insertLayout(3, thresh_layout)
        self.valid_spin.valueChanged.connect(self._apply_threshold)

        for w in (self.bolt_spin, self.btn_scan, self.temp_spin, self.tof_spin,
                  self.algo_selector, self.threshold_label, self.threshold_spin,
                  self.tof_mode, self.bolt_length_label, self.bolt_length,
                  self.velocity_label, self.velocity):
            self._hide_layout_with_widget(w)

        self.btn_add_db = QPushButton("Add to DB")
        self.btn_add_db.setStyleSheet("background-color: #e67e22; border: 1px solid #d35400; color: white;")
        self.btn_add_db.clicked.connect(self._on_add_db)
        self.layout().addWidget(self.btn_add_db)

        self.best_combo: tuple[float, float, float] | None = None
        self.best_combo_index: int | None = None
        self._table_df = None
        self.bolt_signals: dict[tuple[int, int, int], dict[float, dict[int, dict[str, list]]]] = {}

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

        self.table.clicked.connect(self._on_table_clicked)

    def _on_source_changed(self):
        file_mode = self.src_selector.currentText() == "File"
        self.btn_open.setVisible(file_mode)
        self.btn_start.setVisible(True)
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
            # Fetch with timestamp so we can keep latest per (F,G,P,Bolt,Temp)
            self.db.cur.execute(
                "SELECT freq, gain, pulse, bolt_id, temp, pico1, pct_diff, dat2, dat3, measured_at "
                "FROM temp_measurement WHERE batch_id=%s ORDER BY measured_at DESC",
                (batch_id,),
            )
            rows = self.db.cur.fetchall()
            cols = [d[0] for d in self.db.cur.description]
            df = pd.DataFrame(rows, columns=cols)
            df.rename(columns={'freq': 'Freq', 'gain': 'Gain', 'pulse': 'Pulse', 'bolt_id': 'Bolt ID'}, inplace=True)
            df['Bolt Num'] = df['Bolt ID'].apply(lambda bid: self.db.get_bolt_alias(batch_id, bid))
            # Round temperature to integer groups for Temp process logic
            df['temp'] = df['temp'].astype(float).round().astype(int)
            # Keep only latest measurement per (Freq, Gain, Pulse, Bolt Num, temp)
            try:
                df = df.sort_values('measured_at', ascending=False)
                df = df.drop_duplicates(subset=['Freq', 'Gain', 'Pulse', 'Bolt Num', 'temp'], keep='first').reset_index(drop=True)
            except Exception:
                pass

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

        total_bolts = df.groupby(['Freq', 'Gain', 'Pulse', 'temp'])['Bolt Num'].nunique().rename('total_bolts')
        df['pct_diff_num'] = df['pct_diff'] * 100 if df['pct_diff'].max() <= 1 else df['pct_diff']
        df_pct = df[df['pct_diff_num'] >= float(self.pct_spin.value())]
        if self.peak_chk.isChecked():
            df_core = df_pct[df_pct['pico1'] >= float(self.peak_spin.value())]
        else:
            df_core = df_pct
        valid_bolts = df_core.groupby(['Freq', 'Gain', 'Pulse', 'temp'])['Bolt Num'].nunique().rename('valid_bolts')
        debug_df = pd.concat([total_bolts, valid_bolts], axis=1).reset_index().fillna(0)
        combos_ok_temp = debug_df[debug_df['total_bolts'] == debug_df['valid_bolts']]

        temps_total = df.groupby(['Freq', 'Gain', 'Pulse'])['temp'].nunique()
        temps_ok = combos_ok_temp.groupby(['Freq', 'Gain', 'Pulse'])['temp'].nunique()
        combos_valid = [idx for idx, val in temps_ok.items() if val == temps_total[idx]]
        df_final = df_core[df_core[['Freq', 'Gain', 'Pulse']].apply(tuple, axis=1).isin(combos_valid)].reset_index(drop=True)
        debug_df.rename(columns={'Freq': 'Freq', 'Gain': 'Gain', 'Pulse': 'Pulse'}, inplace=True)
        self._on_finished(df_final, debug_df, "")

    def _hide_layout_with_widget(self, widget):
        lay = None
        main = self.layout()
        for i in range(main.count()):
            item = main.itemAt(i)
            l = item.layout()
            if l:
                for j in range(l.count()):
                    if l.itemAt(j).widget() is widget:
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
            self.db.cur.execute("DELETE FROM temp_valid_combo WHERE batch_id=%s", (batch_id,))
            combos_set: set[tuple[int, int, int]] = set()
            best_combo: tuple[int, int, int] | None = None
            count = 0
            for row_idx in range(model.rowCount()):
                sel_item = model.item(row_idx, 0)
                if sel_item.checkState() != Qt.Checked:
                    continue
                combo = self._table_df.loc[row_idx]
                freq = int(combo['Freq']); gain = int(combo['Gain']); pulse = int(combo['Pulse'])
                key = (freq, gain, pulse)
                if key in combos_set:
                    continue
                combos_set.add(key)
                best_state = model.item(row_idx, 1).checkState() == Qt.Checked
                self.db.cur.execute(
                    "INSERT INTO temp_valid_combo (batch_id, freq, gain, pulse, is_best) VALUES (%s, %s, %s, %s, %s)",
                    (batch_id, freq, gain, pulse, best_state),
                )
                if best_state:
                    best_combo = key
                count += 1
            if best_combo:
                f, g, p = best_combo
                self.db.cur.execute(
                    "UPDATE batch SET frequency=%s, gain=%s, cycles_coarse=%s, cycles_fine=%s WHERE batch_id=%s",
                    (f, g, p, p, batch_id),
                )
            QMessageBox.information(self, "Guardado", f"{count} combinaciones guardadas en la base de datos.")
        except Exception as e:
            QMessageBox.warning(self, "DB Error", f"Error guardando combinaciones: {e}")
        
    def _load_dataframe_into_table(self, df):
        df = df.reset_index(drop=True)
        self._table_df = df
        super()._load_dataframe_into_table(df)
        model = self._table_model
        model.insertColumn(0)
        model.insertColumn(1)
        model.setHeaderData(0, Qt.Horizontal, "Select")
        model.setHeaderData(1, Qt.Horizontal, "Best")
        try:
            best_idx = df['pct_ok_frac'].idxmax()
            self.best_combo_index = int(best_idx)
            self.best_combo = tuple(df.loc[best_idx, ['Freq', 'Gain', 'Pulse']])
        except Exception:
            best_idx = -1
            self.best_combo_index = None
            self.best_combo = None
        for row in range(model.rowCount()):
            sel_item = QtGui.QStandardItem()
            sel_item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            sel_item.setCheckState(Qt.Checked)
            model.setItem(row, 0, sel_item)
            best_item = QtGui.QStandardItem()
            best_item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            best_item.setCheckState(Qt.Checked if row == best_idx else Qt.Unchecked)
            model.setItem(row, 1, best_item)
        self.table.setModel(model)
        model.itemChanged.connect(self._on_best_toggled)

    def _on_finished(self, df_filtered, df_debug, out_path):
        pct_thr = float(self.pct_spin.value())
        use_peak = self.peak_chk.isChecked()
        peak_thr = float(self.peak_spin.value())
        self.current_pct_thr = pct_thr
        self.current_peak_used = use_peak
        self.current_peak_thr = peak_thr

        import pandas as __pd
        df_full = self._db_full_df.copy()
        df_full['pct_diff_num'] = FilterWorker._pct_to_float(df_full['pct_diff'])
        df_full['pct_diff'] = df_full['pct_diff_num'] / 100
        df_full = df_full.drop_duplicates(subset=['Freq', 'Gain', 'Pulse', 'Bolt Num', 'temp'], keep='first')

        pivot_pct = df_full.pivot(index=['Freq', 'Gain', 'Pulse'], columns=['temp', 'Bolt Num'], values='pct_diff')
        temps = sorted(df_full['temp'].unique())
        bolts = sorted(df_full['Bolt Num'].unique())
        mask_pct = pivot_pct.notna()
        frac_pct = ((pivot_pct >= pct_thr / 100) & mask_pct).sum(axis=1) / mask_pct.sum(axis=1)

        df_res = __pd.DataFrame(index=pivot_pct.index)
        df_res['pct_ok_frac'] = frac_pct

        if use_peak:
            pivot_peak = df_full.pivot(index=['Freq', 'Gain', 'Pulse'], columns=['temp', 'Bolt Num'], values='pico1')
            mask_peak = pivot_peak.notna()
            frac_peak = ((pivot_peak >= peak_thr) & mask_peak).sum(axis=1) / mask_peak.sum(axis=1)
            df_res['pico1_ok_frac'] = frac_peak

        for t in temps:
            for b in bolts:
                col = pivot_pct[(t, b)] if (t, b) in pivot_pct.columns else np.nan
                df_res[f'pct_{int(t)}_{int(b)}'] = col
                if use_peak:
                    colp = pivot_peak[(t, b)] if (t, b) in pivot_peak.columns else np.nan
                    df_res[f'pico1_{int(t)}_{int(b)}'] = colp

        table_df = df_res.reset_index()
        all_combos = df_debug[['Freq', 'Gain', 'Pulse']].drop_duplicates()
        table_df = all_combos.merge(table_df, on=['Freq', 'Gain', 'Pulse'], how='left').fillna(0)

        if use_peak:
            perfect = (table_df['pct_ok_frac'] == 1.0) & (table_df['pico1_ok_frac'] == 1.0)
            table_df['perfect'] = perfect
            table_df = table_df.sort_values(by=['perfect', 'pct_ok_frac', 'pico1_ok_frac'],
                                           ascending=[False, False, False]).drop(columns=['perfect'])
        else:
            perfect = table_df['pct_ok_frac'] == 1.0
            table_df['perfect'] = perfect
            table_df = table_df.sort_values(by=['perfect', 'pct_ok_frac'],
                                           ascending=[False, False]).drop(columns=['perfect'])

        self.bolt_signals.clear()
        dat2_cols = [c for c in df_full.columns if c.startswith('dat2_')]
        dat3_cols = [c for c in df_full.columns if c.startswith('dat3_')]
        for _, row in df_full.iterrows():
            combo_key = (int(row['Freq']), int(row['Gain']), int(row['Pulse']))
            temp = float(row['temp'])
            bolt = int(row['Bolt Num'])
            sigs = self.bolt_signals.setdefault(combo_key, {})
            temp_dict = sigs.setdefault(temp, {})
            temp_dict[bolt] = {
                'dat2': row[dat2_cols].to_list(),
                'dat3': row[dat3_cols].to_list(),
            }

        self.valid_combos_df = table_df[['Freq', 'Gain', 'Pulse']].drop_duplicates().reset_index(drop=True)
        self._load_dataframe_into_table(table_df)
        if out_path:
            QMessageBox.information(self, "Export", f"Archivo creado:\n{out_path}\n\nFilas exportadas: {len(df_filtered)}")

        self.bolt_spin.setEnabled(False)
        self.btn_scan.setEnabled(False)

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
                    val = float(item.text().strip('%')) / 100.0
                except Exception:
                    continue
                if val >= threshold:
                    item.setBackground(QtGui.QBrush(QtGui.QColor('green')))

    def _on_table_clicked(self, index):
        if not index.isValid() or self._table_df is None:
            return
        header = self._table_model.headerData(index.column(), Qt.Horizontal)
        if not isinstance(header, str):
            return
        if header.startswith('pct_') or header.startswith('pico1_'):
            if header.endswith('_frac'):
                return
            parts = header.split('_')
            if len(parts) < 3:
                return
            try:
                temp = float(parts[1])
                bolt = int(parts[2])
            except Exception:
                return
            row = index.row()
            if row >= len(self._table_df):
                return
            combo = tuple(int(self._table_df.loc[row, c]) for c in ['Freq', 'Gain', 'Pulse'])
            dtype = self.data_selector.currentText()
            self._open_plot(combo, temp, bolt, dtype)

    def _open_plot(self, combo: tuple[int, int, int], temp: float, bolt: int, dtype: str) -> None:
        data = (self.bolt_signals.get(combo, {}).get(temp, {}).get(bolt, {}).get(dtype))
        if not data:
            QMessageBox.warning(self, "Data", "Signal not available for this selection")
            return
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(f"T{temp} Bolt {bolt} - {dtype} F{combo[0]} G{combo[1]} P{combo[2]}")
        lay = QVBoxLayout(dlg)
        # Plot with green color and hover amplitude readout
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

        # Legend rows: Max amplitude and Cursor value
        max_label_item = None
        cursor_label_item = None
        try:
            import numpy as _np
            max_amp = float(_np.nanmax(_np.asarray(list(data), dtype=float))) if len(data) else float('nan')
            sample_max = pg.PlotDataItem([], [])
            sample_max.setPen(pg.mkPen(0, 0, 0, 0))  # hide sample
            legend.addItem(sample_max, f"Max amplitude: {max_amp:.3f}")
            max_label_item = legend.items[-1][1] if getattr(legend, 'items', None) else None

            sample_cur = pg.PlotDataItem([], [])
            sample_cur.setPen(pg.mkPen(0, 0, 0, 0))
            legend.addItem(sample_cur, "Cursor: -")
            cursor_label_item = legend.items[-1][1] if getattr(legend, 'items', None) else None
        except Exception:
            pass

        # Hover helpers: vertical line and text label
        vline = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('#00c853', width=1))
        plot.addItem(vline); vline.setVisible(False)
        # Use light text color for dark background
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
                # Update legend cursor row
                try:
                    if cursor_label_item is not None:
                        cursor_label_item.setText(f"Cursor: {amp:.3f}")
                except Exception:
                    pass
            except Exception:
                pass

        # Connect hover event
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
        # Make dialog resizable by user
        try:
            dlg.setSizeGripEnabled(True)
        except Exception:
            pass
        dlg.resize(600, 400)
        dlg.exec_()

    def _on_best_toggled(self, item: QtGui.QStandardItem):
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


class TemperatureCSVViewerTab(QWidget):
    def __init__(self, *, db: BatchDB | None = None,
                 batch_selector: QComboBox | None = None, parent=None):
        super().__init__(parent)
        self.db = db
        self.batch_selector = batch_selector

        self.viewer = CSVViewerTab(db=db, batch_selector=batch_selector)
        self._temp_map: dict[tuple[int, int, float], set[float]] = {}
        # Temperature grouping for Temp dropdown (±0.5 °C)
        self._temp_group_tol: float = 0.5
        # Five process setpoints for the batch (fallback to defaults)
        self._setpoints: list[float] = [-20.0, 0.0, 20.0, 40.0, 50.0]

        # Monkey-patch viewer filtering to use grouped temperatures in this tab only
        def _filter_df_current_grouped(_self_v):
            import numpy as _np
            import pandas as _pd
            if not getattr(_self_v, '_df_list', None):
                return []
            # Parse selections
            try:
                f = int(float(_self_v.freq_cb.currentText()))
                g = int(float(_self_v.gain_cb.currentText()))
                p = float(_self_v.pulse_cb.currentText())
                t = float(_self_v.temp_cb.currentText())
            except Exception:
                return []
            # Selected bolts
            model = _self_v.bolt_cb.model()
            bolts = []
            for i in range(model.rowCount()):
                it = model.item(i)
                if it.checkState() == Qt.Checked:
                    bolts.append(it.text())
            if not bolts:
                return []
            if 'All' in bolts:
                bolt_vals = _self_v._df_list[0]['Bolt Num'].unique()
            else:
                bolt_vals = [int(v) for v in bolts]
            tol = float(self._temp_group_tol)
            df_subs: list[_pd.DataFrame] = []
            for df in _self_v._df_list:
                try:
                    pulse_series = _pd.to_numeric(df['Pulse'], errors='coerce')
                    temp_series = _pd.to_numeric(df['temp'], errors='coerce')
                    cmask = (
                        (df['Freq'] == f) &
                        (df['Gain'] == g) &
                        (_np.isclose(pulse_series, p, rtol=0, atol=1e-6)) &
                        (_np.isfinite(temp_series)) & (abs(temp_series - t) <= tol) &
                        (df['Bolt Num'].isin(bolt_vals))
                    )
                    sub = df[cmask]
                except Exception:
                    sub = _pd.DataFrame()
                df_subs.append(sub)
            return df_subs

        try:
            self.viewer._filter_df_current = types.MethodType(_filter_df_current_grouped, self.viewer)
        except Exception:
            pass
        for cb in (self.viewer.freq_cb, self.viewer.gain_cb, self.viewer.pulse_cb):
            try:
                cb.currentIndexChanged.connect(self._sync_temp_options)
            except Exception:
                pass

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

    def _on_source_changed(self):
        db_mode = self.src_selector.currentText() == "Database"
        for w in (self.viewer.file1_edit, self.viewer.browse1_btn,
                  self.viewer.file2_edit, self.viewer.browse2_btn,
                  self.viewer.load_btn):
            w.setVisible(not db_mode)
        self.load_db_btn.setVisible(db_mode)

    def _sync_temp_options(self):
        """Show the five process temperatures in the Temp dropdown.

        The plot/table will include rows whose measured temperature falls
        within ±self._temp_group_tol of the selected setpoint.
        """
        cb = self.viewer.temp_cb
        cb.blockSignals(True)
        cb.clear()
        cb.addItems([f"{t:g}" for t in self._setpoints])
        cb.blockSignals(False)
        if self._setpoints:
            # Prefer 20 °C when present
            try:
                idx_20 = self._setpoints.index(20.0)
            except ValueError:
                idx_20 = 0
            cb.setCurrentIndex(idx_20)
        try:
            self.viewer._update_plot(); self.viewer._update_table()
        except Exception:
            pass

    def _load_from_db(self):
        if not self.batch_selector or not self.db:
            QMessageBox.warning(self, "Batch ID", "Selecciona un Batch antes de cargar.")
            return
        batch_id = self.batch_selector.currentText()
        try:
            import pandas as pd, numpy as np
            self.db.cur.execute(
                "SELECT freq, gain, pulse, bolt_id, temp, pico1, pct_diff, tof, dat2, dat3, measured_at "
                "FROM temp_measurement WHERE batch_id=%s ORDER BY measured_at DESC",
                (batch_id,),
            )
            rows = self.db.cur.fetchall()
            cols = [d[0] for d in self.db.cur.description]
            df = pd.DataFrame(rows, columns=cols)
            df.rename(columns={'freq': 'Freq', 'gain': 'Gain', 'pulse': 'Pulse', 'bolt_id': 'Bolt ID'}, inplace=True)
            df['Bolt Num'] = df['Bolt ID'].apply(lambda bid: self.db.get_bolt_alias(batch_id, bid))
            df[['Freq', 'Gain', 'Bolt Num']] = df[['Freq', 'Gain', 'Bolt Num']].astype(int)
            # Normalize numeric fields
            df['Pulse'] = pd.to_numeric(df['Pulse'], errors='coerce')
            df['temp'] = pd.to_numeric(df['temp'], errors='coerce')
            # Define processed temperature (rounded integer) for case grouping
            df['temp_proc'] = df['temp'].round().astype(int)
            # Keep only latest measurement per (Freq, Gain, Pulse, Bolt Num, processed temp)
            try:
                df = df.sort_values('measured_at', ascending=False)
                df = df.drop_duplicates(subset=['Freq', 'Gain', 'Pulse', 'Bolt Num', 'temp_proc'], keep='first').reset_index(drop=True)
            except Exception:
                pass
            # (dat arrays converted below)

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
            dat2_df = pd.DataFrame([(arr.tolist() + [0]*(n2 - arr.size))[:n2] for arr in dat2_arrays],
                                   columns=[f'dat2_{i}' for i in range(n2)])
            dat3_df = pd.DataFrame([(arr.tolist() + [0]*(n3 - arr.size))[:n3] for arr in dat3_arrays],
                                   columns=[f'dat3_{i}' for i in range(n3)])
            df = pd.concat([df.drop(columns=['dat2', 'dat3']), dat2_df, dat3_df], axis=1)
        except Exception as e:
            QMessageBox.critical(self, "DB Error", f"Error cargando de DB: {e}")
            return

        # Ensure uniqueness by case using processed temperature again after concatenation
        try:
            if 'temp_proc' not in df.columns:
                df['temp'] = pd.to_numeric(df['temp'], errors='coerce')
                df['temp_proc'] = df['temp'].round().astype(int)
            df = df.drop_duplicates(subset=['Freq', 'Gain', 'Pulse', 'Bolt Num', 'temp_proc'], keep='first')
        except Exception:
            df = df.drop_duplicates(subset=['Freq', 'Gain', 'Pulse', 'Bolt Num', 'temp'], keep='first')
        # Do not expose helper column in the viewer
        try:
            df = df.drop(columns=['temp_proc'])
        except Exception:
            pass
        self.viewer._df_list = [df]

        temp_map: dict[tuple[int, int, float], set[float]] = {}
        for _, row in df.iterrows():
            key = (int(row['Freq']), int(row['Gain']), float(row['Pulse']))
            temp_map.setdefault(key, set()).add(float(row['temp']))
        self._temp_map = temp_map

        freqs = sorted(df['Freq'].astype(int).unique())
        self.viewer.freq_cb.clear(); self.viewer.freq_cb.addItems([str(f) for f in freqs]); self.viewer.freq_cb.setEnabled(True)
        gains = sorted(df['Gain'].astype(int).unique())
        self.viewer.gain_cb.clear(); self.viewer.gain_cb.addItems([str(g) for g in gains]); self.viewer.gain_cb.setEnabled(True)
        pulses = sorted(pd.to_numeric(df['Pulse'], errors='coerce').unique())
        self.viewer.pulse_cb.clear(); self.viewer.pulse_cb.addItems([f"{p:g}" for p in pulses]); self.viewer.pulse_cb.setEnabled(True)
        # Determine process setpoints for Temp grouping from batch attrs
        try:
            b = self.db.get_batch(batch_id) or {"attrs": {}}
            attrs = b.get("attrs", {}) or {}
            def _to_float(v):
                try:
                    return float(v)
                except Exception:
                    return None
            tmin = _to_float(attrs.get("min_temp"))
            tmax = _to_float(attrs.get("max_temp"))
            self._setpoints = _compute_batch_setpoints(tmin, tmax)
        except Exception:
            self._setpoints = [-20.0, 0.0, 20.0, 40.0, 50.0]

        # Fill Temp dropdown with grouped setpoints (not raw measured temps)
        self.viewer.temp_cb.clear(); self.viewer.temp_cb.addItems([f"{t:g}" for t in self._setpoints]); self.viewer.temp_cb.setEnabled(True)

        bolt_model: QtGui.QStandardItemModel = self.viewer.bolt_cb.model()
        bolt_model.clear()
        all_item = QtGui.QStandardItem('All')
        all_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
        all_item.setCheckState(Qt.Checked)
        bolt_model.appendRow(all_item)
        for b in sorted(df['Bolt Num'].astype(int).unique()):
            it = QtGui.QStandardItem(str(b))
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

        self._sync_temp_options()
        self.viewer._update_plot()
        self.viewer._update_table()


class TemperatureTab(QWidget):
    """Container tab for temperature qualification with sub-tabs."""

    def __init__(self, com_selector=None, db: BatchDB | None = None, parent: QWidget | None = None):
        super().__init__(parent)
        self.db = db or BatchDB()
        self.com_selector = com_selector

        layout = QVBoxLayout(self)
        batch_row = QHBoxLayout()
        batch_row.addWidget(QLabel("Batch Num:"))
        self.batch_cb = RefreshComboBox(self._refresh_batches, self)
        batch_row.addWidget(self.batch_cb)
        layout.addLayout(batch_row)

        tabs = QTabWidget()
        self.process_tab = TemperatureProcessTab(db=self.db, com_selector=self.com_selector,
                                                 batch_selector=self.batch_cb)
        tabs.addTab(self.process_tab, "Process")
        # New summary tab (scrollable)
        self.summary_tab = TemperatureSummaryTab(db=self.db, batch_selector=self.batch_cb)
        scroll_summary = QScrollArea()
        scroll_summary.setWidgetResizable(True)
        scroll_summary.setWidget(self.summary_tab)
        tabs.addTab(scroll_summary, "Summary")
        filter_tab = TemperatureFilterTab(db=self.db, batch_selector=self.batch_cb)
        scroll_filter = QScrollArea()
        scroll_filter.setWidgetResizable(True)
        scroll_filter.setWidget(filter_tab)
        tabs.addTab(scroll_filter, "Valid Combinations")
        csv_tab = TemperatureCSVViewerTab(db=self.db, batch_selector=self.batch_cb)
        tabs.addTab(csv_tab, "CSV Viewer")
        layout.addWidget(tabs)

        self._refresh_batches()

    def _refresh_batches(self):
        if self.db is None:
            items: list[str] = []
        else:
            try:
                items = self.db.batch_numbers()
            except Exception:
                items = []
        self.batch_cb.blockSignals(True)
        self.batch_cb.clear()
        self.batch_cb.addItems(items)
        self.batch_cb.blockSignals(False)
        if items:
            self.batch_cb.setCurrentIndex(0)
            self.process_tab._refresh_batches()
        else:
            self.process_tab._batch_id = None

    # ------------------------ Manual AlphaTemp helpers -----------------------

