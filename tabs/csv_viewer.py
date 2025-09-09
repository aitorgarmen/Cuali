from __future__ import annotations

"""
CSVViewerTab - Tab para cargar un CSV generado por la app y visualizar
gráficas de dat2 / dat3 por combinación Freq‑Gain‑Pulse‑Bolt.

Requisitos:
    pip install pandas pyqt5 pyqtgraph
"""

from pathlib import Path
from typing import List, Tuple, Optional

import logging
import numpy as np
import pandas as pd
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QStandardItemModel, QStandardItem
import pyqtgraph as pg
from pgdb import BatchDB


logger = logging.getLogger(__name__)


class CSVViewerTab(QtWidgets.QWidget):
    """Pestaña de exploración de CSV.

    Flujo de uso:
        1. Buscar fichero (.csv) ➜ ruta en QLineEdit.
        2. "Cargar" ➜ se lee el CSV en un DataFrame.
        3. Se detectan todas las combinaciones únicas de
           (Freq, Gain, Pulse, Bolt Num) y se listan en un QComboBox.
        4. Al seleccionar una combinación:
              • Se trazan dat2 y/o dat3 según los checkboxes.
              • Se actualiza el panel con el resto de parámetros
                (pico1, pct_diff, tof, force, maxcorrx, maxcorry, ...).
    """

    def __init__(self, *, db: Optional[BatchDB] = None,
                 batch_selector: Optional[QtWidgets.QComboBox] = None):
        super().__init__()
        self.db = db
        self.batch_selector = batch_selector
        self._df_list: List[pd.DataFrame] = []
        # Track per-file origin to distinguish loaded/unloaded groups
        self._file_origins: List[str] = []
        self._build_ui()

    # ---------------------------------------------------------------- UI
    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(12)

        # ── 1. Selectors de fichero + botón Cargar ───────────────────────
        # Soporte para hasta 2 ficheros
        frow1 = QtWidgets.QHBoxLayout()
        self.file1_edit = QtWidgets.QLineEdit(); self.file1_edit.setPlaceholderText('Archivo 1')
        self.browse1_btn = QtWidgets.QPushButton('...')
        self.browse1_btn.clicked.connect(lambda: self._choose_file(self.file1_edit))
        frow1.addWidget(self.file1_edit, 1); frow1.addWidget(self.browse1_btn)
        frow2 = QtWidgets.QHBoxLayout()
        self.file2_edit = QtWidgets.QLineEdit(); self.file2_edit.setPlaceholderText('Archivo 2 (opcional)')
        self.browse2_btn = QtWidgets.QPushButton('...')
        self.browse2_btn.clicked.connect(lambda: self._choose_file(self.file2_edit))
        frow2.addWidget(self.file2_edit, 1); frow2.addWidget(self.browse2_btn)
        load_row = QtWidgets.QHBoxLayout()
        self.load_btn = QtWidgets.QPushButton('Cargar')
        self.load_btn.clicked.connect(self._load_csv)
        load_row.addStretch(); load_row.addWidget(self.load_btn)
        layout.addLayout(frow1); layout.addLayout(frow2); layout.addLayout(load_row)

        # ── 2. Parámetros de selección individuales ─────────────────────
        crow = QtWidgets.QHBoxLayout()
        # Dropdowns for each parameter
        self.freq_cb = QtWidgets.QComboBox(); self.gain_cb = QtWidgets.QComboBox()
        self.pulse_cb = QtWidgets.QComboBox(); self.bolt_cb = QtWidgets.QComboBox()
        self.type_cb = QtWidgets.QComboBox()
        # New: scale selector (Normalized/Raw)
        self.scale_cb = QtWidgets.QComboBox()
        self.scale_cb.addItems(['Normalized (1e6)', 'Raw'])
        self.scale_cb.setCurrentIndex(0)
        self.scale_cb.setEnabled(True)
        # New: plot mode selector (Alligned/Not alligned)
        self.align_cb = QtWidgets.QComboBox()
        # Populate plot mode at build time so it's visible always
        self.align_cb.addItems(['No', 'Yes'])
        self.align_cb.setCurrentIndex(0)
        self.align_cb.setEnabled(True)
        self.temp_cb = QtWidgets.QComboBox()  # Dropdown for temperature
        # Insert temp_cb into selectors layout
        for label, cb in [
            ("Freq:", self.freq_cb), ("Gain:", self.gain_cb),
            ("Pulse:", self.pulse_cb), ("Bolt Num:", self.bolt_cb),
            ("Data Type:", self.type_cb), ("Scale:", self.scale_cb), ("Alligned:", self.align_cb), ("Temp:", self.temp_cb)
        ]:
            crow.addWidget(QtWidgets.QLabel(label)); crow.addWidget(cb)
        layout.addLayout(crow)
        # Disable until CSV loaded
        # Only bolt_cb will use multiselect with checkable items; others single-select
        # Keep align_cb enabled so options are visible even without CSV
        for cb in (self.freq_cb, self.gain_cb, self.pulse_cb, self.type_cb, self.temp_cb):
            cb.setEnabled(False)
        # Bolt dropdown: checkable multi-select list
        bolt_model = QStandardItemModel(self.bolt_cb)
        self.bolt_cb.setModel(bolt_model)
        # Editable display to show selected items
        self.bolt_cb.setEditable(True)
        self.bolt_cb.lineEdit().setReadOnly(True)
        self.bolt_cb.setEnabled(False)

        # Connect changes to update handlers
        # Connect updates
        for cb in (self.freq_cb, self.gain_cb, self.pulse_cb, self.type_cb, self.scale_cb, self.align_cb, self.temp_cb):
            cb.currentIndexChanged.connect(lambda _: (self._update_plot(), self._update_table()))
        # Bolt multiselect triggers: manage 'All' logic and display
        def on_bolt_item_changed(item: QStandardItem):
            # If 'All' checked, uncheck others; if any other checked, uncheck 'All'
            if item.text() == 'All' and item.checkState() == Qt.Checked:
                for r in range(bolt_model.rowCount()):
                    it = bolt_model.item(r)
                    if it.text() != 'All': it.setCheckState(Qt.Unchecked)
            elif item.text() != 'All' and item.checkState() == Qt.Checked:
                all_item = next((bolt_model.item(r) for r in range(bolt_model.rowCount()) if bolt_model.item(r).text()=='All'), None)
                if all_item: all_item.setCheckState(Qt.Unchecked)
            # Update display text
            checked = [bolt_model.item(r).text() for r in range(bolt_model.rowCount()) if bolt_model.item(r).checkState()==Qt.Checked]
            disp = 'All' if 'All' in checked or not checked else ','.join(checked)
            self.bolt_cb.lineEdit().setText(disp)
            # Trigger update
            self._update_plot(); self._update_table()
        bolt_model.itemChanged.connect(on_bolt_item_changed)

        # ── 3. Gráfica ──────────────────────────────────────────────────
        self.plot = pg.PlotWidget(title="Signals")
        # Leyenda en recuadro arriba a la derecha
        # Leyenda arriba derecha, texto negro
        legend = self.plot.addLegend(offset=(-10, 10), labelTextColor='k')
        legend.setBrush(pg.mkBrush(255, 255, 255, 200))
        legend.setPen(pg.mkPen('k'))
        self.plot.showGrid(x=True, y=True, alpha=0.2)
        self.plot.setLabel('bottom', 'Samples')
        self.plot.setLabel('left', 'Amplitude')
        self.curve_dat2 = self.plot.plot([], [], pen=pg.mkPen('#00b8ff', width=1))
        self.curve_dat3 = self.plot.plot([], [], pen=pg.mkPen('#ff5722', width=2))
        # Disable mouse zoom/pan on viewer plot
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
        layout.addWidget(self.plot, 1)

        # ── 4. Tabla de parámetros ─────────────────────────────────────
        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["Parámetro", "Valor"])
        # Mostrar cabeceras en negro y ajustar tamaño
        self.table.horizontalHeader().setStyleSheet('QHeaderView::section { color: black; }')
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        self.table.setHorizontalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
        self.table.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        layout.addWidget(self.table)

        # ── Conexiones ─────────────────────────────────────────────────
        # self.combo_selector.currentIndexChanged.connect(self._on_combo_changed)
        # self.chk_dat2.stateChanged.connect(lambda _: self._update_plot())
        # self.chk_dat3.stateChanged.connect(lambda _: self._update_plot())

    # ---------------------------------------------------------------- helpers
    def _choose_file(self, edit: QtWidgets.QLineEdit):
        """Diálogo estándar de apertura."""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Selecciona CSV', str(Path.cwd()), 'CSV (*.csv)')
        if path: edit.setText(path)

    # ........................................................................
    def _load_csv(self):
        """Lee el CSV y rellena el selector de combinaciones."""
        paths = [self.file1_edit.text().strip()]
        if self.file2_edit.text().strip(): paths.append(self.file2_edit.text().strip())
        self._df_list = []
        self._file_origins = []
        for p in paths:
            try:
                df = pd.read_csv(p)
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, 'CSV', f'Error leyendo {p}: {e}')
                return
            # Normalize numeric columns
            df['Pulse'] = pd.to_numeric(df.get('Pulse'), errors='coerce')
            df['temp'] = pd.to_numeric(df.get('temp'), errors='coerce')
            self._df_list.append(df)
            # Infer origin: prefer explicit column, else filename
            origin = 'unknown'
            try:
                if 'origin' in df.columns and isinstance(df['origin'].iloc[0], str):
                    origin = str(df['origin'].iloc[0]).strip().lower()
                elif 'table' in df.columns and isinstance(df['table'].iloc[0], str):
                    origin = str(df['table'].iloc[0]).strip().lower()
                else:
                    name = Path(p).stem.lower()
                    if 'one4' in name or '1-4' in name or 'one_four' in name or 'onefour' in name:
                        origin = 'one4'
                    elif 'pre' in name:
                        origin = 'pre'
            except Exception:
                origin = 'unknown'
            if origin.startswith('one4') or origin in ('1-4', 'one_four', 'onefour'):
                origin = 'one4'
            elif origin.startswith('pre'):
                origin = 'pre'
            else:
                origin = 'unknown'
            self._file_origins.append(origin)
            logger.debug("Loaded %s, pulses: %s", p, df['Pulse'].unique())

        # Validar columnas en ambos dataframes
        required = {'Freq','Gain','Pulse','Bolt Num','temp'}
        for df in self._df_list:
            if not required <= set(df.columns):
                QtWidgets.QMessageBox.critical(self,'CSV','Faltan columnas requeridas')
                return

        # Pobla dropdowns con valores únicos combinando todos los ficheros
        df_all = pd.concat(self._df_list, ignore_index=True)
        freqs = sorted(df_all['Freq'].dropna().unique()); self.freq_cb.clear(); self.freq_cb.addItems([str(int(f)) for f in freqs]); self.freq_cb.setEnabled(True)
        gains = sorted(df_all['Gain'].dropna().unique()); self.gain_cb.clear(); self.gain_cb.addItems([str(int(g)) for g in gains]); self.gain_cb.setEnabled(True)
        pulses = sorted(df_all['Pulse'].dropna().unique()); self.pulse_cb.clear()
        # Use full string representation to support non-integer pulse values
        self.pulse_cb.addItems([str(p) for p in pulses])
        self.pulse_cb.setEnabled(True)
        logger.debug("Available pulses after load: %s", pulses)
        # Populate bolt multiselect with 'All' and each bolt
        bolts = sorted(df_all['Bolt Num'].dropna().unique())
        model: QStandardItemModel = self.bolt_cb.model(); model.clear()
        # 'All' option
        it_all = QStandardItem("All"); it_all.setFlags(Qt.ItemIsEnabled|Qt.ItemIsUserCheckable); it_all.setData(Qt.Checked, Qt.CheckStateRole); model.appendRow(it_all)
        for b in bolts:
            it = QStandardItem(str(int(b)))
            it.setFlags(Qt.ItemIsEnabled|Qt.ItemIsUserCheckable)
            # Default unchecked, only 'All' selected
            it.setData(Qt.Unchecked, Qt.CheckStateRole)
            model.appendRow(it)
        self.bolt_cb.setEnabled(True)
        # Set initial display to All
        self.bolt_cb.lineEdit().setText("All")
        # Data type options
        self.type_cb.clear(); self.type_cb.addItems(['dat2','dat3']); self.type_cb.setEnabled(True)
        # Plot mode options
        self.align_cb.clear(); self.align_cb.addItems(['Original','Alineado']); self.align_cb.setEnabled(True)
        # Temperature options
        temps = sorted(df_all['temp'].dropna().unique())
        self.temp_cb.clear(); self.temp_cb.addItems([f"{t}" for t in temps]); self.temp_cb.setEnabled(True)
        # Dispara primera actualización
        if freqs:
            for cb in (self.freq_cb, self.gain_cb, self.pulse_cb, self.bolt_cb, self.type_cb, self.align_cb, self.temp_cb):
                cb.setCurrentIndex(0)

    # ........................................................................
    def _filter_df_current(self) -> List[pd.DataFrame]:
        """Devuelve una lista de DataFrames filtrados por la combinación seleccionada para cada fichero."""
        if not self._df_list:
            return []
        # Get selections
        try:
            f = int(float(self.freq_cb.currentText()))
            g = int(float(self.gain_cb.currentText()))
            # Pulse can be non-integer, so keep full precision
            p = float(self.pulse_cb.currentText())
            t = float(self.temp_cb.currentText())
        except Exception:
            # If selection parsing fails, return empty list
            logger.debug("Failed to parse selections")
            return []
        logger.debug(
            "Filtering with F=%s, G=%s, P=%s, T=%s", f, g, p, t
        )
        # Get checked bolts
        model: QStandardItemModel = self.bolt_cb.model(); bolts = []
        for i in range(model.rowCount()):
            it: QStandardItem = model.item(i)
            if it.checkState() == Qt.Checked:
                bolts.append(it.text())
        # If no bolt selected, no data
        if not bolts:
            return []
        # If 'All' selected, include all bolts
        if 'All' in bolts:
            bolt_vals = self._df_list[0]['Bolt Num'].unique()
        else:
            bolt_vals = [int(v) for v in bolts]
        logger.debug("Bolts selected: %s", bolt_vals)
        # Filtrar por cada fichero usando máscara basada en su propio índice
        df_subs: List[pd.DataFrame] = []
        for df in self._df_list:
            try:
                pulse_series = pd.to_numeric(df['Pulse'], errors='coerce')
                temp_series = pd.to_numeric(df['temp'], errors='coerce')
                cmask = (
                    (df['Freq'] == f) &
                    (df['Gain'] == g) &
                    # Compare pulse with tolerance to avoid float issues
                    (np.isclose(pulse_series, p, rtol=0, atol=1e-6)) &
                    (np.isclose(temp_series, t, rtol=0, atol=1e-6)) &
                    (df['Bolt Num'].isin(bolt_vals))
                )
                sub = df[cmask]
                logger.debug(
                    "DF pulses: %s -> filtered rows: %s",
                    pulse_series.unique(),
                    len(sub)
                )
            except Exception as exc:
                logger.debug("Filtering error: %s", exc)
                sub = pd.DataFrame()
            df_subs.append(sub)
        return df_subs

    # ........................................................................
    def _update_plot(self):
        df_subs = self._filter_df_current()
        # Clear previous
        self.plot.clear()
        # Leyenda arriba derecha, texto negro
        legend = self.plot.addLegend(offset=(-10, 10), labelTextColor='k')
        legend.setBrush(pg.mkBrush(255, 255, 255, 200))
        legend.setPen(pg.mkPen('k'))
        self.plot.showGrid(x=True, y=True, alpha=0.2)
        self.plot.setLabel('bottom', 'Samples'); self.plot.setLabel('left', 'Amplitude')
        # Prepare x-axis matching signal length dynamically
        # Signals will be numeric floats
        dtype = self.type_cb.currentText()
        # Paleta extendida con más de 20 colores distinguibles
        colors = [
            '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6',
            '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3',
            '#808000', '#ffd8b1', '#000075', '#808080', '#ffffff', '#000000', '#800080', '#008000',
            '#0000ff', '#ff0000', '#00ff00', '#ffff00', '#00ffff', '#ff00ff'
        ]

        # Alignment mode (support multiple labels)
        align_mode = False
        try:
            _txt = (self.align_cb.currentText() or '').strip().lower()
            align_mode = _txt in ('alineado', 'yes', 'aligned', 'alligned')
        except Exception:
            align_mode = False

        # Reference signal per group (loaded/unloaded) when aligning
        ref_signals: dict[str, np.ndarray | None] = {'loaded': None, 'unloaded': None}

        def _norm_sig(a: np.ndarray) -> np.ndarray:
            a = np.asarray(a, dtype=float)
            if a.size == 0:
                return a
            # Use absolute amplitude to be robust to sign inversion
            a = np.abs(a)
            m = np.nanmean(a)
            s = np.nanstd(a)
            if not np.isfinite(s) or s == 0:
                s = 1.0
            return (a - m) / s

        def _best_shift(ref: np.ndarray, y: np.ndarray, max_shift: int = 200) -> int:
            """Return integer shift to align y to ref using normalized cross-correlation.
            Positive shift moves y to the right (delays it).
            """
            x0 = _norm_sig(ref)
            y0 = _norm_sig(y)
            try:
                corr = np.correlate(x0, y0, mode='full')
            except Exception:
                return 0
            lags = np.arange(-len(y0) + 1, len(x0))
            if max_shift is not None and max_shift > 0:
                mask = (lags >= -max_shift) & (lags <= max_shift)
                corr = corr[mask]
                lags = lags[mask]
            if corr.size == 0:
                return 0
            k = int(lags[int(np.nanargmax(corr))])
            return k
        # Plot each file's subset; add file index in legend
        for file_idx, df_sub in enumerate(df_subs):
            # choose line style for file: solid for file1, dashed for file2
            line_style = Qt.SolidLine if file_idx == 0 else Qt.DashLine
            for idx, (_, row) in enumerate(df_sub.iterrows()):
                # Select data prefix
                prefix = f"{dtype}_"
                cols = sorted([c for c in row.index if c.startswith(prefix)], key=lambda s: int(s.split('_')[1]))
                # Coerce to float array for plotting
                try:
                    vals = row[cols].to_numpy(dtype=float)
                except TypeError:
                    vals = np.array(row[cols].tolist(), dtype=float)
                # Skip if no data
                if vals.size == 0:
                    continue
                # Optional normalization to 1e6 for visibility
                normalize = True
                try:
                    normalize = (self.scale_cb.currentText().startswith('Normalized'))
                except Exception:
                    normalize = True
                if normalize:
                    vmax = np.nanmax(np.abs(vals))
                    if vmax and vmax != 0:
                        vals = vals / vmax * 1e6
                pen = pg.mkPen(colors[idx % len(colors)], width=1)
                pen.setStyle(line_style)
                # Legend label: only show the bolt number
                label = f"Bolt {int(row['Bolt Num'])}"
                # Plot with x length matching vals; optionally align by load group
                if align_mode:
                    # Decide group based on file origin rather than 'force'
                    origin = (self._file_origins[file_idx] if 0 <= file_idx < len(self._file_origins) else 'unknown')
                    group = 'loaded' if origin == 'one4' else 'unloaded'
                    ref = ref_signals.get(group)
                    if ref is None:
                        ref_signals[group] = vals.copy()
                        shift = 0
                    else:
                        max_shift = max(10, min(int(len(vals) * 0.25), 500))
                        shift = _best_shift(ref, vals, max_shift=max_shift)
                    x = np.arange(vals.size) + int(shift)
                else:
                    x = np.arange(vals.size)
                self.plot.plot(x, vals, pen=pen, name=label)

    # ........................................................................
    def _update_table(self):
        df_subs = self._filter_df_current()
        # Asegurar que df_subs es lista de DataFrames
        if not isinstance(df_subs, list):
            df_subs = [df_subs]
        # Si no hay datos o todos vacíos, limpiar tabla
        if not df_subs or all(getattr(sub, 'empty', True) for sub in df_subs):
            self.table.setRowCount(0)
            return
        # Generar lista de bolts seleccionados por fichero
        bolts_per_file = [sorted(sub['Bolt Num'].astype(int).unique()) for sub in df_subs]
        # Construir cabeceras: Parámetro, luego FicheroX-BoltY
        cols = ['Parámetro']
        for i, bolts in enumerate(bolts_per_file):
            for b in bolts:
                cols.append(f'Fichero {i+1}-Bolt {b}')
        self.table.setColumnCount(len(cols))
        self.table.setHorizontalHeaderLabels(cols)
        # Parámetros a mostrar: unión de columnas distintas de dat2_/dat3_ y metadatos
        skip_pref = ("dat2_", "dat3_")
        skip_cols = {"Freq", "Gain", "Pulse", "Bolt Num"}
        params_set = set()
        for df_sub in df_subs:
            params_set.update([col for col in df_sub.columns if not col.startswith(skip_pref) and col not in skip_cols])
        params = sorted(params_set)
        self.table.setRowCount(len(params))
        # Rellenar tabla: por cada parámetro y cada combinación fichero-bolt
        for i, param in enumerate(params):
            self.table.setItem(i, 0, QtWidgets.QTableWidgetItem(param))
            col_idx = 1
            for sub_idx, df_sub in enumerate(df_subs):
                for b in bolts_per_file[sub_idx]:
                    # filtrar fila de este bolt
                    if df_sub.empty:
                        text = ''
                    else:
                        sel = df_sub[df_sub['Bolt Num'].astype(int) == b]
                        if sel.empty:
                            text = ''
                        else:
                            val = sel.iloc[0].get(param, '')
                            text = f"{val:.4g}" if isinstance(val, float) else str(val)
                    item = QtWidgets.QTableWidgetItem(text)
                    item.setFlags(Qt.ItemIsEnabled)
                    self.table.setItem(i, col_idx, item)
                    col_idx += 1
        self.table.resizeRowsToContents()
