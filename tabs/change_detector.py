# change_detector_tab.py - Signal Change Detector tab for the qualification app
# -----------------------------------------------------------------------------
# This file converts the standalone Tk / CLI signal‑change detector into a PyQt5
# tab that plugs into the qualification GUI (cualificacion.py).  It follows the
# same dark theme, widgets & layout philosophy used by FrequencyScanTab and
# DJTab so the look & feel stays consistent.
# -----------------------------------------------------------------------------
# Extra requirements (already present in the main app):
#     pip install numpy pandas scipy matplotlib pyqt5
# -----------------------------------------------------------------------------
from __future__ import annotations

import sys
# User-configurable parameters for peak detection algorithm
d_n_peaks    = 5       # número de picos a detectar
d_prominence = 0.75    # umbral relativo para detectar picos (0-1)
d_amp_tol    = 0.01     # tolerancia de amplitud (fracción) para comparar picos

from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
from scipy import signal as _signal

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget, QLabel, QLineEdit, QPushButton, QHBoxLayout, QVBoxLayout,
    QFileDialog, QProgressBar, QTableWidget, QTableWidgetItem, QHeaderView,
    QSplitter, QMessageBox, QFrame, QComboBox
)

# Matplotlib for embedded canvas ----------------------------------------------
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Reuse utilities and style from cualificacion.py -----------------------------
try:
    # When running as part of the main application
    from cualificacion import STYLE_SHEET, labeled_spinbox, thin_separator
except ImportError:
    # Minimal fallback for running this tab as a standalone script
    STYLE_SHEET = """*{font-family:'Segoe UI'} QWidget{background:#202124;color:#e6e6e6} QLineEdit,QPushButton{background:#2b2b2f;border:1px solid #444;padding:4px 6px;border-radius:4px;} QPushButton{padding:8px 16px;border:none;border-radius:6px;font-weight:600;}"""
    def labeled_spinbox(*a, **k):
        sb = QtWidgets.QDoubleSpinBox(); sb.setRange(0, 1); return sb
    def thin_separator():
        ln = QFrame(); ln.setFrameShape(QFrame.HLine); ln.setFrameShadow(QFrame.Sunken); ln.setStyleSheet("color:#444;"); return ln

# ---------------------------------------------------------------------------
# Core algorithm (identical to signal_change_detector.py but without Tk) ----
DAT3_COLS = [f"dat3_{i}" for i in range(1024)]
DAT2_COLS = [f"dat2_{i}" for i in range(1024)]
DEFAULT_KEY = "Bolt Num"


# --- Core utility functions for signal comparison ---

# Align two signals using FFT-based cross-correlation and pad to avoid wrap-around
def _align_signal(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Aligns signal y to signal x using cross-correlation to find the best shift.
    The aligned signal is padded with zeros instead of rolling to avoid wrapping.
    Args:
        x: Reference signal.
        y: Signal to align.
    Returns:
        A tuple containing the aligned signal y and the calculated shift.
    """
    n = len(x)
    # Use FFT for efficient cross-correlation
    corr = np.fft.ifft(np.fft.fft(y, 2 * n) * np.conj(np.fft.fft(x, 2 * n)))
    corr = np.real(np.fft.fftshift(corr))
    # Find the shift that maximizes correlation
    shift = int(np.argmax(corr) - n)

    # Pad with zeros instead of rolling to avoid wrapping the signal
    y_aligned = np.zeros_like(y)
    if shift > 0:
        # Shift y left, pad with zeros on the right
        y_aligned[:n - shift] = y[shift:]
    elif shift < 0:
        # Shift y right, pad with zeros on the left
        s = -shift
        y_aligned[s:] = y[:n - s]
    else:
        # No shift needed
        y_aligned = y.copy()

    return y_aligned, shift


# Compute correlation coefficient and RMSE over active regions above threshold
def _metrics_windowed(x: np.ndarray, y: np.ndarray,
                      thr: float = 0.05) -> tuple[float, float]:
    """
    Calcula r y RMSE sólo en la región activa (|x| o |y| mayores que thr·max).
    Si la ventana resulta vacía, usa toda la señal.
    """
    # Máscara de muestras "vivas"
    mask = (np.abs(x) > thr * np.max(np.abs(x))) | \
           (np.abs(y) > thr * np.max(np.abs(y)))

    if not mask.any():          # señal demasiado pequeña
        mask = slice(None)      # usa todo

    x_sel = x[mask]
    y_sel = y[mask]

    r = float(np.corrcoef(x_sel, y_sel)[0, 1]) if np.std(x_sel) and np.std(y_sel) else 0.0
    rmse = float(np.sqrt(np.mean((x_sel - y_sel) ** 2)))

    return r, rmse


# Detect the most prominent peaks in a signal, up to max_peaks
def _get_significant_peaks(
        sig: np.ndarray,
        max_peaks: int = 5,
        prominence: float = 0.75) -> np.ndarray:
    """
    Devuelve las posiciones (índices) de los `max_peaks` picos más prominentes.

    prominence → fracción del máximo (0-1).  Ej.: 0.75 = 75 % de la altura máx.
    """
    prom = prominence * np.max(np.abs(sig))
    peaks, props = _signal.find_peaks(sig, prominence=prom)
    if not len(peaks):
        return np.array([], dtype=int)

    # ordenamos por prominencia y nos quedamos con los max_peaks primeros
    order = np.argsort(props["prominences"])[::-1][:max_peaks]
    return peaks[order]


# Check if all original peaks are preserved in the aligned signal within tolerance
def _peaks_preserved(
        x: np.ndarray,
        y_aligned: np.ndarray,
        idx_peaks_x: np.ndarray,
        pos_tol: int = 5,
        amp_tol: float = 0.01) -> bool:
    """
    Comprueba que cada pico de `x` aparece en `y_aligned`.

    pos_tol:  ±muestras de margen para la posición del pico.
    amp_tol:  tolerancia relativa de amplitud (1 % por defecto).

    Devuelve True si *todos* los picos se preservan.
    """
    if not len(idx_peaks_x):
        return True           # no hay picos significativos, nada que comprobar

    # buscamos picos en y_aligned con el mismo criterio de prominencia
    idx_peaks_y = _get_significant_peaks(y_aligned,
                                         max_peaks=len(idx_peaks_x),
                                         prominence=d_prominence)

    for idx in idx_peaks_x:
        # ventana de búsqueda alrededor de idx
        candidates = idx_peaks_y[np.abs(idx_peaks_y - idx) <= pos_tol]
        if not len(candidates):
            return False      # no hay pico en la vecindad, se perdió
        # analizamos el que más se parezca en altura
        amps_y = np.abs(y_aligned[candidates])
        idx_best = candidates[np.argmax(amps_y)]

        amp_ratio = np.abs(y_aligned[idx_best]) / (np.abs(x[idx]) + 1e-9)
        if not (1 - amp_tol) <= amp_ratio <= (1 + amp_tol):
            return False      # altura cambió demasiado

    return True


# Normalize signal to unit maximum magnitude
def _norm(sig: np.ndarray) -> np.ndarray:
    m = np.max(np.abs(sig))
    return sig / m if m else sig

# Analyze a pair of signals using the peak-based algorithm and return metrics
def analyze_pair_peak_based(
        x: np.ndarray, y: np.ndarray,
        n_peaks: int = 5, prom: float = 0.75,
        pos_tol: int = 5, amp_tol: float = 0.01) -> dict[str, Any]:

    # 1) Normalizar a |1|
    x_n = _norm(x)
    y_n = _norm(y)

    # 2) Alinear
    y_aligned, shift = _align_signal(x_n, y_n)

    # 3) Extraer los primeros n_peaks picos de la señal original
    idx_peaks_x = _get_significant_peaks(
                      x_n,
                      max_peaks=n_peaks,
                      prominence=prom)

    # 4) Comprobar preservación de picos
    peaks_ok = _peaks_preserved(
                   x_n, y_aligned, idx_peaks_x,
                   pos_tol=pos_tol,
                   amp_tol=amp_tol)

    # 5) Métricas
    r, rmse = _metrics_windowed(x_n, y_aligned)

    return dict(pearson_r=r,
                rmse=rmse,
                shift=shift,
                n_peaks=len(idx_peaks_x),
                peaks_preserved=peaks_ok,
                changed=not peaks_ok)


def _get_significant_peaks(sig: np.ndarray,
                           max_peaks: int = 5,
                           prominence: float = 0.75) -> np.ndarray:
    prom = prominence * np.max(np.abs(sig))
    peaks, _ = _signal.find_peaks(sig, prominence=prom)
    # devolver los primeros picos encontrados en orden temporal
    return peaks[:max_peaks]




# ---------------------------------------------------------------------------
# Worker thread to process large CSV files without blocking the GUI ---------
class AnalyzeWorker(QtCore.QThread):
    """
    This worker runs the signal analysis in a separate thread to keep the UI responsive.
    It emits signals for progress updates, completion, and errors.
    """
    progress = QtCore.pyqtSignal(int)                  # Progress signal (0-100%)
    finished = QtCore.pyqtSignal(pd.DataFrame)         # Finished signal with results DataFrame
    error = QtCore.pyqtSignal(str)                     # Error signal with message

    def __init__(self, orig_path: Path, proc_path: Path | None, key: str,
                 data_type: str,
                 n_peaks: int = d_n_peaks,
                 prom: float = d_prominence,
                 amp_tol: float = d_amp_tol):
        super().__init__()
        self._orig_path = orig_path
        self._proc_path = proc_path
        self._key = key
        self._cols = DAT3_COLS if data_type == 'dat3' else DAT2_COLS
        self._data_type = data_type
        self._n_peaks = n_peaks
        self._prom    = prom
        self._amp_tol = amp_tol

    # ................................................................. run
    def run(self):
        """Main processing logic of the worker thread."""
        try:
            # Load CSV files
            orig_df = pd.read_csv(self._orig_path)
            if self._proc_path and self._proc_path != self._orig_path:
                proc_df = pd.read_csv(self._proc_path)
            else:
                proc_df = orig_df.copy()

            # Determine key column or fallback to index
            if self._key not in orig_df.columns or self._key not in proc_df.columns:
                if self._key == DEFAULT_KEY:
                    orig_df['__row_idx__'] = orig_df.index
                    proc_df['__row_idx__'] = proc_df.index
                    key = '__row_idx__'
                else:
                    raise ValueError(f"The key '{self._key}' does not exist in both CSV files")
            else:
                key = self._key

            # Ensure Freq, Gain, Pulse exist
            for col in ('Freq', 'Gain', 'Pulse'):
                if col not in orig_df.columns:
                    orig_df[col] = ''
                if col not in proc_df.columns:
                    proc_df[col] = ''

            results: List[Dict[str, Any]] = []
            # Analyze both data types separately
            for data_type, cols in (('dat3', DAT3_COLS), ('dat2', DAT2_COLS)):
                # Prepare data slices
                orig_sel = orig_df[[key, 'Freq', 'Gain', 'Pulse'] + cols]
                proc_sel = proc_df[[key, 'Freq', 'Gain', 'Pulse'] + cols]
                # Rename for merge
                orig_sel = orig_sel.rename(columns={c: f"orig_{c}" for c in cols})
                proc_sel = proc_sel.rename(columns={c: f"proc_{c}" for c in cols})
                # Merge on full combination
                merged = orig_sel.merge(proc_sel,
                                       on=[key, 'Freq', 'Gain', 'Pulse'],
                                       how='inner')
                # Iterate matches
                total = len(merged) or 1
                for idx, row in merged.iterrows():
                    x = row[[f"orig_{c}" for c in cols]].to_numpy(float)
                    y = row[[f"proc_{c}" for c in cols]].to_numpy(float)
                    res = analyze_pair_peak_based(
                        x, y,
                        n_peaks=self._n_peaks,
                        prom=self._prom,
                        amp_tol=self._amp_tol)

                    # Metadata
                    res['Freq'] = row['Freq']; res['Gain'] = row['Gain']
                    res['Pulse'] = row['Pulse']; res[key] = row[key]
                    res['data_type'] = data_type
                    results.append(res)
                    if idx % 10 == 0:
                        self.progress.emit(int(idx / total * 100))

            # Emit final progress and results
            self.progress.emit(100)
            self.finished.emit(pd.DataFrame(results))
        except Exception as e:
            self.error.emit(str(e))

# ---------------------------------------------------------------------------
# Main tab widget -----------------------------------------------------------
class ChangeDetectorTab(QWidget):
    """The main widget for the 'Signal Change Detector' tab."""

    def __init__(self):
        super().__init__()
        # --- Data storage ---
        # DataFrames to hold the loaded CSV data and analysis results
        self.orig_df: pd.DataFrame | None = None
        self.proc_df: pd.DataFrame | None = None
        self.results_df: pd.DataFrame | None = None
        # --- Worker thread ---
        self._worker: AnalyzeWorker | None = None
        # --- Build the UI ---
        self._build_ui()

    # ................................................................. UI
    def _build_ui(self):
        """Constructs all the UI elements for the tab."""
        # ---- LEFT panel: controls ------------------------------------
        left = QVBoxLayout(); left.setSpacing(12)

        # --- File selectors ---
        # Line edits and buttons for selecting original and processed CSV files
        self.orig_edit = QLineEdit(); self.orig_btn = QPushButton("Original CSV", objectName="File")
        self.proc_edit = QLineEdit(); self.proc_btn = QPushButton("Processed CSV", objectName="File")
        def _file(lbl_edit):
            # File dialog function
            path, _ = QFileDialog.getOpenFileName(self, "Select CSV File", "", "CSV Files (*.csv)")
            if path:
                lbl_edit.setText(path)
        self.orig_btn.clicked.connect(lambda: _file(self.orig_edit))
        self.proc_btn.clicked.connect(lambda: _file(self.proc_edit))

        # Layout for file selectors
        fbox1 = QHBoxLayout(); fbox1.addWidget(self.orig_edit); fbox1.addWidget(self.orig_btn)
        fbox2 = QHBoxLayout(); fbox2.addWidget(self.proc_edit); fbox2.addWidget(self.proc_btn)
        left.addLayout(fbox1); left.addLayout(fbox2)

        # --- Key selector ---
        # A field to specify the column name used to match rows between the two CSVs
        hkey = QHBoxLayout(); hkey.addWidget(QLabel("Key:"))
        self.key_edit = QLineEdit(DEFAULT_KEY); self.key_edit.setFixedWidth(120); hkey.addWidget(self.key_edit); hkey.addStretch()
        left.addLayout(hkey)
        left.addWidget(thin_separator())

        # --- Parámetros de detección de picos ---
        hparams = QHBoxLayout()

        # umbral (0-1) → prominencia relativa
        self.prom_sb = QtWidgets.QDoubleSpinBox()
        self.prom_sb.setRange(0.0, 1.0)
        self.prom_sb.setSingleStep(0.01)
        self.prom_sb.setValue(d_prominence)

        # nº de picos a detectar
        self.npeaks_sb = QtWidgets.QSpinBox()
        self.npeaks_sb.setRange(1, 50)
        self.npeaks_sb.setValue(d_n_peaks)

        # tolerancia de amplitud (0-1)
        self.amp_tol_sb = QtWidgets.QDoubleSpinBox()
        self.amp_tol_sb.setRange(0.0, 1.0)
        self.amp_tol_sb.setSingleStep(0.01)
        self.amp_tol_sb.setValue(d_amp_tol)

        hparams.addWidget(QLabel("Umbral:"));   hparams.addWidget(self.prom_sb)
        hparams.addWidget(QLabel("Nº picos:")); hparams.addWidget(self.npeaks_sb)
        hparams.addWidget(QLabel("Margen:"));   hparams.addWidget(self.amp_tol_sb)
        left.addLayout(hparams)


        # --- Analyze button and progress bar ---
        left.addWidget(QLabel("Data:"))
        # Data type selector
        self.data_cb = QComboBox(); self.data_cb.addItems(["dat3","dat2"])
        left.addWidget(self.data_cb)
        self.analyze_btn = QPushButton("ANALYZE", objectName="Play")
        self.progress = QProgressBar(); self.progress.setRange(0, 100); self.progress.setValue(0)
        left.addWidget(self.analyze_btn)
        left.addWidget(self.progress)

        # --- Parameter selectors for plotting (including Data Type) ---
        # Dropdown menus to select a specific signal combination to visualize
        self.freq_cb = QtWidgets.QComboBox(); self.gain_cb = QtWidgets.QComboBox()
        self.pulse_cb = QtWidgets.QComboBox(); self.bolt_cb = QtWidgets.QComboBox()
        self.type_cb = QComboBox()
        # Layout for parameter selectors
        params_layout = QHBoxLayout()
        params_layout.addWidget(QLabel("Freq:")); params_layout.addWidget(self.freq_cb)
        params_layout.addWidget(QLabel("Gain:")); params_layout.addWidget(self.gain_cb)
        params_layout.addWidget(QLabel("Pulse:")); params_layout.addWidget(self.pulse_cb)
        params_layout.addWidget(QLabel("Bolt Num:")); params_layout.addWidget(self.bolt_cb)
        params_layout.addWidget(QLabel("Data Type:")); params_layout.addWidget(self.type_cb)
        left.addLayout(params_layout)
        # --- Plot button ---
        self.plot_btn = QPushButton("Plot", objectName="Play")
        left.addWidget(self.plot_btn)
        left.addStretch(1) # Pushes all controls to the top

        left_panel = QWidget(); left_panel.setLayout(left)

        # ---- RIGHT: table + plot -------------------------------------
        right_layout = QVBoxLayout(); right_layout.setContentsMargins(0, 0, 0, 0)

        # --- Results table ---
        # Results table with Data Type column
        self.table = QTableWidget(0, 9)
        self.table.setHorizontalHeaderLabels([
            "Freq", "Gain", "Pulse", "Bolt Num", "Data Type",
            "r", "RMSE", "Shift", "Changed"
        ])
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch) # Columns fill width
        # Make header labels bold for readability
        header_font = header.font()
        header_font.setBold(True)
        header.setFont(header_font)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows) # Select whole rows
        right_layout.addWidget(self.table, 2) # Give table 2/5 of vertical space

        # --- Plotting canvas ---
        self.fig = Figure(figsize=(4, 3), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Superimposed Signals"); self.ax.set_xlabel("Sample"); self.ax.set_ylabel("Amplitude")
        self.canvas = FigureCanvas(self.fig)
        right_layout.addWidget(self.canvas, 3) # Give plot 3/5 of vertical space

        right_panel = QWidget(); right_panel.setLayout(right_layout)

        # --- Main Splitter ---
        # Divides the left control panel and the right results panel
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(1, 1) # Ensure right panel resizes
        layout = QHBoxLayout(self); layout.addWidget(splitter)

        # --- Connections ---
        # Connect button clicks and table selections to their handler methods
        self.analyze_btn.clicked.connect(self._on_analyze)
        self.table.itemSelectionChanged.connect(self._on_select_row)
        self.plot_btn.clicked.connect(self._plot_from_selectors)

    # ................................................................. logic
    def _on_analyze(self):
        """Handles the 'ANALYZE' button click. Starts the analysis worker."""
        # Prevent starting a new analysis if one is already running
        if self._worker and self._worker.isRunning():
            QMessageBox.warning(self, "In Progress", "An analysis is already running.")
            return

        # --- Validate inputs ---
        orig_path = Path(self.orig_edit.text()) if self.orig_edit.text() else None
        if not orig_path or not orig_path.exists():
            QMessageBox.warning(self, "Invalid File", "Please select a valid original CSV file.")
            return
        proc_path = Path(self.proc_edit.text()) if self.proc_edit.text() else None
        key = self.key_edit.text().strip() or DEFAULT_KEY

        # --- Load data and validate key ---
        try:
            # Load raw data for plotting and validation
            self.orig_df = pd.read_csv(orig_path)
            if proc_path and proc_path.exists() and proc_path != orig_path:
                self.proc_df = pd.read_csv(proc_path)
            else:
                self.proc_df = self.orig_df.copy()

            # *** KEY VALIDATION ***
            # Check if the key column exists in the dataframe before proceeding
            if key not in self.orig_df.columns:
                QMessageBox.critical(self, "Key Error", f"The key column '{key}' was not found in the original CSV file.")
                self.orig_df = None # Clear dataframe
                return
            if self.proc_df is not None and key not in self.proc_df.columns:
                QMessageBox.critical(self, "Key Error", f"The key column '{key}' was not found in the processed CSV file.")
                self.proc_df = None # Clear dataframe
                return

        except Exception as e:
            QMessageBox.critical(self, "Error Loading CSV", f"Failed to load or parse CSV file.\r\nError: {e}")
            return

        # --- Start worker ---
        self.progress.setValue(0)
        self.analyze_btn.setEnabled(False) # Disable button during analysis
        data_type = self.data_cb.currentText()
        self._cols = DAT3_COLS if data_type == 'dat3' else DAT2_COLS
        n_peaks = self.npeaks_sb.value()
        prom    = self.prom_sb.value()
        amp_tol = self.amp_tol_sb.value()

        self._worker = AnalyzeWorker(
                orig_path, proc_path, key,
                data_type,
                n_peaks=n_peaks,
                prom=prom,
                amp_tol=amp_tol)

        self._worker.progress.connect(self.progress.setValue)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_finished(self, df: pd.DataFrame):
        """Handles the 'finished' signal from the worker."""
        self.analyze_btn.setEnabled(True) # Re-enable analyze button
        self.results_df = df
        self._populate_table()
        # Populate parameter selectors for plotting now that results are available
        self._populate_selectors()
        self.plot_btn.setEnabled(True)
        QMessageBox.information(self, "Analysis Complete", f"Finished: {len(df)} pairs were analyzed.")

    def _on_error(self, msg: str):
        """Handles the 'error' signal from the worker."""
        self.analyze_btn.setEnabled(True)
        QMessageBox.critical(self, "Analysis Error", msg)

    def _populate_selectors(self):
        """Populates the dropdowns with unique values from the results."""
        if self.results_df is None:
            return
        # Clear previous items
        self.freq_cb.clear(); self.gain_cb.clear(); self.pulse_cb.clear(); self.bolt_cb.clear(); self.type_cb.clear()

        # Get unique sorted values for each parameter
        freqs = sorted(self.results_df['Freq'].unique())
        gains = sorted(self.results_df['Gain'].unique())
        pulses = sorted(self.results_df['Pulse'].unique())
        key = self.key_edit.text().strip() or DEFAULT_KEY
        if key not in self.results_df.columns and '__row_idx__' in self.results_df.columns:
            key = '__row_idx__'
        bolts = sorted(self.results_df[key].unique())
        types = sorted(self.results_df['data_type'].unique())

        # Add items to the comboboxes
        self.freq_cb.addItems([str(f) for f in freqs])
        self.gain_cb.addItems([str(g) for g in gains])
        self.pulse_cb.addItems([str(p) for p in pulses])
        self.bolt_cb.addItems([str(b) for b in bolts])
        # Populate data type selector
        self.type_cb.addItems(types)

    def _plot_from_selectors(self):
        """Plots the signal pair selected via the dropdowns."""
        if self.orig_df is None or self.proc_df is None:
            QMessageBox.warning(self, "No Data", "Please load and analyze data first.")
            return

        # --- Get selected values ---
        try:
            freq_val = float(self.freq_cb.currentText())
            gain_val = float(self.gain_cb.currentText())
            pulse_val = float(self.pulse_cb.currentText())
            bolt_text = self.bolt_cb.currentText()
            data_type_text = self.type_cb.currentText()
            key = self.key_edit.text().strip() or DEFAULT_KEY
            # The key value could be an integer or a string
            try:
                bolt_val: Any = int(bolt_text)
            except ValueError:
                bolt_val = bolt_text
        except ValueError:
            QMessageBox.warning(self, "Invalid Selection", "Could not find a match for the selected parameters.")
            return

        # --- Find matching signals ---
        # Find the row in the original dataframe that matches the selected parameters
        orig_match = self.orig_df[
            (self.orig_df['Freq'] == freq_val) &
            (self.orig_df['Gain'] == gain_val) &
            (self.orig_df['Pulse'] == pulse_val) &
            (self.orig_df[key] == bolt_val)
        ]
        # Find the corresponding row in the processed dataframe
        proc_match = self.proc_df[
            (self.proc_df['Freq'] == freq_val) &
            (self.proc_df['Gain'] == gain_val) &
            (self.proc_df['Pulse'] == pulse_val) &
            (self.proc_df[key] == bolt_val)
        ]

        # Determine which data columns to plot
        cols = DAT3_COLS if data_type_text == 'dat3' else DAT2_COLS
        # If original missing
        if orig_match.empty:
            # If original is missing, create a dummy signal of zeros
            x = np.zeros(len(cols))
        else:
            x = orig_match.iloc[0][cols].to_numpy(float)

        # If processed missing
        if proc_match.empty:
            y = np.zeros(len(cols))
        else:
            y = proc_match.iloc[0][cols].to_numpy(float)

        if orig_match.empty and proc_match.empty:
            QMessageBox.warning(self, "Not Found", "Could not find a matching signal pair for the selected parameters.")
            return

        # --- Plotting ---
        # Pass plot_type for title consistency
        params = {'Freq': freq_val, 'Gain': gain_val, 'Pulse': pulse_val, key: bolt_val}
        self._plot_signals(x, y, params)

    # ---------------------------------------------------------------- table
    def _populate_table(self):
        """Fills the results table with data from the results_df DataFrame."""
        # Clear existing rows
        self.table.setRowCount(0)
        if self.results_df is None or self.results_df.empty:
            return
        # Determine key column fallback
        key = self.key_edit.text().strip() or DEFAULT_KEY
        if key not in self.results_df.columns and '__row_idx__' in self.results_df.columns:
            key = '__row_idx__'
        # Remove duplicate combinations (Freq, Gain, Pulse, key, data_type)
        df = self.results_df.drop_duplicates(subset=['Freq', 'Gain', 'Pulse', key, 'data_type'])
        # Populate rows
        for i, row in df.iterrows():
            self.table.insertRow(i)
            # Prepare cell values including Data Type column
            vals = [
                row['Freq'], row['Gain'], row['Pulse'], row[key], row['data_type'],
                f"{row['pearson_r']:.3f}", f"{row['rmse']:.4f}", str(row['shift']),
                "Yes" if row['changed'] else "No"
            ]
            for j, v in enumerate(vals):
                item = QTableWidgetItem(str(v))
                item.setTextAlignment(Qt.AlignCenter)
                # Highlight the 'Changed' cell (last column) in green if changed
                if j == len(vals) - 1 and row['changed']:
                    item.setBackground(QtCore.Qt.darkGreen)
                self.table.setItem(i, j, item)

    # ---------------------------------------------------------------- plot
    def _on_select_row(self):
        """Plots the signals corresponding to the currently selected row in the table."""
        if not self.table.selectedItems() or self.results_df is None or self.orig_df is None or self.proc_df is None:
            return
        # Determine key column and selected data type
        key = self.key_edit.text().strip() or DEFAULT_KEY
        # Extract metadata from table row
        row_idx = self.table.currentRow()
        data_type_text = self.table.item(row_idx, 4).text()
        self.type_cb.setCurrentText(data_type_text)
        # Sync Freq, Gain, and Pulse selectors to match the selected row
        self.freq_cb.setCurrentText(self.table.item(row_idx, 0).text())
        self.gain_cb.setCurrentText(self.table.item(row_idx, 1).text())
        self.pulse_cb.setCurrentText(self.table.item(row_idx, 2).text())
        bolt_text = self.table.item(row_idx, 3).text()
        self.bolt_cb.setCurrentText(bolt_text)
        # Auto-plot when a row is selected
        self._plot_from_selectors()
         # The key value could be an integer or a string
        try:
            bolt_val: Any = int(bolt_text)
        except ValueError:
            bolt_val = bolt_text

        # Determine which data columns to plot based on row's Data Type
        cols = DAT3_COLS if data_type_text == 'dat3' else DAT2_COLS

        # Filter original and processed DataFrames to find the matching rows
        orig_match = self.orig_df[
            (self.orig_df['Freq'] == float(self.table.item(row_idx, 0).text())) &
            (self.orig_df['Gain'] == float(self.table.item(row_idx, 1).text())) &
            (self.orig_df['Pulse'] == float(self.table.item(rowIdx, 2).text())) &
            (self.orig_df[key].astype(str) == str(bolt_val))
        ]
        proc_match = self.proc_df[
            (self.proc_df['Freq'] == float(self.table.item(row_idx, 0).text())) &
            (self.proc_df['Gain'] == float(self.table.item(row_idx, 1).text())) &
            (self.proc_df['Pulse'] == float(self.table.item(row_idx, 2).text())) &
            (self.proc_df[key].astype(str) == str(bolt_val))
        ]

        # Extract signals using selected data columns or zeros if missing
        x = orig_match.iloc[0][cols].to_numpy(float) if not orig_match.empty else np.zeros(len(cols))
        y = proc_match.iloc[0][cols].to_numpy(float) if not proc_match.empty else np.zeros(len(cols))

        if orig_match.empty and proc_match.empty:
            # Obtener valores seleccionados directamente de la tabla
            row = self.table.currentRow()
            f0 = self.table.item(row, 0).text()
            g0 = self.table.item(row, 1).text()
            p0 = self.table.item(row, 2).text()
            QMessageBox.warning(
                self, "No Data Found",
                f"Could not find signal data for the combination: Freq={f0}, Gain={g0}, Pulse={p0}, {key}={bolt_val}"
            )
            return

        # --- Update plot ---
        params = {'Freq': float(self.table.item(row_idx, 0).text()), 'Gain': float(self.table.item(row_idx, 1).text()), 'Pulse': float(self.table.item(row_idx, 2).text()), key: bolt_val}
        self._plot_signals(x, y, params)

    def _plot_signals(self, x: np.ndarray, y: np.ndarray, params: Dict[str, Any]):
        """
        Helper function to plot the original and processed signals.
        Args:
            x: The original signal.
            y: The processed signal.
            params: A dictionary of parameters for the plot title.
        """
        y_aligned, shift = _align_signal(x, y)
        r, rmse = _metrics_windowed(x, y_aligned)
        # Normalize signals to max absolute = 1 for consistent plotting
        def _norm(sig):
            m = np.max(np.abs(sig))
            return sig / m if m != 0 else sig
        x_plot = _norm(x)
        y_plot = _norm(y_aligned)

        # --- Update plot ---
        self.ax.clear()
        self.ax.plot(x_plot, label="Original")
        self.ax.plot(y_plot, label=f"Processed (aligned, shift={shift})")

        # Build title from parameters
        key = self.key_edit.text().strip() or DEFAULT_KEY
        title = f"Freq={params['Freq']} Gain={params['Gain']} Pulse={params['Pulse']} {key}: {params[key]}"
        title += f"\n(r={r:.3f}, RMSE={rmse:.4f})" # Add metrics to title
        self.ax.set_title(title)

        self.ax.legend()
        self.ax.set_xlabel("Samples"); self.ax.set_ylabel("Amplitude")
        self.canvas.draw()

    def _populate_selectors(self):
        """Fills the dropdown selectors with unique parameter values from the results."""
        if self.results_df is None or self.results_df.empty:
            return
        # Get unique, sorted values for each parameter
        freqs = sorted(self.results_df['Freq'].unique())
        gains = sorted(self.results_df['Gain'].unique())
        pulses = sorted(self.results_df['Pulse'].unique())
        key = self.key_edit.text().strip() or DEFAULT_KEY
        # Use fallback key if necessary
        if key not in self.results_df.columns and '__row_idx__' in self.results_df.columns:
            key = '__row_idx__'
        bolts = sorted(self.results_df[key].unique())
        data_types = sorted(self.results_df['data_type'].unique())
        # Populate comboboxes, converting all values to strings
        self.freq_cb.clear(); self.freq_cb.addItems([str(v) for v in freqs])
        self.gain_cb.clear(); self.gain_cb.addItems([str(v) for v in gains])
        self.pulse_cb.clear(); self.pulse_cb.addItems([str(v) for v in pulses])
        self.bolt_cb.clear(); self.bolt_cb.addItems([str(v) for v in bolts])
        self.type_cb.clear(); self.type_cb.addItems([str(v) for v in data_types])

    # Duplicate _plot_from_selectors removed; use updated version above

# ---------------------------------------------------------------------------
# Standalone execution for debugging ----------------------------------------
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(STYLE_SHEET)
    w = ChangeDetectorTab(); w.resize(1200, 700); w.show()
    sys.exit(app.exec_())
