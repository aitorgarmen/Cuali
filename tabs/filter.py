# filter_tab.py â€" Valid Combos + Oneâ€'click Scan on Valid Combos
# ------------------------------------------------------------------
# 1)  from filter_tab import FilterTab
# 2)  tabs.addTab(FilterTab(com_selector), "Valid Combos")
#
# Requisitos: PyQt5, pandas, openpyxl, numpy, pyserial (heredado), pyqtgraph
# ------------------------------------------------------------------

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    # Import Device type for annotations only, avoid circular import at runtime
    from device import Device

import os
import threading
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional

import pandas as pd
import numpy as np

from PyQt5 import QtCore
from PyQt5.QtCore import Qt, pyqtSignal, QObject
from PyQt5.QtGui import QStandardItemModel, QStandardItem, QBrush, QColor
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QFileDialog, QProgressBar,
    QTableView, QMessageBox, QHBoxLayout, QLabel, QCheckBox,
    QDoubleSpinBox, QSpinBox, QComboBox, QLineEdit, QAbstractScrollArea
)

# â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
#  Dependencias de la pestaÃ±a de escaneo                              
# â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
#  Reutilizamos Device y _pack32 ya implementados en device.py
#  para enviar los comandos al equipo.
# -------------------------------------------------------------------


# â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
#  Worker de filtrado (igual que antes)                               
# â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
class FilterWorker(QObject):
    """Procesa el Excel/CSV â‡' df_filtered + df_debug en un hilo."""

    finished     = pyqtSignal(pd.DataFrame, pd.DataFrame, str)  # df_filtered, df_debug, out_path
    progress_str = pyqtSignal(str)                              # Mensajes para la barra
    progress_val = pyqtSignal(int)                              # 0â€'100

    def __init__(self, path_excel: str, pct_threshold: float = 20.0,
                 use_peak: bool = False, peak_threshold: float = 0.0,
                 parent: QObject | None = None):
        super().__init__(parent)
        self.path_excel   = path_excel
        self.pct_threshold= pct_threshold
        self.use_peak     = use_peak
        self.peak_threshold = peak_threshold

    # â€•â€• LÃ³gica principal â€•â€•â€•â€•â€•â€•â€•â€•â€•â€•â€•â€•â€•â€•â€•â€•â€•â€•â€•â€•â€•â€•â€•â€•â€•â€•â€•â€•â€•â€•â€•â€•â€•â€•â€•â€•â€•â€•â€•
    def run(self):
        try:
            self.progress_val.emit(10)
            ext = Path(self.path_excel).suffix.lower()
            if ext == '.csv':
                self.progress_str.emit("Leyendo CSV...")
                df = pd.read_csv(self.path_excel)
            else:
                self.progress_str.emit("Leyendo Excel...")
                df = pd.read_excel(self.path_excel)

            # Columnas requeridas
            req = {'Freq', 'Gain', 'Pulse', 'Bolt Num', 'pct_diff'}
            if not req <= set(df.columns):
                raise ValueError(f"Columnas faltantes: {', '.join(req - set(df.columns))}")

            # Normalizar pct_diff
            self.progress_val.emit(30)
            self.progress_str.emit("Normalizando porcentajes...")
            df['pct_diff_num'] = self._pct_to_float(df['pct_diff'])

            # 1) Filtrar filas por porcentaje dinÃ¡mico
            df_pct = df[df['pct_diff_num'] >= self.pct_threshold].copy()
            if df_pct.empty:
                raise ValueError(f"Ninguna fila supera el {self.pct_threshold} %.")
            # 1b) Opcional: filtrar por amplitud
            if self.use_peak:
                self.progress_str.emit("Filtrando por amplitud...")
                df_core = df_pct[df_pct['pico1'] >= self.peak_threshold].copy()
                if df_core.empty:
                    raise ValueError(f"Ninguna fila con amplitud >= {self.peak_threshold}.")
            else:
                df_core = df_pct

            # 2) Comprobar que TODAS las Bolt Num cumplen el pct mÃ­nimo
            self.progress_val.emit(55)
            self.progress_str.emit("Calculando combinaciones vÃ¡lidas...")

            total_bolts = (
                df.groupby(['Freq', 'Gain', 'Pulse'])['Bolt Num']
                  .nunique().rename('total_bolts')
            )
            valid_bolts = (
                df_core.groupby(['Freq', 'Gain', 'Pulse'])['Bolt Num']
                       .nunique().rename('valid_bolts')
            )
            debug_df = pd.concat([total_bolts, valid_bolts], axis=1).reset_index()
            debug_df['valid_bolts'] = debug_df['valid_bolts'].fillna(0).astype(int)

            combos_ok = debug_df.loc[
                debug_df['total_bolts'] == debug_df['valid_bolts'],
                ['Freq', 'Gain', 'Pulse']
            ].set_index(['Freq', 'Gain', 'Pulse']).index

            if combos_ok.empty:
                # No valid combos: emit finished with empty filtered and debug only
                self.progress_val.emit(100)
                self.progress_str.emit("No hay combinaciones vÃ¡lidas, mostrando invÃ¡lidos.")
                df_final = pd.DataFrame()
                out_path = ""
                self.finished.emit(df_final, debug_df, out_path)
                return

            # Filas finales
            df_final = (
                df_core.set_index(['Freq', 'Gain', 'Pulse'])
                       .loc[combos_ok]
                       .reset_index()
            )

            # Formateo
            if 'force' in df_final.columns:
                df_final['force'] = df_final['force'].round(1)
            df_final['pct_diff'] = df_final['pct_diff_num'] / 100  # fracciÃ³n
            df_final.drop(columns=['pct_diff_num'], inplace=True)

            # Exportar
            self.progress_val.emit(80)
            self.progress_str.emit("Exportando Excel...")
            out_path = self._export_excel(df_final, debug_df)

            self.progress_val.emit(100)
            self.progress_str.emit("Â¡Hecho!")
            self.finished.emit(df_final, debug_df, out_path)

        except Exception as e:
            QMessageBox.critical(None, "Error", str(e))
            self.progress_val.emit(0)
            self.progress_str.emit("")

    # â€•â€• Utilidades internas â€•â€•â€•â€•â€•â€•â€•â€•â€•â€•â€•â€•â€•â€•â€•â€•â€•â€•â€•â€•â€•â€•â€•â€•â€•â€•â€•â€•â€•â€•â€•â€•â€•â€•â€•
    @staticmethod
    def _pct_to_float(col: pd.Series) -> pd.Series:
        if pd.api.types.is_numeric_dtype(col):
            pct = col.astype(float).copy()
            if pct.max(skipna=True) <= 1:
                pct *= 100
            return pct
        pct = (col.astype(str)
                 .str.replace('%', '', regex=False)
                 .str.replace(',', '.', regex=False)
                 .astype(float))
        return pct

    def _export_excel(self, df_final: pd.DataFrame, debug_df: pd.DataFrame) -> str:
        base, ext = os.path.splitext(self.path_excel)
        ext_low = ext.lower()
        out_path = f"{base}_filtered{ext_low}"
        if ext_low == '.csv':
            df_final.to_csv(out_path, index=False)
            return out_path
        # else Excel
        with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
            df_final.to_excel(writer, sheet_name='filtered', index=False)
            # porcentaje 1 decimal
            wb = writer.book; ws = writer.sheets['filtered']
            pct_col = df_final.columns.get_loc('pct_diff') + 1
            for cell in ws.iter_cols(min_col=pct_col, max_col=pct_col,
                                    min_row=2, max_row=ws.max_row):
                for c in cell:
                    c.number_format = '0.0%'
            if 'force' in df_final.columns:
                f_col = df_final.columns.get_loc('force') + 1
                for cell in ws.iter_cols(min_col=f_col, max_col=f_col,
                                        min_row=2, max_row=ws.max_row):
                    for c in cell:
                        c.number_format = '0.0'
        return out_path


# â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
#  Worker para escanear SOLO las combinaciones vÃ¡lidas                
# â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
class ValidScanWorker(QtCore.QThread):
    """Bucle de escaneado sÃ³lo sobre las (Freq, Gain, Pulse) vÃ¡lidas."""

    progress = pyqtSignal(int)       # 0â€'100
    finished = pyqtSignal(list)      # rows (list[list])
    error    = pyqtSignal(str)

    def __init__(self, device: 'Device', combos: List[Tuple[float, float, float]],
                 bolt_num: float, config_params: Dict[str, Any], parent: QObject | None = None):
        super().__init__(parent)
        self._device   = device
        self._combos   = combos
        self._bolt     = bolt_num
        self._config   = config_params  # device configuration from UI
        self._rows: list = []
        self._stop     = False

    # ..................................................................
    def stop(self):
        self._stop = True

    # ------------------------------------------------------------------
    #  RUN â€" execuciÃ³n en hilo
    # ------------------------------------------------------------------
    def run(self):
        """Escanear cada combo enviando la configuraciÃ³n completa antes de medir."""
        try:
            total = len(self._combos)
            for idx, (freq, gain, pulse) in enumerate(self._combos, start=1):
                if self._stop:
                    break
                # Prepara parÃ¡metros combinando config base con valores variables
                p = {
                    'temp': self._config['temp'],
                    'diftemp': self._config['diftemp'],
                    'long_corr': self._config['long_corr'],
                    'tof': self._config['tof'],
                    'freq': np.float32(freq),
                    'pulse': np.float32(pulse),
                    'gain': np.float32(gain),
                    'algo': self._config.get('algo', 0),
                    'threshold': self._config.get('threshold', 0)
                }
                # EnvÃ­a toda la config y adquiere frame
                self._send_full_config(p)
                frame = self._acquire_frame()
                if not frame:
                    continue
                # Registra fila con datos base y arrays completos
                row = [
                    freq, gain, pulse, self._bolt,
                    frame['pico1'], frame['porcentaje_diferencia'],
                    frame['tof'], p['temp'], frame['force'],
                    frame['maxcorrx'], frame['maxcorry']
                ]
                row.extend(frame['dat3'].tolist())
                row.extend(frame['dat2'].tolist())
                self._rows.append(row)
                # Emite progreso
                self.progress.emit(int(idx/total*100))
                self.msleep(10)
            self.finished.emit(self._rows)
        except Exception as e:
            self.error.emit(str(e))
        finally:
            # Asegura cerrar puerto al terminar
            if hasattr(self._device, 'ser') and self._device.ser.is_open:
                self._device.ser.close()

    # ---------------- Helpers internos --------------------------------
    def _send_full_config(self, p: Dict[str, Any]):
        """EnvÃ­a configuraciÃ³n completa (constantes + variables + algoritmo) antes de medir"""
        d = self._device
        from device import _pack32
        try:
            d.modo_standby(); d.modo_configure()
            # parÃ¡metros constantes
            d.enviar(_pack32(p['temp']),    "10")  # temp
            d.enviar(_pack32(p['diftemp']), "11")  # diftemp
            d.enviar(_pack32(p['tof']),     "12")  # tof
            # parÃ¡metros variables
            d.enviar(_pack32(p['freq']),    "14")  # freq
            d.enviar(_pack32(p['pulse']),   "15")  # pulse1
            d.enviar(_pack32(p['pulse']),   "16")  # pulse2
            d.enviar(_pack32(p['gain']),    "17")  # gain
            # constantes de correcciÃ³n
            d.enviar(_pack32(p.get('xi', np.float32(0.0))), "18")  # xi
            d.enviar(_pack32(p.get('alpha', np.float32(0.0))), "19")  # alpha
            d.enviar(_pack32(np.uint32(p.get('short_corr', 900))),  "1A")  # short_corr
            d.enviar(_pack32(p['long_corr']),  "1B")  # long_corr
            d.enviar(_pack32(np.uint32(p.get('short_temp', 990))),  "1C")  # short_temp
            d.enviar(_pack32(np.uint32(p.get('long_temp', 990))),  "1D")  # long_temp
            # algoritmo y umbral
            d.enviar(_pack32(np.uint32(p.get('algo', 0))),      "2C")
            d.enviar(_pack32(np.uint32(p.get('threshold', 0))), "2D")
            # guarda y entra en modo single
            d.modo_save(); d.modo_standby(); d.modo_single()
        except Exception as e:
            raise RuntimeError(f"Error enviando config en batch: {e}")

    def _acquire_frame(self):
        d = self._device
        try:
            # Enviar temperatura previo a la mediciÃ³n, para paridad con DJTab y FrequencyScanTab
            d.enviar_temp()
            d.start_measure()
            return d.lectura()
        except Exception as e:
            self.error.emit(f"Lectura fallÃ³: {e}")
            return {}


# â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
#  Widget principal                                                   
# â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
class FilterTab(QWidget):
    """Tab 'Valid Combos' con filtrado + escaneo en lote sobre combos vÃ¡lidos."""

    STYLE_SHEET = """
        QPushButton   { min-height: 28px; font-size: 14px; }
        QPushButton#Open       { background-color: #ff8f00; color: #ffffff; border-radius: 6px; }
        QPushButton#Start      { background-color: #2962ff; color: #ffffff; border-radius: 6px; }
        QPushButton#StartScan  { background-color: #00c853; color: #ffffff; border-radius: 6px; }
        QProgressBar  { height: 18px; }
        QTableView    { gridline-color: #DDD; font-size: 13px; }
    """

    def __init__(self, com_selector: QComboBox | None = None,
                 parent: QWidget | None = None,
                 db: Optional[Any] = None, batch_id: Optional[str] = None):
        super().__init__(parent)
        self.setStyleSheet(self.STYLE_SHEET)
        self.selected_path: str | None = None
        self.com_selector = com_selector
        self.db = db
        self.batch_id = batch_id
        # Campo para mostrar ruta de fichero seleccionado
        self.file_edit = QLineEdit()
        self.file_edit.setReadOnly(True)
        self.valid_combos_df: pd.DataFrame | None = None
        self._setup_ui()

        # Mantiene modelo vivo
        self._table_model: QStandardItemModel | None = None

    # ------------------------------------------------------------------
    #  setup UI
    # ------------------------------------------------------------------
    def _setup_ui(self):
        # â€•â€• Layout principal â€•â€•
        lay = QVBoxLayout(self)
        # â€•â€• Botones superiores â€•â€•
        self.btn_open   = QPushButton("Seleccionar Fichero...", objectName="Open")
        # Layout con campo de ruta y botÃ³n
        file_row = QHBoxLayout()
        file_row.addWidget(self.file_edit)
        file_row.addWidget(self.btn_open)
        lay.addLayout(file_row)
        self.btn_start  = QPushButton("Start Filter",     objectName="Start")
        self.btn_start.setEnabled(False)

        # â€•â€• ConfiguraciÃ³n de filtro â€•â€•
        self.pct_label = QLabel("Min pct diff (%)")
        self.pct_spin  = QDoubleSpinBox(); self.pct_spin.setRange(0.0, 100.0)
        self.pct_spin.setDecimals(1); self.pct_spin.setSingleStep(1.0)
        self.pct_spin.setValue(20.0)
        self.peak_chk  = QCheckBox("Filtrar por amplitud")
        self.peak_spin = QDoubleSpinBox(); self.peak_spin.setRange(0.0, 1e9)
        self.peak_spin.setDecimals(1); self.peak_spin.setSingleStep(1.0)
        self.peak_spin.setValue(0.0); self.peak_spin.hide()

        # Layout pct + peak threshold
        pct_lay = QHBoxLayout(); pct_lay.addWidget(self.pct_label); pct_lay.addWidget(self.pct_spin)
        peak_lay= QHBoxLayout(); peak_lay.addWidget(QLabel("Threshold")); peak_lay.addWidget(self.peak_spin)
        peak_widget = QWidget(); peak_widget.setLayout(peak_lay); peak_widget.hide()
        # â€•â€• ConfiguraciÃ³n de dispositivo para scan â€•â€•
        dev_cfg_lay = QHBoxLayout()
        # Temperature
        dev_cfg_lay.addWidget(QLabel("Temperature"))
        self.temp_spin = QDoubleSpinBox(); self.temp_spin.setRange(-20.0, 50.0)
        self.temp_spin.setDecimals(1); self.temp_spin.setSingleStep(0.1)
        self.temp_spin.setValue(20.0)
        dev_cfg_lay.addWidget(self.temp_spin)
        # ToF
        dev_cfg_lay.addWidget(QLabel("ToF"))
        self.tof_spin = QDoubleSpinBox(); self.tof_spin.setRange(0.0, 1e6)
        self.tof_spin.setDecimals(1); self.tof_spin.setSingleStep(1.0)
        self.tof_spin.setValue(244500.0)
        dev_cfg_lay.addWidget(self.tof_spin)
        # Algorithm
        dev_cfg_lay.addWidget(QLabel("Algorithm"))
        self.algo_selector = QComboBox(); self.algo_selector.addItems([
            "Absolute maximum", "First maximum", "Second maximum"])
        dev_cfg_lay.addWidget(self.algo_selector)
        # Threshold
        self.threshold_label = QLabel("Threshold")
        dev_cfg_lay.addWidget(self.threshold_label)
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.0, 1e6)
        self.threshold_spin.setDecimals(0); self.threshold_spin.setSingleStep(1.0)
        self.threshold_spin.setValue(0.0)
        dev_cfg_lay.addWidget(self.threshold_spin)
        # hide threshold widgets for default algorithm (Absolute maximum)
        self.threshold_label.hide(); self.threshold_spin.hide()

        # â€•â€• ToF mode (Manual or Calculate)
        dev_cfg_lay.addWidget(QLabel("ToF mode"))
        self.tof_mode = QComboBox()
        self.tof_mode.addItems(["Manual", "Calculate"])
        dev_cfg_lay.addWidget(self.tof_mode)
        # Bolt length and velocity for Calculate mode
        self.bolt_length_label = QLabel("Bolt Length (mm)")
        self.bolt_length = QDoubleSpinBox()
        self.bolt_length.setRange(0.0, 1000.0); self.bolt_length.setDecimals(1); self.bolt_length.setSingleStep(1.0)
        self.bolt_length.setValue(720.0)
        self.velocity_label = QLabel("Velocity (m/s)")
        self.velocity = QDoubleSpinBox()
        self.velocity.setRange(1000.0, 10000.0); self.velocity.setDecimals(1); self.velocity.setSingleStep(10.0)
        self.velocity.setValue(5900.0)
        self.bolt_length_label.hide(); self.bolt_length.hide()
        self.velocity_label.hide(); self.velocity.hide()
        dev_cfg_lay.addWidget(self.bolt_length_label)
        dev_cfg_lay.addWidget(self.bolt_length)
        dev_cfg_lay.addWidget(self.velocity_label)
        dev_cfg_lay.addWidget(self.velocity)

        # connect ToF mode toggling visibility and calculation
        self.tof_mode.currentIndexChanged.connect(self._update_tof_mode)
        # recalc ToF when bolt length or velocity change
        self.bolt_length.valueChanged.connect(lambda _: self._recalc_tof())
        self.velocity.valueChanged.connect(lambda _: self._recalc_tof())
        # initialize ToF mode UI state
        self._update_tof_mode(self.tof_mode.currentIndex())

        # Show/hide only threshold when selecting algorithm
        self.algo_selector.currentIndexChanged.connect(lambda idx: (
            self.threshold_label.setVisible(idx in (1,2)),
            self.threshold_spin.setVisible(idx in (1,2))
        ))

        # â€•â€• Barra de progreso â€•â€•
        self.progress = QProgressBar(); self.progress.setAlignment(Qt.AlignCenter); self.progress.setValue(0)

        # â€•â€• Tabla de combinaciones vÃ¡lidas â€•â€•
        self.table = QTableView(); self.table.setEditTriggers(QTableView.NoEditTriggers)
        self.table.horizontalHeader().setVisible(True); self.table.verticalHeader().setVisible(True)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setStyleSheet("QHeaderView::section{background-color:#CCC;color:#000;font-weight:bold;}")
        # Make table scroll only horizontally; vertical scrolling handled by parent tab
        try:
            self.table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            self.table.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            self.table.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)
        except Exception:
            pass

        # â€•â€• NUEVOS controles: Bolt Num + Start Scan â€•â€•
        bolt_row = QHBoxLayout()
        bolt_row.addWidget(QLabel("Bolt Num"))
        self.bolt_spin = QSpinBox(); self.bolt_spin.setRange(0, 1_000_000); self.bolt_spin.setValue(0)
        self.bolt_spin.setEnabled(False)
        bolt_row.addWidget(self.bolt_spin)
        bolt_row.addStretch()

        self.btn_scan = QPushButton("Start Scan", objectName="StartScan")
        self.btn_scan.setEnabled(False)
        bolt_row.addWidget(self.btn_scan)

        lay.addWidget(self.btn_open)
        lay.addWidget(self.btn_start)
        lay.addLayout(pct_lay)
        lay.addWidget(self.peak_chk)
        lay.addWidget(peak_widget)
        lay.addWidget(self.progress)
        lay.addWidget(self.table)
        lay.addLayout(bolt_row)
        # Add device config (ToF/Algorithm/Threshold) below table
        lay.addLayout(dev_cfg_lay)

        # â€•â€• Conexiones â€•â€•
        self.btn_open.clicked.connect(self._choose_file)
        self.btn_start.clicked.connect(self._start_filter)
        self.btn_scan.clicked.connect(self._start_scan)
        self.peak_chk.toggled.connect(lambda chk: (peak_widget.setVisible(chk), self.peak_spin.setVisible(chk)))

    def _update_tof_mode(self, idx: int):
        """Update ToF field and related inputs based on mode selection."""
        manual = (idx == 0)
        calc = (idx == 1)
        # always show ToF field; enable only in manual mode
        self.tof_spin.setVisible(True)
        self.tof_spin.setEnabled(manual)
        # show/hide bolt length and velocity inputs
        self.bolt_length_label.setVisible(calc)
        self.bolt_length.setVisible(calc)
        self.velocity_label.setVisible(calc)
        self.velocity.setVisible(calc)
        if calc:
            self._recalc_tof()

    def _recalc_tof(self):
        """Calculate ToF from bolt length (mm) and velocity (m/s) and update ToF spin box."""
        # ToF in microseconds: bolt_length(mm)*2000/velocity(m/s)
        length_mm = float(self.bolt_length.value())
        vel_m_s   = float(self.velocity.value())
        if self.velocity.value() != 0:
            tof_us    = (2*length_mm) / vel_m_s * 1e6
        else:
            tof_us = 0.0
        self.tof_spin.setValue(tof_us)

    # ------------------------------------------------------------------
    #  MÃ©todos pÃºblicos auxiliares
    # ------------------------------------------------------------------
    def set_com_selector(self, combo: QComboBox):
        """Permite inyectar el selector de puertos tras crear la pestaÃ±a."""
        self.com_selector = combo

    # ------------------------------------------------------------------
    #  ACCIONES: 1) abrir archivo  2) filtrar  3) escanear vÃ¡lidas
    # ------------------------------------------------------------------
    def _choose_file(self):
        start_dir = Path(self.selected_path).parent if self.selected_path else Path(__file__).parent
        path, _ = QFileDialog.getOpenFileName(
            self, "Selecciona Excel/CSV", str(start_dir), "Archivos (*.xlsx *.xls *.csv)")
        if path:
            self.selected_path = path
            self.file_edit.setText(path)
            self.btn_start.setEnabled(True)

    # .................................................................
    # 1) Start filter
    def _start_filter(self):
        if not self.selected_path:
            QMessageBox.warning(self, "Sin fichero", "Selecciona primero un fichero.")
            return
        self.btn_open.setEnabled(False); self.btn_start.setEnabled(False); self.progress.setValue(0)

        pct = float(self.pct_spin.value())
        use_peak = self.peak_chk.isChecked(); peak_thr = float(self.peak_spin.value())
        self.worker = FilterWorker(self.selected_path, pct_threshold=pct,
                                   use_peak=use_peak, peak_threshold=peak_thr)
        thread = threading.Thread(target=self.worker.run, daemon=True)
        self.worker.progress_str.connect(self.progress.setFormat)
        self.worker.progress_val.connect(self.progress.setValue)
        self.worker.finished.connect(self._on_finished)
        thread.start()

    # .................................................................
    # Callback tras filtrado OK
    def _on_finished(self, df_filtered: pd.DataFrame, df_debug: pd.DataFrame, out_path: str):
        # Store filter settings
        pct_thr = float(self.pct_spin.value())
        use_peak = self.peak_chk.isChecked()
        peak_thr = float(self.peak_spin.value())
        self.current_pct_thr = pct_thr
        self.current_peak_used = use_peak
        self.current_peak_thr = peak_thr
        # Load full data and build interleaved columns with fraction metrics
        import pandas as __pd, numpy as __np; from pathlib import Path
        ext = Path(self.selected_path).suffix.lower()
        df_full = __pd.read_csv(self.selected_path) if ext == '.csv' else __pd.read_excel(self.selected_path)
        # Normalize pct_diff and fraction
        df_full['pct_diff_num'] = FilterWorker._pct_to_float(df_full['pct_diff'])
        df_full['pct_diff'] = df_full['pct_diff_num'] / 100
        # Pivot full data
        pivot_pct = df_full.pivot(index=['Freq','Gain','Pulse'], columns='Bolt Num', values='pct_diff')
        bolts = list(pivot_pct.columns)
        # Compute fraction of bolts meeting pct threshold (0-1)
        frac_pct = (pivot_pct >= pct_thr/100).sum(axis=1) / len(bolts)
        if use_peak:
            pivot_peak = df_full.pivot(index=['Freq','Gain','Pulse'], columns='Bolt Num', values='pico1')
            # Compute fraction of bolts meeting peak threshold (0-1)
            frac_peak = (pivot_peak >= peak_thr).sum(axis=1) / len(bolts)
        # Build ordered DataFrame
        df_res = __pd.DataFrame(index=pivot_pct.index)
        df_res['pct_ok_frac'] = frac_pct
        if use_peak:
            df_res['pico1_ok_frac'] = frac_peak
        for b in bolts:
            df_res[f'pct_{b}'] = pivot_pct[b]
            if use_peak:
                df_res[f'pico1_{b}'] = pivot_peak[b]
        # Merge with all combos to include invalid
        table_df = df_res.reset_index()
        all_combos = df_debug[['Freq','Gain','Pulse']].drop_duplicates()
        table_df = all_combos.merge(table_df, on=['Freq','Gain','Pulse'], how='left')
        # Sort combos: perfect ones first (100% pct and, if used, 100% peak), then others by descending fractions
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
        # Save combos for scan
        self.valid_combos_df = table_df[['Freq','Gain','Pulse']].drop_duplicates().reset_index(drop=True)
        # Load with coloring
        self._load_dataframe_into_table(table_df)
        # Mostrar mensaje de exportaciÃ³n solo si se generÃ³ fichero
        if out_path:
            QMessageBox.information(self, "Export", f"Archivo creado:\n{out_path}\n\nFilas exportadas: {len(df_filtered)}")
        # Permitir re-filtrar con mismo fichero
        self.btn_open.setEnabled(True)
        self.btn_start.setEnabled(True)
         
         # Habilita controles de escaneo
        ok = not self.valid_combos_df.empty
        self.bolt_spin.setEnabled(ok); self.btn_scan.setEnabled(ok)

    # .................................................................
    # 2) Start scan over valid combos
    def _start_scan(self):
        if self.valid_combos_df is None or self.valid_combos_df.empty:
            QMessageBox.warning(self, "Sin combos", "Primero ejecuta el filtrado de combinaciones vÃ¡lidas.")
            return
        if self.com_selector is None or not self.com_selector.currentText():
            QMessageBox.warning(self, "COM", "Selecciona un puerto COM vÃ¡lido desde la barra superior.")
            return
        # SelecciÃ³n de fichero CSV
        csv_path, _ = QFileDialog.getSaveFileName(self, "Guardar CSV de escaneo", "valid_scan.csv", "CSV (*.csv)")
        if not csv_path:
            return

        # Conecta dispositivo
        try:
            # import Device locally to avoid circular import
            from device import Device
            self.device = Device(self.com_selector.currentText(), baudrate=115200, timeout=1)
        except Exception as e:
            QMessageBox.critical(self, "UART", f"No se pudo abrir el puerto:\n{e}")
            return

        # Desactiva botones
        self.btn_scan.setEnabled(False); self.progress.setValue(0); self.progress.setFormat("Escaneando... %p%")

        combos = [tuple(r) for r in self.valid_combos_df.values.tolist()]
        bolt_num = float(self.bolt_spin.value())
        # ConfiguraciÃ³n del dispositivo tomada de la UI
        import numpy as _np
        diftemp, long_corr, xi, alpha = -103.0, 10, 0.0, 0.0
        if self.db and self.batch_id:
            try:
                diftemp, long_corr, xi, alpha = self.db.get_device_params(self.batch_id)
            except Exception:
                pass
        # Resolve window params from DB; fallback to current constants used in worker
        scw = stw = ltw = None
        if self.db and self.batch_id:
            try:
                attrs = (self.db.get_batch(self.batch_id) or {}).get('attrs', {})
                def _to_int(v):
                    try:
                        return int(float(v))
                    except Exception:
                        return None
                scw = _to_int(attrs.get('short_correlation_window'))
                stw = _to_int(attrs.get('short_temporal_window'))
                ltw = _to_int(attrs.get('long_temporal_window'))
            except Exception:
                scw = stw = ltw = None
        config_params = {
            'temp': _np.float32(self.temp_spin.value()),
            'diftemp': _np.float32(diftemp),
            'long_corr': _np.uint32(long_corr),
            'tof': _np.float32(self.tof_spin.value()),
            'xi': _np.float32(xi),
            'alpha': _np.float32(alpha),
            'short_corr': _np.uint32(scw if scw is not None else 900),
            'short_temp': _np.uint32(stw if stw is not None else 990),
            'long_temp': _np.uint32(ltw if ltw is not None else 990),
            'algo': _np.uint32(self.algo_selector.currentIndex()),
            'threshold': _np.uint32(self.threshold_spin.value())
        }
        # Iniciar worker con configuraciÃ³n completa
        self.scan_worker = ValidScanWorker(self.device, combos, bolt_num, config_params)
        self.scan_worker.progress.connect(self.progress.setValue)
        self.scan_worker.finished.connect(lambda rows: self._on_scan_finished(rows, csv_path))
        self.scan_worker.error.connect(lambda msg: QMessageBox.critical(self, "Scan", msg))
        self.scan_worker.start()

    # .................................................................
    def _on_scan_finished(self, rows: list, csv_path: str):
        self.btn_scan.setEnabled(True)
        self.progress.setValue(0); self.progress.setFormat("")

        if not rows:
            QMessageBox.information(self, "Sin datos", "No se recibieron frames vÃ¡lidos.")
            return
        # DataFrame con todos los campos (incluyendo dat3/dat2)
        # Creamos DataFrame original
        df = pd.DataFrame(rows)
        # Columnas: Freq, Gain, Pulse, Bolt Num, pico1, pct_diff (pct%), tof, force, maxcorrx, maxcorry
        base_cols = ['Freq', 'Gain', 'Pulse', 'Bolt Num',
                     'pico1', 'pct_diff', 'tof', 'force',
                     'maxcorrx', 'maxcorry']
        # Asignamos nombres a columnas base
        n_total = len(rows[0])
        n_vals = len(base_cols)
        n_array = (n_total - n_vals) // 2
        dat3_cols = [f'dat3_{i}' for i in range(n_array)]
        dat2_cols = [f'dat2_{i}' for i in range(n_array)]
        df.columns = base_cols + dat3_cols + dat2_cols
        # Convertir pct_diff a fracciÃ³n
        df['pct_diff'] = df['pct_diff'] / 100.0
        # Convertir a entero columnas de parÃ¡metros
        for col in ('Freq', 'Gain', 'Pulse', 'Bolt Num'):
            df[col] = df[col].astype(int)
        # Reordenar columnas en orden exacto y solo columnas requeridas
        df = df[base_cols + dat3_cols + dat2_cols]
        # Guardar CSV
        try:
            # If file exists, append without header; otherwise write new file with header
            if os.path.exists(csv_path):
                df.to_csv(csv_path, mode='a', header=False, index=False, float_format='%.6f')
            else:
                df.to_csv(csv_path, index=False, float_format='%.6f')
            QMessageBox.information(self, "OK", f"Datos guardados en:\n{csv_path}")
        except Exception as e:
            QMessageBox.critical(self, "CSV", f"Error guardando CSV:\n{e}")

    # ------------------------------------------------------------------
    #  Utilidad: cargar DF en QTableView
    # ------------------------------------------------------------------
    def _load_dataframe_into_table(self, df: pd.DataFrame):
        # Create table model with coloring based on thresholds
        model = QStandardItemModel(df.shape[0], df.shape[1], self)
        headers = df.columns.astype(str).tolist()
        model.setHorizontalHeaderLabels(headers)
        for r in range(df.shape[0]):
             for c, col in enumerate(headers):
                 val = df.iloc[r, c]
                # Format text: percent for pct and pico1_ok_frac
                 if col == 'pico1_ok_frac':
                    text = f"{val:.0%}"
                 elif col.startswith('pct_'):
                    text = f"{val:.1%}"
                 else:
                    text = str(val)
                 item = QStandardItem(text)
                # Apply background color: only green if 100% for pct_ok_frac and pico1_ok_frac
                 if col == 'pct_ok_frac' and val == 1.0:
                    item.setBackground(QBrush(QColor('green')))
                 elif col == 'pico1_ok_frac':
                    # Solo verde si 100%, sin color en otro caso
                    if val == 1.0:
                        item.setBackground(QBrush(QColor('green')))
                 elif col.startswith('pct_') and col != 'pct_ok_frac':
                    # Otros pct: rojo si por debajo del umbral, verde si dentro
                    if val*100 < self.current_pct_thr:
                        item.setBackground(QBrush(QColor('red')))
                    else:
                        item.setBackground(QBrush(QColor('green')))
                # Peak columns per bolt (excluye ok_frac)
                 if getattr(self, 'current_peak_used', False) and col.startswith('pico1_') and col != 'pico1_ok_frac':
                    if val < self.current_peak_thr:
                        item.setBackground(QBrush(QColor('red')))
                    else:
                        item.setBackground(QBrush(QColor('green')))
                 model.setItem(r, c, item)
        self.table.setModel(model)
        self._table_model = model
        # Adjust table height so parent scroll area manages vertical scrolling
        self._adjust_table_height()

    def _adjust_table_height(self):
        try:
            # Resize rows to fit contents (safe even if already default)
            try:
                self.table.resizeRowsToContents()
            except Exception:
                pass
            vh = self.table.verticalHeader()
            hh = self.table.horizontalHeader()
            frame = 2 * self.table.frameWidth()
            total_h = int(vh.length() + hh.height() + frame)
            # Guard against zero-height models
            if total_h <= 0:
                total_h = int(hh.height() + 24)
            # Fix the table height to content so outer tab scrolls
            self.table.setMinimumHeight(total_h)
            self.table.setMaximumHeight(total_h)
        except Exception:
            # Fail silently; default behavior still works
            pass
