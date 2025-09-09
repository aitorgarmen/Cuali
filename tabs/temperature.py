# temperature_tab.py - Pestaña para insertar columna "temp" en CSV
# -----------------------------------------------------------------------------
# Este módulo define la clase TemperatureTab, que permite cargar un fichero
# CSV con la cabecera original (... tof, force ...) e insertar un valor de
# temperatura único (con 1 decimal) entre las columnas "tof" y "force".
# El estilo y los componentes siguen el esquema usado en el resto de
# pestañas de la aplicación principal.
# -----------------------------------------------------------------------------

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QDoubleSpinBox,
    QVBoxLayout,
    QWidget,
)


class TemperatureTab(QWidget):
    """Pestaña GUI para añadir la columna *temp* a un CSV existente."""

    def __init__(self) -> None:
        super().__init__()
        self._build_ui()

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(18)

        # ── Selector de fichero CSV ────────────────────────────────────
        file_row = QHBoxLayout()
        file_row.addWidget(QLabel("CSV:"))
        self.file_edit = QLineEdit()
        self.file_edit.setPlaceholderText("Selecciona un archivo .csv...")
        self.file_btn = QPushButton("File", objectName="File")
        self.file_btn.clicked.connect(self._choose_file)
        file_row.addWidget(self.file_edit, 1)
        file_row.addWidget(self.file_btn)
        root.addLayout(file_row)

        # ── SpinBox de temperatura ────────────────────────────────────
        temp_row = QHBoxLayout()
        temp_row.addWidget(QLabel("Temperature (°C):"))
        self.temp_spin = QDoubleSpinBox()
        self.temp_spin.setDecimals(1)
        self.temp_spin.setSingleStep(0.1)
        self.temp_spin.setRange(-100.0, 200.0)
        self.temp_spin.setAlignment(Qt.AlignCenter)
        self.temp_spin.setFixedWidth(120)
        self.temp_spin.setValue(20.0)
        temp_row.addWidget(self.temp_spin)
        temp_row.addStretch()
        root.addLayout(temp_row)

        # ── Botón de ejecución ────────────────────────────────────────
        run_row = QHBoxLayout()
        run_row.addStretch()
        self.run_btn = QPushButton("Insert & Save", objectName="Save")
        self.run_btn.setFixedWidth(160)
        self.run_btn.clicked.connect(self._process_file)
        run_row.addWidget(self.run_btn)
        root.addLayout(run_row)

        root.addStretch()

    # ---------------------------------------------------------------- choose
    def _choose_file(self) -> None:
        # Iniciar diálogo en carpeta del último archivo seleccionado o en home
        last = self.file_edit.text().strip()
        start_dir = Path(last).parent if last else Path.home()
        path, _ = QFileDialog.getOpenFileName(
            self, "Open CSV", str(start_dir), "CSV files (*.csv)"
        )
        if path:
            self.file_edit.setText(path)

    # ---------------------------------------------------------------- process
    def _process_file(self) -> None:
        """Lee el CSV, inserta la columna *temp* y escribe un nuevo archivo."""
        path = self.file_edit.text().strip()
        if not path:
            QMessageBox.warning(self, "CSV", "Selecciona primero un archivo CSV valido.")
            return

        try:
            df = pd.read_csv(path)
        except Exception as e:
            QMessageBox.critical(self, "CSV", f"No se pudo leer el archivo:\n{e}")
            return

        # Validación mínima de cabecera
        for col in ("tof", "force"):
            if col not in df.columns:
                QMessageBox.critical(
                    self,
                    "Cabecera",
                    f"La columna requerida '{col}' no aparece en el CSV.",
                )
                return

        # Eliminar columna previa si existiera para sobrescribir
        if "temp" in df.columns:
            df = df.drop(columns=["temp"])

        # Inserta nueva columna *temp* justo antes de 'force'
        temp_value = round(float(self.temp_spin.value()), 1)
        insert_idx = df.columns.get_loc("force")
        df.insert(insert_idx, "temp", temp_value)

        # Guarda con sufijo _withtemp.csv en misma carpeta
        orig = Path(path)
        out_path = orig.with_name(f"{orig.stem}_withtemp{orig.suffix}")
        try:
            df.to_csv(out_path, index=False)
            QMessageBox.information(
                self,
                "Guardado",
                f"Archivo guardado con la columna 'temp':\n{out_path}",
            )
        except Exception as e:
            QMessageBox.critical(self, "CSV", f"No se pudo guardar el archivo:\n{e}")
