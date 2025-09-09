# bolt_extractor_tab.py - Extrae subconjuntos de un CSV por "Bolt Num"
# -----------------------------------------------------------------------------
#   • Permite al usuario cargar un fichero CSV con la cabecera proporcionada.
#   • Muestra una lista desplegable (multi‑selección mediante checkboxes) con
#     todos los valores únicos de la columna "Bolt Num".
#   • Tras elegir uno o varios números de perno y pulsar "Create CSV", guarda
#     un nuevo fichero que contiene únicamente las filas correspondientes.
#   • La estética, botones y comportamiento imitan el resto de pestañas de la
#     aplicación (mismo STYLE_SHEET, mismos objectName en botones).
# -----------------------------------------------------------------------------
from __future__ import annotations

import pandas as pd
from pathlib import Path
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import Qt

__all__ = ["BoltExtractorTab"]


class BoltExtractorTab(QtWidgets.QWidget):
    """Pestaña para filtrar un CSV por valores de "Bolt Num"."""

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self._df: pd.DataFrame | None = None          # DataFrame cargado
        self._csv_path: Path | None = None            # Ruta al CSV origen
        self._build_ui()

    # --------------------------------------------------------------- UI
    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(14)

        # 1️⃣  Selector de fichero ------------------------------------------------
        file_box = QtWidgets.QHBoxLayout()
        self.file_edit = QtWidgets.QLineEdit()
        self.file_edit.setReadOnly(True)
        self.file_btn = QtWidgets.QPushButton("File", objectName="File")
        file_box.addWidget(self.file_edit)
        file_box.addWidget(self.file_btn)
        layout.addLayout(file_box)

        # 2️⃣  Lista de Bolt Num ---------------------------------------------------
        layout.addWidget(QtWidgets.QLabel("Select Bolt Num:"))
        self.bolt_list = QtWidgets.QListWidget()
        self.bolt_list.setAlternatingRowColors(True)
        # sin selección por filas; usamos checkboxes
        self.bolt_list.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        layout.addWidget(self.bolt_list, 1)

        # 3️⃣  Botón para crear CSV -----------------------------------------------
        self.create_btn = QtWidgets.QPushButton("Create CSV", objectName="Save")
        self.create_btn.setEnabled(False)
        layout.addWidget(self.create_btn, alignment=Qt.AlignRight)

        # 4️⃣  Señales -------------------------------------------------------------
        self.file_btn.clicked.connect(self._load_csv)
        self.create_btn.clicked.connect(self._create_filtered_csv)

    # --------------------------------------------------------------- Helpers
    def _load_csv(self):
        """Abre un QFileDialog, lee el CSV y pobla la lista de Bolt Num."""
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open CSV",
            str(Path(__file__).parent / "Medidas"),
            "CSV files (*.csv)"
        )
        if not file_path:
            return  # usuario canceló

        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "CSV", f"No se pudo leer el fichero:\n{e}")
            return

        if "Bolt Num" not in df.columns:
            QtWidgets.QMessageBox.critical(
                self,
                "CSV",
                "El CSV no contiene la columna 'Bolt Num'. Asegúrate de que la cabecera es correcta."
            )
            return

        self._df = df
        self._csv_path = Path(file_path)
        self.file_edit.setText(file_path)

        # Poblar lista de números únicos (orden ascendente)
        self.bolt_list.clear()
        unique_bolts = sorted(df["Bolt Num"].dropna().unique())
        for num in unique_bolts:
            item = QtWidgets.QListWidgetItem(str(int(num)))
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            self.bolt_list.addItem(item)

        self.create_btn.setEnabled(True)

    # .......................................................................
    def _selected_bolts(self) -> list[int]:
        """Devuelve la lista de Bolt Num marcados en la interfaz."""
        selected: list[int] = []
        for i in range(self.bolt_list.count()):
            item = self.bolt_list.item(i)
            if item.checkState() == Qt.Checked:
                try:
                    selected.append(int(item.text()))
                except ValueError:
                    pass
        return selected

    # .......................................................................
    def _create_filtered_csv(self):
        if self._df is None:
            return

        bolts = self._selected_bolts()
        if not bolts:
            QtWidgets.QMessageBox.information(self, "Bolt Num", "Selecciona al menos un Bolt Num antes de continuar.")
            return

        # Filtrar DataFrame y pedir ruta de guardado
        df_filtered = self._df[self._df["Bolt Num"].isin(bolts)].copy()
        default_name = self._csv_path.stem + "_selected.csv" if self._csv_path else "selected.csv"
        save_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save CSV",
            str(self._csv_path.parent / default_name) if self._csv_path else default_name,
            "CSV files (*.csv)"
        )
        if not save_path:
            return
        if not save_path.lower().endswith(".csv"):
            save_path += ".csv"

        try:
            df_filtered.to_csv(save_path, index=False)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "CSV", f"No se pudo guardar el fichero:\n{e}")
            return

        QtWidgets.QMessageBox.information(self, "Guardado", f"CSV creado en:\n{save_path}")
