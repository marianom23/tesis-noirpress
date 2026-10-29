"""Lectura de las recepciones documentadas en la planilla de Noirpress."""
from pathlib import Path
import pandas as pd


def leer_ingresos(path_control: Path) -> pd.DataFrame:
    """Hoja 'Ingresos': el encabezado real esta en la cuarta fila del archivo."""
    ing = pd.read_excel(path_control, sheet_name="Ingresos", skiprows=3)
    ing = ing.dropna(subset=["ID Lote"]).copy()
    ing["Fecha ingreso"] = pd.to_datetime(ing["Fecha ingreso"])
    ing["Unidades"] = ing["Unidades"].astype(int)
    return ing
