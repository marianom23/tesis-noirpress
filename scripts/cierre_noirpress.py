"""Depuración de cierre, separada de la base histórica usada para evaluar modelos."""
import hashlib
import json
import ast
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
# Se reutiliza únicamente la función de cálculo, sin cargar los gráficos de la auditoría.
audit = (ROOT / "scripts/auditoria_eda_noirpress.py").read_text(encoding="utf-8")
estimate_node = next(n for n in ast.parse(audit).body if isinstance(n, ast.FunctionDef) and n.name == "estimate")
exec(compile(ast.Module(body=[estimate_node], type_ignores=[]), "auditoria_eda_noirpress.py", "exec"))
OUT = ROOT / "resultados_noirpress/cierre"
SOURCE = ROOT / "fuentes/ultimas_ordenes_2026-07-06.xlsx"
ORDERS = ROOT / "Ordenes_Noirpress_limpio.xlsx"
OUT.mkdir(parents=True, exist_ok=True)

source_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
d = pd.read_excel(ORDERS, sheet_name="Ordenes_limpio")
d["Fecha"] = pd.to_datetime(d["Fecha"]).dt.normalize()
assert len(d) == 409 and not d.duplicated(["Origen", "N° Orden"]).any()
zero = d["Total orden"].eq(0)
low = d["Total orden"].gt(0) & d["Costo de productos"].lt(20000)
assert int(zero.sum()) == 10 and int(low.sum()) == 8
assert set(d.loc[low, "N° Orden"].astype(str)) == {
    "265", "266", "293", "295", "332", "340", "344", "350"
}
assert d.loc[low, "Origen"].eq("TiendaNube").all()
assert d.loc[zero, "Origen"].eq("Mercado Libre").all()

# Los archivos de evaluación conservan su versión original: no se recalculan modelos aquí.
clean = d.loc[~(zero | low)].copy()
qty, refs = estimate(clean)
clean["Unidades estimadas cierre"] = qty
clean["Referencia previa cierre"] = refs
clean = clean.drop(columns=["Precio ref. mes", "Ratio (Total/ref)", "Estado", "Cantidad estimada"])
excluded = d.loc[zero | low, list(d.columns[:12])].copy()
excluded["Motivo de exclusión"] = [
    "Cancelada, según confirmación aportada al cierre" if value == 0
    else "Costo bajo; exclusión indicada al cierre, producto no verificado"
    for value in excluded["Total orden"]
]
daily = clean.groupby("Fecha").agg(
    pedidos=("N° Orden", "size"), unidades_estimadas=("Unidades estimadas cierre", "sum")
).reindex(pd.date_range("2026-01-03", "2026-06-30", name="Fecha"), fill_value=0)
assert len(clean) == 391 and len(excluded) == 18 and len(daily) == 179
assert int(qty.sum()) == 405 and int(daily.pedidos.sum()) == len(clean)
assert int(daily.unidades_estimadas.sum()) == int(qty.sum())
assert clean["Total orden"].gt(0).all() and clean["Costo de productos"].ge(20000).all()
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == source_hash

clean.to_csv(OUT / "ordenes_cierre.csv", index=False, encoding="utf-8-sig")
excluded.to_csv(OUT / "ordenes_excluidas.csv", index=False, encoding="utf-8-sig")
daily.to_csv(OUT / "serie_diaria_cierre.csv", encoding="utf-8-sig")
summary = {
    "fecha_cierre": "2026-10-20",
    "ordenes_fuente": len(d), "canceladas_importe_cero": int(zero.sum()),
    "excluidas_costo_bajo": int(low.sum()), "pedidos_cierre": len(clean),
    "unidades_estimadas_cierre": int(qty.sum()), "dias": len(daily),
    "pedidos_por_canal": clean.groupby("Origen").size().to_dict(),
    "unidades_por_mes": daily.groupby(daily.index.strftime("%Y-%m")).unidades_estimadas.sum().to_dict(),
    "fuente_sha256": source_hash,
    "base_evaluacion_anterior": {"pedidos": 399, "unidades_estimadas": 413},
    "modelos_reestimados_en_cierre": False,
    "criterio": "Cancelaciones informadas y exclusión de ocho órdenes de costo bajo indicada por los autores; no se infiere producto.",
}
(OUT / "cierre.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
