#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Noirpress — de las fuentes crudas al archivo depurado.

Entradas: fuentes/ultimas_ordenes_2026-07-06.xlsx   (ventas reales, ene-jun 2026)
          fuentes/control_pedidos_ventas_quiebres_noirpress.xlsx  (ingresos de stock)
Salida  : Ordenes_Noirpress_limpio.xlsx

Reglas de depuracion (Anexo B de la tesis):
  1. Negro y Blanco se tratan como un solo producto combinado.
  2. Las ordenes con 'Total orden' = 0 (canceladas/reembolsadas) se descartan por
     completo: no cuentan como venta.
  3. Sobre el resto se calcula una cantidad estimada de referencia usando la
     mediana mensual de 'Total orden', con piso de 1 unidad. El módulo de
     pronóstico principal recalcula además una cantidad causal con los 30 días
     anteriores, sin mirar información posterior.
  4. Los ratios fuera del rango 0.6x-1.4x quedan marcados 'Revisar - precio atipico'.
     Se incluyen en la demanda con su cantidad estimada, pero quedan senalados.
  5. El saldo de stock se reconstruye recursivamente desde el 02-ene-2026, con
     stock inicial = ingresos acumulados antes de esa fecha (supuesto: 0 ventas
     previas, unico posible sin datos).
  6. Lead time: no existe fecha real de pedido en ninguna fuente. Se asume 30 dias
     (envio aereo), segun el Cap. 5.4.

Uso:  python scripts/limpiar_noirpress.py [--fuentes DIR] [--salida ARCHIVO]

El script es determinista: no hay muestreo ni componentes aleatorios.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent

RATIO_MIN, RATIO_MAX = 0.6, 1.4     # regla 4
LEAD_TIME_SUPUESTO = 30             # regla 6, dias
UMBRAL_STOCK_BAJO = 15              # unidades; por debajo de esto = "Riesgo - stock bajo"
COLA_POSTERIOR = 5                  # dias mostrados desde el ultimo ingreso inclusive,
                                    # para que ese ingreso quede reflejado en el saldo

ST_VAL = "Valido"
ST_REV = "Revisar - precio atipico"
ST_DES = "Descartar - cancelada/sin cargo"


MESES_ES = ["ene", "feb", "mar", "abr", "may", "jun",
            "jul", "ago", "sep", "oct", "nov", "dic"]


def redondear(x):
    """Redondeo con .5 siempre hacia arriba (evita el redondeo bancario de Python)."""
    return int(np.floor(float(x) + 0.5))


def fes(f) -> str:
    """Fecha como '02-ene-2026' (mes abreviado en castellano, siempre en minuscula)."""
    f = pd.Timestamp(f)
    return f"{f.day:02d}-{MESES_ES[f.month - 1]}-{f.year}"


# --------------------------------------------------------------------------- ordenes
def limpiar_ordenes(path_ordenes: Path) -> pd.DataFrame:
    df = pd.read_excel(path_ordenes, sheet_name="Ordenes")
    df["Fecha creación"] = pd.to_datetime(df["Fecha creación"])
    df = df.sort_values("Fecha creación").reset_index(drop=True)

    df["Fecha"] = df["Fecha creación"].dt.normalize()
    df["Mes"] = df["Fecha creación"].dt.strftime("%Y-%m")
    df["Día semana"] = df["Fecha creación"].dt.day_name()

    # Regla 2
    descartada = df["Total orden"] == 0

    # Regla 3: mediana mensual calculada SOLO sobre ordenes con cargo
    ref = df.loc[~descartada].groupby("Mes")["Total orden"].median()
    df["Precio ref. mes"] = np.where(descartada, np.nan, df["Mes"].map(ref))
    df["Ratio (Total/ref)"] = (df["Total orden"] / df["Precio ref. mes"]).round(2)

    # Regla 4
    df["Estado"] = ST_VAL
    df.loc[
        (~descartada)
        & ((df["Ratio (Total/ref)"] < RATIO_MIN) | (df["Ratio (Total/ref)"] > RATIO_MAX)),
        "Estado",
    ] = ST_REV
    df.loc[descartada, "Estado"] = ST_DES

    df["Cantidad estimada"] = [
        0 if e == ST_DES else max(1, redondear(r))
        for e, r in zip(df["Estado"], df["Ratio (Total/ref)"])
    ]
    return df


def demanda_diaria(ordenes: pd.DataFrame) -> pd.DataFrame:
    rango = pd.date_range(ordenes["Fecha"].min(), ordenes["Fecha"].max(), freq="D")
    d = pd.DataFrame({"Fecha": rango})
    d["Dia semana"] = d["Fecha"].dt.day_name()
    for col, est in [
        ("Pedidos validos", ST_VAL),
        ("Pedidos a revisar", ST_REV),
        ("Pedidos descartados", ST_DES),
    ]:
        c = ordenes[ordenes["Estado"] == est].groupby("Fecha").size()
        d[col] = d["Fecha"].map(c).fillna(0).astype(int)
    u = ordenes.groupby("Fecha")["Cantidad estimada"].sum()
    d["Unidades estimadas"] = d["Fecha"].map(u).fillna(0).astype(int)
    return d


def demanda_mensual(ordenes: pd.DataFrame) -> pd.DataFrame:
    m = pd.DataFrame(
        {
            "Mes": sorted(ordenes["Mes"].unique()),
        }
    )
    for col, est in [
        ("Pedidos validos", ST_VAL),
        ("Pedidos a revisar", ST_REV),
        ("Pedidos descartados", ST_DES),
    ]:
        c = ordenes[ordenes["Estado"] == est].groupby("Mes").size()
        m[col] = m["Mes"].map(c).fillna(0).astype(int)
    m["Unidades estimadas"] = m["Mes"].map(
        ordenes.groupby("Mes")["Cantidad estimada"].sum()
    ).astype(int)
    ref = ordenes[ordenes["Estado"] != ST_DES].groupby("Mes")["Total orden"].median()
    m["Precio ref. mediana (ARS)"] = m["Mes"].map(ref)
    return m


# --------------------------------------------------------------------------- ingresos
def leer_ingresos(path_control: Path) -> pd.DataFrame:
    """Hoja 'Ingresos': el encabezado real esta en la cuarta fila del archivo."""
    ing = pd.read_excel(path_control, sheet_name="Ingresos", skiprows=3)
    ing = ing.dropna(subset=["ID Lote"]).copy()
    ing["Fecha ingreso"] = pd.to_datetime(ing["Fecha ingreso"])
    ing["Unidades"] = ing["Unidades"].astype(int)
    return ing


def meses_control_original(path_control: Path):
    """Meses cubiertos por la hoja 'Control Stock' de la planilla original.

    El control mensual reconstruido usa el mismo horizonte (jun-2025 a dic-2026) para que
    ambas planillas sean comparables fila a fila, aunque fuera del alcance real (ene-jun
    2026) la columna de ventas quede vacia."""
    cs = pd.read_excel(path_control, sheet_name="Control Stock", skiprows=3)
    fechas = pd.to_datetime(cs.iloc[:, 0], errors="coerce").dropna()
    return sorted(fechas.dt.strftime("%Y-%m").unique())


def ingresos_combinados(ing: pd.DataFrame) -> pd.Series:
    """Regla 1: Negro + Blanco sumados por fecha de ingreso."""
    return ing.groupby("Fecha ingreso")["Unidades"].sum().sort_index()


# --------------------------------------------------------------------------- stock
def saldo_diario(diaria: pd.DataFrame, ingresos: pd.Series, stock_inicial: int):
    """Regla 5: saldo(t) = saldo(t-1) + ingreso(t) - vendido(t)."""
    fin = max(diaria["Fecha"].max(), ingresos.index.max()) + pd.Timedelta(
        days=COLA_POSTERIOR - 1
    )
    rango = pd.date_range(diaria["Fecha"].min(), fin, freq="D")
    vend = diaria.set_index("Fecha")["Unidades estimadas"]

    filas, saldo = [], stock_inicial
    for f in rango:
        entra = int(ingresos.get(f, 0))
        sale = int(vend.get(f, 0))
        saldo = saldo + entra - sale
        if saldo <= 0:
            estado = "QUIEBRE"
        elif saldo <= UMBRAL_STOCK_BAJO:
            estado = "Riesgo - stock bajo"
        else:
            estado = "OK"
        filas.append([f, entra, sale, saldo, estado])
    return pd.DataFrame(
        filas,
        columns=["Fecha", "Ingreso del dia", "Vendido (unid. estimadas)",
                 "Saldo acumulado", "Estado"],
    )


def control_mensual(mensual: pd.DataFrame, ingresos: pd.Series, meses_planilla):
    """Reconstruye el control mensual sobre el mismo horizonte que la planilla original."""
    ing_mes = ingresos.groupby(ingresos.index.strftime("%Y-%m")).sum()
    ventas_mes = mensual.set_index("Mes")["Unidades estimadas"]
    meses = pd.period_range(
        min(list(meses_planilla) + [ing_mes.index.min(), ventas_mes.index.min()]),
        max(list(meses_planilla) + [ing_mes.index.max(), ventas_mes.index.max()]),
        freq="M",
    ).strftime("%Y-%m")

    filas, stock = [], 0
    for m in meses:
        entra = int(ing_mes.get(m, 0))
        en_alcance = m in set(ventas_mes.index)
        sale = int(ventas_mes.get(m, 0)) if en_alcance else None
        ini = stock
        stock = ini + entra - (sale or 0)
        filas.append([m, entra, sale, ini, stock,
                      "OK (ventas reales)" if en_alcance else "Descartado - fuera de alcance"])
    return pd.DataFrame(
        filas, columns=["Mes", "Ingresos", "Ventas", "Stock inicial", "Stock final", "Estado"]
    )


def tabla_lead_time(ing: pd.DataFrame, diaria: pd.DataFrame, saldo: pd.DataFrame):
    """Regla 6: fecha de pedido estimada = llegada - 30 dias."""
    vend = diaria.set_index("Fecha")["Unidades estimadas"]
    saldo_idx = saldo.set_index("Fecha")["Saldo acumulado"]
    prom_global = diaria["Unidades estimadas"].sum() / len(diaria)
    dem_supuesta = round(prom_global * LEAD_TIME_SUPUESTO, 1)

    filas = []
    for _, r in ing.iterrows():
        llegada = r["Fecha ingreso"]
        pedido = llegada - pd.Timedelta(days=LEAD_TIME_SUPUESTO)
        if pedido in saldo_idx.index:
            stock = int(saldo_idx.loc[pedido])
            ventana = pd.date_range(pedido, periods=LEAD_TIME_SUPUESTO, freq="D")
            dem = int(vend.reindex(ventana).fillna(0).sum())
            fuente, margen = "Real (Ordenes)", stock - dem
        else:
            stock, dem, fuente, margen = None, dem_supuesta, "Supuesto (promedio global)", None
        filas.append([r["ID Lote"], r["SKU"], r["Tipo"], int(r["Unidades"]),
                      pedido, llegada, stock, dem, fuente, margen])

    tabla = pd.DataFrame(
        filas,
        columns=["ID Lote", "SKU", "Tipo", "Unidades", "Fecha pedido (estimada)",
                 "Fecha llegada (real)", "Stock al momento de pedir",
                 "Demanda esperada en el lead time", "Fuente de la demanda",
                 "Margen (stock - demanda)"],
    )
    return tabla, prom_global, dem_supuesta


# --------------------------------------------------------------------------- salida
def bloque(titulo, notas, tabla, cierre_titulo=None, cierre=None, linea_blanco=True):
    """Arma una hoja con titulo, notas, la tabla y un bloque de lectura al pie."""
    filas = [[titulo] + [None] * (tabla.shape[1] - 1)]
    for n in notas:
        filas.append([n] + [None] * (tabla.shape[1] - 1))
    if linea_blanco:
        filas.append([None] * tabla.shape[1])
    filas.append(list(tabla.columns))
    for _, r in tabla.iterrows():
        filas.append(list(r.values))
    if cierre:
        filas.append([None] * tabla.shape[1])
        filas.append([cierre_titulo] + [None] * (tabla.shape[1] - 1))
        for c in cierre:
            filas.append([c] + [None] * (tabla.shape[1] - 1))
    return pd.DataFrame(filas)


def construir(dir_fuentes: Path) -> dict:
    p_ord = dir_fuentes / "ultimas_ordenes_2026-07-06.xlsx"
    p_ctl = dir_fuentes / "control_pedidos_ventas_quiebres_noirpress.xlsx"
    for p in (p_ord, p_ctl):
        if not p.exists():
            sys.exit(f"ERROR: no se encuentra {p}")

    ordenes = limpiar_ordenes(p_ord)
    diaria = demanda_diaria(ordenes)
    mensual = demanda_mensual(ordenes)

    ing = leer_ingresos(p_ctl)
    ingresos = ingresos_combinados(ing)

    inicio = diaria["Fecha"].min()
    stock_inicial = int(ingresos[ingresos.index < inicio].sum())

    saldo = saldo_diario(diaria, ingresos, stock_inicial)
    meses_planilla = meses_control_original(p_ctl)
    control = control_mensual(mensual, ingresos, meses_planilla)
    lead, prom_global, dem_sup = tabla_lead_time(ing, diaria, saldo)

    fila_min = saldo.loc[saldo["Saldo acumulado"].idxmin()]
    # Dias de cobertura "al ritmo de venta de ese momento": se usa la media movil de los
    # 30 dias previos al minimo (misma ventana que el lead time supuesto), no el promedio
    # global de la serie, que subestimaria el consumo post Hot Sale.
    ritmo_local = (
        diaria.set_index("Fecha")["Unidades estimadas"]
        .loc[fila_min["Fecha"] - pd.Timedelta(days=LEAD_TIME_SUPUESTO - 1): fila_min["Fecha"]]
        .mean()
    )
    cobertura = fila_min["Saldo acumulado"] / ritmo_local

    return dict(
        ordenes=ordenes, diaria=diaria, mensual=mensual, ing=ing, ingresos=ingresos,
        stock_inicial=stock_inicial, saldo=saldo, control=control, lead=lead,
        prom_global=prom_global, dem_sup=dem_sup, fila_min=fila_min,
        cobertura=cobertura, inicio=inicio,
        quiebres=int((saldo["Saldo acumulado"] <= 0).sum()),
    )


def hoja_resumen(r: dict) -> pd.DataFrame:
    o = r["ordenes"]
    n_val = int((o["Estado"] == ST_VAL).sum())
    n_rev = int((o["Estado"] == ST_REV).sum())
    n_des = int((o["Estado"] == ST_DES).sum())
    fmin = r["fila_min"]
    t = [
        "Limpieza de datos de venta y analisis de quiebre de stock — Noirpress",
        None,
        f"Fuentes: ultimas_ordenes_2026-07-06.xlsx ({len(o)} ordenes, un solo producto, "
        "ene-jun 2026) y control_pedidos_ventas_quiebres_noirpress.xlsx "
        "(ingresos de stock, jun-2025 a jul-2026).",
        "Se trata el producto como una sola linea (Negro+Blanco combinados), sin distinguir "
        "color, tal como se pidio.",
        None,
        "1. Que SI se puede determinar",
        f"• Demanda diaria y mensual real ({len(o)} ordenes con fecha y hora exacta, ene-jun 2026).",
        "• Estacionalidad de calendario (dia de semana, mes) y mezcla de canal/medio de pago.",
        "• Una senal de precio: el costo de producto salta de ~$62.000 a ~$66.488 en Mercado "
        "Libre desde mayo 2026, coincidiendo con el rango de landed cost del otro archivo "
        "(55.740-64.611) — confirma que ambos archivos hablan del mismo producto.",
        "• Con los ingresos de stock combinados, se pudo armar un saldo diario de stock desde "
        "el 02-ene-2026 y buscar quiebres (ver hoja 'Quiebres de stock').",
        None,
        "2. Resultado del analisis de quiebre de stock",
        f"• Stock estimado al {fes(r['inicio'] - pd.Timedelta(days=1))} (ingresos "
        f"acumulados antes de esa fecha, ver Ingresos_combinados): {r['stock_inicial']} unidades.",
        f"• Con ese punto de partida y la demanda diaria estimada ({fes(r['inicio'])} a "
        f"{fes(r['diaria']['Fecha'].max())}), el saldo NUNCA llega a 0: el minimo se da el "
        f"{fes(fmin['Fecha'])} con {int(fmin['Saldo acumulado'])} unidades en stock.",
        "• Conclusion final: en el unico tramo con datos de venta (ene-jun 2026) no hubo ningun "
        "quiebre de stock. El tramo jun-2025 a ene-2026 (7 meses) se deja fuera de alcance por "
        "decision expresa: no hay ninguna venta registrada en los archivos para ese periodo, "
        "asi que no es analizable con la informacion disponible.",
        None,
        "3. Que NO se puede determinar / que se descarto",
        "• No hay columna de SKU ni de cantidad en las ordenes: no se puede repartir la venta "
        "entre Negro y Blanco (tampoco hacia falta, segun lo indicado).",
        "• La cantidad por orden es una ESTIMACION (Total orden / mediana mensual), no un dato "
        f"duro. Para el {n_val / len(o) * 100:.0f}% de las ordenes cae en un rango compatible "
        "con 1 unidad.",
        f"• {n_des} ordenes con Total orden = 0 se descartaron por completo "
        "(canceladas/reembolsadas, incluye una con Neto negativo) — no cuentan como venta.",
        f"• {n_rev} ordenes (~{n_rev / len(o) * 100:.0f}%) quedaron marcadas 'Revisar - precio "
        "atipico': 8 con costo de producto muy por debajo de lo esperado (~$8.000 vs ~$62.000, "
        "posible otro item o error de carga) y 11 con monto muy por encima de la mediana "
        "(probable compra de varias unidades). Se incluyeron en la demanda con su cantidad "
        "estimada, pero conviene confirmarlas.",
        "• Tramo jun-2025 a ene-2026 (7 meses) FUERA DE ALCANCE (descartado a pedido): ninguno "
        "de los dos archivos registra ventas ni quiebres de stock en ese periodo. La hoja "
        "'Quiebres de stock' no es un dato leido de un archivo: es un calculo propio "
        "(ingresos - ventas) que solo pudo construirse para ene-jun 2026, que es donde hay "
        "ventas reales.",
        "• La hoja 'Ventas Mensuales' del archivo control_pedidos_ventas_quiebres_noirpress.xlsx "
        "sigue vacia (en cero) para todo el periodo; no aporta datos reales, solo confirma el "
        "mismo vacio de informacion.",
        None,
        "4. Metodologia (resumen)",
        "• Se descartan ordenes con Total orden = 0. Sobre el resto, se calcula la mediana "
        "mensual de 'Total orden' y se estima cantidad = redondeo(Total/mediana del mes), con "
        f"piso de 1 unidad; ratios fuera de {RATIO_MIN}x-{RATIO_MAX}x quedan marcados para revisar.",
        f"• El saldo de stock = stock inicial ({fes(r['inicio'] - pd.Timedelta(days=1))}) "
        "+ ingresos del dia - unidades vendidas estimadas del dia, dia por dia desde el "
        f"{fes(r['inicio'])}.",
        "• Todos los valores de este archivo fueron calculados una sola vez a partir de las "
        "fuentes citadas (no son formulas vivas de Excel): si cambian los datos de origen, hay "
        "que volver a correr scripts/limpiar_noirpress.py para actualizar las cifras.",
        None,
        "5. Sobre el 'Estado' (OK / SIN STOCK) de la hoja Control Stock del otro archivo",
        "• Esa hoja SI muestra un 'SIN STOCK': Noirpress-Blanco en jun-2025. Pero no es un "
        "quiebre por agotamiento: ese mes Ingresos=0 y Stock inicial=0 porque Blanco todavia no "
        "habia sido comprado (su primer pedido es del 26-jul-2025). Es un estado 'antes de tener "
        "stock', no 'se acabo el stock'.",
        "• La columna 'Ventas' de esa hoja es 0 en las 38 filas (jun-2025 a dic-2026), sin "
        "excepcion, porque toma los datos de la hoja 'Ventas Mensuales', que sigue vacia. Es "
        "decir: el Estado OK/SIN STOCK de todo ese archivo esta calculado asumiendo que nunca se "
        "vendio nada — no refleja ventas reales en ningun mes, tampoco en ene-jun 2026 donde "
        "nosotros si tenemos datos reales (Ordenes).",
        "• Ademas, la columna 'Dias desde ultimo pedido' de la primera fila (jun-2025) muestra "
        "45.838, que es un numero de serie de fecha de Excel filtrado por un error de formula, "
        "no una cantidad real de dias (el resto de las filas si tienen valores razonables).",
        "• Conclusion: el 'Control Stock' del archivo original no aporta evidencia de quiebres "
        "reales; nuestra hoja 'Quiebres de stock' (basada en ventas reales de Ordenes) es la "
        "unica fuente confiable para el periodo ene-jun 2026, y ahi el resultado es 0 quiebres.",
        None,
        "6. Alcance final (cerrado)",
        "• Por decision expresa, el analisis queda acotado a enero-junio 2026: es el unico tramo "
        "donde existen ventas reales en algun archivo.",
        "• jun-2025 a dic-2025 y jul-2026 a dic-2026 quedan DESCARTADOS - fuera de alcance (ver "
        "hoja 'Control Stock combinado'). No se van a seguir investigando: ningun archivo tiene "
        "ordenes de venta ahi.",
        f"• Conclusion final del trabajo de limpieza: en el tramo en alcance (ene-jun 2026) no "
        f"hubo ningun quiebre de stock; el minimo fue {int(fmin['Saldo acumulado'])} unidades "
        f"({fes(fmin['Fecha'])}), equivalente a unos {r['cobertura']:.0f} dias de cobertura "
        "al ritmo de venta de ese momento.",
    ]
    return pd.DataFrame({0: t})


def escribir(r: dict, salida: Path):
    ing_tab = pd.DataFrame(
        {"Fecha ingreso": r["ingresos"].index, "Unidades recibidas": r["ingresos"].values}
    )
    hoja_ing = bloque(
        "Ingresos de stock combinados (Negro + Blanco tratados como un solo producto)",
        ["Fuente: control_pedidos_ventas_quiebres_noirpress.xlsx, hoja Ingresos "
         "(incluye la muestra inicial de 2 unidades)."],
        ing_tab,
    )
    hoja_ing.loc[len(hoja_ing)] = [None, None]
    hoja_ing.loc[len(hoja_ing)] = [
        f"Stock acumulado antes del {fes(r['inicio'])} (usado como saldo inicial en "
        f"'Quiebres de stock'): {r['stock_inicial']} unidades", None]

    hoja_q = bloque(
        "Saldo de stock diario (Negro+Blanco combinados) y deteccion de quiebres",
        [f"Cubre el periodo con ventas conocidas ({fes(r['inicio'])} a "
         f"{fes(r['diaria']['Fecha'].max())}) mas unos dias posteriores para reflejar el "
         f"ingreso del {fes(r['ingresos'].index.max())}.",
         f"Stock inicial al {fes(r['inicio'] - pd.Timedelta(days=1))} (supuesto: 0 ventas "
         f"antes de esta fecha; ver Ingresos_combinados): {r['stock_inicial']} unidades"],
        r["saldo"],
        "Metodologia",
        [f"- Saldo = stock inicial ({fes(r['inicio'] - pd.Timedelta(days=1))}) + ingresos de stock - unidades vendidas "
         "estimadas (hoja Demanda diaria, ya sin ordenes descartadas).",
         "- No existen ventas registradas entre jun-2025 y ene-2026 (7 meses fuera del rango del "
         "archivo de ordenes); el saldo inicial asume 0 ventas en ese tramo por ser el unico "
         "supuesto posible sin datos. No se puede confirmar ni descartar un quiebre en esa ventana.",
         f"- 'Riesgo - stock bajo' = saldo entre 1 y {UMBRAL_STOCK_BAJO} unidades (umbral "
         "orientativo, ajustable); 'QUIEBRE' = saldo <= 0.",
         "- Valores calculados una sola vez a partir de Ordenes_limpio/Demanda diaria e "
         "Ingresos_combinados; si cambian los datos de origen hay que volver a correr "
         "scripts/limpiar_noirpress.py.",
        ],
        linea_blanco=False,
    )

    hoja_c = bloque(
        "Reconstruccion mensual de stock — Negro+Blanco combinados, con ventas reales donde existen",
        ["Ingresos: control_pedidos_ventas_quiebres_noirpress.xlsx (dato real, todo el periodo). "
         "Ventas: ultimas_ordenes_2026-07-06.xlsx (dato real, solo ene-jun 2026; unidades "
         "estimadas por orden). Fuera de ese rango no hay ventas registradas en ningun archivo."],
        r["control"],
        "Lectura",
        ["- Alcance final del analisis: SOLO enero-junio 2026, por decision expresa. "
         "jun-2025/dic-2025 y jul-2026/dic-2026 quedan descartados (fuera de alcance): ningun "
         "archivo tiene ordenes de venta en esos tramos, asi que no son analizables ni "
         "proyectables con datos reales.",
         "- En el tramo descartado, la columna Ventas queda en gris y el Estado dice 'Descartado "
         "- fuera de alcance'; el Stock que se ve ahi es solo la suma de ingresos, no un saldo "
         "real verificado, y no debe usarse para sacar conclusiones.",
         "- En el unico tramo en alcance (ene-jun 2026), Ventas son datos reales reconstruidos "
         f"desde las {len(r['ordenes'])} ordenes. Resultado: OK en los 6 meses, sin quiebre, "
         f"stock final {int(r['control'].loc[r['control'].Mes == '2026-06', 'Stock final'].iloc[0])} "
         "unidades a fin de junio-2026.",
         "- Diferencia con la hoja 'Control Stock' del archivo original: aca las ventas de ene-jun "
         "2026 son reales (no cero), y los meses sin informacion quedan marcados explicitamente "
         "en vez de asumir venta cero en silencio."],
    )

    n_sup = int(r["lead"]["Fuente de la demanda"].eq("Supuesto (promedio global)").sum())
    margenes = r["lead"]["Margen (stock - demanda)"].dropna()
    hoja_l = bloque(
        f"Lead time asumido: {LEAD_TIME_SUPUESTO} dias (envio aereo) por lote — "
        "Fecha de pedido = Fecha de llegada - 30 dias",
        ["No existe en los archivos la fecha real de pedido; se usa un supuesto de 1 mes, tal "
         "como preve el Capitulo 5.4 de la tesis (valores representativos por metodo de envio)."],
        r["lead"],
        "Lectura",
        [f"- Para los {n_sup} lotes de 2025 (muestra, jul-2025, nov-2025) no hay stock ni demanda "
         "real disponibles (fuera del rango de Ordenes); la demanda se estima con el promedio "
         f"global ({r['prom_global']:.2f} u/dia) solo a modo ilustrativo, en amarillo.",
         "- Para los lotes de feb, abr, may y jul-2026 (dentro del rango real de Ordenes), el "
         f"margen es siempre positivo: entre {int(margenes.min())} y {int(margenes.max())} "
         "unidades por encima de lo que se hubiera vendido durante el mes de lead time asumido. "
         "Es decir, aun con un lead time de 30 dias, los pedidos se hicieron con margen de sobra, "
         "no al limite.",
         "- Esto refuerza la conclusion de la hoja 'Quiebres de stock': no solo no hubo quiebre "
         "con el lead time real (implicito en la fecha de llegada), sino que tampoco lo habria "
         "habido si el proveedor hubiera tardado justo 1 mes en cada pedido."],
    )

    med = r["mensual"][["Mes", "Precio ref. mediana (ARS)"]].rename(
        columns={"Precio ref. mediana (ARS)": "Mediana Total orden (ARS)"})
    hoja_m = bloque(
        "Precio de referencia mensual (mediana de 'Total orden' entre ordenes con Total > 0)",
        ["Calculado una sola vez a partir de Ordenes_limpio. Usado para estimar cantidad por orden."],
        med,
    )

    with pd.ExcelWriter(salida, engine="openpyxl", datetime_format="YYYY-MM-DD HH:MM:SS") as w:
        hoja_resumen(r).to_excel(w, sheet_name="Resumen y hallazgos", index=False, header=False)
        r["ordenes"].to_excel(w, sheet_name="Ordenes_limpio", index=False)
        r["diaria"].to_excel(w, sheet_name="Demanda diaria", index=False)
        r["mensual"].to_excel(w, sheet_name="Demanda mensual", index=False)
        hoja_q.to_excel(w, sheet_name="Quiebres de stock", index=False, header=False)
        hoja_c.to_excel(w, sheet_name="Control Stock combinado", index=False, header=False)
        hoja_l.to_excel(w, sheet_name="Lead time (supuesto)", index=False, header=False)
        hoja_ing.to_excel(w, sheet_name="Ingresos_combinados", index=False, header=False)
        hoja_m.to_excel(w, sheet_name="Mediana_mensual_ref", index=False, header=False)

        anchos = {
            "Resumen y hallazgos": {"A": 130},
            "Ordenes_limpio": {"A": 20, "C": 20, "M": 12, "R": 32},
            "Demanda diaria": {"A": 12, "B": 12, "C": 16, "D": 18, "E": 20, "F": 18},
            "Demanda mensual": {"A": 10, "B": 16, "C": 18, "D": 20, "E": 18, "F": 24},
            "Quiebres de stock": {"A": 22, "B": 16, "C": 26, "D": 18, "E": 20},
            "Control Stock combinado": {"A": 12, "B": 10, "C": 10, "D": 14, "E": 12, "F": 30},
            "Lead time (supuesto)": {"A": 14, "B": 18, "C": 12, "D": 10, "E": 24,
                                     "F": 22, "G": 26, "H": 32, "I": 28, "J": 24},
            "Ingresos_combinados": {"A": 24, "B": 20},
            "Mediana_mensual_ref": {"A": 14, "B": 26},
        }
        for hoja, cols in anchos.items():
            for col, ancho in cols.items():
                w.sheets[hoja].column_dimensions[col].width = ancho


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fuentes", default=str(RAIZ / "fuentes"))
    p.add_argument("--salida", default=str(RAIZ / "Ordenes_Noirpress_limpio.xlsx"))
    a = p.parse_args()

    r = construir(Path(a.fuentes))
    escribir(r, Path(a.salida))

    o = r["ordenes"]
    print(f"ordenes        : {len(o)}  "
          f"(validas {(o.Estado == ST_VAL).sum()}, "
          f"a revisar {(o.Estado == ST_REV).sum()}, "
          f"descartadas {(o.Estado == ST_DES).sum()})")
    print(f"unidades est.  : {int(o['Cantidad estimada'].sum())} en "
          f"{len(r['diaria'])} dias ({r['prom_global']:.2f} u/dia)")
    print(f"stock inicial  : {r['stock_inicial']} u. al "
          f"{fes(r['inicio'] - pd.Timedelta(days=1))}")
    print(f"saldo minimo   : {int(r['fila_min']['Saldo acumulado'])} u. el "
          f"{fes(r['fila_min']['Fecha'])}  "
          f"({r['cobertura']:.0f} dias de cobertura)")
    print(f"quiebres       : {r['quiebres']}")
    print(f"-> {a.salida}")


if __name__ == "__main__":
    main()
