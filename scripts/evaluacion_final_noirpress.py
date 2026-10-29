"""Reevaluación retrospectiva de la serie final depurada, sin modificar fuentes.

Ejecutar con python scripts/evaluacion_final_noirpress.py desde el entorno de calculo.
Los pesos se leen exclusivamente de la caché de Hugging Face.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import time
from pathlib import Path

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import numpy as np
import pandas as pd

from modelos_noirpress_avanzados import (
    MODELS, MODEL_ID, REVISION, prepare_orders, forecast_classic,
    load_foundation, foundation_forecasts, metric_values, save_csv,
)
from auditoria_eda_noirpress import estimate, benchmark_forecasts, simulate_order, stats_row
from limpiar_noirpress import leer_ingresos

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'resultados_noirpress/evaluacion_final'
ORDERS = ROOT / 'resultados_noirpress/cierre/ordenes_cierre.csv'
SERIES = ROOT / 'resultados_noirpress/cierre/serie_diaria_cierre.csv'
STOCK = ROOT / 'fuentes/control_pedidos_ventas_quiebres_noirpress.xlsx'


def write_json(name, value):
    portable = json.loads(pd.Series(value).to_json(force_ascii=False, default_handler=str))
    (OUT / name).write_text(json.dumps(portable, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    began = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    orders = pd.read_csv(ORDERS, parse_dates=['Fecha'])
    q, _ = estimate(orders)
    assert len(orders) == 391 and orders['Total orden'].gt(0).all()
    assert orders['Costo de productos'].ge(20000).all()
    assert np.array_equal(q, orders['Unidades estimadas cierre'])
    prepared = prepare_orders(orders)
    assert prepared.unidades_causales.sum() == q.sum() == 405
    index = pd.date_range('2026-01-03', '2026-06-30', freq='D', name='fecha')
    d = pd.DataFrame(index=index)
    d['demanda'] = orders.assign(q=q).groupby('Fecha').q.sum().reindex(index, fill_value=0).astype(float)
    saved = pd.read_csv(SERIES, parse_dates=['Fecha']).set_index('Fecha')
    assert np.array_equal(d.demanda, saved.unidades_estimadas)
    assert len(d) == 179 and d.demanda.sum() == 405
    d['pedidos'] = orders.groupby('Fecha').size().reindex(index, fill_value=0)
    d['sin_pedidos'] = d.pedidos.eq(0)
    d['importe_ars'] = orders.groupby('Fecha')['Total orden'].sum().reindex(index, fill_value=0)
    for channel, group in orders.groupby('Origen'):
        d[channel] = group.groupby('Fecha').size().reindex(index, fill_value=0)
    d['mes'] = index.strftime('%Y-%m')
    d['evento'] = 'Fuera de ventanas'
    d.loc['2026-05-11':'2026-05-17', 'evento'] = 'Ventana mayo'
    d.loc['2026-06-14':'2026-06-17', 'evento'] = 'Ventana junio'
    d['etapa'] = ['historial inicial'] * 70 + ['validacion'] * 55 + ['prueba'] * 54
    # Cambiar o retirar futuro no puede modificar las unidades reconstruidas del pasado.
    prefix = orders[orders.Fecha < '2026-05-01']
    prefix_q, _ = estimate(prefix)
    assert np.array_equal(prefix_q, q.loc[prefix.index])
    y = d.demanda

    monthly = d.groupby('mes').agg(dias=('demanda', 'size'), unidades=('demanda', 'sum'),
        pedidos=('pedidos', 'sum'), dias_sin_pedidos=('sin_pedidos', 'sum'), media=('demanda', 'mean'), desvio=('demanda', 'std'),
        importe=('importe_ars', 'sum'), TiendaNube=('TiendaNube', 'sum'), Mercado_Libre=('Mercado Libre', 'sum')).reset_index()
    save_csv(monthly, OUT / 'resumen_mensual.csv')
    channels = orders.assign(unidades=q).groupby('Origen').agg(pedidos=('N° Orden', 'size'),
        unidades=('unidades', 'sum'), importe=('Total orden', 'sum'), ultima_fecha=('Fecha', 'max')).reset_index()
    save_csv(channels, OUT / 'canales.csv')
    save_csv(d.groupby('evento').demanda.agg(['size', 'sum', 'mean', 'median', 'std']).reset_index(), OUT / 'eventos.csv')
    save_csv(d.groupby(d.index.dayofweek).demanda.agg(['size', 'sum', 'mean', 'median', 'std']).reset_index(), OUT / 'dia_semana.csv')
    save_csv(pd.DataFrame([stats_row('Total', y)] + [stats_row(stage, g.demanda) for stage, g in d.groupby('etapa')]), OUT / 'estadistica_serie.csv')

    ing = leer_ingresos(STOCK)
    flow = ing.groupby('Fecha ingreso').Unidades.sum().reindex(index, fill_value=0)
    initial = int(ing.loc[ing['Fecha ingreso'] < index[0], 'Unidades'].sum())
    assert initial == 98 and flow.sum() == 420
    d['ingresos_unidades'] = flow
    d['saldo_inicial_98'] = initial + flow.cumsum() - y.cumsum()
    min_initial = int(max(0, (y.cumsum() - flow.cumsum()).max()))
    stock_rows = []
    for q0 in sorted(set([0, 25, 32, 50, 96, 98, min_initial])):
        balance = q0 + flow.cumsum() - y.cumsum()
        stock_rows.append({'Stock inicial supuesto': q0, 'Mínimo contable': balance.min(),
            'Fecha mínimo': str(balance.idxmin().date()), 'Días saldo negativo': int(balance.lt(0).sum()),
            'Días saldo cero': int(balance.eq(0).sum()), 'Saldo 30 junio': balance.iloc[-1]})
    save_csv(pd.DataFrame(stock_rows), OUT / 'sensibilidad_inventario.csv')
    decisions, trajectories = [], []
    for issued in [pd.Timestamp('2026-04-27'), pd.Timestamp('2026-06-01')]:
        history = y.loc[:issued]
        start_stock = int(np.floor(.30 * d.loc[issued, 'saldo_inicial_98']))
        future = y.loc[issued + pd.Timedelta(days=1):min(issued + pd.Timedelta(days=60), index[-1])]
        for name, rate in [('Media 7 días', history.iloc[-7:].mean()), ('Media 28 días', history.iloc[-28:].mean()), ('Mediana 28 días', history.iloc[-28:].median())]:
            quantity = int(np.ceil(max(0, rate * 60 - start_stock)))
            sim = simulate_order(future, start_stock, issued, quantity, 30)
            sim['metodo'] = name
            trajectories.append(sim)
            decisions.append({'Emisión': str(issued.date()), 'Llegada': str((issued + pd.Timedelta(days=30)).date()),
                'Método': name, 'Stock inicial simulado': start_stock, 'Pedido calculado': quantity,
                'Días puntuados': len(sim), 'Faltantes antes llegada': sim.loc[sim.antes_de_llegada, 'faltante_simulado'].sum(),
                'Faltantes después llegada': sim.loc[~sim.antes_de_llegada, 'faltante_simulado'].sum(),
                'Stock medio': sim.stock_cierre.mean(), 'Atendido_pct': 100 * sim.atendido.sum() / sim.demanda_proxy.sum()})
    save_csv(pd.DataFrame(decisions), OUT / 'reposicion_resumen.csv')
    save_csv(pd.concat(trajectories), OUT / 'reposicion_dia_a_dia.csv')
    check = simulate_order(pd.Series([2., 2., 2.], index=pd.date_range('2026-01-02', periods=3)), 1, '2026-01-01', 10, 3)
    assert check.loc[check.antes_de_llegada, 'faltante_simulado'].sum() == 3
    save_csv(d.reset_index(), OUT / 'serie_diaria.csv')
    write_json('resumen_descriptivo.json', dict(unidades_estimadas=405, pedidos=391, dias=179,
        media_diaria=float(y.mean()), dias_sin_ordenes=int(d.sin_pedidos.sum()),
        mensuales=monthly.to_dict('records'), canales=channels.to_dict('records'),
        eventos=d.groupby('evento').demanda.agg(['size', 'sum', 'mean', 'median', 'std']).reset_index().to_dict('records'),
        maximo_diario=float(y.max()), fechas_maximo=[str(t.date()) for t in y[y == y.max()].index],
        inventario=stock_rows, stock_minimo_inicial=min_initial, escenarios=decisions))
    print('Base confirmada: 391 pedidos, 405 unidades, 179 días; inventario y escenarios recalculados.', flush=True)

    model = load_foundation('cpu')
    rows, status = [], []
    for stage, start, end in [('validacion', 70, 125), ('prueba', 125, 179)]:
        for i in range(start, end):
            history = y.iloc[:i]
            dates = pd.date_range(y.index[i], periods=30)
            fpoint, flo, fhi = foundation_forecasts(model, [history])[0]
            for name in MODELS:
                tick = time.perf_counter()
                if name == 'TimesFM 3.0':
                    point, lo, hi = fpoint, flo, fhi
                    ok, message = True, 'zero-shot; revisión fija; caché local'
                else:
                    point, ok, message = forecast_classic(name, history, dates)
                    lo = hi = np.full(30, np.nan)
                status.append(dict(etapa=stage, origen=history.index[-1], modelo=name, convergio=ok,
                                   advertencias=message, segundos=time.perf_counter() - tick))
                for step in range(min(30, end - i)):
                    rows.append(dict(etapa=stage, origen=history.index[-1], fecha=dates[step], horizonte=step + 1,
                        modelo=name, real=y.iloc[i + step], pronostico=point[step], p10=lo[step], p90=hi[step]))
            if (i - start) % 10 == 0 or i == end - 1:
                print(f'{stage}: {i - start + 1}/{end - start} orígenes, {time.perf_counter() - began:.1f}s', flush=True)
    pred = pd.concat([pd.DataFrame(rows), benchmark_forecasts(y)], ignore_index=True)
    assert pred[['real', 'pronostico']].notna().all().all() and pred.pronostico.ge(0).all()
    assert not pred.duplicated(['etapa', 'origen', 'modelo', 'horizonte']).any()
    daily = pred[pred.horizonte == 1].copy()
    cumulative = pred.groupby(['etapa', 'modelo', 'origen'], as_index=False).agg(
        dias=('horizonte', 'size'), real=('real', 'sum'), pronostico=('pronostico', 'sum'))
    cumulative = cumulative[cumulative.dias == 30]
    metric_rows = []
    for horizon, frame in [(1, daily), (30, cumulative)]:
        for (stage, name), group in frame.groupby(['etapa', 'modelo']):
            expected = (55 if stage == 'validacion' else 54) if horizon == 1 else (26 if stage == 'validacion' else 25)
            assert len(group) == expected
            metric_rows.append(dict(horizonte_dias=horizon, etapa=stage, modelo=name,
                                    **metric_values(group.real, group.pronostico)))
    all_metrics = pd.DataFrame(metric_rows)
    save_csv(pred, OUT / 'pronosticos_por_origen.csv')
    save_csv(daily, OUT / 'pronostico_demanda_diaria.csv')
    save_csv(cumulative, OUT / 'demanda_acumulada_30_dias.csv')
    save_csv(all_metrics, OUT / 'comparacion_modelos.csv')
    for h, filename in [(1, 'metricas_diarias.csv'), (30, 'metricas_30_dias.csv')]:
        save_csv(all_metrics[all_metrics.horizonte_dias == h], OUT / filename)
    status_df = pd.DataFrame(status)
    eligible = status_df[status_df.etapa == 'validacion'].groupby('modelo').convergio.all()
    selections = {}
    for group_name, candidates in [('cuatro_originales', MODELS), ('nueve_exploratorios', list(pred.modelo.unique()))]:
        allowed = [n for n in candidates if n not in eligible.index or eligible[n]]
        selections[group_name] = {}
        for horizon in [1, 30]:
            selected = all_metrics[(all_metrics.etapa == 'validacion') & (all_metrics.horizonte_dias == horizon) & all_metrics.modelo.isin(allowed)].sort_values(['MAE', 'RMSE', 'modelo']).iloc[0]
            selections[group_name][str(horizon)] = selected.modelo
    write_json('seleccion_modelos.json', selections)

    futures = []
    future_dates = pd.date_range(y.index[-1] + pd.Timedelta(days=1), periods=30)
    for name in MODELS:
        if name == 'TimesFM 3.0':
            point, lo, hi = foundation_forecasts(model, [y])[0]
        else:
            point, ok, message = forecast_classic(name, y, future_dates)
            lo = hi = np.full(30, np.nan)
            status.append(dict(etapa='futuro', origen=y.index[-1], modelo=name, convergio=ok, advertencias=message, segundos=np.nan))
        for i, day in enumerate(future_dates):
            futures.append(dict(origen=y.index[-1], fecha=day, modelo=name, pronostico=point[i], p10=lo[i], p90=hi[i], sin_ventas_observadas=True))
    save_csv(pd.DataFrame(futures), OUT / 'pronostico_30_dias_desde_ultimo_dato.csv')
    save_csv(pd.DataFrame(status), OUT / 'diagnostico_ajustes.csv')

    test = daily[daily.etapa == 'prueba'].copy()
    test['tramo'] = test.fecha.map(d.evento)
    segmented = pd.DataFrame([dict(modelo=name, tramo=segment, **metric_values(g.real, g.pronostico))
        for (name, segment), g in test.groupby(['modelo', 'tramo'])])
    save_csv(segmented, OUT / 'errores_por_ventana.csv')
    tf = test[test.modelo == 'TimesFM 3.0']
    coverage = dict(modelo='TimesFM 3.0', n=len(tf), cubiertos=int(tf.real.between(tf.p10, tf.p90).sum()),
        nominal=.8, cobertura_observada=float(tf.real.between(tf.p10, tf.p90).mean()), ancho_medio=float((tf.p90 - tf.p10).mean()))
    save_csv(pd.DataFrame([coverage]), OUT / 'cobertura_intervalos.csv')
    shares = []
    for name, g in test.groupby('modelo'):
        err = (g.pronostico - g.real).abs()
        event = g.tramo.ne('Fuera de ventanas')
        shares.append(dict(modelo=name, error_absoluto_total=float(err.sum()),
            error_en_ventanas=float(err[event].sum()), fraccion_error_ventanas=float(err[event].sum() / err.sum())))
    save_csv(pd.DataFrame(shares), OUT / 'concentracion_error.csv')
    manifest = dict(fecha_revision='2026-10-20', python=platform.python_version(),
        paquetes={n: importlib.metadata.version(n) for n in ['numpy', 'pandas', 'scipy', 'statsmodels', 'timesfm', 'torch']},
        hashes={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in [ORDERS, SERIES, STOCK, Path(__file__), ROOT / 'scripts/modelos_noirpress_avanzados.py', ROOT / 'scripts/auditoria_eda_noirpress.py']},
        modelo=MODEL_ID, revision=REVISION, device='cpu', pesos_solo_cache=True,
        licencia_pesos='timesfm-non-commercial-license-v1.0; uso comercial excluido',
        alcance='Reevaluación retrospectiva de 405 unidades estimadas; partición de prueba previamente conocida; sin nueva prueba prospectiva.',
        unidades_estimadas=405, pedidos=391, dias=179, historial=70, validacion=55, prueba=54,
        ventanas30_validacion=26, ventanas30_prueba=25, seleccion=selections,
        modelos_no_convergentes=status_df.loc[~status_df.convergio, ['etapa', 'origen', 'modelo', 'advertencias']].to_dict('records'),
        inventario_inicial_supuesto=initial, ingresos_periodo=int(flow.sum()), stock_inicial_minimo_contable=min_initial,
        saldo_minimo=float(d.saldo_inicial_98.min()), fecha_saldo_minimo=str(d.saldo_inicial_98.idxmin().date()), saldo_final=float(d.saldo_inicial_98.iloc[-1]),
        dias_sin_ordenes=int(d.pedidos.eq(0).sum()), cobertura=coverage,
        duracion_segundos=time.perf_counter() - began,
        limitaciones=['Cantidades inferidas, no líneas de producto verificadas.', 'Stock inicial contable, no conteo físico.',
            'Lead time fijo de30 días informado, no distribución observada.', 'Ceros de calendario bajo supuesto de cobertura.',
            'Ventanas acumuladas solapadas.', 'Cinco referencias añadidas tras conocer el período.',
            'TimesFM publicado después del período de ventas; uso retrospectivo y licencia no comercial.'])
    write_json('resumen_final.json', manifest)
    print(all_metrics[['horizonte_dias', 'etapa', 'modelo', 'MAE', 'RMSE']].to_string(index=False), flush=True)
    print(json.dumps(selections, ensure_ascii=False), flush=True)
    print(f'Finalizado en {time.perf_counter() - began:.1f}s. Sin fuentes modificadas.', flush=True)


if __name__ == '__main__':
    main()
