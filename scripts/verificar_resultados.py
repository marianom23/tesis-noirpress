"""Verifica fuentes, predicciones guardadas y simulaciones, sin reentrenar modelos."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / 'resultados_noirpress/evaluacion_final'


def verify():
    manifest = json.loads((R / 'resumen_final.json').read_text(encoding='utf-8'))
    for path, expected in manifest['hashes'].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected, path
    raw = pd.read_excel(ROOT / 'fuentes/ultimas_ordenes_2026-07-06.xlsx', sheet_name='Ordenes')
    orders = pd.read_csv(ROOT / 'resultados_noirpress/cierre/ordenes_cierre.csv', parse_dates=['Fecha'])
    excluded = pd.read_csv(ROOT / 'resultados_noirpress/cierre/ordenes_excluidas.csv')
    keys = ['Origen', 'N° Orden']
    keyset = lambda df: set(df[keys].astype(str).itertuples(index=False, name=None))
    assert len(raw) == 409 and len(orders) == 391 and len(excluded) == 18
    assert keyset(raw) == keyset(orders) | keyset(excluded)
    assert not keyset(orders) & keyset(excluded)
    assert raw['Total orden'].eq(0).sum() == 10
    assert (raw['Total orden'].gt(0) & raw['Costo de productos'].lt(20000)).sum() == 8
    for day, group in orders.groupby('Fecha'):
        past = orders[(orders.Fecha < day) & orders['Total orden'].gt(0)]
        recent = past[past.Fecha >= day-pd.Timedelta(days=30)]
        ref = (recent if len(recent) else past)['Total orden'].median()
        quantities = np.maximum(1, np.floor(np.round(group['Total orden']/ref, 2)+.5)) if pd.notna(ref) else np.ones(len(group))
        np.testing.assert_array_equal(quantities, group['Unidades estimadas cierre'])
    d = pd.read_csv(R / 'serie_diaria.csv', parse_dates=['fecha']).set_index('fecha')
    pd.testing.assert_index_equal(d.index, pd.date_range('2026-01-03', '2026-06-30', name='fecha'), check_names=True)
    np.testing.assert_array_equal(d.demanda, orders.groupby('Fecha')['Unidades estimadas cierre'].sum().reindex(d.index, fill_value=0))
    assert d.demanda.sum() == 405 and d.pedidos.sum() == 391 and d.pedidos.eq(0).sum() == 34
    assert orders.groupby('Origen').size().to_dict() == {'Mercado Libre': 177, 'TiendaNube': 214}
    assert d.demanda.nlargest(10).sum() == 100
    assert d.loc['2026-05-11':'2026-05-17', 'demanda'].sum() == 59
    assert d.loc['2026-06-14':'2026-06-17', 'demanda'].sum() == 33
    monthly = pd.read_csv(R / 'resumen_mensual.csv').set_index('mes')
    for month, g in d.groupby(d.index.strftime('%Y-%m')):
        row = monthly.loc[month]
        np.testing.assert_allclose([len(g), g.demanda.sum(), g.pedidos.sum(), g.pedidos.eq(0).sum(), g.demanda.mean()],
                                   row[['dias', 'unidades', 'pedidos', 'dias_sin_pedidos', 'media']].astype(float))

    receipts = pd.read_excel(ROOT / 'fuentes/control_pedidos_ventas_quiebres_noirpress.xlsx', sheet_name='Ingresos', skiprows=3).dropna(subset=['ID Lote'])
    receipts['Fecha ingreso'] = pd.to_datetime(receipts['Fecha ingreso'])
    initial = receipts.loc[receipts['Fecha ingreso'] < d.index[0], 'Unidades'].sum()
    flow = receipts.groupby('Fecha ingreso').Unidades.sum().reindex(d.index, fill_value=0)
    np.testing.assert_array_equal(flow, d.ingresos_unidades)
    balance = initial + (flow-d.demanda).cumsum()
    np.testing.assert_allclose(balance, d.saldo_inicial_98)
    assert initial == 98 and flow.sum() == 420 and balance.iloc[-1] == 113 and balance.min() == 68
    assert balance.idxmin() == pd.Timestamp('2026-05-25')
    assert -(flow-d.demanda).cumsum().min() == 30
    for _, row in pd.read_csv(R / 'sensibilidad_inventario.csv').iterrows():
        b = row['Stock inicial supuesto'] + (flow-d.demanda).cumsum()
        np.testing.assert_allclose([b.min(), b.lt(0).sum(), b.iloc[-1]],
                                   [row['Mínimo contable'], row['Días saldo negativo'], row['Saldo 30 junio']])

    p = pd.read_csv(R / 'pronosticos_por_origen.csv', parse_dates=['origen', 'fecha'])
    assert p.modelo.nunique() == 9 and not p.duplicated(['etapa', 'modelo', 'origen', 'horizonte']).any()
    np.testing.assert_array_equal((p.fecha-p.origen).dt.days, p.horizonte)
    np.testing.assert_array_equal(p.real, p.fecha.map(d.demanda))
    assert p.pronostico.ge(0).all()
    daily = p[p.horizonte == 1]
    totals = p.groupby(['etapa', 'modelo', 'origen']).agg(dias=('horizonte', 'size'), real=('real', 'sum'), pronostico=('pronostico', 'sum')).reset_index()
    totals = totals[totals.dias == 30]
    metrics = pd.read_csv(R / 'comparacion_modelos.csv').set_index(['horizonte_dias', 'etapa', 'modelo'])
    assert len(metrics) == 36
    for horizon, frame in [(1, daily), (30, totals)]:
        for (stage, model), group in frame.groupby(['etapa', 'modelo']):
            e = group.pronostico-group.real
            expected = metrics.loc[(horizon, stage, model)]
            np.testing.assert_allclose([e.abs().mean(), np.sqrt((e**2).mean()), e.mean(), 100*e.abs().sum()/group.real.sum(), len(e)],
                                       expected[['MAE', 'RMSE', 'sesgo_pred_menos_real', 'WAPE_pct', 'n']], rtol=1e-9)
            assert len(e) == ({'validacion': 55, 'prueba': 54} if horizon == 1 else {'validacion': 26, 'prueba': 25})[stage]
    for horizon in [1, 30]:
        assert metrics.loc[(horizon, 'validacion')].MAE.idxmin() == 'Mediana 28 días'
        assert metrics.loc[(horizon, 'prueba')].MAE.idxmin() == ('SARIMAX' if horizon == 1 else 'Media 28 días')
    test = daily[daily.etapa == 'prueba']
    tf = test[test.modelo == 'TimesFM 3.0']
    assert tf.real.between(tf.p10, tf.p90).sum() == 37
    np.testing.assert_allclose((tf.p90-tf.p10).mean(), manifest['cobertura']['ancho_medio'])
    for _, row in pd.read_csv(R / 'concentracion_error.csv').iterrows():
        g = test[test.modelo == row.modelo]
        err = (g.pronostico-g.real).abs()
        event = g.fecha.map(d.evento).ne('Fuera de ventanas')
        np.testing.assert_allclose(err[event].sum()/err.sum(), row.fraccion_error_ventanas)

    trajectories = pd.read_csv(R / 'reposicion_dia_a_dia.csv', parse_dates=['fecha', 'emision', 'llegada'])
    summary = pd.read_csv(R / 'reposicion_resumen.csv', parse_dates=['Emisión', 'Llegada'])
    for _, row in summary.iterrows():
        issued, arrival = row['Emisión'], row['Llegada']
        g = trajectories[(trajectories.emision == issued) & (trajectories.metodo == row['Método'])].sort_values('fecha')
        history = d.loc[:issued, 'demanda']
        rate = {'Media 7 días': history.iloc[-7:].mean(), 'Media 28 días': history.iloc[-28:].mean(), 'Mediana 28 días': history.iloc[-28:].median()}[row['Método']]
        stock = int(np.floor(.3*balance.loc[issued]))
        assert stock == row['Stock inicial simulado']
        assert row['Pedido calculado'] == np.ceil(max(0, 60*rate-stock))
        assert arrival == issued+pd.Timedelta(days=30)
        assert g.fecha.min() == issued+pd.Timedelta(days=1)
        assert g.fecha.max() == min(issued+pd.Timedelta(days=60), d.index[-1])
        for _, day in g.iterrows():
            receipt = row['Pedido calculado'] if day.fecha == arrival else 0
            stock += receipt
            demand = d.loc[day.fecha, 'demanda']
            served = min(stock, demand)
            stock -= served
            np.testing.assert_allclose([receipt, demand, served, demand-served, stock],
                                       day[['recepcion', 'demanda_proxy', 'atendido', 'faltante_simulado', 'stock_cierre']].astype(float))
        before = g.fecha < arrival
        assert before.equals(g.antes_de_llegada)
        np.testing.assert_allclose([g.loc[before, 'faltante_simulado'].sum(), g.loc[~before, 'faltante_simulado'].sum(), g.stock_cierre.mean(), 100*g.atendido.sum()/g.demanda_proxy.sum()],
                                   row[['Faltantes antes llegada', 'Faltantes después llegada', 'Stock medio', 'Atendido_pct']].astype(float))
        assert before.sum() == 29 and (~before).sum() == (31 if issued.month == 4 else 0)
    print('OK: fuentes, depuración, serie, meses, inventario, 36 combinaciones de métricas, cobertura y seis escenarios.')
    print('Convención comprobada: 29 días antes y 31 desde la recepción en el escenario completo de 60 días.')


if __name__ == '__main__':
    verify()
