"""Estimación causal, referencias simples y escenarios de reposición."""
import numpy as np
import pandas as pd


def estimate(orders, window=30, by_channel=False):
    """Recalcula la regla publicada independientemente y la contrasta con su auditoría."""
    refs = pd.Series(np.nan, index=orders.index, dtype=float)
    qty = pd.Series(0, index=orders.index, dtype=int)
    for day, group in orders.groupby('Fecha', sort=True):
        groups = group.groupby('Origen') if by_channel else [('todos', group)]
        for channel, g in groups:
            mask = (orders.Fecha < day) & (orders['Total orden'] > 0)
            if by_channel:
                mask &= orders.Origen.eq(channel)
            past = orders.loc[mask & (orders.Fecha >= day-pd.Timedelta(days=window)), 'Total orden']
            if past.empty:
                past = orders.loc[mask, 'Total orden']
            ref = float(past.median()) if len(past) else np.nan
            refs.loc[g.index] = ref
            q = np.maximum(1, np.floor(np.round(g['Total orden']/ref, 2)+.5)) if np.isfinite(ref) else np.ones(len(g))
            qty.loc[g.index] = np.where(g['Total orden'] > 0, q, 0).astype(int)
    return qty, refs


def benchmark_forecasts(y):
    rows = []
    split = int(len(y)*.7)
    for stage, start, end in [('validacion', 70, split), ('prueba', split, len(y))]:
        for i in range(start, end):
            h = y.iloc[:i]
            candidates = {
                'Ingenuo último día': np.repeat(h.iloc[-1], 30),
                'Ingenuo semanal': np.resize(h.iloc[-7:].to_numpy(), 30),
                'Media 28 días': np.repeat(h.iloc[-28:].mean(), 30),
                'Mediana 28 días': np.repeat(h.iloc[-28:].median(), 30),
                'Media histórica': np.repeat(h.mean(), 30),
            }
            for name, pred in candidates.items():
                for step in range(min(30, end-i)):
                    rows.append(dict(etapa=stage, modelo=name, origen=h.index[-1],
                                     fecha=y.index[i+step], horizonte=step+1,
                                     real=y.iloc[i+step], pronostico=pred[step]))
    return pd.DataFrame(rows)


def stats_row(label, v):
    v = pd.Series(v)
    pos = v[v>0]
    return {'Grupo': label, 'Días': len(v), 'Total': float(v.sum()), 'Media': v.mean(),
            'Mediana': v.median(), 'Desvío': v.std(), 'Ceros': int(v.eq(0).sum()),
            'Varianza/media': v.var()/v.mean(), 'ADI': len(v)/len(pos),
            'CV² positivos': (pos.std()/pos.mean())**2}


def simulate_order(demand, initial_stock, issued, quantity, lead_days):
    """Escenario de ventas perdidas: recepción al inicio del día; emisión tras su cierre.

    La trayectoria observada se usa sólo para puntuar el escenario; nunca para decidir cantidad.
    """
    issued=pd.Timestamp(issued)
    arrival=issued+pd.Timedelta(days=lead_days)
    stock=float(initial_stock)
    rows=[]
    for day,value in demand.items():
        received=quantity if day==arrival else 0
        stock+=received
        served=min(stock,float(value))
        lost=float(value)-served
        stock-=served
        rows.append({'fecha':day,'emision':issued,'llegada':arrival,'pedido':quantity,
                     'recepcion':received,'demanda_proxy':value,'atendido':served,
                     'faltante_simulado':lost,'stock_cierre':stock,'antes_de_llegada':day<arrival})
    result=pd.DataFrame(rows)
    assert np.isclose(result.demanda_proxy.sum(),result.atendido.sum()+result.faltante_simulado.sum())
    assert np.isclose(initial_stock+result.recepcion.sum(),result.atendido.sum()+result.stock_cierre.iloc[-1])
    assert result.stock_cierre.ge(0).all()
    return result
