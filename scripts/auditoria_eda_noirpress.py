"""Auditoría exploratoria de Noirpress desde las fuentes, sin alterar resultados previos.

Ejecutar desde cualquier directorio: python scripts/auditoria_eda_noirpress.py
La comparación adicional es retrospectiva: no constituye una nueva prueba reservada.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import platform
from html import escape
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from limpiar_noirpress import limpiar_ordenes, leer_ingresos

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'resultados_noirpress'
OUT = RESULTS / 'auditoria_eda'
DATE = '2026-10-07'
SALES = ROOT / 'fuentes/ultimas_ordenes_2026-07-06.xlsx'
STOCK = ROOT / 'fuentes/control_pedidos_ventas_quiebres_noirpress.xlsx'
DAYS = ['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom']


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


def metrics(g):
    e = g.pronostico.to_numpy()-g.real.to_numpy()
    return dict(n=len(g), MAE=float(np.abs(e).mean()), RMSE=float(np.sqrt(np.mean(e**2))),
                sesgo=float(e.mean()), WAPE_pct=float(100*np.abs(e).sum()/g.real.sum()))


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


def fmt(value, decimals=2):
    return f'{float(value):,.{decimals}f}'.replace(',', 'X').replace('.', ',').replace('X', '.')


def table(df):
    return '<div class="table">'+df.to_html(index=False, border=0, escape=True,
        float_format=lambda v: fmt(v, 3), na_rep='—')+'</div>'


def picture(fig, alt):
    fig.tight_layout()
    b = io.BytesIO()
    fig.savefig(b, format='png', dpi=150, bbox_inches='tight')
    folder=OUT/'figuras'
    folder.mkdir(exist_ok=True)
    filename={'Pedidos diarios, unidades aproximadas y canales':'01_serie_canales.png',
              'Patrón semanal condicionado por las ventanas y por el tramo':'02_semana.png',
              'Comparación con referencias adicionales en la misma prueba retrospectiva':'03_referencias.png',
              'Saldos reconstruidos con diferentes existencias iniciales':'04_stock.png'}[alt]
    (folder/filename).write_bytes(b.getvalue())
    plt.close(fig)
    return '<figure><img alt="'+escape(alt)+'" src="data:image/png;base64,'+base64.b64encode(b.getvalue()).decode()+'"></figure>'


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


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_excel(SALES, sheet_name='Ordenes')
    cost_cols=['Costo de productos','Costo de envío Total','Costo envío privado (Flex)','Comisiones de pago','Comisión de tienda']
    net_simple=raw['Total orden']-raw[cost_cols].apply(pd.to_numeric,errors='raise').fillna(0).sum(axis=1)
    net_mismatches=int((net_simple-raw.Neto).abs().gt(.05).sum())
    o = limpiar_ordenes(SALES)
    assert not o.duplicated(['Origen', 'N° Orden']).any()
    assert o['Fecha'].notna().all() and o['Total orden'].ge(0).all()
    o['unidades_causales'], o['referencia_pasada'] = estimate(o)
    o['bajo_costo_a_verificar'] = (o['Costo de productos'] < 20000) & o['Total orden'].gt(0)
    o['ratio_pasado'] = o['Total orden']/o.referencia_pasada
    o['atipico_ref_pasada'] = o['Total orden'].gt(0) & ~o.ratio_pasado.between(.6, 1.4) & o.referencia_pasada.notna()
    positive = o[o['Total orden']>0]
    idx = pd.date_range(o.Fecha.min(), o.Fecha.max(), freq='D', name='fecha')
    d = pd.DataFrame(index=idx)
    for col, series in {
        'pedidos_positivos': positive.groupby('Fecha').size(),
        'unidades_estimadas': o.groupby('Fecha').unidades_causales.sum(),
        'unidades_ref_mensual': o.groupby('Fecha')['Cantidad estimada'].sum(),
        'importe_ars': positive.groupby('Fecha')['Total orden'].sum(),
        'ordenes_sin_cargo': o[o['Total orden']==0].groupby('Fecha').size(),
        'ordenes_bajo_costo': o[o.bajo_costo_a_verificar].groupby('Fecha').size(),
        'alerta_importe_pasada': o[o.atipico_ref_pasada].groupby('Fecha').size(),
    }.items():
        d[col] = series.reindex(idx, fill_value=0)
    for c, g in positive.groupby('Origen'):
        d[c] = g.groupby('Fecha').size().reindex(idx, fill_value=0)
    d['mes'] = idx.strftime('%Y-%m')
    d['dia_semana'] = [DAYS[n] for n in idx.dayofweek]
    d['dow'] = idx.dayofweek
    d['semana'] = idx.to_period('W-SUN').astype(str)
    d['ventana'] = 'Fuera de ventanas'
    d.loc['2026-05-11':'2026-05-17', 'ventana'] = 'Ventana mayo'
    d.loc['2026-06-14':'2026-06-17', 'ventana'] = 'Ventana junio'
    d['etapa'] = ['historial inicial']*70+['validacion']*55+['prueba']*54
    for w in [7, 14, 28, 30]:
        d[f'media_pasada_{w}'] = d.unidades_estimadas.shift(1).rolling(w, min_periods=w).mean()
    d['desvio_pasado_28'] = d.unidades_estimadas.shift(1).rolling(28).std()
    d['pico_vs_pasado_28'] = d.unidades_estimadas > d.media_pasada_28 + 3*d.desvio_pasado_28
    d['sin_pedidos'] = d.pedidos_positivos.eq(0)
    d['racha_sin_pedidos'] = d.sin_pedidos.groupby((~d.sin_pedidos).cumsum()).cumsum()
    d['cero_sin_estado_operativo'] = d.sin_pedidos
    d['lectura'] = [
        ('Sin pedidos positivos registrados; verificar actividad de tienda y cobertura del export.' if r.sin_pedidos else
         'Contiene órdenes de bajo costo; confirmar producto y cantidad.' if r.ordenes_bajo_costo else
         'Pico frente a 28 días anteriores; investigar campaña, canal y pedidos.' if r.pico_vs_pasado_28 else
         'Dentro de ventana exploratoria; asociación comercial por confirmar.' if r.ventana != 'Fuera de ventanas' else
         'Pedidos positivos registrados; unidades inferidas, sin validación física.') for r in d.itertuples()]

    # Reconciliar fuentes, cálculo independiente, Excel derivado y resultados publicados.
    cached = pd.read_csv(RESULTS/'eda_dia_a_dia.csv', parse_dates=['fecha']).set_index('fecha')
    np.testing.assert_array_equal(d.unidades_estimadas, cached.demanda)
    np.testing.assert_array_equal(d.pedidos_positivos, cached.pedidos)
    clean = pd.read_excel(ROOT/'Ordenes_Noirpress_limpio.xlsx', sheet_name='Ordenes_limpio')
    assert len(raw)==len(clean)==len(o)==409 and len(positive)==399
    for frame in [clean, o]:
        frame['clave'] = frame.Origen.astype(str)+'|'+frame['N° Orden'].astype(str)
    for col in raw.columns:
        a, b = o.set_index('clave')[col].sort_index(), clean.set_index('clave')[col].sort_index()
        if col=='Fecha creación':
            np.testing.assert_array_equal(pd.to_datetime(a), pd.to_datetime(b))
        else:
            pd.testing.assert_series_equal(a, b, check_dtype=False, check_names=False)
    manifest = json.loads((RESULTS/'reproducibilidad.json').read_text(encoding='utf-8'))
    assert hashlib.sha256((ROOT/'Ordenes_Noirpress_limpio.xlsx').read_bytes()).hexdigest()==manifest['sha256_entrada']
    script_hash=hashlib.sha256((ROOT/'scripts/modelos_noirpress_avanzados.py').read_bytes()).hexdigest()
    if script_hash != manifest['sha256_script']:
        editorial=json.loads((RESULTS/'revision_editorial.json').read_text(encoding='utf-8'))
        assert editorial['sha256_script_original']==manifest['sha256_script']
        assert script_hash==editorial['sha256_script_actual']
    # Comprobar que cambios futuros no alteran la reconstrucción del pasado.
    changed=o.copy(); cutoff=idx[100]; changed.loc[changed.Fecha>=cutoff,'Total orden'] *= 2
    np.testing.assert_array_equal(estimate(changed)[0][o.Fecha<cutoff], o.loc[o.Fecha<cutoff,'unidades_causales'])

    sensitivity=[]
    for w,c in [(7,False),(14,False),(30,False),(60,False),(30,True)]:
        qty,_=estimate(o,w,c); daily=qty.groupby(o.Fecha).sum().reindex(idx,fill_value=0)
        sensitivity.append({'Referencia': f'Mediana pasada {w} días'+(' por canal' if c else ' conjunta'),
            'Unidades estimadas': int(qty.sum()), 'Órdenes que cambian vs 30 días': int(qty.ne(o.unidades_causales).sum()),
            'Días que cambian': int(daily.ne(d.unidades_estimadas).sum())})
    sensitivity=pd.DataFrame(sensitivity)
    sans=o[~o.bajo_costo_a_verificar].copy(); sans['q'],_=estimate(sans)
    sensitivity=pd.concat([sensitivity,pd.DataFrame([{'Referencia':'Excluir 8 órdenes de bajo costo y recalcular 30 días (sólo sensibilidad)',
        'Unidades estimadas':int(sans.q.sum()),'Órdenes que cambian vs 30 días':np.nan,'Días que cambian':np.nan}])],ignore_index=True)
    monthly=d.groupby('mes').agg(dias=('pedidos_positivos','size'),pedidos=('pedidos_positivos','sum'),
        unidades=('unidades_estimadas','sum'),media_diaria=('pedidos_positivos','mean'),
        ceros=('sin_pedidos','sum'),TiendaNube=('TiendaNube','sum'),Mercado_Libre=('Mercado Libre','sum'),
        importes_ARS=('importe_ars','sum')).reset_index()
    monthly['proporcion_TiendaNube_pct']=100*monthly.TiendaNube/monthly.pedidos
    dist=pd.DataFrame([stats_row(name,v) for name,v in [('Pedidos positivos',d.pedidos_positivos),('Unidades estimadas',d.unidades_estimadas)]])
    stages=pd.DataFrame([stats_row(stage,g.unidades_estimadas) for stage,g in d.groupby('etapa',sort=False)])
    channels=positive.groupby('Origen').agg(pedidos=('Total orden','size'),primer_registro=('Fecha','min'),
        ultimo_registro=('Fecha','max'),ticket_mediano_ARS=('Total orden','median')).reset_index()
    weekly=[]
    for label,mask in [('Todos',pd.Series(True,index=idx)),('Fuera de ventanas',d.ventana.eq('Fuera de ventanas')),
                       ('Enero a abril',idx<pd.Timestamp('2026-05-01')),('Mayo a junio',idx>=pd.Timestamp('2026-05-01'))]:
        g=d.loc[mask].groupby('dow').unidades_estimadas.agg(['size','mean','median'])
        for k,r in g.iterrows():weekly.append({'Tramo':label,'Día':DAYS[k],'n':int(r['size']),'Media':r['mean'],'Mediana':r['median']})
    weekly=pd.DataFrame(weekly)
    # Conserva la distancia temporal: no concatenar días después de excluir eventos.
    autocorr=[]
    for lag in [1,7,14,28]:
        pair=pd.DataFrame({'y':d.unidades_estimadas,'pasado':d.unidades_estimadas.shift(lag)})
        normal=d.ventana.eq('Fuera de ventanas')
        for label,mask in [('Todos',pair.notna().all(axis=1)),('Ambas fechas fuera de ventanas',normal & normal.shift(lag,fill_value=False))]:
            g=pair.loc[mask].dropna();autocorr.append({'Ámbito':label,'Rezago días':lag,'Pares':len(g),'Correlación':g.y.corr(g.pasado)})
    autocorr=pd.DataFrame(autocorr)
    events=[]
    for label,start,end in [('Mayo','2026-05-11','2026-05-17'),('Junio','2026-06-14','2026-06-17')]:
        a,b=pd.Timestamp(start),pd.Timestamp(end);g=d.loc[a:b]
        for refname,ref in [('Todos los días fuera de ventanas',d[d.ventana.eq('Fuera de ventanas')]),
                            ('28 días previos',d.loc[a-pd.Timedelta(days=28):a-pd.Timedelta(days=1)])]:
            events.append({'Ventana':label,'Inicio':start,'Fin':end,'Días evento':len(g),
                'Unidades':int(g.unidades_estimadas.sum()),'Media evento':g.unidades_estimadas.mean(),
                'Referencia':refname,'Media referencia':ref.unidades_estimadas.mean(),
                'Cociente descriptivo':g.unidades_estimadas.mean()/ref.unidades_estimadas.mean()})
    events=pd.DataFrame(events)

    cached_pred=pd.read_csv(RESULTS/'pronosticos_por_origen.csv',parse_dates=['origen','fecha'])
    assert cached_pred.origen.lt(cached_pred.fecha).all()
    np.testing.assert_array_equal(cached_pred.real,d.unidades_estimadas.reindex(cached_pred.fecha))
    extra=benchmark_forecasts(d.unidades_estimadas)
    combined=pd.concat([cached_pred[['etapa','modelo','origen','fecha','horizonte','real','pronostico']],extra],ignore_index=True)
    assert not combined.duplicated(['etapa','modelo','origen','horizonte']).any()
    assert np.isfinite(combined.pronostico).all()
    assert combined.fecha.le(pd.Timestamp('2026-05-07')).equals(combined.etapa.eq('validacion'))
    metric_rows=[]
    for horizon in [1,30]:
        if horizon==1: scored=combined[combined.horizonte.eq(1)]
        else:
            scored=combined.groupby(['etapa','modelo','origen'],as_index=False).agg(
                dias=('horizonte','size'),real=('real','sum'),pronostico=('pronostico','sum'))
            scored=scored[scored.dias.eq(30)]
        for (stage,name),g in scored.groupby(['etapa','modelo']):
            metric_rows.append({'etapa':stage,'modelo':name,'horizonte_dias':horizon,
                                'procedencia':'ejecución previa verificada' if name in cached_pred.modelo.unique() else 'referencia adicional posterior',**metrics(g)})
    metric=pd.DataFrame(metric_rows)
    for horizon,file in [(1,'metricas_diarias.csv'),(30,'metricas_30_dias.csv')]:
        old=pd.read_csv(RESULTS/file)
        check=metric[metric.horizonte_dias.eq(horizon)].merge(old,on=['etapa','modelo'],suffixes=('_nuevo','_anterior'))
        np.testing.assert_allclose(check.MAE_nuevo,check.MAE_anterior,rtol=1e-10)
    ranking=metric.pivot(index='modelo',columns=['horizonte_dias','etapa'],values='MAE')
    ranking=ranking.reindex(columns=[(1,'validacion'),(1,'prueba'),(30,'validacion'),(30,'prueba')])
    ranking.columns=['MAE diario validación','MAE diario prueba','MAE 30 días validación','MAE 30 días prueba']
    ranking=ranking.sort_values('MAE 30 días validación').reset_index()
    # Sensibilidad del error a ventanas seleccionadas ex post; no se vuelve a entrenar.
    p=combined[(combined.etapa=='prueba') & (combined.horizonte==1)].copy()
    p['ventana']=d.ventana.reindex(p.fecha).to_numpy()
    segment=[]
    for (name,ev),g in p.groupby(['modelo','ventana']):
        segment.append({'modelo':name,'tramo':ev,**metrics(g)})
    segment=pd.DataFrame(segment)
    # Contraste pareado exploratorio: bloques contiguos de siete días.
    wide=p.pivot(index='fecha',columns='modelo',values='pronostico')
    truth=d.unidades_estimadas.reindex(wide.index)
    errors=wide.sub(truth,axis=0).abs();n=len(errors)
    rng=np.random.default_rng(20261006)
    blocks=np.array([np.concatenate([np.arange(s,s+7) for s in rng.integers(0,n-6,size=int(np.ceil(n/7)))])[:n] for _ in range(3000)])
    uncertainty=[]
    for a,b in [('SARIMAX','Media 28 días'),('TimesFM 3.0','Mediana 28 días'),('SARIMAX','TimesFM 3.0')]:
        delta=(errors[a]-errors[b]).to_numpy();lo,hi=np.quantile(delta[blocks].mean(axis=1),[.025,.975])
        uncertainty.append({'Comparación A menos B':a+' menos '+b,'Diferencia MAE':delta.mean(),'Percentil 2,5':lo,'Percentil 97,5':hi})
    uncertainty=pd.DataFrame(uncertainty)
    ing=leer_ingresos(STOCK)
    flow=ing.groupby('Fecha ingreso').Unidades.sum().reindex(idx,fill_value=0)
    initial=int(ing.loc[ing['Fecha ingreso']<idx[0],'Unidades'].sum())
    d['ingresos_unidades']=flow
    d['saldo_supone_inicial_98']=initial+flow.cumsum()-d.unidades_estimadas.cumsum()
    stock_rows=[]
    for q0 in [0,25,32,50,96,98]:
        balance=q0+flow.cumsum()-d.unidades_estimadas.cumsum()
        stock_rows.append({'Stock inicial supuesto':q0,'Mínimo contable':balance.min(),
            'Fecha mínimo':str(balance.idxmin().date()),'Días saldo negativo':int(balance.lt(0).sum()),
            'Días saldo cero':int(balance.eq(0).sum()),'Saldo 30 junio':balance.iloc[-1]})
    stock_sensitivity=pd.DataFrame(stock_rows)
    # Comparación causal de cantidades en dos decisiones, bajo estrés definido y sin otros pedidos.
    decisions=[];trajectories=[]
    for issued in [pd.Timestamp('2026-04-27'),pd.Timestamp('2026-06-01')]:
        history=d.loc[:issued,'unidades_estimadas']
        start_stock=int(np.floor(.30*d.loc[issued,'saldo_supone_inicial_98']))
        future=d.loc[issued+pd.Timedelta(days=1):min(issued+pd.Timedelta(days=60),idx[-1]),'unidades_estimadas']
        for name,rate in [('Media 7 días',history.iloc[-7:].mean()),('Media 28 días',history.iloc[-28:].mean()),('Mediana 28 días',history.iloc[-28:].median())]:
            # Posición objetivo para lead time 30 + cobertura adicional 30; sin compras en tránsito.
            quantity=int(np.ceil(max(0,rate*60-start_stock)))
            sim=simulate_order(future,start_stock,issued,quantity,30)
            sim['metodo']=name;trajectories.append(sim)
            decisions.append({'Emisión':str(issued.date()),'Llegada':str((issued+pd.Timedelta(days=30)).date()),
                'Método':name,'Stock inicial simulado':start_stock,'Pedido calculado':quantity,'Días puntuados':len(sim),
                'Faltantes antes llegada':sim.loc[sim.antes_de_llegada,'faltante_simulado'].sum(),
                'Faltantes después llegada':sim.loc[~sim.antes_de_llegada,'faltante_simulado'].sum(),
                'Stock medio':sim.stock_cierre.mean(),'Atendido_pct':100*sim.atendido.sum()/sim.demanda_proxy.sum()})
    decisions=pd.DataFrame(decisions)
    # Un pedido emitido en A y recibido en A+L no puede atender consumo anterior a A+L.
    a=simulate_order(pd.Series([2.,2.,2.],index=pd.date_range('2026-01-02',periods=3)),1,'2026-01-01',10,3)
    assert a.loc[a.antes_de_llegada,'faltante_simulado'].sum()==3
    end=idx[-1]
    low=o[o.bajo_costo_a_verificar]
    claims=pd.DataFrame([
        ['Alta','4.2 y 4.4','Único producto; 10 operaciones canceladas','El titular identifica la cafetera portátil como producto principal y posibles ventas de soportes o café. Faltan líneas de pedido, SKU, cantidad y estado para separarlas.'],
        ['Alta','4.4 y anexo B','19 marcas explicadas con referencia pasada','Las 19 marcas proceden de la mediana mensual retrospectiva; documentar esa diferencia con la estimación de unidades pasada.'],
        ['Alta','6.4 y 6.5','SARIMAX es la mejor opción para 30 días','La media y mediana de 28 días adicionales obtienen menor MAE de validación y prueba que SARIMAX. Comparación revisada, posterior.'],
        ['Alta','6.7','Inventario real; no hubo quiebres efectivos','Hay saldo reconstruido. La ausencia de ventas previas, dos muestras y ausencia de ajustes limitan esa conclusión.'],
        ['Alta','6.7','Las alertas no constituyen falsos positivos','No puede comprobarse sin inventario real, pedidos en tránsito y quiebres observados. Un contrafactual no valida la clasificación.'],
        ['Alta','6.9.1','Reposición del 27 de abril cubre Hot Sale','Con 30 días llega el 27 de mayo, después de la ventana del 11 al 17. Para llegar el 11 debe emitirse el 11 de abril.'],
        ['Media','4.6.1','Otros modelos no pueden estimarse con esta historia','No hay comparación que pruebe esa imposibilidad. Explicar alcance y costo de ajuste; Prophet puede desactivar estacionalidad anual.'],
        ['Media','6.9 y anexo C','Referencias a 6.6.2, 6.6.3 y 6.6.4','No existen esos subapartados actuales. El anexo mezcla código ilustrativo antiguo con métodos actuales.'],
        ['Alta','4.2','Indumentaria; nueve lotes entre junio 2025 y julio 2026','El titular confirma que el producto principal es una cafetera portátil. Hay diez filas de ingreso: nueve comerciales y una muestra.'],
    ],columns=['Prioridad','Ubicación','Texto o conclusión a revisar','Corrección necesaria'])

    parts=[]
    def section(num,title,text):
        parts.append(f'<h2 id="s{num}">{num}. {title}</h2><p>{text}</p>')
    parts.append('<h1>EDA y revisión técnica de Noirpress</h1><p class="subtitle">Fuentes del 3 de enero al 30 de junio de 2026 · Revisión del '+DATE+'</p>')
    parts.append('<p>El caso permite estudiar el pronóstico de pedidos registrados y la sensibilidad de una aproximación a unidades vendidas. El principal trabajo pendiente es establecer qué productos y cantidades contiene cada orden, justificar la estructura temporal y vincular el pronóstico con decisiones de reposición que lleguen a tiempo. Los resultados siguientes se recalcularon desde las fuentes de Noirpress.</p>')
    parts.append('<nav>'+ ' · '.join(f'<a href="#s{n}">{label}</a>' for n,label in [(1,'Problema'),(2,'Fuentes'),(3,'Calidad'),(4,'Unidades'),(5,'Comportamiento'),(6,'Canales'),(7,'Calendario'),(8,'Eventos'),(9,'Pronóstico'),(10,'Inventario'),(11,'Tesis'),(12,'Mejoras'),(13,'Día a día'),(14,'Reproducibilidad')])+'</nav>')
    section(1,'El problema que la tesis debe resolver','La unidad de estudio es Noirpress, cuyo producto principal es la cafetera portátil, según la aclaración recibida el 6 de octubre de 2026. Algunas órdenes podrían corresponder a soportes metálicos para taza o café. La decisión de negocio es cuándo emitir una reposición y cuántas cafeteras pedir, considerando existencias, compras en tránsito y plazo de llegada. El pronóstico de demanda es una entrada de esa decisión; por sí solo no acredita una mejora de inventario.')
    parts.append('<p>Pregunta propuesta: ¿qué precisión puede alcanzar el pronóstico diario y acumulado a 30 días para la operación de Noirpress con registros incompletos, y cuánto dependen sus resultados de la estimación de unidades, los canales y los picos comerciales? Con las fuentes actuales, la decisión física de reposición se estudia como escenario condicionado a supuestos.</p>')
    section(2,'Qué se recibió y cómo se transformó','El archivo de órdenes contiene 409 registros y 12 columnas. La clave verificable es canal más número de orden. Se reconciliaron las columnas originales con el Excel depurado y los totales diarios publicados. La reconstrucción independiente reproduce 399 órdenes positivas, 412 unidades de referencia mensual y 413 unidades con referencia pasada.')
    parts.append(table(pd.DataFrame([
        ['Export de órdenes','409 órdenes','Fecha y hora, canal, pago, importes, costos, neto y cuotas','No contiene producto, SKU, cantidad ni estado de entrega/cancelación.'],
        ['Planilla de ingresos',f'{len(ing)} filas, {ing["Fecha ingreso"].nunique()} fechas','Recepción, SKU, tipo, unidades y costos de importación','9 filas comerciales y 1 muestra. Falta fecha de emisión al proveedor.'],
        ['Ventas Mensuales','38 filas en plantilla','Dos variantes en meses de 2025 y 2026','Todas las cantidades son cero; no usar como ventas observadas.'],
        ['Control Stock','38 filas calculadas','Saldos sobre la plantilla mensual','No sustituye un conteo físico de existencias.'],
    ],columns=['Fuente','Grano','Qué aporta','Limitación'])))
    parts.append('<p>Secuencia reproducible: export original → ordenar y verificar claves/fechas/importes → marcar importes cero y atípicos → construir pedidos diarios → calcular cantidades aproximadas con referencia previa → completar calendario bajo supuesto de cobertura → comparar pronósticos en fechas futuras respecto del origen → explorar reposición bajo inventario y lead time supuestos.</p>')
    section(3,'Calidad de los datos y preguntas abiertas','No aparecen duplicados de la clave canal–orden ni fechas o importes totales vacíos; tampoco importes totales negativos. Esto valida controles concretos, pero no demuestra que el export esté completo. Los 222 valores ausentes de envío Flex corresponden a TiendaNube y los 187 de comisión de tienda a Mercado Libre: distinguir ausencia por canal de dato faltante por error.')
    parts.append('<p>Las diez órdenes de importe cero pertenecen a Mercado Libre. Nueve tienen neto cero y una neto negativo; la fuente no registra su estado. La exclusión por importe cero es una regla analítica. La afirmación “todas fueron canceladas o reembolsadas” requiere contrastar estados en el sistema de origen.</p>')
    parts.append('<p>Ocho órdenes positivas tienen costo de producto entre $7.999 y $10.000, frente al rango habitual cercano a $62.000–$67.000. El titular indicó que pueden corresponder a soportes metálicos para la taza o café. Esa aclaración permite formular una hipótesis comercial concreta, pero no identifica cada orden. Se conservan y se marcan hasta revisar sus líneas. Que Noirpress tenga un producto principal no convierte todas las órdenes del negocio en ventas de cafeteras.</p>')
    parts.append(table(low[['N° Orden','Origen','Fecha','Total orden','Costo de productos','unidades_causales']].assign(Fecha=lambda f:f.Fecha.dt.strftime('%Y-%m-%d'))))
    parts.append(f'<p>El campo Neto tampoco debe interpretarse automáticamente como margen de contribución: la identidad simple “importe menos los cinco campos de costo” no concilia {net_mismatches} filas (tolerancia $0,05; campos de costo vacíos tratados como cero sólo para este contraste). Esto no demuestra un error contable: hace falta la definición del export antes de calcular rentabilidad o costo de faltantes.</p>')
    section(4,'Qué se mide cuando se pronostica demanda','Hay tres objetivos distintos: número de órdenes positivas (observado), importes en pesos (observados) y unidades por orden (aproximadas). La regla publicada divide importe por mediana de importes de días anteriores, redondea y aplica un mínimo de uno. Usa 30 días previos; si la ventana está vacía recurre al historial positivo previo; al inicio asigna uno a cada orden positiva.')
    parts.append('<p>Esa referencia es un importe típico por orden, no un precio unitario verificado. Descuentos, financiación, composición del pedido y varios productos pueden modificar el importe. El piso de una unidad hace que las ocho órdenes de bajo costo cuenten como una unidad de la misma serie. La proximidad de 412 y 413 no valida las cantidades físicas: ambas reglas comparten la misma hipótesis.</p>')
    parts.append(table(sensitivity))
    parts.append(f'<p>La depuración anterior marca 19 órdenes por referencia mensual. La regla de referencia pasada marca {int(o.atipico_ref_pasada.sum())}; hay {int(o.atipico_ref_pasada.ne(o.Estado.str.startswith("Revisar")).sum())} diferencias de etiqueta. El texto de la tesis debe explicar qué regla produce cada cifra. La recomendación es presentar pedidos positivos como medición directa y conservar las unidades aproximadas como análisis de sensibilidad hasta recuperar cantidades por línea.</p>')
    section(5,'Cómo cambia la serie día a día','Los 179 días incluyen 32 sin pedidos positivos. Diecisiete de esos días se concentran en enero. El intervalo medio por día con ventas (ADI) y la variación de las cantidades positivas (CV²) se calculan como descriptores; los ceros por sí solos no justifican clasificar toda la serie como demanda intermitente estable.')
    parts.append(table(dist));parts.append(table(monthly))
    parts.append('<p>Enero sólo cubre 29 días y su media de pedidos es 0,52. Febrero, marzo y abril se sitúan cerca de 1,93–2,19. Mayo y junio aumentan a 3,32 y 3,13. El cambio puede combinar crecimiento, campañas, mezcla de canales o distinta cobertura del export; no identifica una causa. El contraste debe hacerse por día observado, además de comparar totales.</p>')
    top=d.nlargest(10,'unidades_estimadas')
    parts.append(f'<p>Los diez días de mayor volumen acumulan {int(top.unidades_estimadas.sum())} de 413 unidades estimadas ({fmt(100*top.unidades_estimadas.sum()/413)}%). El máximo es 22 el 11 de mayo. Los extremos se conservan y se investigan; eliminarlos sin evidencia borraría parte del problema comercial.</p>')
    fig,axs=plt.subplots(2,1,figsize=(12,6.6),sharex=True)
    axs[0].plot(idx,d.pedidos_positivos,label='Pedidos positivos',color='#173b59');axs[0].plot(idx,d.unidades_estimadas,label='Unidades estimadas',alpha=.65)
    axs[0].plot(idx,d.unidades_estimadas.rolling(28).mean(),label='Media 28 días descriptiva',color='#c47624',lw=2)
    axs[0].legend(fontsize=8);axs[0].set_ylabel('Por día');axs[0].set_title('Cambio de nivel y concentración de picos')
    for date in [idx[70],idx[125]]:axs[0].axvline(date,color='gray',ls='--')
    axs[1].bar(idx,d.TiendaNube,label='TiendaNube',color='#17827d');axs[1].bar(idx,d['Mercado Libre'],bottom=d.TiendaNube,label='Mercado Libre',color='#deb34b')
    axs[1].set_ylabel('Pedidos por canal');axs[1].legend(fontsize=8);parts.append(picture(fig,'Pedidos diarios, unidades aproximadas y canales'))
    parts.append(table(top.reset_index()[['fecha','pedidos_positivos','unidades_estimadas','TiendaNube','Mercado Libre','ventana']].assign(fecha=lambda x:x.fecha.dt.strftime('%Y-%m-%d'))))
    section(6,'Los canales cambian la interpretación','TiendaNube aporta 222 órdenes positivas y Mercado Libre 177. La proporción de TiendaNube cambia de 33,3% en febrero a 75,7% en mayo. Una serie agregada mezcla procesos comerciales distintos. No se atribuye la variación a un cambio general de demanda sin revisar antes cada canal.')
    parts.append(table(channels.assign(primer_registro=lambda f:f.primer_registro.dt.strftime('%Y-%m-%d'),ultimo_registro=lambda f:f.ultimo_registro.dt.strftime('%Y-%m-%d'))))
    parts.append('<p>El primer registro de Mercado Libre es del 19 de enero y el último del 19 de junio. No hay órdenes allí entre el 20 y el 30 de junio, mientras TiendaNube continúa. Eso puede ser ausencia de ventas, publicación pausada o export parcial. Los archivos no permiten decidirlo. Los ceros de canal quedan identificados para consultar con el negocio.</p><p>“Método de pago” está ligado al canal: Mercado Libre usa un único valor de pago. No conviene interpretar canal y pago como efectos independientes sin estudiar esa dependencia. El ticket mediano es importe por orden; no mide elasticidad de precio.</p>')
    section(7,'Qué evidencia hay de un patrón semanal','El lunes presenta la mayor media global, pero los picos de mayo y junio caen al inicio de la semana. Fuera de esas ventanas su media baja de 3,46 a 2,46 unidades estimadas. Se comparan medias, medianas, tamaños de grupo y tramos temporales antes de justificar una estacionalidad semanal estable.')
    fig,axs=plt.subplots(1,2,figsize=(12,4))
    for lab in ['Todos','Fuera de ventanas']:
        g=weekly[weekly.Tramo.eq(lab)];axs[0].plot(range(7),g.Media,marker='o',label=lab)
    axs[0].set(xticks=range(7),xticklabels=DAYS,ylabel='Media unidades estimadas',title='Semana con y sin ventanas comerciales');axs[0].legend(fontsize=8)
    for lab in ['Enero a abril','Mayo a junio']:
        g=weekly[weekly.Tramo.eq(lab)];axs[1].plot(range(7),g.Media,marker='o',label=lab)
    axs[1].set(xticks=range(7),xticklabels=DAYS,ylabel='Media unidades estimadas',title='Estabilidad entre tramos');axs[1].legend(fontsize=8)
    parts.append(picture(fig,'Patrón semanal condicionado por las ventanas y por el tramo'))
    parts.append(table(autocorr))
    parts.append('<p>La autocorrelación de siete días de la serie completa es aproximadamente 0,026. No basta para demostrar una estructura semanal útil, ni descarta por sí sola cualquier efecto de calendario. STL con período siete impone esa periodicidad al descomponer: verla en un gráfico no prueba su valor predictivo. Corresponde contrastar modelos equivalentes con y sin componente semanal usando el período de desarrollo.</p>')
    station=pd.read_csv(RESULTS/'eda_estacionariedad.csv')
    station['p_valor']=station.p_valor.map(lambda v:'<0,001' if v<.001 else fmt(v,3))
    parts.append(table(station[['ambito','prueba','estadistico','p_valor','hipotesis_nula']]))
    parts.append('<p>ADF en el tramo inicial no rechaza raíz unitaria (p≈0,832), mientras KPSS rechaza estacionariedad de nivel (p≤0,01). En la serie completa ADF rechaza raíz unitaria, pero KPSS sigue rechazando estacionariedad. La lectura es compatible con cambios de nivel o picos que requieren inspección; no autoriza a declarar automáticamente una serie estacionaria ni a elegir diferenciación usando la prueba futura.</p>')
    section(8,'Qué se puede decir de Hot Sale y Día del Padre','Se auditan las mismas ventanas que empleó el proyecto: 11–17 de mayo y 14–17 de junio. Son ventanas descriptivas; su pertenencia a campañas efectivamente activas debe documentarse con el calendario del negocio. Hay una sola ocurrencia disponible de cada evento.')
    parts.append(table(events))
    parts.append('<p>El cociente cambia al comparar contra todo el semestre o contra los 28 días previos. Esto muestra sensibilidad a la referencia. Crecimiento, publicidad, canal, descuentos y disponibilidad pueden intervenir simultáneamente. Las ventanas no son un efecto causal ni un multiplicador aprendido para el próximo año. Seleccionarlas mirando los picos también introduce una decisión posterior a los hechos.</p>')
    section(9,'Revisión del sustento del pronóstico de demanda','La evaluación existente sí respeta el orden temporal dentro de cada origen. Se verificaron fechas, objetivos, métricas y hashes de entrada y programa. Se reutilizaron sus pronósticos almacenados; esta auditoría no volvió a entrenar ni descargó los modelos. Se agregaron cinco referencias simples sobre los mismos orígenes y horizontes: último día, repetición de la última semana, media histórica, media y mediana de 28 días.')
    parts.append('<p>Las referencias de 28 días son comparadores posteriores elegidos para revisar la robustez de la conclusión. No se buscaron automáticamente ventanas para minimizar el error. El proyecto ya había inspeccionado todo el semestre: esta ampliación es exploratoria y su resultado necesita confirmación futura con protocolo congelado.</p>')
    parts.append(table(ranking))
    sar=metric.query('modelo=="SARIMAX" and etapa=="prueba" and horizonte_dias==30').MAE.iloc[0]
    mean=metric.query('modelo=="Media 28 días" and etapa=="prueba" and horizonte_dias==30').MAE.iloc[0]
    parts.append(f'<p>La media de 28 días obtiene MAE acumulado de 6,036 en validación y {fmt(mean,3)} en prueba; SARIMAX alcanza 9,983 y {fmt(sar,3)}. La reducción de MAE de prueba es {fmt(100*(1-mean/sar),1)}%. La mediana de 28 días también mejora a SARIMAX en ambas particiones acumuladas y alcanza MAE diario de validación de 1,036, frente a 1,118 de TimesFM. Esto obliga a acotar la conclusión anterior al conjunto original de cuatro métodos.</p>')
    parts.append('<p>Con estas referencias adicionales, el mínimo de validación corresponde a la mediana de 28 días en ambos horizontes; la prueba diaria favorece a SARIMAX y la prueba acumulada a la media de 28 días. No corresponde elegir retrospectivamente la media porque ganó la prueba. MAE favorece estimaciones de la mediana; RMSE favorece la media. Multiplicar una mediana diaria por treinta tampoco garantiza obtener la mediana del total mensual. Se evalúa el total emitido y se informa sesgo antes de recomendar cantidades.</p>')
    fig,axs=plt.subplots(1,2,figsize=(12,5))
    for ax,(h,title) in zip(axs,[(1,'MAE de prueba diaria'),(30,'MAE de prueba del total de 30 días')]):
        g=metric[(metric.horizonte_dias==h)&metric.etapa.eq('prueba')].sort_values('MAE')
        ax.barh(g.modelo,g.MAE,color=['#17827d' if '28 días' in n else '#42647a' for n in g.modelo]);ax.invert_yaxis();ax.set_title(title);ax.set_xlabel('Unidades estimadas')
    parts.append(picture(fig,'Comparación con referencias adicionales en la misma prueba retrospectiva'))
    parts.append('<h3>La cantidad de prueba efectiva</h3>'+table(stages))
    parts.append('<p>Historial inicial: 3 de enero–13 de marzo. Validación diaria: 14 de marzo–7 de mayo (55 días). Prueba diaria: 8 de mayo–30 de junio (54 días). Las dos ventanas de picos quedan en prueba. Su media es mayor, por lo que un único MAE mezcla funcionamiento ordinario con cambios comerciales.</p>')
    parts.append('<p>En 30 días hay 26 ventanas completas de validación y 25 de prueba; se solapan. Los orígenes de prueba van del 7 al 31 de mayo y todos sus horizontes intersectan mayo. Sólo cabe una ventana completa no solapada en cada partición. Los 25 resultados no equivalen a 25 meses independientes; la solidez de un ranking de reposición sigue siendo limitada.</p>')
    parts.append(table(segment[segment.modelo.isin(['M1 histórico','SARIMAX','TimesFM 3.0','Media 28 días'])][['modelo','tramo','n','MAE','sesgo']]))
    parts.append('<h3>Incertidumbre y diagnóstico de errores</h3>'+table(uncertainty))
    parts.append('<p>Las diferencias anteriores usan 3.000 remuestras pareadas de bloques de siete días de los 54 errores de prueba. Un valor negativo favorece A; un intervalo que contiene cero no separa claramente A de B. Es una sensibilidad exploratoria con pocos bloques y picos, sin ajuste por comparaciones múltiples; no una prueba confirmatoria. No se extrapola a las ventanas acumuladas solapadas.</p>')
    parts.append(table(pd.read_csv(RESULTS/'cobertura_intervalos.csv')[['modelo','n','nominal','cobertura_observada','ancho_medio']]))
    parts.append('<p>El intervalo diario nominal de 80% de TimesFM cubre 38 de 54 observaciones (70,4%). No puede usarse como nivel de servicio garantizado. El sesgo diario de SARIMAX y TimesFM es negativo: un MAE menor puede convivir con subestimación de ventas y no asegura menos faltantes. Los ajustes clásicos guardados informan convergencia en todos los orígenes; converger no demuestra que la especificación sea adecuada.</p>')
    section(10,'Qué falta para sostener la reposición','El stock inicial de 98 proviene de ingresos previos: incluye dos unidades de muestra y supone ninguna venta anterior al 3 de enero de 2026. La planilla no contiene un conteo físico inicial, devoluciones, pérdidas ni cantidades vendidas verificadas. Durante la ventana entran 420 unidades registradas.')
    parts.append(table(stock_sensitivity))
    parts.append('<p>Con 98 iniciales, el saldo mínimo reconstruido es 66 el 25 de mayo y el cierre es 105 con unidades de referencia pasada (106 con referencia mensual). Se necesitan al menos 32 unidades iniciales para mantener el balance al cierre de cada día no negativo bajo esta reconstrucción. Esto es sensibilidad contable, no evidencia de quiebres ni de demanda no atendida. La evolución intradía de stock no se conoce.</p>')
    fig,ax=plt.subplots(figsize=(12,4))
    for q in [0,32,98]:ax.plot(idx,q+flow.cumsum()-d.unidades_estimadas.cumsum(),label=f'Inicial supuesto {q}')
    ax.axhline(0,color='black',lw=1);ax.legend();ax.set(title='El diagnóstico de stock depende del saldo inicial supuesto',ylabel='Saldo contable, unidades aproximadas')
    parts.append(picture(fig,'Saldos reconstruidos con diferentes existencias iniciales'))
    parts.append('<p>El plazo de 30 días no se observa: se fija. Los días entre dos recepciones miden cadencia de compras, no demora de un pedido. Para usar el resultado se necesita la posición de inventario: disponible más compras en tránsito menos compromisos, junto con las fechas esperadas de llegada.</p>')
    parts.append(table(pd.DataFrame([
        ['Llegar al comienzo de la ventana de mayo','11/05/2026','11/04/2026'],
        ['Pedido reconstruido del documento','27/05/2026','27/04/2026'],
        ['Llegar al comienzo de la ventana de junio','14/06/2026','15/05/2026'],
    ],columns=['Objetivo o ejercicio','Llegada','Emisión con 30 días naturales'])))
    parts.append('<p>La cantidad ordenada el 27 de abril llegaría el 27 de mayo y no podría resolver un faltante del 11 de mayo. Comparar el faltante de los treinta días con la cantidad pedida ignora que se consume antes de la llegada. El inventario previo debe cubrir la espera. La evaluación operativa debe simular día por día fechas de emisión, llegada, stock, ventas atendidas, faltantes y compras ya en tránsito.</p>')
    parts.append('<h3>Escenario implementado con recepción explícita</h3><p>La simulación arranca después del cierre del día de emisión, con 30% del saldo reconstruido, redondeado hacia abajo. No incluye otros pedidos. El pedido busca cubrir 30 días de espera y 30 días posteriores: cantidad = techo de máximo entre cero y 60 × tasa pasada menos stock simulado. Se calcula sólo con historia cerrada. El lote llega al comienzo del día de recepción y después se atiende la demanda aproximada observada; los faltantes se pierden. No hay costos o stock de seguridad calibrados.</p>')
    parts.append(table(decisions))
    parts.append('<p>La simulación muestra faltantes durante la espera que ninguna cantidad recibida posteriormente puede reparar. El escenario del 1 de junio termina el 30 de junio, antes de la llegada del 1 de julio: sólo permite puntuar la espera. Las tasas de atención son resultados simulados contra una aproximación a demanda realizada; no son ventas perdidas observadas ni una validación operativa. El modelo omite compras reales ya en tránsito porque su fecha de emisión se desconoce.</p>')
    section(11,'Correcciones aplicadas a la tesis','La actualización del documento incorpora esta revisión: corrige la descripción del producto, amplía la comparación de pronósticos y sustituye los ejercicios anteriores de reposición por una simulación con fechas explícitas. Las confirmaciones de producto por orden, cantidades, cobertura y stock siguen pendientes de datos del negocio.')
    parts.append(table(claims))
    section(12,'Cómo fortalecer el trabajo usando sólo Noirpress','La mejora prioritaria es construir una cadena defendible entre pregunta, variable observada, exploración, elección metodológica, validación y decisión. Incorporar algoritmos adicionales sólo tiene valor si resuelve una necesidad que el diagnóstico haya mostrado.')
    plan=pd.DataFrame([
        ['1','Cerrar la unidad de estudio','El producto principal ya está identificado como cafetera portátil. Confirmar líneas/SKU de las 8 órdenes de bajo costo, las 10 de importe cero y cobertura por canal.','Diccionario de datos y registro de decisiones confirmados por el negocio.'],
        ['2','Fijar el objetivo medible','Usar pedidos positivos como objetivo observado; unidades como aproximación mientras falten líneas de pedido.','Resultados separados y trazabilidad entre orden, cantidad aproximada y día.'],
        ['3','Explicar cambios de nivel y picos','Documentar cambios de catálogo/canal, campañas, precios publicados, actividad y disponibilidad.','Cada pico y cambio con una explicación verificada o hipótesis explícita.'],
        ['4','Contrastar estructura semanal','Comparar nivel solo, tendencia amortiguada y componentes semanales; usar igual calendario de evaluación.','Ablación del componente semanal dentro del período de desarrollo, con MAE/RMSE y sesgo.'],
        ['5','Revisar elección de métodos','Incluir ingenuos y referencias de 28 días; justificar límites de SARIMAX, Holt-Winters y TimesFM.','Ranking por objetivo, costos y estabilidad; no por modernidad.'],
        ['6','Congelar una evaluación futura','Fijar reglas y métricas después de esta revisión; usar ventas posteriores si se consiguen.','Período nuevo de Noirpress sin reutilizarlo para ajustar parámetros.'],
        ['7','Validar reposición','Registrar stock inicial, órdenes de compra, llegada real, ventas/estado y faltantes por fecha.','Simulación causal con tránsito y plazos; luego contraste operativo con nivel de servicio y stock medio.'],
        ['8','Reescribir método y resultados','Usar el EDA para fundamentar decisiones y corregir las afirmaciones de la sección anterior.','Cada conclusión con su tabla, definición, origen y límite correspondiente.'],
    ],columns=['Paso','Mejora','Trabajo concreto','Criterio de cierre'])
    plan['Estado']=['Parcial: producto confirmado; detalle pendiente','Aplicado: objetivos y trazabilidad separados','EDA aplicado; causas comerciales pendientes','Pendiente de evaluación específica','Aplicado: nueve referencias y límites documentados','Pendiente de nuevos datos','Simulación aplicada; validación operativa pendiente','Aplicado en la tesis del 7 de octubre']
    parts.append(table(plan))
    parts.append('<p>No hace falta sumar otro comercio para estudiar bien este caso. Sí hacen falta mejores registros de Noirpress para validar cantidades físicas y decisiones reales. Tampoco hay una regla universal de “dos años para cualquier modelo”: seis meses permiten explorar patrones semanales; efectos anuales repetidos necesitan otra cobertura. La ausencia de dos años no prohíbe por sí sola Prophet o árboles; las alternativas no evaluadas deben presentarse como fuera del alcance, sin atribuirles una inferioridad demostrada.</p>')
    parts.append('<h3>Cómo explicar el trabajo</h3><p>Primero auditamos qué observaba realmente la fuente. Después reconstruimos una serie diaria y distinguimos pedidos registrados de unidades aproximadas. El EDA mostró cambios de nivel, mezcla de canales y picos concentrados. Evaluamos el pronóstico respetando las fechas y el horizonte de compra. Al ampliar las referencias simples, algunas superaron a los modelos complejos para el total mensual. La reposición queda condicionada a confirmar inventario, cantidades y plazos. Ese recorrido permite justificar las decisiones y señalar exactamente qué falta validar.</p>')
    section(13,'EDA día por día y diccionario','Buscador por fecha, mes, ventana o lectura. Los 179 días se mantienen completos. Las medias pasadas excluyen la fecha analizada. “Sin pedidos” significa ausencia en la fuente bajo supuesto de cobertura; no confirma cierre, falta de interés ni quiebre de stock.')
    display=d.reset_index()[['fecha','etapa','pedidos_positivos','unidades_estimadas','TiendaNube','Mercado Libre','importe_ars','ordenes_sin_cargo','ordenes_bajo_costo','media_pasada_28','pico_vs_pasado_28','ventana','saldo_supone_inicial_98','lectura']].copy()
    display['fecha']=display.fecha.dt.strftime('%Y-%m-%d')
    parts.append('<label>Buscar un día <input type="search" oninput="filtrar(this.value)" placeholder="2026-05-11, junio, bajo costo..."></label><div id="diario">'+table(display)+'</div>')
    parts.append(table(pd.DataFrame([
        ['pedidos_positivos','conteo observado','Órdenes con Total orden mayor que cero; no certifica entrega.'],
        ['unidades_estimadas','aproximación','Cantidad inferida por importe y referencia pasada, no unidades verificadas.'],
        ['importe_ars','pesos nominales','Total de importes de órdenes positivas; no precio unitario ni margen.'],
        ['media_pasada_28','unidades aproximadas/día','Media de los 28 días cerrados anteriores. Vacío al inicio.'],
        ['pico_vs_pasado_28','alerta heurística','Supera media pasada más tres desvíos. No es test de significación ni motivo de borrado.'],
        ['saldo_supone_inicial_98','escenario contable','98 más ingresos acumulados menos unidades aproximadas.'],
        ['ventana','clasificación descriptiva','Fechas comerciales empleadas en el proyecto; requiere validar campañas reales.'],
        ['Variables futuras disponibles','calendario y decisiones documentadas','Día de semana conocido. Canal, cuotas, ticket, ventas y costos del futuro no se conocen antes.'],
    ],columns=['Campo','Naturaleza','Definición'])))
    section(14,'Reproducibilidad y sustento metodológico','El cálculo conserva fuentes y resultados previos. Los archivos de esta auditoría contienen el detalle diario, las órdenes que requieren revisión, las métricas y las predicciones de referencias adicionales. Los modelos existentes se contrastaron con sus métricas y hashes guardados; no se vuelve a certificar aquí el origen de sus pesos ni su licencia.')
    parts.append('<p>Fuentes: ultimas_ordenes_2026-07-06.xlsx, hoja Ordenes; control_pedidos_ventas_quiebres_noirpress.xlsx, hojas Ingresos, Ventas Mensuales y Control Stock; Excel depurado, programa y salidas existentes. Aclaración del titular en conversación del 6 de octubre de 2026: producto principal cafetera portátil; órdenes de bajo costo posiblemente de soportes metálicos o café. No se recibió aún identificación de esas órdenes ni confirmación de cobertura por canal. Semilla de remuestreo: 20261006. La elección adicional de métodos y las interpretaciones quedan fechadas después de la evaluación original.</p>')
    parts.append('<ul><li><a href="https://otexts.com/fpp3/simple-methods.html">Hyndman y Athanasopoulos, referencias simples de pronóstico</a>: sirven para comprobar el valor agregado del método.</li><li><a href="https://otexts.com/fpp3/tscv.html">Validación con origen móvil</a>: entrenar con pasado y evaluar el horizonte de interés.</li><li><a href="https://otexts.com/fpp3/accuracy.html">Métricas de precisión</a>: MAE y RMSE responden a objetivos diferentes; MAPE es problemática con ceros.</li><li><a href="https://otexts.com/fpp3/regression-evaluation.html">Diagnóstico de modelos</a>: revisar residuos, atípicos y relaciones espurias.</li><li><a href="https://facebook.github.io/prophet/docs/seasonality,_holiday_effects,_and_regressors.html">Documentación oficial de Prophet</a>: estacionalidades configurables y variables externas.</li></ul>')
    style='body{max-width:1180px;margin:38px auto;padding:0 24px;font:16px/1.65 system-ui,sans-serif;color:#203340}h1{font-size:34px;line-height:1.2}h2{margin-top:2.6em;line-height:1.25}h3{margin-top:1.6em}p{max-width:1050px}.subtitle{color:#50636c}nav{padding:18px 0;border-block:1px solid #d7e1e5}a{color:#00746c}table{border-collapse:collapse;font-size:13px;width:100%;margin:20px 0}th,td{border:1px solid #d7e1e5;padding:9px 12px;text-align:left;vertical-align:top}th{background:#edf3f5}tr:nth-child(even){background:#f8fafb}.table{overflow:auto}figure{margin:20px 0}img{width:100%;height:auto}#diario{max-height:650px;overflow:auto}input{padding:10px;width:310px} @media print{nav,input{display:none}body{max-width:none;font-size:10pt}h2{break-after:avoid}figure,tr{break-inside:avoid}.table,#diario{overflow:visible;max-height:none}}'
    html='<!doctype html><html lang="es"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>EDA de Noirpress</title><style>'+style+'</style><body>'+''.join(parts)+'<script>function filtrar(v){v=v.toLowerCase();document.querySelectorAll("#diario tbody tr").forEach(r=>r.hidden=!r.textContent.toLowerCase().includes(v));}</script></body></html>'
    (OUT/'Informe_EDA_Noirpress.html').write_text(html,encoding='utf-8')
    d.reset_index().to_csv(OUT/'dia_a_dia.csv',index=False,encoding='utf-8-sig',date_format='%Y-%m-%d')
    review=o[o.Estado.ne('Valido')|o.atipico_ref_pasada|o.bajo_costo_a_verificar]
    review[['N° Orden','Origen','Fecha','Total orden','Costo de productos','Estado','Cantidad estimada','referencia_pasada','ratio_pasado','unidades_causales','atipico_ref_pasada','bajo_costo_a_verificar']].to_csv(OUT/'ordenes_a_revisar.csv',index=False,encoding='utf-8-sig',date_format='%Y-%m-%d')
    metric.to_csv(OUT/'comparacion_referencias.csv',index=False,encoding='utf-8-sig')
    extra.to_csv(OUT/'referencias_por_origen.csv',index=False,encoding='utf-8-sig',date_format='%Y-%m-%d')
    decisions.to_csv(OUT/'reposicion_resumen.csv',index=False,encoding='utf-8-sig')
    pd.concat(trajectories,ignore_index=True).to_csv(OUT/'reposicion_dia_a_dia.csv',index=False,encoding='utf-8-sig',date_format='%Y-%m-%d')
    evidence={'fecha_revision':DATE,'python':platform.python_version(),'paquetes':{'numpy':np.__version__,'pandas':pd.__version__,'matplotlib':matplotlib.__version__},
        'hashes':{str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [SALES,STOCK,ROOT/'Ordenes_Noirpress_limpio.xlsx',ROOT/'scripts/modelos_noirpress_avanzados.py',Path(__file__).resolve(),RESULTS/'pronosticos_por_origen.csv']},
        'comprobaciones':['Columnas originales reconciliadas con Excel depurado','Totales diarios reconstruidos iguales a EDA previo','Cambiar importes futuros no altera unidades anteriores','Hashes verificados contra manifiesto original y registro editorial cuando corresponde','Métricas recalculadas iguales a las publicadas','Mismas fechas y horizontes para referencias adicionales'],
        'calidad':{'filas_originales':len(raw),'nulos_por_columna':raw.isna().sum().to_dict(),
                   'duplicados_canal_orden':int(raw.duplicated(['Origen','N° Orden']).sum()),
                   'filas_no_concilian_formula_simple_neto':net_mismatches,
                   'ordenes_de_bajo_costo':len(low),'marcas_mensuales':int(o.Estado.str.startswith('Revisar').sum()),
                   'marcas_referencia_pasada':int(o.atipico_ref_pasada.sum())},
        'sensibilidad_unidades':sensitivity.to_dict('records'),'resumen_mensual':monthly.to_dict('records'),
        'calendario':weekly.to_dict('records'),'autocorrelacion':autocorr.to_dict('records'),
        'eventos':events.to_dict('records'),'incertidumbre':uncertainty.to_dict('records'),
        'inventario':stock_sensitivity.to_dict('records'),'correcciones_tesis':claims.to_dict('records')}
    evidence['simulacion_reposicion']=decisions.to_dict('records')
    # Normalizar NaN de campos no aplicables: JSON portable.
    evidence=json.loads(pd.Series(evidence).to_json(force_ascii=False,default_handler=str))
    (OUT/'evidencia.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf-8')
    print(ranking.to_string(index=False))
    print(sensitivity.to_string(index=False))
    print('Días:',len(d),'Órdenes a revisar:',len(review),'Informe:',OUT/'Informe_EDA_Noirpress.html')


if __name__=='__main__':
    main()
