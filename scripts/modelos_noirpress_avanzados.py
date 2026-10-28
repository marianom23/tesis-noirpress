"""Pronóstico de demanda diario reproducible y evaluación temporal de Noirpress."""
from __future__ import annotations

import argparse
import base64
import hashlib
from html import escape
import importlib.metadata
import json
import platform
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.statespace.sarimax import SARIMAX

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / 'Ordenes_Noirpress_limpio.xlsx'
MODEL_ID = 'google/timesfm-3.0-pytorch'
REVISION = '43046b85ec22d584a13f8098c2ed39c889e129c2'
N_INITIAL = 70
HORIZON = 30
MODELS = ['M1 histórico', 'Holt-Winters', 'SARIMAX', 'TimesFM 3.0']
DAYS = ['lunes', 'martes', 'miércoles', 'jueves', 'viernes', 'sábado', 'domingo']


def prepare_orders(orders):
    """Referencia de importe de los 30 días ANTERIORES; nunca del día en curso."""
    o = orders.copy()
    o['Fecha'] = pd.to_datetime(o['Fecha']).dt.normalize()
    o['Total orden'] = pd.to_numeric(o['Total orden'], errors='raise')
    if o['Fecha'].isna().any() or not np.isfinite(o['Total orden']).all():
        raise ValueError('Fechas o importes ausentes/no finitos: revisar fuente.')
    if (o['Total orden'] < 0).any():
        raise ValueError('Importes negativos: requieren una regla de devolución explícita.')
    if o.duplicated(['Origen', 'N° Orden']).any():
        raise ValueError('Identificador de orden repetido dentro del canal. Revisar duplicados.')
    o = o.sort_values('Fecha', kind='stable')
    o['referencia_pasada'] = np.nan
    o['unidades_causales'] = 0
    for day, group in o.groupby('Fecha', sort=True):
        past = o.loc[(o.Fecha < day) & (o.Fecha >= day - pd.Timedelta(days=30)) &
                     (o['Total orden'] > 0), 'Total orden']
        if past.empty:
            past = o.loc[(o.Fecha < day) & (o['Total orden'] > 0), 'Total orden']
        ref = float(past.median()) if len(past) else np.nan
        o.loc[group.index, 'referencia_pasada'] = ref
        amounts = group['Total orden'].to_numpy()
        # Arranque sin referencia: una unidad por orden positiva. Supuesto auditado.
        qty = np.maximum(1, np.floor(np.round(amounts / ref, 2) + .5)) if np.isfinite(ref) else np.ones(len(group))
        o.loc[group.index, 'unidades_causales'] = np.where(amounts > 0, qty, 0).astype(int)
    return o


def load_series(path=INPUT):
    source = pd.read_excel(path, sheet_name='Ordenes_limpio')
    o = prepare_orders(source)
    idx = pd.date_range(o.Fecha.min(), o.Fecha.max(), freq='D', name='fecha')
    d = pd.DataFrame(index=idx)
    positive = o[o['Total orden'] > 0]
    d['demanda'] = o.groupby('Fecha')['unidades_causales'].sum().reindex(idx, fill_value=0).astype(float)
    d['unidades_retrospectivas'] = o.groupby('Fecha')['Cantidad estimada'].sum().reindex(idx, fill_value=0)
    d['pedidos'] = positive.groupby('Fecha').size().reindex(idx, fill_value=0)
    d['importe_ordenes'] = positive.groupby('Fecha')['Total orden'].sum().reindex(idx, fill_value=0)
    d['pedidos_descartados'] = o[o['Total orden'] == 0].groupby('Fecha').size().reindex(idx, fill_value=0)
    d['pedidos_revision'] = o[o.Estado.str.startswith('Revisar')].groupby('Fecha').size().reindex(idx, fill_value=0)
    for channel, group in positive.groupby('Origen'):
        d['pedidos_' + str(channel)] = group.groupby('Fecha').size().reindex(idx, fill_value=0)
    d['dow'] = idx.dayofweek
    d['dia_semana'] = [DAYS[x] for x in d.dow]
    d['mes'] = idx.strftime('%Y-%m')
    d['evento'] = 'normal'
    d.loc['2026-05-11':'2026-05-17', 'evento'] = 'ventana promocional'
    d.loc['2026-06-14':'2026-06-17', 'evento'] = 'ventana previa Día del Padre'
    d.attrs['orders'] = o.to_dict('records')
    assert d.pedidos.sum() == len(positive)
    assert d.demanda.sum() == o.unidades_causales.sum()
    assert d.index.is_unique and d.demanda.ge(0).all()
    return d


def calendar(index):
    angle = 2 * np.pi * pd.DatetimeIndex(index).dayofweek.to_numpy() / 7
    return np.column_stack([np.sin(angle), np.cos(angle)])


def forecast_classic(name, history, future_dates):
    """La función recibe SOLO historia y fechas futuras, nunca ventas futuras."""
    h = len(future_dates)
    if name == MODELS[0]:
        return np.repeat(history.iloc[-7:].mean(), h), True, ''
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        if name == 'Holt-Winters':
            fit = ExponentialSmoothing(history, trend='add', damped_trend=True,
                                       seasonal='add', seasonal_periods=7,
                                       initialization_method='estimated').fit(optimized=True)
            result = fit.forecast(h)
            converged = bool(fit.mle_retvals.get('success', True))
        else:
            fit = SARIMAX(history, exog=calendar(history.index), order=(1, 0, 1),
                          seasonal_order=(1, 0, 0, 7), trend='c',
                          enforce_stationarity=True, enforce_invertibility=True).fit(disp=False, maxiter=300)
            result = fit.forecast(h, exog=calendar(future_dates))
            converged = bool(fit.mle_retvals.get('converged', False))
        message = ' | '.join(sorted(set(str(w.message) for w in caught)))
    values = np.asarray(result, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f'Pronóstico no finito: {name}')
    return np.maximum(values, 0), converged, message


def load_foundation(device='cpu'):
    import torch
    from timesfm3 import ModelConfig, TimesFM3Evaluator
    torch.set_num_threads(4)
    torch.manual_seed(0)
    return TimesFM3Evaluator(ModelConfig(checkpoint_path=MODEL_ID, revision=REVISION,
                                        per_core_batch_size=4, device=device))


def foundation_forecasts(model, histories):
    """Cada origen es una tarea independiente; sin aprendizaje entre ventanas."""
    contexts = [y.to_numpy(dtype=np.float32) for y in histories]
    covariates = []
    for y in histories:
        dates = pd.date_range(y.index[0], periods=len(y) + HORIZON, freq='D')
        covariates.append(calendar(dates).T.astype(np.float32))
    # Aislar cada origen también de cambios de padding al mezclar longitudes en un lote.
    # Prueba: modificar o añadir un historial futuro no debe cambiar un origen anterior.
    outputs = []
    for context, covariate in zip(contexts, covariates):
        outputs.extend(model.predict_batch(contexts=[context], horizon=HORIZON,
                       past_future_covariates=[covariate], return_quantiles=True,
                       use_symmetric_averaging=False, make_positive=True))
    assert len(outputs) == len(histories)
    results = []
    for item in outputs:
        point = np.asarray(item.forecast).reshape(HORIZON)
        q = np.asarray(item.quantiles).reshape(HORIZON, 9)
        if not np.isfinite(point).all() or not np.isfinite(q).all():
            raise ValueError('TimesFM devolvió valores no finitos.')
        results.append((np.maximum(point, 0), np.maximum(q[:, 0], 0), np.maximum(q[:, 8], 0)))
    return results


def metric_values(real, pred):
    y, p = np.asarray(real, float), np.asarray(pred, float)
    err = p - y
    nz = y != 0
    return {'MAE': np.abs(err).mean(), 'RMSE': np.sqrt(np.mean(err ** 2)),
            'sesgo_pred_menos_real': err.mean(),
            'WAPE_pct': 100 * np.abs(err).sum() / np.abs(y).sum() if np.abs(y).sum() else np.nan,
            'MAPE_sin_ceros_pct': 100 * np.mean(np.abs(err[nz] / y[nz])) if nz.any() else np.nan,
            'n': len(y)}


def run_models(d, out, device='cpu', foundation=True):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    split = int(len(d) * .70)
    if split - N_INITIAL < HORIZON or len(d) - split < HORIZON:
        raise ValueError('Serie insuficiente para validación y prueba separadas de 30 días.')
    names = MODELS if foundation else MODELS[:3]
    y = d.demanda
    rows, status = [], []
    forecaster = load_foundation(device) if foundation else None
    for stage, start, end in [('validacion', N_INITIAL, split), ('prueba', split, len(d))]:
        origins = list(range(start, end))
        found = foundation_forecasts(forecaster, [y.iloc[:i] for i in origins]) if foundation else None
        for j, i in enumerate(origins):
            history = y.iloc[:i]
            dates = pd.date_range(y.index[i], periods=HORIZON, freq='D')
            # Sólo se puntúan horizontes cuyos valores reales pertenecen a la misma partición.
            count = min(HORIZON, end - i)
            for name in names:
                began = time.perf_counter()
                if name == 'TimesFM 3.0':
                    point, lo, hi = found[j]
                    converged, message = True, 'zero-shot; sin ajuste sobre Noirpress'
                else:
                    point, converged, message = forecast_classic(name, history, dates)
                    lo = hi = np.repeat(np.nan, HORIZON)
                status.append({'etapa': stage, 'origen': history.index[-1], 'modelo': name,
                               'convergio': converged, 'advertencias': message,
                               'segundos_ajuste_clasico': time.perf_counter() - began if name != 'TimesFM 3.0' else np.nan})
                for step in range(count):
                    rows.append({'etapa': stage, 'origen': history.index[-1], 'fecha': dates[step],
                                 'horizonte': step + 1, 'modelo': name, 'real': y.iloc[i + step],
                                 'pronostico': point[step], 'p10': lo[step], 'p90': hi[step],
                                 'convergio': converged})
            if j % 10 == 0:
                print(f'{stage}: {j+1}/{len(origins)} orígenes diarios', flush=True)
    pred = pd.DataFrame(rows)
    daily = pred[pred.horizonte == 1].copy()
    metrics = []
    for (stage, name), g in daily.groupby(['etapa', 'modelo'], sort=False):
        metrics.append({'etapa': stage, 'modelo': name, **metric_values(g.real, g.pronostico)})
    metrics = pd.DataFrame(metrics)
    cumulative = pred.groupby(['etapa', 'modelo', 'origen'], as_index=False).agg(
        dias=('horizonte', 'size'), real=('real', 'sum'), pronostico=('pronostico', 'sum'))
    cumulative = cumulative[cumulative.dias == HORIZON]
    totals = pd.DataFrame([{'etapa': stage, 'modelo': name, **metric_values(g.real, g.pronostico)}
                           for (stage, name), g in cumulative.groupby(['etapa', 'modelo'], sort=False)])
    statuses = pd.DataFrame(status)
    eligible = statuses[statuses.etapa == 'validacion'].groupby('modelo').convergio.all()
    chosen = {}
    for purpose, table in [('diario', metrics), ('reposicion_30_dias', totals)]:
        validation = table[(table.etapa == 'validacion') & table.modelo.isin(eligible[eligible].index)]
        chosen[purpose] = validation.sort_values(['MAE', 'RMSE', 'modelo']).iloc[0].modelo
    future = []
    dates = pd.date_range(y.index[-1] + pd.Timedelta(days=1), periods=HORIZON)
    for name in names:
        if name == 'TimesFM 3.0':
            point, lo, hi = foundation_forecasts(forecaster, [y])[0]
        else:
            point, ok, message = forecast_classic(name, y, dates)
            lo = hi = np.repeat(np.nan, HORIZON)
            status.append({'etapa': 'futuro', 'origen': y.index[-1], 'modelo': name, 'convergio': ok,
                           'advertencias': message, 'segundos_ajuste_clasico': np.nan})
        for j, day in enumerate(dates):
            future.append({'origen': y.index[-1], 'fecha': day, 'modelo': name, 'pronostico': point[j],
                           'p10': lo[j], 'p90': hi[j], 'sin_ventas_observadas': True,
                           'seleccionado_reposicion': name == chosen['reposicion_30_dias']})
    save_csv(pred, out / 'pronosticos_por_origen.csv')
    save_csv(daily, out / 'pronostico_demanda_avanzado.csv')
    save_csv(metrics, out / 'metricas_diarias.csv')
    save_csv(cumulative, out / 'demanda_acumulada_30_dias.csv')
    save_csv(totals, out / 'metricas_30_dias.csv')
    save_csv(pd.DataFrame(status), out / 'diagnostico_ajustes.csv')
    save_csv(pd.DataFrame(future), out / 'pronostico_30_dias_desde_ultimo_dato.csv')
    (out / 'seleccion_modelos.json').write_text(json.dumps(chosen, ensure_ascii=False, indent=2), encoding='utf-8')
    diagnostics(daily, d, out)
    plot_forecasts(daily, pd.DataFrame(future), chosen, out)
    print(metrics.to_string(index=False), flush=True)
    print('Selección anterior a la prueba:', chosen, flush=True)
    return daily, metrics, totals, chosen


def save_csv(frame, path):
    frame.to_csv(path, index=False, encoding='utf-8-sig', date_format='%Y-%m-%d')


def save_fig(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def run_eda(d, out):
    out = Path(out)
    figs = out / 'figuras_eda'
    figs.mkdir(parents=True, exist_ok=True)
    daily = d.copy()
    daily.attrs = {}
    y = d.demanda
    daily['cambio_diario'] = y.diff()
    daily['acumulado'] = y.cumsum()
    daily['sin_pedidos'] = d.pedidos.eq(0)
    daily['racha_sin_pedidos'] = daily.sin_pedidos.groupby((~daily.sin_pedidos).cumsum()).cumsum()
    daily['diferencia_estimacion'] = y - d.unidades_retrospectivas
    for window in (7, 14, 30):
        past = y.shift(1).rolling(window, min_periods=window)
        daily[f'media_pasada_{window}'] = past.mean()
        daily[f'desvio_pasado_{window}'] = past.std()
        daily[f'media_descriptiva_{window}'] = y.rolling(window, min_periods=window).mean()
    daily['residuo_referencia_7'] = y - daily.media_pasada_7
    q1, q3 = y.quantile([.25, .75])
    threshold = q3 + 1.5 * (q3 - q1)
    daily['atipico_IQR_descriptivo'] = y > threshold
    daily['alerta_pasada_30'] = y > (daily.media_pasada_30 + 3 * daily.desvio_pasado_30)
    daily['lectura'] = [
        ('Sin órdenes registradas; cero asumido dentro de la ventana del export.' if row.pedidos == 0 else
         'Pico respecto a 30 días anteriores; revisar promoción y órdenes.' if row.alerta_pasada_30 else
         'Órdenes con importe atípico; revisar antes de atribuir variación a demanda.' if row.pedidos_revision else
         'Venta registrada; cantidad de unidades estimada a partir del importe.')
        for row in daily.itertuples()]
    save_csv(daily.reset_index(), out / 'eda_dia_a_dia.csv')
    audit = pd.DataFrame(d.attrs['orders'])[['Origen', 'N° Orden', 'Fecha', 'Total orden', 'Estado',
                                'Cantidad estimada', 'referencia_pasada', 'unidades_causales']]
    save_csv(audit, out / 'auditoria_estimacion_unidades.csv')
    monthly = d.groupby('mes').agg(dias=('demanda', 'size'), unidades=('demanda', 'sum'),
                                   pedidos=('pedidos', 'sum'), media=('demanda', 'mean'),
                                   desvio=('demanda', 'std'), importe=('importe_ordenes', 'sum'))
    save_csv(monthly.reset_index(), out / 'eda_mensual.csv')
    weekday = d.groupby(['dow', 'dia_semana']).demanda.agg(['size', 'sum', 'mean', 'median', 'std'])
    save_csv(weekday.reset_index(), out / 'eda_dia_semana.csv')
    normal = d[d.evento == 'normal'].demanda
    events = d.groupby('evento').demanda.agg(['size', 'sum', 'mean', 'median', 'std']).reset_index()
    events['ratio_vs_normal_descriptivo'] = events['mean'] / normal.mean()
    save_csv(events, out / 'eda_eventos_descriptivos.csv')
    train = y.iloc[:N_INITIAL]
    tests = []
    for label, series in [('entrenamiento_inicial', train), ('serie_completa_descriptiva', y)]:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            a, k = adfuller(series, autolag='AIC'), kpss(series, regression='c', nlags='auto')
        tests.extend([{'ambito': label, 'prueba': 'ADF', 'estadistico': a[0], 'p_valor': a[1],
                       'hipotesis_nula': 'raíz unitaria', 'nota': str([str(x.message) for x in w])},
                      {'ambito': label, 'prueba': 'KPSS', 'estadistico': k[0], 'p_valor': k[1],
                       'hipotesis_nula': 'estacionariedad de nivel', 'nota': 'p puede estar limitado por tabla crítica'}])
    save_csv(pd.DataFrame(tests), out / 'eda_estacionariedad.csv')
    stl = STL(y, period=7, robust=True).fit()
    components = pd.DataFrame({'fecha': y.index, 'demanda': y, 'tendencia': stl.trend,
                                'semanal': stl.seasonal, 'residuo': stl.resid})
    save_csv(components, out / 'eda_stl_descriptiva.csv')
    summary = {'inicio': str(y.index[0].date()), 'fin': str(y.index[-1].date()), 'dias': len(y),
               'pedidos_positivos': int(d.pedidos.sum()), 'pedidos_descartados': int(d.pedidos_descartados.sum()),
               'unidades_estimadas_causales': float(y.sum()),
               'unidades_estimadas_retrospectivas': float(d.unidades_retrospectivas.sum()),
               'dias_con_estimacion_distinta': int(daily.diferencia_estimacion.ne(0).sum()),
               'media': float(y.mean()), 'mediana': float(y.median()), 'desvio': float(y.std()),
               'dias_sin_pedidos': int(daily.sin_pedidos.sum()), 'racha_sin_pedidos': int(daily.racha_sin_pedidos.max()),
               'maximo': float(y.max()), 'fecha_maximo': str(y.idxmax().date()),
               'atipicos_IQR': int(daily.atipico_IQR_descriptivo.sum()),
               'umbral_IQR': float(threshold), 'concentracion_10_dias_pct': float(100 * y.nlargest(10).sum()/y.sum()),
               'correlacion_lag7': float(y.autocorr(7)),
               'fuente_cantidad': 'estimación; no hay cantidades observadas por línea de pedido'}
    (out / 'eda_resumen.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(y.index, y, lw=1, label='Unidades estimadas con referencia pasada')
    ax.plot(y.index, daily.media_descriptiva_7, label='Media 7 días descriptiva')
    ax.plot(y.index, daily.media_descriptiva_30, label='Media 30 días descriptiva')
    ax.scatter(y.index[daily.atipico_IQR_descriptivo], y[daily.atipico_IQR_descriptivo], color='red', s=18, label='Atípico IQR; conservado')
    for day, label in [(y.index[N_INITIAL], 'Validación'), (y.index[int(len(y)*.7)], 'Prueba')]:
        ax.axvline(day, color='gray', ls='--'); ax.text(day, y.max(), label, fontsize=9)
    ax.set(title='Demanda diaria de Noirpress y particiones temporales', ylabel='Unidades estimadas')
    ax.legend(fontsize=8)
    save_fig(fig, figs/'01_demanda_rolling.png')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].boxplot([d[d.dow == i].demanda for i in range(7)], tick_labels=[s[:3] for s in DAYS])
    axes[0].set(title='Distribución por día de semana', ylabel='Unidades estimadas')
    axes[1].hist(y, bins=np.arange(0, y.max()+2)-.5, color='#168b88')
    axes[1].set(title='Frecuencia de demanda diaria', xlabel='Unidades estimadas', ylabel='Días')
    save_fig(fig, figs/'02_demanda_dia_semana.png')
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(events.evento, events['mean'], color=['#168b88', '#c47725', '#416184'])
    for i, row in events.iterrows():
        ax.text(i, row['mean'], f"n={row['size']}", ha='center', va='bottom')
    ax.set(title='Ventanas comerciales: asociación descriptiva, sin inferencia causal', ylabel='Media de unidades estimadas')
    save_fig(fig, figs/'03_demanda_eventos.png')
    fig, axes = plt.subplots(3, 2, figsize=(12, 8))
    for ax, (month, g) in zip(axes.flat, d.groupby('mes')):
        grid = np.full((6, 7), np.nan)
        first_weekday = pd.Timestamp(month+'-01').dayofweek
        for date, value in g.demanda.items():
            cell = first_weekday + date.day - 1
            grid[cell//7, cell%7] = value
        ax.imshow(grid, cmap='YlGnBu', vmin=0, vmax=y.max())
        for date, value in g.demanda.items():
            cell = first_weekday + date.day - 1
            ax.text(cell%7, cell//7, f'{date.day}\n{value:g}', ha='center', va='center', fontsize=7,
                    color='white' if value > y.max()*.5 else 'black')
        ax.set(title=month, xticks=range(7), xticklabels=[s[:2] for s in DAYS], yticks=[])
    fig.suptitle('Cada celda muestra día del mes y unidades estimadas; blanco = fuera de muestra')
    save_fig(fig, figs/'05_calendario_diario.png')
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for row, (label, series) in enumerate([('Sólo entrenamiento inicial', train), ('Serie completa descriptiva', y)]):
        plot_acf(series, lags=28, ax=axes[row, 0], title=label+' — ACF', zero=False)
        plot_pacf(series, lags=28, ax=axes[row, 1], title=label+' — PACF', method='ywm', zero=False)
    save_fig(fig, figs/'06_autocorrelacion.png')
    fig = stl.plot(); fig.set_size_inches(12, 8)
    fig.suptitle('STL semanal descriptiva de toda la serie; no se usa como predictor')
    save_fig(fig, figs/'07_descomposicion_stl.png')
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(y.index, d.unidades_retrospectivas, label='Referencia del mes completo', alpha=.7)
    axes[0].plot(y.index, y, label='Referencia de días anteriores', alpha=.7)
    axes[0].legend(); axes[0].set_ylabel('Unidades estimadas')
    axes[1].bar(y.index, daily.diferencia_estimacion)
    axes[1].set_ylabel('Diferencia'); axes[0].set_title('Sensibilidad a la estimación de unidades')
    save_fig(fig, figs/'08_sensibilidad_unidades.png')
    print('EDA diario completo:', summary, flush=True)
    return daily, summary


def diagnostics(daily, d, out):
    rows, segments = [], []
    test = daily[daily.etapa == 'prueba'].copy()
    for name, g in test.groupby('modelo'):
        residual = (g.real - g.pronostico).to_numpy()
        for lag, row in acorr_ljungbox(residual, lags=[7, 14], return_df=True).iterrows():
            rows.append({'modelo': name, 'lag': lag, 'estadistico': row.lb_stat, 'p_valor': row.lb_pvalue,
                         'nota': 'Exploratorio: errores walk-forward, no residuos de un único ajuste fijo.'})
        g = g.copy(); g['evento'] = d.evento.reindex(g.fecha).to_numpy()
        g['dia_semana'] = [DAYS[x] for x in pd.DatetimeIndex(g.fecha).dayofweek]
        for column in ('evento', 'dia_semana'):
            for value, part in g.groupby(column):
                segments.append({'modelo': name, 'agrupacion': column, 'grupo': value, **metric_values(part.real, part.pronostico)})
    save_csv(pd.DataFrame(rows), out/'diagnostico_errores_ljungbox.csv')
    save_csv(pd.DataFrame(segments), out/'errores_por_evento_y_dia.csv')
    calibration = []
    q = test[test.modelo == 'TimesFM 3.0']
    if len(q):
        calibration.append({'modelo': 'TimesFM 3.0', 'n': len(q), 'nominal': .8,
                            'cobertura_observada': ((q.real >= q.p10) & (q.real <= q.p90)).mean(),
                            'ancho_medio': (q.p90-q.p10).mean(),
                            'nota': 'Intervalo marginal diario, no intervalo de demanda acumulada.'})
    save_csv(pd.DataFrame(calibration, columns=['modelo', 'n', 'nominal', 'cobertura_observada', 'ancho_medio', 'nota']), out/'cobertura_intervalos.csv')
    # Comparación pareada con bloques semanales: dependencia temporal, no días iid.
    wide = test.pivot(index='fecha', columns='modelo', values='pronostico')
    truth = test.drop_duplicates('fecha').set_index('fecha').real.reindex(wide.index)
    absolute = wide.sub(truth, axis=0).abs()
    rng = np.random.default_rng(2026)
    n = len(wide)
    samples = np.array([np.concatenate([np.arange(s, s+7) for s in
                        rng.integers(0, n-6, size=int(np.ceil(n/7)))])[:n] for _ in range(2000)])
    comparisons = []
    for name in absolute.columns:
        if name == MODELS[0]:
            continue
        delta = (absolute[name]-absolute[MODELS[0]]).to_numpy()
        lo, hi = np.quantile(delta[samples].mean(axis=1), [.025, .975])
        comparisons.append({'modelo': name, 'referencia': MODELS[0], 'delta_MAE': delta.mean(),
                            'intervalo_inferior_95': lo, 'intervalo_superior_95': hi,
                            'nota': 'Negativo favorece al modelo. Bootstrap por bloques de 7 días, 2000 réplicas; exploratorio.'})
    save_csv(pd.DataFrame(comparisons), out/'comparacion_incertidumbre.csv')


def plot_forecasts(daily, future, chosen, out):
    figs = out/'figuras_eda'
    figs.mkdir(exist_ok=True)
    test = daily[daily.etapa == 'prueba']
    fig, ax = plt.subplots(figsize=(12, 5))
    observed = test.drop_duplicates('fecha')
    ax.plot(observed.fecha, observed.real, color='black', label='Real (unidades estimadas)', lw=2)
    for name, g in test.groupby('modelo'):
        ax.plot(g.fecha, g.pronostico, label=name, alpha=.85)
    ax.set(title='Pronóstico de demanda del día siguiente: misma información en cada fecha', ylabel='Unidades estimadas')
    ax.legend(fontsize=8)
    save_fig(fig, figs/'04_pronosticos_comparados.png')
    fig, axes = plt.subplots(len(test.modelo.unique()), 1, figsize=(12, 9), sharex=True)
    for ax, (name, g) in zip(np.atleast_1d(axes), test.groupby('modelo')):
        ax.bar(g.fecha, g.real-g.pronostico, color='#168b88')
        ax.axhline(0, color='black', lw=.7); ax.set_ylabel(name, fontsize=8)
    fig.suptitle('Errores diarios en prueba: positivo indica demanda subestimada')
    save_fig(fig, figs/'09_errores_diarios.png')
    fig, ax = plt.subplots(figsize=(12, 4))
    for name, g in future.groupby('modelo'):
        ax.plot(g.fecha, g.pronostico, label=name, lw=2.5 if name == chosen['reposicion_30_dias'] else 1)
        if name == 'TimesFM 3.0':
            ax.fill_between(g.fecha, g.p10, g.p90, alpha=.15, label='TimesFM p10–p90 diario')
    ax.set(title='Pronóstico desde el último dato disponible; sin ventas futuras para verificar', ylabel='Unidades estimadas')
    ax.legend(fontsize=8)
    save_fig(fig, figs/'10_pronostico_30_dias.png')


def self_check(check_foundation=False):
    """Cambiar ventas futuras no puede cambiar cantidades ni predicciones pasadas."""
    dates = pd.date_range('2026-01-01', periods=90)
    o = pd.DataFrame({'Fecha': dates, 'Total orden': np.repeat(100., 90),
                      'Origen': 'prueba', 'N° Orden': np.arange(90)})
    changed = o.copy(); changed.loc[changed.index >= 75, 'Total orden'] = 10000
    a, b = prepare_orders(o), prepare_orders(changed)
    pd.testing.assert_series_equal(a.unidades_causales.iloc[:75], b.unidades_causales.iloc[:75])
    hist = pd.Series(2.+np.sin(np.arange(70)*2*np.pi/7), index=dates[:70])
    pred, _, _ = forecast_classic('M1 histórico', hist, dates[70:])
    assert np.allclose(pred, hist.iloc[-7:].mean())
    try:
        prepare_orders(pd.concat([o, o.iloc[[0]]]))
        raise AssertionError('Se aceptó una orden duplicada.')
    except ValueError:
        pass
    assert metric_values([0, 2], [1, 1])['MAE'] == 1
    if check_foundation:
        model = load_foundation()
        alone = foundation_forecasts(model, [hist])[0][0]
        other = pd.Series(np.repeat(1000., len(dates)), index=dates)
        with_other = foundation_forecasts(model, [hist, other])[0][0]
        np.testing.assert_allclose(alone, with_other, rtol=1e-5, atol=1e-5)
        print('TimesFM: orígenes aislados sin dependencia de otros historiales: OK')
    print('Controles de causalidad, duplicados y métricas: OK')


def write_report(out):
    """Informe portátil: contiene las imágenes y todas las filas del EDA diario."""
    out = Path(out)
    summary = json.loads((out/'eda_resumen.json').read_text(encoding='utf-8'))
    chosen = json.loads((out/'seleccion_modelos.json').read_text(encoding='utf-8'))
    daily = pd.read_csv(out/'eda_dia_a_dia.csv')
    metrics = pd.read_csv(out/'metricas_diarias.csv')
    totals = pd.read_csv(out/'metricas_30_dias.csv')
    def table(df):
        return df.to_html(index=False, border=0, float_format=lambda v: f'{v:.3f}', na_rep='—', escape=True)
    def csv_table(name, columns=None):
        df = pd.read_csv(out/name)
        return table(df[columns] if columns else df)
    def picture(name):
        path = out/'figuras_eda'/name
        if not path.exists():
            return ''
        encoded = base64.b64encode(path.read_bytes()).decode('ascii')
        return '<img alt="Gráfico del análisis de Noirpress" src="data:image/png;base64,'+encoded+'">'
    validation_start = daily.fecha.iloc[N_INITIAL]
    test_start = daily.fecha.iloc[int(len(daily)*.7)]
    cols = ['modelo', 'MAE', 'RMSE', 'sesgo_pred_menos_real', 'n']
    parts = [f'<h1>Pronóstico de demanda de Noirpress</h1><p>EDA diario y evaluación de Holt-Winters, SARIMAX y TimesFM 3.0. '
             f'Período observado: {summary["inicio"]} a {summary["fin"]}.</p>',
             '<nav><a href="#datos">Datos</a> · <a href="#modelos">Modelos</a> · <a href="#eda">EDA</a> · '
             '<a href="#diario">Buscar un día</a> · <a href="#limites">Límites</a></nav>',
             '<h2 id="datos">Qué información tenemos</h2>',
             f'<p>Son {summary["dias"]} días, {summary["pedidos_positivos"]} órdenes con importe positivo y '
             f'{summary["pedidos_descartados"]} órdenes sin cargo excluidas. El archivo original contiene '
             f'{summary["unidades_estimadas_retrospectivas"]:.0f} unidades estimadas con la mediana del mes completo. '
             f'La reconstrucción con referencia disponible hasta el día anterior da '
             f'{summary["unidades_estimadas_causales"]:.0f} unidades: cambia '
             f'{summary["dias_con_estimacion_distinta"]} día.</p>',
             '<p>La cantidad física vendida no figura explícitamente en la fuente. El objetivo de pronóstico '
             'es una aproximación a la demanda realizada. La mediana de importes de los 30 días anteriores '
             'se usa como referencia; si no hay operaciones en esa ventana, se usa toda la historia anterior. '
             'En el primer día sin referencia se asigna una unidad por orden positiva. '
             'La ausencia de órdenes en una fecha dentro del export se representa con cero, bajo el supuesto '
             'de que el registro es completo. No demuestra ausencia de demanda latente ni permite medir ventas perdidas.</p>',
             '<p>Se rechazan fechas inválidas, importes negativos/no finitos y órdenes duplicadas por canal e identificador. '
             'Los picos de ventas se conservan. Los importes atípicos quedan identificados para revisión. '
             'La auditoría por orden conserva ambos cálculos de unidades y su referencia pasada.</p>',
             '<h2 id="modelos">Cómo se evaluó el pronóstico de demanda</h2>',
             f'<p>Los primeros {N_INITIAL} días forman el historial inicial. La validación empieza el '
             f'{validation_start}; la prueba empieza el {test_start} y termina el {summary["fin"]}. '
             'En cada origen se utilizan únicamente los días cerrados anteriores y se pronostican hasta 30 días. '
             'Los cuatro métodos reciben las mismas fechas. Cada día se amplía el historial con ventas ya observadas.</p>',
             '<p>Se elige por MAE de validación y, en empate, RMSE. La selección para reposición minimiza '
             'el error absoluto de la suma de 30 días. Las ventanas de validación no cruzan al período de prueba. '
             'Hay 26 ventanas completas de validación y 25 de prueba; se superponen y no equivalen a 51 experimentos independientes. '
             'El error diario usa 55 y 54 fechas distintas respectivamente.</p>',
             '<ul><li>M1 histórico: promedio de los últimos siete días, constante durante cada horizonte emitido.</li>'
             '<li>Holt-Winters: suavizado exponencial con tendencia aditiva amortiguada y componente semanal aditivo de siete días.</li>'
             '<li>SARIMAX: orden (1,0,1), estacional (1,0,0,7), intercepto y dos variables externas conocidas: seno y coseno del día de semana.</li>'
             '<li>TimesFM 3.0: Transformer preentrenado de Google, ejecutado localmente sin ajuste fino sobre Noirpress; '
             'incorpora las mismas dos variables de calendario. Pesos y versión están fijados para reproducir el experimento.</li></ul>',
             '<p>Las configuraciones se fijaron antes de esta nueva ejecución; no se declara una búsqueda exhaustiva de hiperparámetros. '
             'Se registran advertencias y convergencia. No se inyectaron ventas futuras, importes futuros, publicidad ni dólar inexistentes. '
             'Los eventos se describen en el EDA; sus factores calculados sobre ventas futuras no se usan para pronosticar.</p>',
             f'<p>Modelo elegido en validación para el día siguiente: <strong>{escape(chosen["diario"])}</strong>. '
             f'Para demanda acumulada de 30 días: <strong>{escape(chosen["reposicion_30_dias"])}</strong>.</p>',
             '<h3>Validación diaria</h3>', table(metrics[metrics.etapa=='validacion'][cols]),
             '<h3>Prueba diaria posterior</h3>', table(metrics[metrics.etapa=='prueba'][cols]),
             '<p>MAE y RMSE se expresan en unidades estimadas por día. El sesgo negativo indica subestimación. '
             'La selección diaria previa se mantiene aunque otro modelo consiga mejor MAE en prueba; no se reelige con esos resultados.</p>',
             picture('04_pronosticos_comparados.png'),
             '<h3>Demanda acumulada en 30 días</h3>', table(totals[['etapa']+cols]),
             '<p>Estos errores están expresados en unidades acumuladas por ventana, no en unidades por día. '
             'Son la medida más cercana al supuesto de reposición de 30 días. No demuestran una reducción real de quiebres.</p>',
             '<h3>Incertidumbre de la comparación diaria</h3>', csv_table('comparacion_incertidumbre.csv'),
             '<p>El intervalo de diferencia de MAE se estima con bloques semanales. Si incluye cero, '
             'la muestra no separa claramente ese modelo del promedio móvil. Las fechas comerciales son escasas '
             'y pueden dominar el resultado; estos intervalos son exploratorios.</p>',
             '<h3>Errores día a día</h3>', picture('09_errores_diarios.png'),
             csv_table('cobertura_intervalos.csv'),
             '<p>p10 y p90 forman un intervalo predictivo marginal nominal del 80%. '
             'No debe sumarse ese intervalo para afirmar cobertura del total de 30 días; falta modelar '
             'la dependencia de los errores entre días. Se informa su cobertura observada, no se presupone calibración.</p>',
             '<h2 id="eda">Exploración de los 179 días</h2>',
             f'<p>Media {summary["media"]:.2f}, mediana {summary["mediana"]:.0f}, desvío {summary["desvio"]:.2f}. '
             f'Hay {summary["dias_sin_pedidos"]} días sin pedidos, con una racha máxima de '
             f'{summary["racha_sin_pedidos"]} días. El pico es {summary["maximo"]:.0f} unidades el '
             f'{summary["fecha_maximo"]}. Los diez días de mayor demanda reúnen '
             f'{summary["concentracion_10_dias_pct"]:.1f}% del volumen.</p>',
             picture('01_demanda_rolling.png'), picture('05_calendario_diario.png'),
             '<h3>Nivel mensual y días de semana</h3>', csv_table('eda_mensual.csv'),
             '<p>Enero comienza el día 3: no debe compararse su total como si fuera un mes completo. '
             'La media por día observado ayuda a distinguir duración y nivel de ventas.</p>',
             csv_table('eda_dia_semana.csv'), picture('02_demanda_dia_semana.png'),
             '<h3>Ventanas comerciales</h3>', csv_table('eda_eventos_descriptivos.csv'), picture('03_demanda_eventos.png'),
             '<p>Las ventanas promocional del 11 al 17 de mayo y previa al Día del Padre del 14 al 17 de junio '
             'provienen del análisis previo del proyecto. No se verificó aquí el calendario particular de campañas del comercio. '
             'Los cocientes contra días normales son asociaciones descriptivas: también pueden reflejar tendencia, '
             'publicidad o variaciones de canal. No hay campañas anuales repetidas para estimar un efecto estable.</p>',
             '<h3>Dependencia temporal y cambios de nivel</h3>', picture('06_autocorrelacion.png'),
             csv_table('eda_estacionariedad.csv', ['ambito','prueba','estadistico','p_valor','hipotesis_nula']),
             '<p>ADF parte de la hipótesis de raíz unitaria; KPSS parte de estacionariedad de nivel. '
             'Son pruebas complementarias y pueden discrepar en una serie corta con cambios de nivel y picos. '
             'Los resultados de toda la serie son descriptivos y no se utilizaron para seleccionar configuraciones con conocimiento del futuro.</p>',
             picture('07_descomposicion_stl.png'),
             '<p>STL separa tendencia, componente semanal y residuo. Se aplica a toda la serie para describirla; '
             'su suavizado puede usar observaciones posteriores y por eso no se introduce como variable de los modelos. '
             'La estructura semanal extraída no prueba por sí sola que sea estable o útil para pronosticar.</p>',
             '<h3>Qué cambia al estimar las unidades sin mirar el resto del mes</h3>', picture('08_sensibilidad_unidades.png')]
    sensitivity = out/'sensibilidad_pedidos'/'metricas_diarias.csv'
    if sensitivity.exists():
        sm = pd.read_csv(sensitivity)
        parts += ['<h3>Sensibilidad usando pedidos observados</h3>', table(sm[['etapa']+cols]),
                  '<p>Esta ejecución usa el número de órdenes con importe positivo, sin estimar unidades. '
                  'Sirve para evaluar sensibilidad al objetivo elegido; pedidos y unidades no son intercambiables '
                  'para calcular reposición física.</p>']
    view = daily[['fecha','dia_semana','pedidos','demanda','pedidos_revision','evento',
                  'media_pasada_7','media_pasada_30','alerta_pasada_30','lectura']]
    parts += ['<h2 id="diario">Buscar una fecha y explicar qué pasó</h2>',
              '<p>Las medias pasadas excluyen el día analizado. Los campos vacíos al inicio indican historial insuficiente.</p>',
              '<label for="buscar">Filtrar por fecha, evento o lectura</label> '
              '<input id="buscar" placeholder="Ejemplo: 2026-05-11" oninput="filtrar(this.value)">',
              '<div id="tablaDiaria">'+table(view)+'</div>',
              '<h2 id="limites">Alcance de las conclusiones</h2>',
              '<p>La serie fue explorada en versiones anteriores de la tesis. Esta evaluación respeta el orden temporal '
              'pero no equivale a una prueba prospectiva completamente nueva. TimesFM 3.0 se publicó después del período '
              'de ventas: la comparación es una reconstrucción retrospectiva con tecnología actual. '
              'No se afirma que estuviera disponible para tomar decisiones reales en abril de 2026.</p>',
              '<p>Los pesos de TimesFM 3.0 se usan para investigación no comercial, con licencia '
              'timesfm-non-commercial-license-v1.0. El código es Apache 2.0, pero esa licencia no habilita '
              'usar los pesos en producción comercial. No se realizó despliegue productivo.</p>',
              '<p>Las cantidades son estimadas; no se dispone de demanda no atendida ni tiempos reales de compra '
              'al proveedor. Los 30 días de reposición son un supuesto. Dólar, inversión publicitaria y precios '
              'futuros no se inventan. Un mejor MAE tampoco garantiza un menor costo total de inventario.</p>',
              '<h3>Pronóstico desde el último día observado</h3>', picture('10_pronostico_30_dias.png'),
              '<p>El origen es el 30 de junio de 2026 y el horizonte va del 1 al 30 de julio. '
              'No es un pronóstico actualizado a septiembre: no hay ventas posteriores al 30 de junio en las fuentes. '
              'Se muestra una proyección sin verificación y no se usa como evidencia de precisión.</p>',
              '<h3>Fuentes técnicas</h3><ul>'
              '<li><a href="https://github.com/google-research/timesfm">Google Research: TimesFM 3.0, uso y licencia</a></li>'
              '<li><a href="https://huggingface.co/google/timesfm-3.0-pytorch">Ficha oficial de pesos y preentrenamiento</a></li>'
              '<li><a href="https://www.research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/">Publicación de TimesFM 3, agosto de 2026</a></li>'
              '<li><a href="https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html">statsmodels SARIMAX</a></li>'
              '<li><a href="https://www.statsmodels.org/stable/generated/statsmodels.tsa.holtwinters.ExponentialSmoothing.html">statsmodels Holt-Winters</a></li></ul>']
    html = '<!doctype html><html lang="es"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
    html += '<title>Pronóstico de demanda Noirpress</title><style>body{max-width:1150px;margin:32px auto;padding:0 20px;font:16px/1.6 Arial;color:#172c3c}'
    html += 'h1,h2,h3{line-height:1.25}h2{margin-top:48px}table{border-collapse:collapse;font-size:13px;width:100%;margin:20px 0}'
    html += 'td,th{border:1px solid #d4dce0;padding:7px;text-align:left}th{background:#e9f2f3}tr:nth-child(even){background:#f7f9fa}'
    html += 'img{max-width:100%;height:auto}nav{position:sticky;top:0;background:white;padding:12px 0;border-bottom:1px solid #ddd}input{padding:8px;min-width:260px}'
    html += '#tablaDiaria{overflow:auto;max-height:700px}#tablaDiaria th{position:sticky;top:0}a{color:#007774}</style><body>'
    html += ''.join(parts)
    html += '<script>function filtrar(v){v=v.toLowerCase();document.querySelectorAll("#tablaDiaria tbody tr").forEach(r=>{r.style.display=r.textContent.toLowerCase().includes(v)?"":"none"})}</script></body></html>'
    (out/'Informe_pronostico_demanda.html').write_text(html, encoding='utf-8')
    return out/'Informe_pronostico_demanda.html'


def record_manifest(entrada, salida, device='cpu', scope='comparacion_completa'):
    packages = ['numpy', 'pandas', 'statsmodels', 'scipy', 'timesfm', 'torch']
    manifest = {'python': platform.python_version(), 'paquetes': {x: importlib.metadata.version(x) for x in packages},
                'sha256_entrada': hashlib.sha256(Path(entrada).read_bytes()).hexdigest(),
                'sha256_script': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'modelo': MODEL_ID, 'revision': REVISION, 'device': device, 'alcance': scope,
                'licencia_pesos': 'timesfm-non-commercial-license-v1.0',
                'evaluacion': 'retrospectiva con pesos publicados después del período de ventas',
                'inicial': N_INITIAL, 'fraccion_corte_prueba': .7, 'horizonte': HORIZON,
                'ejecucion': pd.Timestamp.now().isoformat()}
    (Path(salida)/'reproducibilidad.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--entrada', type=Path, default=INPUT)
    parser.add_argument('--salida', type=Path, default=ROOT/'resultados_noirpress')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--solo-eda', action='store_true')
    parser.add_argument('--sin-foundation', action='store_true', help='Diagnóstico de clásicos; no completa comparación moderna.')
    parser.add_argument('--self-check', action='store_true')
    parser.add_argument('--self-check-foundation', action='store_true')
    parser.add_argument('--sensibilidad-pedidos', action='store_true', help='Repite la comparación con pedidos observados como objetivo.')
    args = parser.parse_args()
    if args.self_check or args.self_check_foundation:
        self_check(args.self_check_foundation); return
    d = load_series(args.entrada)
    run_eda(d, args.salida)
    if not args.solo_eda:
        run_models(d, args.salida, args.device, not args.sin_foundation)
        if args.sensibilidad_pedidos:
            orders = d.copy()
            orders['demanda'] = orders.pedidos.astype(float)
            run_models(orders, args.salida/'sensibilidad_pedidos', args.device, not args.sin_foundation)
        write_report(args.salida)
    record_manifest(args.entrada, args.salida, args.device,
                    'solo_eda' if args.solo_eda else 'clasicos' if args.sin_foundation else 'comparacion_completa')


if __name__ == '__main__':
    main()
