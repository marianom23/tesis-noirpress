"""Funciones de pronóstico utilizadas por la evaluación de Noirpress."""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX


MODEL_ID = 'google/timesfm-3.0-pytorch'


REVISION = '43046b85ec22d584a13f8098c2ed39c889e129c2'


HORIZON = 30


MODELS = ['M1 histórico', 'Holt-Winters', 'SARIMAX', 'TimesFM 3.0']


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


def save_csv(frame, path):
    frame.to_csv(path, index=False, encoding='utf-8-sig', date_format='%Y-%m-%d')
