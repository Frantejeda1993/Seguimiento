"""Análisis puro de clientes con periodos cerrados comparables."""
import numpy as np
import pandas as pd

MESES = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']


def _var(a, b):
    return (a - b) / b if b > 0 else (np.nan if a > 0 else 0.0)


def build_clientes_table(ventas: pd.DataFrame, as_of) -> pd.DataFrame:
    """Construye YTD comparable y tendencias de seis meses cerrados por cliente."""
    v = ventas.copy()
    v['_p'] = pd.to_datetime(dict(year=v['Año Factura'].astype(int), month=v['Mes Factura'].astype(int), day=1)).dt.to_period('M')
    end = min(v['_p'].max(), pd.Period(as_of, freq='M') - 1)
    year = end.year
    monthly = v.groupby(['Cliente', '_p'])['Importe Neto'].sum().unstack(fill_value=0)
    clients = pd.DataFrame({'Cod': monthly.index})
    clients['Cliente'] = clients.Cod.map(v.groupby('Cliente')['Nombre Cliente'].first())
    current_months = list(pd.period_range(f'{year}-01', end, freq='M'))
    for period in current_months:
        clients[MESES[period.month - 1]] = clients.Cod.map(monthly.get(period, pd.Series(0, index=monthly.index))).fillna(0)
    current = pd.Period(as_of, freq='M')
    clients['Mes en curso (parcial)'] = clients.Cod.map(monthly.get(current, pd.Series(0, index=monthly.index))).fillna(0)
    ytd = monthly.reindex(columns=current_months, fill_value=0).sum(axis=1)
    py_months = [p - 12 for p in current_months]
    ytd_py = monthly.reindex(columns=py_months, fill_value=0).sum(axis=1)
    clients['YTD'] = clients.Cod.map(ytd).fillna(0); clients['YTD_PY'] = clients.Cod.map(ytd_py).fillna(0)
    clients['Var_YTD_abs'] = clients.YTD - clients.YTD_PY
    clients['Var_YTD_%'] = [_var(a, b) * 100.0 if pd.notna(_var(a, b)) else np.nan for a, b in zip(clients.YTD, clients.YTD_PY)]
    last3 = list(pd.period_range(end - 2, end, freq='M'))
    clients['L3M'] = clients.Cod.map(monthly.reindex(columns=last3, fill_value=0).sum(axis=1)).fillna(0)
    clients['L3M_PY'] = clients.Cod.map(monthly.reindex(columns=[p - 12 for p in last3], fill_value=0).sum(axis=1)).fillna(0)
    clients['Var_L3M_YoY_%'] = [_var(a, b) * 100.0 if pd.notna(_var(a, b)) else np.nan for a, b in zip(clients.L3M, clients.L3M_PY)]
    six = monthly.reindex(columns=pd.period_range(end - 5, end, freq='M'), fill_value=0)
    x = np.arange(6)
    def regression(row):
        slope, intercept = np.polyfit(x, row.to_numpy(float), 1); fitted = slope * x + intercept; total = ((row - row.mean()) ** 2).sum()
        return pd.Series({'Pend_%/mes': slope / row.mean() if row.mean() > 0 else 0.0, 'R2': 1 - ((row - fitted) ** 2).sum() / total if total else 0.0})
    clients = clients.join(six.apply(regression, axis=1).set_index(clients.index))
    last_purchase = v[v['Importe Neto'] > 0].groupby('Cliente')['_p'].max()
    raw_months_sin_compra = (end - clients.Cod.map(last_purchase)).map(lambda value: value.n if pd.notna(value) else np.nan)
    clients['Meses_sin_compra'] = np.where(raw_months_sin_compra < 0, 0, raw_months_sin_compra)
    prior = monthly.reindex(columns=pd.period_range(end - 12, end - 3, freq='M'), fill_value=0).sum(axis=1).gt(0)
    has_prior = clients.Cod.map(prior).fillna(False)
    cv_6m = six.std(axis=1).div(six.mean(axis=1).replace(0, np.nan)); cv_6m = clients.Cod.map(cv_6m)
    clients['Tendencia'] = np.select(
        [
            (clients.YTD <= 0) & (clients.YTD_PY > 0),
            (clients.Meses_sin_compra >= 3) & has_prior,
            clients['Var_YTD_%'].isna(),
            (clients['Pend_%/mes'] > .02) & (clients.R2 >= .5),
            (clients['Pend_%/mes'] < -.02) & (clients.R2 >= .5),
            cv_6m > .5
        ],
        [
            'Inactivo',
            'Sin compra reciente',
            'Nuevo',
            'Creciente',
            'Decreciente',
            'Irregular'
        ],
        default='Estable'
    )
    total_ytd = clients.YTD.sum()
    clients['Cuota_%'] = (clients.YTD / total_ytd * 100.0).clip(lower=0.0) if total_ytd else 0.0
    clients[f'Año {year-2}'] = clients.Cod.map(monthly.reindex(columns=pd.period_range(f'{year-2}-01', f'{year-2}-12', freq='M'), fill_value=0).sum(axis=1)).fillna(0)
    clients[f'Año {year-1}'] = clients.Cod.map(monthly.reindex(columns=pd.period_range(f'{year-1}-01', f'{year-1}-12', freq='M'), fill_value=0).sum(axis=1)).fillna(0)
    
    # Ordenar de mayor a menor ventas
    clients = clients.sort_values(by=['YTD', 'YTD_PY'], ascending=[False, False]).reset_index(drop=True)
    clients['Ranking YTD'] = np.arange(1, len(clients) + 1)
    clients['Ranking PY'] = clients['YTD_PY'].rank(ascending=False, method='first').astype(int)
    clients['Cambio Ranking'] = clients['Ranking PY'] - clients['Ranking YTD']
    
    meses_con_compra = clients[[MESES[m - 1] for m in range(1, end.month + 1)]].gt(0).sum(axis=1)
    clients['Ticket Medio'] = np.where(meses_con_compra > 0, clients['YTD'] / meses_con_compra, 0)
    clients['Recurrencia %'] = (meses_con_compra / end.month * 100.0).round(1)
    
    cum_share = clients['Cuota_%'].cumsum()
    clients['ABC'] = np.select([cum_share <= 80.0, cum_share <= 95.0], ['A', 'B'], default='C')
    return clients


def client_sku_drivers(ventas: pd.DataFrame, clientes_sel: list, as_of, top: int = 10):
    """Devuelve los SKU que más elevan y reducen el importe YTD seleccionado."""
    cols = ['Artículo', 'Delta Importe Neto', 'Delta Unidades']
    empty_res = pd.DataFrame(columns=cols)
    if ventas is None or ventas.empty or not clientes_sel:
        return empty_res.copy(), empty_res.copy()

    v = ventas[ventas.Cliente.isin(clientes_sel)].copy()
    if v.empty:
        return empty_res.copy(), empty_res.copy()

    p = pd.to_datetime(dict(year=v['Año Factura'].astype(int), month=v['Mes Factura'].astype(int), day=1)).dt.to_period('M')
    max_p = p.max()
    as_of_prev = pd.Period(as_of, freq='M') - 1
    end = min(max_p, as_of_prev) if pd.notna(max_p) else as_of_prev

    a = v[(p.dt.year == end.year) & (p.dt.month <= end.month)].groupby('Artículo')[['Importe Neto', 'Unidades Venta']].sum()
    b = v[(p.dt.year == end.year - 1) & (p.dt.month <= end.month)].groupby('Artículo')[['Importe Neto', 'Unidades Venta']].sum()
    d = a.sub(b, fill_value=0).rename(columns={'Importe Neto': 'Delta Importe Neto', 'Unidades Venta': 'Delta Unidades'})
    return d.nlargest(top, 'Delta Importe Neto').reset_index(), d.nsmallest(top, 'Delta Importe Neto').reset_index()
