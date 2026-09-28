"""Análisis puro de clientes con periodos cerrados comparables."""
import numpy as np
import pandas as pd
MESES=['Ene','Feb','Mar','Abr','May','Jun','Jul','Ago','Sep','Oct','Nov','Dic']
def _var(a,b): return (a-b)/b if b>0 else (np.nan if a>0 else 0.)
def build_clientes_table(ventas, as_of):
    """Construye YTD comparable y tendencia OLS de seis meses cerrados por cliente."""
    v=ventas.copy(); v['_p']=pd.to_datetime(dict(year=v['Año Factura'].astype(int), month=v['Mes Factura'].astype(int), day=1)).dt.to_period('M'); end=min(v['_p'].max(),pd.Period(as_of,freq='M')-1); year=end.year
    clients=pd.DataFrame({'Cod':v['Cliente'].dropna().unique()}); clients['Cliente']=clients.Cod.map(v.groupby('Cliente')['Nombre Cliente'].first())
    for m in range(1,end.month+1): clients[MESES[m-1]]=clients.Cod.map(v[(v['_p'].dt.year==year)&(v['_p'].dt.month==m)].groupby('Cliente')['Importe Neto'].sum()).fillna(0)
    current=pd.Period(as_of,freq='M'); clients['Mes en curso (parcial)']=clients.Cod.map(v[v['_p']==current].groupby('Cliente')['Importe Neto'].sum()).fillna(0)
    ytd=v[(v['_p'].dt.year==year)&(v['_p'].dt.month<=end.month)].groupby('Cliente')['Importe Neto'].sum(); py=v[(v['_p'].dt.year==year-1)&(v['_p'].dt.month<=end.month)].groupby('Cliente')['Importe Neto'].sum(); clients['YTD']=clients.Cod.map(ytd).fillna(0); clients['YTD_PY']=clients.Cod.map(py).fillna(0); clients['Var_YTD_abs']=clients.YTD-clients.YTD_PY; clients['Var_YTD_%']=[_var(a,b) for a,b in zip(clients.YTD,clients.YTD_PY)]
    ps=pd.period_range(end-2,end,freq='M'); l3=v[v._p.isin(ps)].groupby('Cliente')['Importe Neto'].sum(); l3py=v[v._p.isin(ps-12)].groupby('Cliente')['Importe Neto'].sum(); clients['L3M']=clients.Cod.map(l3).fillna(0);clients['L3M_PY']=clients.Cod.map(l3py).fillna(0);clients['Var_L3M_YoY_%']=[_var(a,b) for a,b in zip(clients.L3M,clients.L3M_PY)]
    slopes=[]; r2s=[]; gaps=[]
    for c in clients.Cod:
      arr=v[v.Cliente==c].groupby('_p')['Importe Neto'].sum().reindex(pd.period_range(end-5,end,freq='M'),fill_value=0).values; x=np.arange(6); sl,ic=np.polyfit(x,arr,1); ss=((arr-(sl*x+ic))**2).sum(); tt=((arr-arr.mean())**2).sum(); slopes.append(sl/arr.mean() if arr.mean()>0 else 0);r2s.append(1-ss/tt if tt else 0)
      allp=v[v.Cliente==c]._p; gaps.append(next((i for i in range(0,100) if end-i not in set(allp)),100))
    clients['Pend_%/mes']=slopes;clients['R2']=r2s;clients['Meses_sin_compra']=gaps; clients['Tendencia']='Estable'
    for i,r in clients.iterrows():
      prior=((v.Cliente==r.Cod)&(v._p.between(end-12,end-3))).any(); last=v[v.Cliente==r.Cod].groupby('_p')['Importe Neto'].sum().reindex(pd.period_range(end-5,end,freq='M'),fill_value=0)
      if r.Meses_sin_compra>=3 and prior:t='Sin compra reciente'
      elif pd.isna(r['Var_YTD_%']):t='Nuevo'
      elif r['Pend_%/mes']>.02 and r.R2>=.5:t='Creciente'
      elif r['Pend_%/mes']<-.02 and r.R2>=.5:t='Decreciente'
      elif last.mean()>0 and last.std()/last.mean()>.5:t='Irregular'
      else:t='Estable'
      clients.at[i,'Tendencia']=t
    clients['Cuota_%']=clients.YTD/clients.YTD.sum() if clients.YTD.sum() else 0
    clients[f'Año {year-2}']=clients.Cod.map(v[v._p.dt.year==year-2].groupby('Cliente')['Importe Neto'].sum()).fillna(0); clients[f'Año {year-1}']=clients.Cod.map(v[v._p.dt.year==year-1].groupby('Cliente')['Importe Neto'].sum()).fillna(0)
    return clients
def client_sku_drivers(ventas, clientes_sel, as_of, top=10):
    """Devuelve los SKU que más elevan y reducen el importe YTD seleccionado."""
    v=ventas[ventas.Cliente.isin(clientes_sel)].copy(); end=min(pd.to_datetime(dict(year=v['Año Factura'].astype(int), month=v['Mes Factura'].astype(int), day=1)).dt.to_period('M').max(),pd.Period(as_of,freq='M')-1); p=pd.to_datetime(dict(year=v['Año Factura'].astype(int), month=v['Mes Factura'].astype(int), day=1)).dt.to_period('M'); a=v[(p.year==end.year)&(p.month<=end.month)].groupby('Artículo')[['Importe Neto','Unidades Venta']].sum(); b=v[(p.year==end.year-1)&(p.month<=end.month)].groupby('Artículo')[['Importe Neto','Unidades Venta']].sum(); d=a.sub(b,fill_value=0).rename(columns={'Importe Neto':'Delta Importe Neto','Unidades Venta':'Delta Unidades'}); return d.nlargest(top,'Delta Importe Neto').reset_index(),d.nsmallest(top,'Delta Importe Neto').reset_index()
