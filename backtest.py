"""Walk-forward de demanda. No valida stock: no existe histórico de existencias y
las roturas de stock pueden censurar las ventas observadas."""
import argparse
import pandas as pd
import numpy as np
from forecasting import ForecastConfig, forecast_sku, tendencia_label
def run_backtest(ventas, recepciones=None, horizon=2, origins=None, methods=('actual','media_12m','ultimos_3m','propuesto')):
    """Compara previsiones contra unidades reales futuras por SKU y mes de origen."""
    v=ventas.copy(); v['_p']=pd.to_datetime(dict(year=v['Año Factura'].astype(int), month=v['Mes Factura'].astype(int), day=1)).dt.to_period('M'); months=pd.period_range(v._p.min(),v._p.max(),freq='M'); origins=origins or months[12:-int(np.ceil(horizon))]
    rows=[]
    for origin in origins:
      for sku,g in v.groupby('Artículo'):
       hist=g[g._p<=origin].groupby('_p')['Unidades Venta'].sum().reindex(pd.period_range(max(months[0],origin-11),origin,freq='M'),fill_value=0).values
       actual=g[g._p.isin(pd.period_range(origin+1,origin+int(np.ceil(horizon)),freq='M'))]['Unidades Venta'].sum()
       fc=forecast_sku(hist,ForecastConfig(horizon_months=horizon)); forecasts={'propuesto':fc['demanda_H'],'media_12m':hist.mean()*horizon,'ultimos_3m':hist[-3:].mean()*horizon,'actual':hist.mean()*horizon}
       for method in methods: rows.append({'origen':str(origin),'SKU':sku,'metodo':method,'previsto':forecasts[method],'real':actual,'Patrón demanda':fc['patron'],'Tendencia':tendencia_label(fc['tendencia_pct']),'Volumen 12M':hist.sum()})
    detail=pd.DataFrame(rows)
    if detail.empty:return detail,pd.DataFrame()
    def metrics(x):
      a=x.real.sum(); return pd.Series({'WAPE':(x.previsto-x.real).abs().sum()/a if a else np.nan,'sesgo':(x.previsto-x.real).sum()/a if a else np.nan,'sobreestimación':np.maximum(x.previsto-x.real,0).sum()/a if a else np.nan,'subestimación':np.maximum(x.real-x.previsto,0).sum()/a if a else np.nan})
    summary=detail.groupby('metodo').apply(metrics,include_groups=False).reset_index(); return detail,summary
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('excel');p.add_argument('--horizon',type=float,default=2);p.add_argument('--output',default='backtest.xlsx');a=p.parse_args();v=pd.read_excel(a.excel,sheet_name='3 - INPUT Ventas');d,s=run_backtest(v,horizon=a.horizon); 
 with pd.ExcelWriter(a.output) as w:d.to_excel(w,'DETALLE',index=False);s.to_excel(w,'RESUMEN',index=False)
