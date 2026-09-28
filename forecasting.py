"""Previsión mensual pura para recomendaciones de compra."""
from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class ForecastConfig:
    horizon_months: float = 2.0
    z: float = 0.84
    phi_up: float = 0.5
    phi_down: float = 1.0
    winsor_k: float = 2.0
    window: int = 12
    wma_months: int = 6
    trend_r2_min: float = 0.5
    min_sales_months: int = 3
    gap_reset_months: int = 6
    young_months: int = 6

def _horizon_sum(f_of_k, H):
    full = int(np.floor(H)); frac = H-full
    total = sum(f_of_k(k) for k in range(1, full+1))
    return total + (frac * f_of_k(full+1) if frac > 1e-9 else 0)

def _restart_index(s, gap):
    for i in range(len(s)-1, gap-1, -1):
        if s[i] > 0 and not s[i-gap:i].any(): return i
    return 0

def forecast_sku(monthly, cfg=ForecastConfig()):
    """Predice unidades en H con WMA, tendencia amortiguada y stock de seguridad.

    La demanda es la suma de los niveles mensuales previstos; el stock de
    seguridad es ``z * sqrt(demanda_H)`` para una cobertura aproximada del 80%.
    """
    s=np.asarray(monthly,float)[-cfg.window:]; H=cfg.horizon_months
    out=dict(patron='SIN_ROTACION',n_meses=len(s),meses_con_venta=int((s>0).sum()),nivel=0.,tendencia_abs=0.,tendencia_pct=0.,demanda_H=0.,ss=0.)
    if len(s)==0 or s.sum()<=0:return out
    r=_restart_index(s,cfg.gap_reset_months); reactivado=r>0; s=s[r:]; n=len(s); ms=int((s>0).sum()); out.update(n_meses=n,meses_con_venta=ms)
    young=n<cfg.young_months
    if reactivado and (n<3 or ms<3): out['patron']='ESPORADICO'; return out
    if young and n<=2:
        lvl=s.mean(); out.update(patron='NUEVO',nivel=lvl,demanda_H=lvl*min(H,1.),ss=0.); return out
    if ms<cfg.min_sales_months: out['patron']='ESPORADICO'; return out
    if n>=cfg.young_months:
        med=np.median(s); mad=1.4826*np.median(np.abs(s-med)); cap=med+cfg.winsor_k*max(mad,np.sqrt(med))
        if (s[-3:]>cap).sum()<2:s=np.minimum(s,cap)
    l=s[-cfg.wma_months:]; k=len(l); w=np.arange(1,k+1); wma=(l*w).sum()/w.sum(); lag=(w*(k-1-np.arange(k))).sum()/w.sum(); b=0.
    if k>=cfg.wma_months and l.var()>0:
        x=np.arange(k); sl,ic=np.polyfit(x,l,1); r2=1-((l-(sl*x+ic))**2).sum()/((l-l.mean())**2).sum()
        if r2>=cfg.trend_r2_min:b=sl*(cfg.phi_down if sl<0 else cfg.phi_up)
    dem=_horizon_sum(lambda j:max(0.,wma+b*(lag+j)),H)
    out.update(patron='REACTIVADO' if reactivado else ('NUEVO' if young else 'MADURO'),nivel=wma,tendencia_abs=b,tendencia_pct=b/wma if wma>0 else 0.,demanda_H=dem,ss=cfg.z*np.sqrt(dem))
    return out

def calcular_pedido(fc, stock, recibir, cartera, reservas, comprable):
    """Pedido = max(necesidad - stock - recibir, 0), redondeado; necesidad es
    max(demanda prevista, comprometido) + stock de seguridad."""
    comprometido=cartera+reservas; disponible=stock+recibir-comprometido; faltante=max(0,-disponible)
    necesidad=max(fc['demanda_H'],comprometido)+fc['ss']; raw=necesidad-(stock+recibir)
    return (0 if not comprable else int(max(0,np.floor(raw+.5)))), disponible, faltante, necesidad

def closed_end(ventas, today):
    """Devuelve el último mes cerrado presente en ventas, nunca el mes en curso."""
    periods=pd.to_datetime(dict(year=ventas['Año Factura'].astype(int), month=ventas['Mes Factura'].astype(int), day=1)).dt.to_period('M')
    return min(periods.max(),pd.Period(today,freq='M')-1)

def build_monthly_matrix(ventas, recepciones, closed_end):
    """Matriz SKU × meses cerrados, sin ceros antes de la primera actividad."""
    sales=ventas.copy(); sales['_p']=pd.to_datetime(dict(year=sales['Año Factura'].astype(int), month=sales['Mes Factura'].astype(int), day=1)).dt.to_period('M')
    sales=sales[sales['_p']<=closed_end]; rec=pd.DataFrame(columns=['Artículo','_p'])
    if recepciones is not None and not recepciones.empty and 'Fecha Recepción' in recepciones:
        rec=recepciones.copy(); rec['_p']=pd.to_datetime(rec['Fecha Recepción']).dt.to_period('M'); rec=rec[rec['_p']<=closed_end]
    rows={}
    for sku in set(sales['Artículo']).union(rec['Artículo'] if 'Artículo' in rec else []):
        sp=sales.loc[sales['Artículo']==sku]; rp=rec.loc[rec['Artículo']==sku] if 'Artículo' in rec else rec
        starts=[x for x in [sp['_p'].min() if not sp.empty else None,rp['_p'].min() if not rp.empty else None] if pd.notna(x)]
        if not starts: continue
        start=max(min(starts), min(starts))
        idx=pd.period_range(start,closed_end,freq='M'); rows[sku]=sp.groupby('_p')['Unidades Venta'].sum().reindex(idx,fill_value=0)
    return pd.DataFrame(rows).T.reindex(columns=pd.period_range(min((x.index.min() for x in rows.values()), default=closed_end),closed_end,freq='M'))

def tendencia_label(value): return 'Creciente' if value>.02 else ('Decreciente' if value<-.02 else 'Estable')
def build_motivo(row):
    estado=str(row.get('Estado','')).strip(); f=row.get('Faltante',0); suffix=f' Faltante vendido sin cubrir: {f:.0f} uds.' if f>0 else ''
    if estado in ('D','O'): return f'Situación {estado}: no se compra.'+suffix
    p=row.get('Patrón demanda','');
    if p=='SIN_ROTACION': return 'Sin ventas en los últimos 12 meses: no se compra.'+suffix
    if p=='ESPORADICO': return 'Ventas esporádicas (<3 meses con venta en 12): sin compra automática.'+suffix
    if p=='NUEVO': return f"Producto nuevo ({row.get('Meses activos', row.get('n_meses',0))} meses de datos): previsión basada en ventas recientes, revisar."+suffix
    if row.get('PEDIDO',0)>0:return f"Comprar {row['PEDIDO']:.0f}: demanda prevista {row.get('Demanda prevista H',0):.0f} uds en {row.get('Meses de compra',0):g} meses ({row.get('Tendencia','Estable')}), comprometido {row.get('Comprometido',0):.0f}, stock de seguridad {row.get('Stock Seguridad',0):.0f}, disponible {row.get('Stock',0)+row.get('Pendiente Recibir',0):.0f} (stock {row.get('Stock',0):.0f} + recibir {row.get('Pendiente Recibir',0):.0f})."+suffix
    return f"No comprar: disponible {row.get('Stock',0)+row.get('Pendiente Recibir',0):.0f} cubre la necesidad de {row.get('Necesidad',0):.0f} uds."+suffix
