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

PATRON_LABELS = {
    'MADURO': 'Histórico continuo',
    'REACTIVADO': 'Reactivado tras inactividad',
    'NUEVO': 'Nuevo (<6 meses)',
    'ESPORADICO': 'Esporádico',
    'SIN_ROTACION': 'Sin rotación',
}

def forecast_sku(monthly, cfg=ForecastConfig()):
    """Predice unidades en H con WMA, tendencia amortiguada y stock de seguridad.

    La demanda es la suma de los niveles mensuales previstos; el stock de
    seguridad es ``z * sqrt(demanda_H)`` para una cobertura aproximada del 80%.
    """
    s=np.asarray(monthly,float)[-cfg.window:]; H=cfg.horizon_months
    out=dict(patron=PATRON_LABELS['SIN_ROTACION'],n_meses=len(s),meses_con_venta=int((s>0).sum()),nivel=0.,tendencia_abs=0.,tendencia_pct=0.,demanda_H=0.,ss=0.)
    if len(s)==0 or s.sum()<=0:return out
    r=_restart_index(s,cfg.gap_reset_months); reactivado=r>0; s=s[r:]; n=len(s); ms=int((s>0).sum()); out.update(n_meses=n,meses_con_venta=ms)
    young=n<cfg.young_months
    if reactivado and (n<3 or ms<3): out['patron']=PATRON_LABELS['ESPORADICO']; return out
    if young and n<=2:
        lvl=s.mean(); out.update(patron=PATRON_LABELS['NUEVO'],nivel=lvl,demanda_H=lvl*min(H,1.),ss=round(cfg.z*np.sqrt(max(0.,lvl*min(H,1.))), 1)); return out
    if ms<cfg.min_sales_months: out['patron']=PATRON_LABELS['ESPORADICO']; return out
    if n>=cfg.young_months:
        med=np.median(s); mad=1.4826*np.median(np.abs(s-med)); cap=med+cfg.winsor_k*max(mad,np.sqrt(med))
        if (s[-3:]>cap).sum()<2:s=np.minimum(s,cap)
    l=s[-cfg.wma_months:]; k=len(l); w=np.arange(1,k+1); wma=(l*w).sum()/w.sum(); lag=(w*(k-1-np.arange(k))).sum()/w.sum(); b=0.
    if k>=cfg.wma_months and l.var()>0:
        x=np.arange(k); sl,ic=np.polyfit(x,l,1); r2=1-((l-(sl*x+ic))**2).sum()/((l-l.mean())**2).sum()
        if r2>=cfg.trend_r2_min:b=sl*(cfg.phi_down if sl<0 else cfg.phi_up)
    dem=_horizon_sum(lambda j:max(0.,wma+b*(lag+j)),H)
    patron_key = 'REACTIVADO' if reactivado else ('NUEVO' if young else 'MADURO')
    out.update(patron=PATRON_LABELS[patron_key],nivel=wma,tendencia_abs=b,tendencia_pct=b/wma if wma>0 else 0.,demanda_H=dem,ss=round(cfg.z*np.sqrt(max(0.,dem)), 1))
    return out

def calcular_pedido(fc, stock, recibir, cartera, reservas, comprable):
    """Pedido = max(demanda_H - disponible_teorico, 0), redondeado.
    El stock de seguridad se mantiene a modo informativo pero no fuerza compras adicionales."""
    comprometido = cartera + reservas
    disponible = stock + recibir - comprometido
    faltante = max(0, -disponible)
    raw = max(0.0, fc['demanda_H'] - disponible)
    necesidad = fc['demanda_H'] + comprometido
    pedido = 0 if not comprable else int(max(0, np.floor(raw + .5)))
    return pedido, disponible, faltante, necesidad

def closed_end(ventas, today):
    """Devuelve el último mes cerrado presente en ventas, nunca el mes en curso."""
    periods=pd.to_datetime(dict(year=ventas['Año Factura'].astype(int), month=ventas['Mes Factura'].astype(int), day=1)).dt.to_period('M')
    return min(periods.max(),pd.Period(today,freq='M')-1)

def build_monthly_matrix(ventas, recepciones, closed_end):
    """Matriz SKU × meses cerrados construida con operaciones vectorizadas."""
    sales = ventas.copy()
    sales['_p'] = pd.to_datetime(dict(year=sales['Año Factura'].astype(int), month=sales['Mes Factura'].astype(int), day=1)).dt.to_period('M')
    sales = sales[sales['_p'] <= closed_end]
    matrix = sales.groupby(['Artículo', '_p'])['Unidades Venta'].sum().unstack(fill_value=0)
    if recepciones is not None and not recepciones.empty and {'Fecha Recepción', 'Artículo'}.issubset(recepciones.columns):
        rec = recepciones.copy(); rec['_p'] = pd.to_datetime(rec['Fecha Recepción']).dt.to_period('M'); rec = rec[rec['_p'] <= closed_end]
        rec_skus = set(rec['Artículo'].dropna()) - set(matrix.index)
        if rec_skus: matrix = pd.concat([matrix, pd.DataFrame(0, index=list(rec_skus), columns=matrix.columns)])
    if matrix.empty: return matrix
    return matrix.reindex(columns=pd.period_range(matrix.columns.min(), closed_end, freq='M'), fill_value=0)

def is_descatalogado_o_obsoleto(estado) -> bool:
    """Comprueba si el estado corresponde a descatalogado u obsoleto."""
    s = str(estado).strip().upper()
    if s in ('', 'NAN', 'NONE'):
        return False
    return s.startswith(('D', 'O')) or 'DESCATALOG' in s or 'OBSOLET' in s

def tendencia_label(value): return 'Creciente' if value>.02 else ('Decreciente' if value<-.02 else 'Estable')

def build_motivo(row):
    estado = str(row.get('Estado', '')).strip()
    disp = row.get('Disponible Teorico', 0)
    stock = row.get('Stock', 0)
    f = max(0, -disp)
    
    # 1. Artículos descatalogados u obsoletos
    if is_descatalogado_o_obsoleto(estado):
        if f > 0:
            return f"Situación {estado}: no se compra. Faltante vendido sin cubrir: {f:.0f} uds (Alerta FALTANTE_DO: requiere gestionar con proveedor o cancelar con cliente)."
        if stock > 0 or disp > 0:
            disp_liq = max(stock, disp)
            return f"Situación {estado}: no se compra. Stock disponible para liquidar: {disp_liq:.0f} uds (Alerta LIQUIDACION)."
        return f"Situación {estado}: no se compra."

    # 2. Artículos sin rotación
    p = str(row.get('Patrón demanda', ''))
    suffix_f = f" Faltante vendido sin cubrir: {f:.0f} uds." if f > 0 else ""
    if p in ('Sin rotación', 'SIN_ROTACION'):
        return f"Sin ventas en los últimos 12 meses: no se compra.{suffix_f}"

    # 3. Explicación de revisión para la alerta REVISAR
    revisar_motivo = ""
    if p in ('Nuevo (<6 meses)', 'NUEVO'):
        meses_act = row.get('Meses activos', row.get('n_meses', 0))
        revisar_motivo = f" [REVISAR: Producto nuevo con solo {meses_act} meses de datos; verificar si la demanda inicial se mantendrá antes de pedir]."
    elif p in ('Reactivado tras inactividad', 'REACTIVADO'):
        revisar_motivo = " [REVISAR: Reactivado tras +6 meses sin compras; validar si la venta reciente fue puntual o recurrente antes de cursar pedido]."
    elif p in ('Esporádico', 'ESPORADICO'):
        revisar_motivo = " [REVISAR: Demanda esporádica (<3 meses con venta en el último año); confirmar necesidad con cliente antes de comprar]."

    pedido = row.get('PEDIDO', 0)
    demanda_col = row.get('Demanda prevista meses de compra', row.get('Demanda prevista H', 0))
    pend_servir = row.get('Pendiente Servir', row.get('Comprometido', 0))

    if pedido > 0:
        base_msg = (
            f"Comprar {pedido:.0f}: demanda prevista {demanda_col:.0f} uds en {row.get('Meses de compra',0):g} meses ({row.get('Tendencia','Estable')}), "
            f"pendiente servir {pend_servir:.0f}, disponible teórico {disp:.0f} (stock {stock:.0f} + recibir {row.get('Pendiente Recibir',0):.0f} - servir {pend_servir:.0f}), "
            f"stock seguridad informativo {row.get('Stock Seguridad',0):.0f}."
        )
        return base_msg + suffix_f + revisar_motivo

    # Pedido 0
    if p in ('Esporádico', 'ESPORADICO'):
        return f"Ventas esporádicas (<3 meses con venta en 12): sin compra automática.{suffix_f}{revisar_motivo if f > 0 else ''}"

    return f"No comprar: disponible teórico {disp:.0f} cubre la demanda prevista de {demanda_col:.0f} uds.{suffix_f}{revisar_motivo if f > 0 else ''}"
