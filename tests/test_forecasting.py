import pytest
from forecasting import ForecastConfig, forecast_sku, calcular_pedido
@pytest.mark.parametrize('series,stock,expected', [([10,11,9,10,10,11,9,10,10],12,12),([10,11,12,14,16,18,20,22,24],30,23),([24,22,20,18,16,14,12,10,8],10,3),([10,11,10,12,11,10,11,10,50],15,14),([8,12,16],10,21)])
def test_orders(series,stock,expected):
 fc=forecast_sku(series); assert calcular_pedido(fc,stock,0,0,0,True)[0] == pytest.approx(expected,abs=1)
def test_special_patterns():
 assert forecast_sku([0]*12)['patron']=='SIN_ROTACION'
 assert forecast_sku([0]*11+[5])['patron']=='ESPORADICO'
 assert forecast_sku([0]*8+[3,5,8,12])['patron']=='REACTIVADO'
def test_fractional_horizon(): assert forecast_sku([10]*12,ForecastConfig(horizon_months=2.5))['demanda_H']==25
