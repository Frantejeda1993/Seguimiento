# 📦 Seguimiento de inventario

Aplicación Streamlit, CLI y pandas para convertir las hojas Excel **Stock**,
**Recepciones**, **Ventas** y opcionalmente **Stock_Value** en recomendaciones
revisables. `inventory_manager.py` contiene los cálculos, `dashboard.py` la UI
y `run_analysis.py` la ejecución por línea de comandos. El mapeo de columnas por
alias se conserva para los formatos históricos.

## Flujo y definiciones

- **Stock** es exclusivamente el stock físico de la hoja Stock. `Stock_Value`
  sólo aporta `Stock Valor` y permite reconciliar sus unidades con el físico.
- **Comprometido** = Cartera + Reservas.
- **Disponible Teorico** = Stock + Pendiente Recibir − Comprometido.
- **Faltante** = `max(0, −Disponible Teorico)`: venta comprometida sin cubrir.
- Pendiente Recibir usa `Total Pendiente Recibir` cuando está disponible; en
  caso contrario suma compra, fabricación y tránsito.

La situación vacía (incluido `NaN` y texto en blanco) es comprable. `D`
(descatalogado) y `O` (obsoleto) nunca generan pedido, incluso con faltante.

## Previsión y pedido

`forecasting.ForecastConfig` centraliza el horizonte H, nivel de servicio
`z=0.84`, WMA de seis meses, ventana de 12, winsorización al alza y tendencia
OLS con R² mínimo 0.5. Los patrones son `MADURO`, `NUEVO`, `REACTIVADO`,
`ESPORADICO` y `SIN_ROTACION`.

La fórmula es `necesidad = max(demanda prevista H, comprometido) + stock de
seguridad`; el pedido es la necesidad menos stock y recibir, redondeado. Por
ejemplo, stock 0, cartera 3, reservas 6 y recibir 6 deja Disponible Teorico
−3/Faltante 3. El pedido cubre como mínimo ese faltante para un SKU comprable.

La salida conserva la ejecución paralela: **PEDIDO ACTUAL** es el cálculo
histórico y **PEDIDO** el propuesto; el método seleccionado decide `VALOR
PEDIDO` y `MARGEN PEDIDO`. El KPI Expected Margin ahora es moneda:
`pedido × precio de venta medio × margen`, no el porcentaje de margen.

Las alertas son `FALTANTE_DO`, `LIQUIDACION` y `REVISAR`.

## Clientes

`clientes.build_clientes_table()` produce meses cerrados comparables, YTD/YTD
PY, L3M, pendiente y R² de seis meses, cuota y tendencia. `client_sku_drivers`
explica las variaciones mediante los artículos que suben o bajan.

## Uso

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
streamlit run dashboard.py
python run_analysis.py SEGUIMIENTO_3_0.xlsx --months 2.5
pytest
python backtest.py SEGUIMIENTO_3_0.xlsx --horizon 2 --output backtest.xlsx
```

El backtest walk-forward valida demanda, no disponibilidad: no hay histórico de
stock y las roturas pueden censurar ventas observadas.

## Modificar fórmulas

- `forecasting.forecast_sku()` y `calcular_pedido()` para la previsión.
- `InventoryManager._calculate_pedido_legacy()` para la comparación histórica.
- `clientes.build_clientes_table()` para los indicadores de clientes.
