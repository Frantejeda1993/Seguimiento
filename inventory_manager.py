"""
Inventory Management System - Core Module
Replicates Excel functionality with improved performance
"""

import pandas as pd
import numpy as np
from datetime import datetime, date
import warnings
from typing import Dict, Tuple
import re
import unicodedata


MARGIN_COLUMN = 'CR3: % Margen s/Venta + Transport'
LEGACY_MARGIN_COLUMN = 'CR2: %Margen s/Venta sin Transporte Athena'
MARGIN_COLUMN_ALIASES = (
    MARGIN_COLUMN,
    'CR3: %Margen s/Venta + Transport',
    'CR3: %Margen s/Venta + Transporte',
    'CR5: % Margen s/Venta + Marketing',
    'CR3: % Margen s/Venta + Transport',
    'CR3:% Margen s/Venta + Transport',
    LEGACY_MARGIN_COLUMN,
    'CR2: % Margen s/Venta sin Transporte Athena',
)

# All columns required from the Ventas input, including the margin column.
# Each key is the logical name shown to the user; the tuple lists accepted
# column aliases (checked in order, first match wins).
REQUIRED_VENTAS_COLUMNS: Dict[str, Tuple[str, ...]] = {
    'Artículo':             ('Artículo', 'Articulo'),
    'Clave 1':              ('Clave 1',),
    'Descripción Artículo': ('Descripción Artículo', 'Descripcion Articulo'),
    'Precio Coste':         ('Precio Coste',),
    'Nombre Cliente':       ('Nombre Cliente', 'Nombre_Cliente'),
    'Cliente':              ('Cliente',),
    'Año Factura':          ('Año Factura', 'Ano Factura', 'Año_Factura'),
    'Mes Factura':          ('Mes Factura', 'Mes_Factura'),
    'Importe Neto':         ('Importe Neto', 'Importe_Neto'),
    'Unidades Venta':       ('Unidades Venta', 'Unidades_Venta'),
    # Margen is validated here so it appears alongside the rest in the mapping UI
    'Margen':               MARGIN_COLUMN_ALIASES,
}

REQUIRED_STOCK_COLUMNS: Dict[str, Tuple[str, ...]] = {
    'Artículo':                    ('Artículo', 'Articulo'),
    'Situación':                   ('Situación', 'Situacion'),
    'Stock':                       ('Stock',),
    'Cartera':                     ('Cartera',),
    'Reservas':                    ('Reservas',),
    'Pendiente Recibir Compra':    ('Pendiente Recibir Compra',),
    'Pendiente Entrar Fabricación': (
        'Pendiente Entrar Fabricación',
        'Pendiente Entrar Fabricacion',
    ),
    'En Tránsito':                 ('En Tránsito', 'En Transito'),
}


def get_export_compras_columns(current_year: int) -> list[str]:
    """Columnas esenciales para el export del pedido."""
    return ['SKU', 'Marca', 'Descripción', 'Estado', 'Demanda 12M',
            f'Ventas {current_year - 2}', f'Ventas {current_year - 1}', f'Ventas {current_year}',
            'Demanda mensual prevista', 'Tendencia', 'Patrón demanda', 'Stock',
            'Pendiente Recibir', 'Comprometido', 'Disponible Teorico', 'Meses de Stock',
            'PEDIDO', 'PEDIDO ACTUAL', 'Dif PEDIDO', 'Precio Compra', 'VALOR PEDIDO',
            'MARGEN PEDIDO', 'Motivo', 'Alerta', 'Stock Seguridad']


class ColumnMappingError(Exception):
    def __init__(
        self,
        input_name: str,
        resolved: dict[str, str],
        missing: list[str],
        available: list[str],
    ):
        self.input_name = input_name
        self.resolved = resolved
        self.missing = missing
        self.available = available
        super().__init__(
            f"No se pudieron resolver columnas para '{input_name}'. "
            f"Faltan: {', '.join(missing) if missing else 'ninguna'}."
        )


class MultiColumnMappingError(Exception):
    """Raised when multiple inputs have unresolved columns at the same time."""

    def __init__(self, errors: list[ColumnMappingError]):
        self.errors = errors
        names = ', '.join(e.input_name for e in errors)
        super().__init__(f"Columnas sin resolver en: {names}")


class InventoryManager:
    """
    Main class for inventory management calculations.
    Replicates the Excel SEGUIMIENTO functionality.
    """

    def __init__(self, meses_compras: float = 2, today: date | datetime | None = None):
        self.meses_compras = float(meses_compras)
        self.today = today or date.today()
        self.current_month = self.today.month
        self.current_year = self.today.year

        self.stock_df = None
        self.recepciones_df = None
        self.ventas_df = None
        self.stock_value_df = None
        self.compras_df = None
        self.clientes_df = None
        self.extra_margin_aliases: tuple[str, ...] = tuple()

    def set_extra_margin_aliases(self, aliases: list[str] | tuple[str, ...] | None):
        """Allow dynamically configured margin aliases from the UI."""
        self.extra_margin_aliases = tuple(aliases or ())

    def load_data(
        self,
        stock_file: str = None,
        recepciones_file: str = None,
        ventas_file: str = None,
        stock_value_file: str = None,
        excel_file: str = None,
    ):
        if excel_file:
            self.stock_df = pd.read_excel(excel_file, sheet_name='1 - INPUT Stock')
            self.recepciones_df = pd.read_excel(excel_file, sheet_name='2 - INPUT Recepciones')
            self.ventas_df = pd.read_excel(excel_file, sheet_name='3 - INPUT Ventas')
            try:
                self.stock_value_df = pd.read_excel(excel_file, sheet_name='Stock_Value')
            except ValueError:
                self.stock_value_df = None
        else:
            if stock_file:
                self.stock_df = pd.read_csv(stock_file)
            if recepciones_file:
                self.recepciones_df = pd.read_csv(recepciones_file)
            if ventas_file:
                self.ventas_df = pd.read_csv(ventas_file)
            if stock_value_file:
                self.stock_value_df = pd.read_csv(stock_value_file)

        for attr in ('stock_df', 'recepciones_df', 'ventas_df', 'stock_value_df'):
            df = getattr(self, attr)
            if df is not None:
                df.columns = [self._normalize_column_name(c) for c in df.columns]

    # ------------------------------------------------------------------
    # Column-name helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_column_name(column_name: str) -> str:
        return re.sub(r"\s+", " ", str(column_name).strip())

    @staticmethod
    def _column_key(column_name: str) -> str:
        normalized = InventoryManager._normalize_column_name(column_name).casefold()
        normalized = unicodedata.normalize("NFKD", normalized)
        return "".join(ch for ch in normalized if not unicodedata.combining(ch))

    def _find_existing_column(self, df: pd.DataFrame, aliases: Tuple[str, ...]) -> str | None:
        key_to_column = {self._column_key(col): col for col in df.columns}
        for alias in aliases:
            found = key_to_column.get(self._column_key(alias))
            if found:
                return found
        return None

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def _resolve_columns(
        self,
        input_name: str,
        df: pd.DataFrame,
        required: Dict[str, Tuple[str, ...]],
        overrides: dict[str, str] | None = None,
    ) -> dict[str, str]:
        """
        Try to resolve every logical column name in *required* to an actual
        column present in *df*.  Applies *overrides* (logical_name → actual_col)
        before falling back to alias scanning.

        Returns a mapping  logical_name → actual_col  for all required fields.
        Raises ColumnMappingError listing every logical name that could not be
        resolved.
        """
        overrides = overrides or {}
        resolved: dict[str, str] = {}
        missing: list[str] = []

        for logical_name, aliases in required.items():
            # 1. User-supplied override takes highest priority
            override_col = overrides.get(logical_name)
            if override_col and override_col in df.columns:
                resolved[logical_name] = override_col
                continue

            # 2. Try aliases (including the override as the first alias if given)
            effective_aliases = (
                (override_col, *aliases) if override_col else aliases
            )
            found = self._find_existing_column(df, effective_aliases)
            if found:
                resolved[logical_name] = found
            else:
                missing.append(logical_name)

        if missing:
            raise ColumnMappingError(
                input_name=input_name,
                resolved=resolved,
                missing=missing,
                available=list(df.columns),
            )

        return resolved

    def _validate_all_inputs(
        self,
        column_overrides: dict[str, dict[str, str]],
    ) -> tuple[dict[str, str], dict[str, str]]:
        """
        Validate Ventas and Stock columns in a single pass.
        If either (or both) have unresolved columns, raises MultiColumnMappingError
        so the UI can show all problems at once.
        """
        ventas_required = dict(REQUIRED_VENTAS_COLUMNS)
        if self.extra_margin_aliases:
            ventas_required['Margen'] = (
                *self.extra_margin_aliases,
                *ventas_required['Margen'],
            )

        stock_required = dict(REQUIRED_STOCK_COLUMNS)

        errors: list[ColumnMappingError] = []
        ventas_map: dict[str, str] = {}
        stock_map: dict[str, str] = {}

        try:
            ventas_map = self._resolve_columns(
                'Ventas',
                self.ventas_df,
                ventas_required,
                overrides=column_overrides.get('Ventas', {}),
            )
        except ColumnMappingError as exc:
            errors.append(exc)

        try:
            stock_map = self._resolve_columns(
                'Stock',
                self.stock_df,
                stock_required,
                overrides=column_overrides.get('Stock', {}),
            )
        except ColumnMappingError as exc:
            errors.append(exc)

        if errors:
            raise MultiColumnMappingError(errors)

        return ventas_map, stock_map

    # ------------------------------------------------------------------
    # Public calculation methods
    # ------------------------------------------------------------------

    def calculate_compras(
        self,
        column_overrides: dict[str, dict[str, str]] | None = None,
        metodo_pedido: str = "propuesto",
    ) -> pd.DataFrame:
        """Calcula pedidos propuestos y conserva el algoritmo anterior para comparar."""
        from forecasting import (ForecastConfig, build_monthly_matrix, build_motivo,
                                 calcular_pedido, closed_end, forecast_sku, tendencia_label)
        if metodo_pedido not in {"propuesto", "actual"}:
            raise ValueError("metodo_pedido debe ser 'propuesto' o 'actual'")
        if self.ventas_df is None or self.stock_df is None:
            raise ValueError("Sales and stock data must be loaded first")
        ventas_map, stock_map = self._validate_all_inputs(column_overrides or {})
        ventas=self.ventas_df.rename(columns={v:k for k,v in ventas_map.items()}).copy()
        stock=self.stock_df.rename(columns={v:k for k,v in stock_map.items()}).copy()
        for c in ('Unidades Venta','Precio Coste','Importe Neto','Margen'): ventas[c]=pd.to_numeric(ventas[c],errors='coerce').fillna(0)
        for c in ('Stock','Cartera','Reservas','Pendiente Recibir Compra','Pendiente Entrar Fabricación','En Tránsito'): stock[c]=pd.to_numeric(stock[c],errors='coerce').fillna(0)
        skus=pd.Index(ventas['Artículo']).union(pd.Index(stock['Artículo'])).unique(); compras=pd.DataFrame({'SKU':skus})
        for target,col in [('Marca','Clave 1'),('Descripción','Descripción Artículo')]: compras[target]=compras.SKU.map(ventas.groupby('Artículo')[col].first())
        cutoff=closed_end(ventas,self.today); periods=pd.to_datetime(dict(year=ventas['Año Factura'].astype(int), month=ventas['Mes Factura'].astype(int), day=1)).dt.to_period('M'); recent=ventas.loc[(periods<=cutoff)&(periods>=cutoff-11)]
        base=recent if not recent.empty else ventas
        def weighted(frame, value, weight):
            """Media ponderada vectorizada, evitando un ``apply`` por SKU."""
            weights = frame[weight]
            numerator = (frame[value] * weights).groupby(frame['Artículo']).sum()
            denominator = weights.groupby(frame['Artículo']).sum().replace(0, np.nan)
            return numerator / denominator
        compras['Precio Compra']=compras.SKU.map(weighted(base,'Precio Coste','Unidades Venta')).fillna(compras.SKU.map(ventas.groupby('Artículo')['Precio Coste'].mean())).fillna(0)
        compras['Margen']=compras.SKU.map(weighted(base,'Margen','Importe Neto')).fillna(compras.SKU.map(ventas.groupby('Artículo')['Margen'].mean())).fillna(0)
        margen_es_porcentaje = compras.Margen.dropna().median() > 1.5
        revenue=base.groupby('Artículo')['Importe Neto'].sum(); units=base.groupby('Artículo')['Unidades Venta'].sum().replace(0,np.nan); compras['Precio Venta medio']=compras.SKU.map(revenue/units).fillna(0)
        indexed=stock.drop_duplicates('Artículo').set_index('Artículo'); compras['Estado']=compras.SKU.map(indexed['Situación']).fillna('')
        compras['Stock']=compras.SKU.map(indexed['Stock']).fillna(0); compras['Stock Unidades']=compras['Stock']
        compras=self._attach_stock_units_and_value(compras,stock)
        compras['Cartera']=compras.SKU.map(indexed['Cartera']).fillna(0); compras['Reservas']=compras.SKU.map(indexed['Reservas']).fillna(0); compras['Comprometido']=compras.Cartera+compras.Reservas; compras['Pendiente Servir']=compras.Comprometido
        compras['Pendiente Recibir']=self._calc_pending_receive(compras.SKU,stock); compras['Disponible Teorico']=compras.Stock+compras['Pendiente Recibir']-compras.Comprometido
        # Legacy output remains available, but it no longer supplies Meses de Stock.
        legacy=self._calculate_sales_metrics(compras.copy(),ventas); avg=f'Promedio {self.current_year-2} - {self.current_year}'; cur=f'Ventas {self.current_year}'; legacy['COMPRAR']=legacy.Estado.astype(str).str.strip().eq(''); compras['PEDIDO ACTUAL']=self._calculate_pedido_legacy(legacy,avg,cur)
        # Preserve the legacy monthly/annual sales fields in the purchase export.
        sales_columns = [c for c in legacy.columns if c.startswith('Ventas ') or c.startswith('Promedio ')]
        compras[sales_columns] = legacy[sales_columns]
        matrix=build_monthly_matrix(ventas,self.recepciones_df,cutoff); cfg=ForecastConfig(horizon_months=self.meses_compras)
        matrix_values = {sku: values for sku, values in zip(matrix.index, matrix.to_numpy())}
        records = [forecast_sku(matrix_values.get(sku, np.array([])), cfg) for sku in compras.SKU]
        fcdf=pd.DataFrame(records); compras['Meses activos']=fcdf.n_meses; compras['Meses con venta 12M']=fcdf.meses_con_venta; compras['Demanda prevista H']=fcdf.demanda_H; compras['Demanda mensual prevista']=fcdf.demanda_H/self.meses_compras; compras['Tendencia %/mes']=fcdf.tendencia_pct; compras['Tendencia']=fcdf.tendencia_pct.map(tendencia_label); compras['Patrón demanda']=fcdf.patron; compras['Stock Seguridad']=fcdf.ss
        compras['Demanda 12M']=compras.SKU.map(recent.groupby('Artículo')['Unidades Venta'].sum()).fillna(0)
        vals = [calcular_pedido(fc, stock, recibir, cartera, reservas, str(estado).strip() == '') for fc, stock, recibir, cartera, reservas, estado in zip(records, compras['Stock'].to_numpy(), compras['Pendiente Recibir'].to_numpy(), compras['Cartera'].to_numpy(), compras['Reservas'].to_numpy(), compras['Estado'].to_numpy())]
        compras['PEDIDO']=[v[0] for v in vals]; compras['Faltante']=[v[2] for v in vals]; compras['Necesidad']=[v[3] for v in vals]
        compras['Meses de Stock']=np.where(compras['Demanda mensual prevista']>0,np.maximum(compras['Disponible Teorico'],0)/compras['Demanda mensual prevista'],np.nan)
        compras['Dif PEDIDO']=compras.PEDIDO-compras['PEDIDO ACTUAL']; do=compras.Estado.astype(str).str.strip().isin(['D','O']); compras['Alerta']=np.select([do&(compras.Faltante>0),do&(compras.Stock>0)&compras['Patrón demanda'].isin(['SIN_ROTACION','ESPORADICO']),compras['Patrón demanda'].isin(['NUEVO','REACTIVADO','ESPORADICO'])&((compras.PEDIDO>0)|(compras.Faltante>0))],['FALTANTE_DO','LIQUIDACION','REVISAR'],'')
        compras['Meses de compra']=self.meses_compras; compras['Motivo']=compras.apply(build_motivo,axis=1); active='PEDIDO' if metodo_pedido=='propuesto' else 'PEDIDO ACTUAL'; compras['VALOR PEDIDO']=compras[active]*compras['Precio Compra']
        margen_pct = compras['Margen'].copy()
        if not margen_es_porcentaje:
            margen_pct *= 100.0
        factor_margen = 1.0 - ((100.0 - margen_pct) / 100.0)
        precio_venta = compras['Precio Venta medio'].copy()
        sin_venta = (precio_venta <= 0) & (compras['Precio Compra'] > 0) & factor_margen.between(0, 1, inclusive='neither')
        precio_venta.loc[sin_venta] = compras.loc[sin_venta, 'Precio Compra'] / (1.0 - factor_margen.loc[sin_venta])
        compras['MARGEN PEDIDO'] = compras[active] * precio_venta * factor_margen
        self.compras_df=compras; return compras

    def _calculate_pedido_legacy(self, compras, avg_3y_col, current_year_col):
        """Algoritmo histórico conservado exclusivamente en PEDIDO ACTUAL."""
        return self._calculate_pedido_vectorized(compras, avg_3y_col, current_year_col)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _attach_stock_units_and_value(
        self, compras: pd.DataFrame, stock: pd.DataFrame
    ) -> pd.DataFrame:
        if self.stock_value_df is not None and not self.stock_value_df.empty:
            article_col = self._find_existing_column(
                self.stock_value_df,
                ('Código Artículo', 'Codigo Articulo', 'Artículo', 'Articulo'),
            )
            units_col = self._find_existing_column(self.stock_value_df, ('Unidades',))
            amount_col = self._find_existing_column(self.stock_value_df, ('Importe',))

            if article_col and units_col and amount_col:
                agg = (
                    self.stock_value_df
                    .groupby(article_col, as_index=False)
                    .agg({
                        units_col: lambda s: pd.to_numeric(s, errors='coerce').fillna(0).sum(),
                        amount_col: lambda s: pd.to_numeric(s, errors='coerce').fillna(0).sum(),
                    })
                )
                compras['Stock Unidades (valor)'] = compras['SKU'].map(
                    agg.set_index(article_col)[units_col].to_dict()
                ).fillna(0)
                # Physical stock always comes from the Stock input sheet.
                compras['Stock Unidades'] = compras['SKU'].map(
                    stock.set_index('Artículo')['Stock'].to_dict()
                ).fillna(0)
                compras['Stock Valor'] = compras['SKU'].map(
                    agg.set_index(article_col)[amount_col].to_dict()
                ).fillna(0)
                return compras

        # Fall back to the Stock input sheet
        compras['Stock Unidades'] = compras['SKU'].map(
            stock.set_index('Artículo')['Stock'].to_dict()
        ).fillna(0)
        compras['Stock Valor'] = compras['Stock Unidades'] * compras['Precio Compra']
        compras['Stock Unidades (valor)'] = compras['Stock Unidades']
        return compras

    @staticmethod
    def reconcile_stock(compras: pd.DataFrame) -> pd.DataFrame:
        """Devuelve SKUs cuya valoración no coincide con el stock físico."""
        required = {'SKU', 'Stock Unidades', 'Stock Unidades (valor)'}
        if not required.issubset(compras.columns):
            return pd.DataFrame(columns=['SKU', 'Stock Unidades', 'Stock Unidades (valor)', 'Diferencia Stock'])
        result = compras.loc[(compras['Stock Unidades'] - compras['Stock Unidades (valor)']).abs() > 0,
                             ['SKU', 'Stock Unidades', 'Stock Unidades (valor)']].copy()
        result['Diferencia Stock'] = result['Stock Unidades'] - result['Stock Unidades (valor)']
        return result

    def _calc_pending_receive(
        self, skus: pd.Series, stock: pd.DataFrame
    ) -> pd.Series:
        stock_indexed = stock.set_index('Artículo')
        total_col = next(
            (
                col for col in stock_indexed.columns
                if self._normalize_column_name(col).casefold() == 'total pendiente recibir'
            ),
            None,
        )
        if total_col:
            mapping = stock_indexed[total_col].fillna(0).to_dict()
            return skus.map(lambda x: mapping.get(x, 0))

        pend = stock_indexed['Pendiente Recibir Compra'].fillna(0).to_dict()
        fab = stock_indexed['Pendiente Entrar Fabricación'].fillna(0).to_dict()
        trans = stock_indexed['En Tránsito'].fillna(0).to_dict()
        return skus.map(lambda x: pend.get(x, 0) + fab.get(x, 0) + trans.get(x, 0))

    def _calculate_sales_metrics(self, df: pd.DataFrame, ventas: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate sales metrics.  *ventas* must already have canonical column names.
        """
        ventas_grouped = (
            ventas
            .groupby(['Artículo', 'Año Factura', 'Mes Factura'])['Unidades Venta']
            .sum()
            .reset_index()
        )

        current_year = self.current_year
        current_month = self.current_month

        month_m2 = current_month - 2 if current_month > 2 else current_month - 2 + 12
        year_m2 = current_year if current_month > 2 else current_year - 1
        month_m1 = current_month - 1 if current_month > 1 else 12
        year_m1 = current_year if current_month > 1 else current_year - 1

        def _month_sales(yr, mo):
            return (
                ventas_grouped[
                    (ventas_grouped['Año Factura'] == yr)
                    & (ventas_grouped['Mes Factura'] == mo)
                ]
                .set_index('Artículo')['Unidades Venta']
            )

        df['Ventas -2 meses'] = df['SKU'].map(_month_sales(year_m2, month_m2)).fillna(0)
        df['Ventas -1 mes'] = df['SKU'].map(_month_sales(year_m1, month_m1)).fillna(0)
        df['Ventas mes'] = df['SKU'].map(_month_sales(current_year, current_month)).fillna(0)

        def _year_sales(yr):
            return (
                ventas_grouped[ventas_grouped['Año Factura'] == yr]
                .groupby('Artículo')['Unidades Venta']
                .sum()
            )

        sales_year = _year_sales(current_year)
        sales_prev = _year_sales(current_year - 1)
        sales_y2 = _year_sales(current_year - 2)

        df[f'Ventas {current_year}'] = df['SKU'].map(sales_year).fillna(0)
        df[f'Ventas {current_year - 1}'] = df['SKU'].map(sales_prev).fillna(0)
        df[f'Ventas {current_year - 2}'] = df['SKU'].map(sales_y2).fillna(0)

        # CV histórico sin producto cartesiano: los meses ausentes aportan cero.
        historical_years = [current_year - 2, current_year - 1]
        hist = ventas_grouped[ventas_grouped['Año Factura'].isin(historical_years)]
        hist_stats = hist.assign(_sq=hist['Unidades Venta'] ** 2).groupby('Artículo').agg(total=('Unidades Venta', 'sum'), total_sq=('_sq', 'sum'))
        n_periods = 24
        hist_stats['mean'] = hist_stats['total'] / n_periods
        hist_stats['std'] = np.sqrt(np.maximum((hist_stats['total_sq'] - hist_stats['total'] ** 2 / n_periods) / (n_periods - 1), 0))
        cv = (hist_stats['std'] / hist_stats['mean'].replace(0, np.nan)).fillna(0)
        current_values = df['SKU'].map(sales_year).fillna(0)
        previous_values = df['SKU'].map(sales_prev).fillna(0)
        previous2_values = df['SKU'].map(sales_y2).fillna(0)
        cv_values = df['SKU'].map(cv).fillna(0)
        valid_count = (previous_values > 0).astype(int) + (previous2_values > 0).astype(int)
        valid_mean = (previous_values.where(previous_values > 0, 0) + previous2_values.where(previous2_values > 0, 0)) / valid_count.replace(0, np.nan)
        annualized_current = current_values / current_month * 12 if current_month else current_values * 0
        annualized = np.where(valid_count.eq(0), annualized_current.where(current_values > 0, 0), np.where(cv_values > 1.5, np.maximum(current_values, valid_mean), annualized_current))

        sales_3y = (
            df['SKU'].map(sales_y2).fillna(0)
            + df['SKU'].map(sales_prev).fillna(0)
            + annualized
        ) / 3

        no_recent = (
            df['SKU'].map(sales_year).fillna(0).eq(0)
            & df['SKU'].map(sales_prev).fillna(0).eq(0)
        )
        sales_3y = sales_3y.where(~no_recent, 0)

        df[f'Promedio {current_year - 2} - {current_year}'] = sales_3y
        return df

    def _calculate_pedido_vectorized(
        self,
        compras: pd.DataFrame,
        avg_3y_col: str,
        current_year_col: str,
    ) -> np.ndarray:
        promedio_total = compras[avg_3y_col]
        ventas_corriente = compras[current_year_col]
        stock_actual = compras['Disponible Teorico']

        monthly_sales_total = ((promedio_total - ventas_corriente) / 12) * self.meses_compras
        monthly_need = monthly_sales_total - stock_actual

        decimal_part = np.abs(np.mod(monthly_need, 1))
        rounded = np.where(decimal_part >= 0.9, np.ceil(monthly_need), np.floor(monthly_need))

        should_zero = (
            (stock_actual >= monthly_sales_total)
            | (monthly_need < 0)
            | (~compras['COMPRAR'])
        )
        return np.where(should_zero, 0, rounded)

    # Used by run_analysis.py CLI; dashboard calls build_clientes_table directly.
    def calculate_clientes(self) -> pd.DataFrame:
        """Calcula la tabla única de clientes usando meses cerrados comparables."""
        if self.ventas_df is None: raise ValueError("Sales data must be loaded first")
        from clientes import build_clientes_table
        ventas_map=self._resolve_columns('Ventas', self.ventas_df, REQUIRED_VENTAS_COLUMNS)
        ventas=self.ventas_df.rename(columns={v:k for k,v in ventas_map.items()})
        self.clientes_df=build_clientes_table(ventas,self.today)
        return self.clientes_df

    # Used by run_analysis.py CLI; dashboard computes stats inline.
    def get_summary_stats(self) -> Dict:
        stats = {}

        if self.compras_df is not None:
            stats['total_stock_value'] = self.compras_df['Stock Valor'].sum()
            stats['total_pedido_value'] = self.compras_df['VALOR PEDIDO'].sum()
            stats['total_pedido_margin'] = self.compras_df['MARGEN PEDIDO'].sum()
            stats['items_to_order'] = int((self.compras_df['PEDIDO'] > 0).sum())
            stats['total_stock_units'] = self.compras_df['Stock Unidades'].sum()

        if self.clientes_df is not None:
            for col in (f'Año {self.current_year}', f'Año {self.current_year - 1}'):
                if col in self.clientes_df.columns:
                    stats[f'total_sales_{col}'] = self.clientes_df[col].sum()

        return stats

    def export_results(self, output_file: str = 'inventory_results.xlsx'):
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            if self.compras_df is not None:
                self.compras_df[[c for c in get_export_compras_columns(self.current_year) if c in self.compras_df]].to_excel(writer, sheet_name='COMPRAS', index=False)
            if self.clientes_df is not None:
                self.clientes_df.to_excel(writer, sheet_name='CLIENTES', index=False)
            if self.stock_df is not None:
                self.stock_df.to_excel(writer, sheet_name='Stock Input', index=False)
        print(f"Results exported to {output_file}")


if __name__ == "__main__":
    print("Inventory Manager Module - Ready to use")
    print("Import this module and use InventoryManager class")
