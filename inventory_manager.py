"""
Inventory Management System - Core Module
Replicates Excel functionality with improved performance
"""

import pandas as pd
import numpy as np
from datetime import datetime
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

    def __init__(self, meses_compras: float = 2):
        self.meses_compras = float(meses_compras)
        self.current_month = datetime.now().month
        self.current_year = datetime.now().year

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
        contemplar_sobre_stock: bool = False,
        column_overrides: dict[str, dict[str, str]] | None = None,
    ) -> pd.DataFrame:
        if self.ventas_df is None or self.stock_df is None:
            raise ValueError("Sales and stock data must be loaded first")

        column_overrides = column_overrides or {}

        # Resolve + validate all columns at once
        ventas_map, stock_map = self._validate_all_inputs(column_overrides)

        # ---- Rename ventas/stock columns to canonical names ----
        ventas = self.ventas_df.rename(columns={v: k for k, v in ventas_map.items()})
        stock = self.stock_df.rename(columns={v: k for k, v in stock_map.items()})

        # Get unique SKUs from sales
        skus = ventas['Artículo'].unique()
        compras = pd.DataFrame({'SKU': skus})

        # Brand / Marca
        compras['Marca'] = compras['SKU'].map(ventas.groupby('Artículo')['Clave 1'].first())

        # Description
        compras['Descripción'] = compras['SKU'].map(
            ventas.groupby('Artículo')['Descripción Artículo'].first()
        )

        # Purchase price
        compras['Precio Compra'] = compras['SKU'].map(
            ventas.groupby('Artículo')['Precio Coste'].mean()
        )

        # Margin — now always available via the canonical 'Margen' column
        compras['Margen'] = compras['SKU'].map(
            ventas.groupby('Artículo')['Margen'].mean()
        )

        # Stock status
        compras['Estado'] = compras['SKU'].map(
            stock.set_index('Artículo')['Situación'].to_dict()
        )

        # Last reception date
        if self.recepciones_df is not None and not self.recepciones_df.empty:
            last_recep = self.recepciones_df.groupby('Artículo')['Fecha Recepción'].max()
            compras['Ultima recepción'] = compras['SKU'].map(last_recep)
        else:
            compras['Ultima recepción'] = None

        # Stock units and value — prefer dedicated stock_value sheet
        compras = self._attach_stock_units_and_value(compras, stock)

        # Pending to serve (Cartera + Reservas)
        cartera_map = stock.set_index('Artículo')['Cartera'].fillna(0).to_dict()
        reservas_map = stock.set_index('Artículo')['Reservas'].fillna(0).to_dict()
        compras['Pendiente Servir'] = compras['SKU'].map(
            lambda x: cartera_map.get(x, 0) + reservas_map.get(x, 0)
        )

        # Pending to receive
        compras['Pendiente Recibir'] = self._calc_pending_receive(compras['SKU'], stock)

        # Theoretical available
        compras['Disponible Teorico'] = (
            compras['Stock Unidades']
            + compras['Pendiente Recibir']
            - compras['Pendiente Servir']
        )

        # Sales metrics (uses canonical ventas columns)
        compras = self._calculate_sales_metrics(compras, ventas)

        # Months of stock
        avg_3y_col = f'Promedio {self.current_year - 2} - {self.current_year}'
        current_year_col = f'Ventas {self.current_year}'

        denom = compras[avg_3y_col] - compras['Disponible Teorico'] - compras[current_year_col]
        compras['Meses de Stock'] = np.where(denom > 0, compras['Disponible Teorico'] / denom, 0)

        # Purchase flag
        compras['COMPRAR'] = compras['Estado'].isna()

        # Order quantity (vectorized)
        compras['PEDIDO'] = self._calculate_pedido_vectorized(
            compras, contemplar_sobre_stock, avg_3y_col, current_year_col
        )

        compras['VALOR PEDIDO'] = compras['PEDIDO'] * compras['Precio Compra']
        compras['MARGEN PEDIDO'] = compras['PEDIDO'] * compras['Margen']

        self.compras_df = compras
        return compras

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
                compras['Stock Unidades'] = compras['SKU'].map(
                    agg.set_index(article_col)[units_col].to_dict()
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
        return compras

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

        # Three-year average with seasonality detection (vectorized)
        historical_years = [current_year - 2, current_year - 1]
        all_skus = df['SKU'].unique()

        timeline_df = pd.MultiIndex.from_product(
            [all_skus, historical_years, range(1, 13)],
            names=['Artículo', 'Año Factura', 'Mes Factura'],
        ).to_frame(index=False)

        historical_ventas = ventas_grouped[
            ventas_grouped['Año Factura'].isin(historical_years)
        ][['Artículo', 'Año Factura', 'Mes Factura', 'Unidades Venta']].copy()

        complete_sales = timeline_df.merge(
            historical_ventas,
            on=['Artículo', 'Año Factura', 'Mes Factura'],
            how='left',
        )
        complete_sales['Unidades Venta'] = complete_sales['Unidades Venta'].fillna(0)

        cv_stats = complete_sales.groupby('Artículo')['Unidades Venta'].agg(['mean', 'std'])
        cv_stats['cv'] = cv_stats['std'] / cv_stats['mean'].replace(0, np.nan)
        cv_stats['cv'] = cv_stats['cv'].fillna(0)

        def get_annualized_value(sku):
            current_sales = sales_year.get(sku, 0)
            prev_year_val = sales_prev.get(sku, 0)
            prev_2_year_val = sales_y2.get(sku, 0)
            cv_value = cv_stats.loc[sku, 'cv'] if sku in cv_stats.index else 0
            valid_years = [y for y in [prev_2_year_val, prev_year_val] if y > 0]

            if not valid_years:
                return (current_sales / current_month) * 12 if current_month > 0 and current_sales > 0 else 0

            if cv_value > 1.5:
                return max(current_sales, np.mean(valid_years))
            else:
                return (current_sales / current_month) * 12 if current_month > 0 else 0

        annualized = df['SKU'].apply(get_annualized_value)

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
        contemplar_sobre_stock: bool,
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
            | ((monthly_need < 0) & (not contemplar_sobre_stock))
            | (~compras['COMPRAR'])
        )
        return np.where(should_zero, 0, rounded)

    def calculate_clientes(self) -> pd.DataFrame:
        if self.ventas_df is None:
            raise ValueError("Sales data must be loaded first")

        customers = self.ventas_df['Cliente'].unique()
        clientes = pd.DataFrame({'Cod': customers})

        name_map = self.ventas_df.groupby('Cliente')['Nombre Cliente'].first()
        clientes['Cliente'] = clientes['Cod'].map(name_map)

        for year_offset in [2, 1, 0]:
            year = self.current_year - year_offset
            year_sales = (
                self.ventas_df[self.ventas_df['Año Factura'] == year]
                .groupby('Cliente')['Importe Neto']
                .sum()
            )
            clientes[f'Año {year}'] = clientes['Cod'].map(year_sales).fillna(0)

        year_cols = [f'Año {self.current_year - i}' for i in [2, 1, 0]]

        clientes[f'Dif {self.current_year - 2} - {self.current_year - 1}'] = clientes.apply(
            lambda row: (row[year_cols[1]] - row[year_cols[0]]) / row[year_cols[0]]
            if row[year_cols[0]] != 0 else 1,
            axis=1,
        )
        clientes[f'Dif {self.current_year - 1} - {self.current_year}'] = clientes.apply(
            lambda row: (row[year_cols[2]] - row[year_cols[1]]) / row[year_cols[1]]
            if row[year_cols[1]] != 0 else 1,
            axis=1,
        )

        self.clientes_df = clientes
        return clientes

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
                self.compras_df.to_excel(writer, sheet_name='COMPRAS', index=False)
            if self.clientes_df is not None:
                self.clientes_df.to_excel(writer, sheet_name='CLIENTES', index=False)
            if self.stock_df is not None:
                self.stock_df.to_excel(writer, sheet_name='Stock Input', index=False)
        print(f"Results exported to {output_file}")


if __name__ == "__main__":
    print("Inventory Manager Module - Ready to use")
    print("Import this module and use InventoryManager class")
