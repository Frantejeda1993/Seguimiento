"""
Inventory Management System - Core Module
Replicates Excel functionality with improved performance
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Tuple
import re


MARGIN_COLUMN = 'CR3: % Margen s/Venta + Transport'
LEGACY_MARGIN_COLUMN = 'CR2: %Margen s/Venta sin Transporte Athena'
MARGIN_COLUMN_ALIASES = (
    MARGIN_COLUMN,
    'CR3: %Margen s/Venta + Transport',
    'CR3:% Margen s/Venta + Transport',
    LEGACY_MARGIN_COLUMN,
    'CR2: % Margen s/Venta sin Transporte Athena',
)


class InventoryManager:
    """
    Main class for inventory management calculations.
    Replicates the Excel SEGUIMIENTO functionality.
    """
    
    def __init__(self, meses_compras: float = 2):
        """
        Initialize the inventory manager.
        
        Args:
            meses_compras: Number of months to calculate purchase recommendations
        """
        self.meses_compras = float(meses_compras)
        self.current_month = datetime.now().month
        self.current_year = datetime.now().year
        
        # DataFrames
        self.stock_df = None
        self.recepciones_df = None
        self.ventas_df = None
        self.compras_df = None
        self.clientes_df = None
    
    def load_data(self, stock_file: str = None, recepciones_file: str = None, 
                  ventas_file: str = None, excel_file: str = None):
        """
        Load data from CSV files or Excel sheets.
        
        Args:
            stock_file: Path to stock CSV
            recepciones_file: Path to recepciones CSV
            ventas_file: Path to ventas CSV
            excel_file: Path to Excel file (if using Excel format)
        """
        if excel_file:
            # Load from Excel workbook
            self.stock_df = pd.read_excel(excel_file, sheet_name='1 - INPUT Stock')
            self.recepciones_df = pd.read_excel(excel_file, sheet_name='2 - INPUT Recepciones')
            self.ventas_df = pd.read_excel(excel_file, sheet_name='3 - INPUT Ventas')
        else:
            # Load from CSV files
            if stock_file:
                self.stock_df = pd.read_csv(stock_file)
            if recepciones_file:
                self.recepciones_df = pd.read_csv(recepciones_file)
            if ventas_file:
                self.ventas_df = pd.read_csv(ventas_file)
        
        # Clean column names
        if self.stock_df is not None:
            self.stock_df.columns = [self._normalize_column_name(col) for col in self.stock_df.columns]
        if self.recepciones_df is not None:
            self.recepciones_df.columns = [self._normalize_column_name(col) for col in self.recepciones_df.columns]
        if self.ventas_df is not None:
            self.ventas_df.columns = [self._normalize_column_name(col) for col in self.ventas_df.columns]

    @staticmethod
    def _normalize_column_name(column_name: str) -> str:
        """Normalize input column names removing line breaks and duplicated spaces."""
        return re.sub(r"\s+", " ", str(column_name).strip())
    

    def _resolve_margin_column(self) -> str:
        """Resolve margin column name allowing legacy labels and spacing variants."""
        normalized_to_original = {
            self._normalize_column_name(col).casefold(): col
            for col in self.ventas_df.columns
        }

        for alias in MARGIN_COLUMN_ALIASES:
            normalized_alias = self._normalize_column_name(alias).casefold()
            if normalized_alias in normalized_to_original:
                return normalized_to_original[normalized_alias]

        for col in self.ventas_df.columns:
            normalized_col = self._normalize_column_name(col).casefold()
            if normalized_col.startswith('cr3:') and 'margen s/venta' in normalized_col:
                return col

        for col in self.ventas_df.columns:
            normalized_col = self._normalize_column_name(col).casefold()
            if normalized_col.startswith('cr2:') and 'margen s/venta' in normalized_col:
                return col

        raise KeyError(
            "Column not found for margin percentage. Expected CR3/CR2 margin column variants."
        )

    def calculate_compras(self, contemplar_sobre_stock: bool = False) -> pd.DataFrame:
        """
        Calculate purchase recommendations (COMPRAS sheet).
        
        Args:
            contemplar_sobre_stock: Consider items already in stock
            
        Returns:
            DataFrame with purchase recommendations
        """
        if self.ventas_df is None or self.stock_df is None:
            raise ValueError("Sales and stock data must be loaded first")
        
        # Get unique SKUs from sales
        skus = self.ventas_df['Artículo'].unique()
        
        # Initialize results dataframe
        compras = pd.DataFrame({
            'SKU': skus
        })
        
        # Add brand/marca from ventas (Clave 1)
        marca_map = self.ventas_df.groupby('Artículo')['Clave 1'].first()
        compras['Marca'] = compras['SKU'].map(marca_map)
        
        # Add description
        desc_map = self.ventas_df.groupby('Artículo')['Descripción Artículo'].first()
        compras['Descripción'] = compras['SKU'].map(desc_map)
        
        # Add purchase price (Precio Coste from ventas)
        precio_map = self.ventas_df.groupby('Artículo')['Precio Coste'].mean()
        compras['Precio Compra'] = compras['SKU'].map(precio_map)
        
        # Add margin (from ventas)
        margin_column = self._resolve_margin_column()
        margin_map = self.ventas_df.groupby('Artículo')[margin_column].mean()
        compras['Margen'] = compras['SKU'].map(margin_map)
        
        # Add stock status from stock_df
        if not self.stock_df.empty:
            stock_map = self.stock_df.set_index('Artículo')['Situación'].to_dict()
            compras['Estado'] = compras['SKU'].map(stock_map)
        else:
            compras['Estado'] = None
        
        # Add last reception date
        if self.recepciones_df is not None and not self.recepciones_df.empty:
            last_recep = self.recepciones_df.groupby('Artículo')['Fecha Recepción'].max()
            compras['Ultima recepción'] = compras['SKU'].map(last_recep)
        else:
            compras['Ultima recepción'] = None
        
        # Stock units and value
        if not self.stock_df.empty:
            stock_units_map = self.stock_df.set_index('Artículo')['Stock'].to_dict()
            compras['Stock Unidades'] = compras['SKU'].map(stock_units_map).fillna(0)
        else:
            compras['Stock Unidades'] = 0
        
        compras['Stock Valor'] = compras['Stock Unidades'] * compras['Precio Compra']
        
        # Pending to serve (Cartera + Reservas)
        if not self.stock_df.empty:
            cartera_map = self.stock_df.set_index('Artículo')['Cartera'].fillna(0).to_dict()
            reservas_map = self.stock_df.set_index('Artículo')['Reservas'].fillna(0).to_dict()
            compras['Pendiente Servir'] = compras['SKU'].map(lambda x: cartera_map.get(x, 0) + reservas_map.get(x, 0))
        else:
            compras['Pendiente Servir'] = 0
        
        # Pending to receive
        if not self.stock_df.empty:
            stock_indexed = self.stock_df.set_index('Artículo')
            pending_receive_column = next(
                (
                    col for col in stock_indexed.columns
                    if self._normalize_column_name(col).casefold() == 'total pendiente recibir'
                ),
                None,
            )

            if pending_receive_column:
                total_pend_map = stock_indexed[pending_receive_column].fillna(0).to_dict()
                compras['Pendiente Recibir'] = compras['SKU'].map(total_pend_map).fillna(0)
            else:
                pend_map = stock_indexed['Pendiente Recibir Compra'].fillna(0).to_dict()
                fab_map = stock_indexed['Pendiente Entrar Fabricación'].fillna(0).to_dict()
                trans_map = stock_indexed['En Tránsito'].fillna(0).to_dict()
                compras['Pendiente Recibir'] = compras['SKU'].map(
                    lambda x: pend_map.get(x, 0) + fab_map.get(x, 0) + trans_map.get(x, 0)
                )
        else:
            compras['Pendiente Recibir'] = 0
        
        # Theoretical available
        compras['Disponible Teorico'] = (compras['Stock Unidades'] + 
                                         compras['Pendiente Recibir'] - 
                                         compras['Pendiente Servir'])
        
        # Calculate sales by period
        compras = self._calculate_sales_metrics(compras)
        
        # Calculate months of stock
        avg_3y_col = f'Promedio {self.current_year - 2} - {self.current_year}'
        current_year_col = f'Ventas {self.current_year}'

        meses_stock_denominator = (
            compras[avg_3y_col] - compras['Disponible Teorico'] - compras[current_year_col]
        )
        compras['Meses de Stock'] = np.where(
            meses_stock_denominator > 0,
            compras['Disponible Teorico'] / meses_stock_denominator,
            0
        )
        
        # Determine if item should be purchased
        compras['COMPRAR'] = compras['Estado'].isna()
        
        # Calculate purchase order quantity
        compras['PEDIDO'] = self._calculate_pedido_vectorized(
            compras,
            contemplar_sobre_stock,
            avg_3y_col,
            current_year_col,
        )
        
        # Calculate order value and margin
        compras['VALOR PEDIDO'] = compras['PEDIDO'] * compras['Precio Compra']
        compras['MARGEN PEDIDO'] = compras['PEDIDO'] * compras['Margen']
        
        self.compras_df = compras
        return compras
    
    def _calculate_sales_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate sales metrics for different periods with improved annualization logic.
        Uses coefficient of variation to detect seasonal/sporadic vs regular sales patterns.
        """
        # Group sales by SKU and month/year
        ventas_grouped = self.ventas_df.groupby(['Artículo', 'Año Factura', 'Mes Factura'])['Unidades Venta'].sum().reset_index()
        
        current_year = self.current_year
        current_month = self.current_month
        
        # ========== VENTAS MENSUALES RECIENTES ==========
        
        # Sales 2 months ago
        month_m2 = current_month - 2 if current_month > 2 else current_month - 2 + 12
        year_m2 = current_year if current_month > 2 else current_year - 1
        
        sales_m2 = ventas_grouped[
            (ventas_grouped['Año Factura'] == year_m2) & 
            (ventas_grouped['Mes Factura'] == month_m2)
        ].set_index('Artículo')['Unidades Venta']
        
        # Sales 1 month ago
        month_m1 = current_month - 1 if current_month > 1 else 12
        year_m1 = current_year if current_month > 1 else current_year - 1
        
        sales_m1 = ventas_grouped[
            (ventas_grouped['Año Factura'] == year_m1) & 
            (ventas_grouped['Mes Factura'] == month_m1)
        ].set_index('Artículo')['Unidades Venta']
        
        # Current month sales
        sales_current = ventas_grouped[
            (ventas_grouped['Año Factura'] == current_year) & 
            (ventas_grouped['Mes Factura'] == current_month)
        ].set_index('Artículo')['Unidades Venta']
        
        df['Ventas -2 meses'] = df['SKU'].map(sales_m2).fillna(0)
        df['Ventas -1 mes'] = df['SKU'].map(sales_m1).fillna(0)
        df['Ventas mes'] = df['SKU'].map(sales_current).fillna(0)
        
        # ========== VENTAS ANUALES ==========
        
        # Current year total
        sales_year = ventas_grouped[
            ventas_grouped['Año Factura'] == current_year
        ].groupby('Artículo')['Unidades Venta'].sum()
        df[f'Ventas {current_year}'] = df['SKU'].map(sales_year).fillna(0)
        
        # Previous year total sales
        sales_prev_year = ventas_grouped[
            ventas_grouped['Año Factura'] == current_year - 1
        ].groupby('Artículo')['Unidades Venta'].sum()
        df[f'Ventas {current_year - 1}'] = df['SKU'].map(sales_prev_year).fillna(0)

        # Two years ago total sales
        sales_year_minus_2 = ventas_grouped[
            ventas_grouped['Año Factura'] == current_year - 2
        ].groupby('Artículo')['Unidades Venta'].sum()
        df[f'Ventas {current_year - 2}'] = df['SKU'].map(sales_year_minus_2).fillna(0)

        # ========== PROMEDIO TRIANUAL CON DETECCIÓN DE ESTACIONALIDAD (OPTIMIZADO) ==========
        
        # OPTIMIZACIÓN: Calcular CV para todos los SKUs de una vez en lugar de uno por uno
        
        # Paso 1: Crear timeline completo para años históricos (solo años anteriores completos)
        historical_years = [current_year - 2, current_year - 1]
        
        # Paso 2: Para cada SKU, crear timeline completo (cross join) de forma vectorizada
        all_skus = df['SKU'].unique()
        timeline_df = pd.MultiIndex.from_product(
            [all_skus, historical_years, range(1, 13)],
            names=['Artículo', 'Año Factura', 'Mes Factura']
        ).to_frame(index=False)
        
        # Paso 3: Merge con ventas históricas reales
        historical_ventas = ventas_grouped[
            ventas_grouped['Año Factura'].isin(historical_years)
        ][['Artículo', 'Año Factura', 'Mes Factura', 'Unidades Venta']].copy()
        
        complete_sales = timeline_df.merge(
            historical_ventas,
            on=['Artículo', 'Año Factura', 'Mes Factura'],
            how='left'
        )
        complete_sales['Unidades Venta'] = complete_sales['Unidades Venta'].fillna(0)
        
        # Paso 4: Calcular CV para cada SKU de una vez
        cv_stats = complete_sales.groupby('Artículo')['Unidades Venta'].agg(['mean', 'std'])
        cv_stats['cv'] = cv_stats['std'] / cv_stats['mean'].replace(0, np.nan)
        cv_stats['cv'] = cv_stats['cv'].fillna(0)  # Si mean=0, cv=0
        
        # Paso 5: Determinar estrategia de anualización para cada SKU
        def get_annualized_value(sku):
            current_sales = sales_year.get(sku, 0)
            prev_year_sales_val = sales_prev_year.get(sku, 0)
            prev_2_year_sales_val = sales_year_minus_2.get(sku, 0)
            
            # Obtener CV del SKU
            cv_value = cv_stats.loc[sku, 'cv'] if sku in cv_stats.index else 0
            
            # Verificar si hay suficiente historial
            years_with_sales = [prev_2_year_sales_val, prev_year_sales_val]
            valid_years = [y for y in years_with_sales if y > 0]
            
            # Si no hay historial suficiente
            if len(valid_years) == 0:
                if current_month > 0 and current_sales > 0:
                    return (current_sales / current_month) * 12
                return 0
            
            # DECISIÓN BASADA EN CV:
            # CV > 1.5: Ventas esporádicas → usar promedio histórico
            # CV <= 1.5: Ventas regulares → anualizar año actual
            
            if cv_value > 1.5:
                # VENTAS ESPORÁDICAS
                avg_historical = np.mean(valid_years)
                # Si el año actual ya superó el promedio histórico, usarlo
                return max(current_sales, avg_historical)
            else:
                # VENTAS REGULARES
                if current_month > 0:
                    return (current_sales / current_month) * 12
                return 0
        
        # Aplicar la función optimizada
        annualized_sales = df['SKU'].apply(get_annualized_value)
        
        # Calcular promedio trianual
        sales_3y = (
            df['SKU'].map(sales_year_minus_2).fillna(0) + 
            df['SKU'].map(sales_prev_year).fillna(0) + 
            annualized_sales
        ) / 3

        # Regla de negocio: si el SKU no vendió ni en el año corriente ni en el
        # año anterior, no debe contemplarse en el promedio de compra.
        no_recent_sales_mask = (
            df['SKU'].map(sales_year).fillna(0).eq(0)
            & df['SKU'].map(sales_prev_year).fillna(0).eq(0)
        )
        sales_3y = sales_3y.where(~no_recent_sales_mask, 0)
        
        df[f'Promedio {current_year - 2} - {current_year}'] = sales_3y
        
        return df
    
    def _calculate_pedido(self, row, contemplar_sobre_stock: bool) -> float:
        """
        Calculate the purchase order quantity.
        
        Formula: ((Promedio 2023-2026 - Ventas 2026) / 12) * Meses_Compra - Stock
        Round up only if decimal >= 0.9, otherwise round down
        Return 0 if negative unless contemplar_sobre_stock is True
        """
        try:
            # 1. Get the 3-year average (Promedio 2023-2026)
            promedio_total = row[f'Promedio {self.current_year - 2} - {self.current_year}']
            
            # 2. Get current year sales (Ventas 2026)
            ventas_corriente = row[f'Ventas {self.current_year}']
            
            # 3. Get current stock
            stock_actual = row['Disponible Teorico']
            
            # 4. Calculate period need first, then discount the current stock.
            # This ensures stock always reduces the final purchase quantity.
            cantidad_base = promedio_total - ventas_corriente
            monthly_sales_total = (cantidad_base / 12) * self.meses_compras
            monthly_need = monthly_sales_total - stock_actual

            # 5. If stock covers period demand, return 0
            if stock_actual >= monthly_sales_total:
                return 0
            
            # 6. If negative and contemplar_sobre_stock is False, return 0
            if monthly_need < 0 and not contemplar_sobre_stock:
                return 0
            
            # 7. If item should not be purchased (has active status), return 0
            if not row['COMPRAR']:
                return 0
            
            # 8. Round: up only if decimal >= 0.9, otherwise down
            decimal_part = abs(monthly_need % 1)
            if decimal_part >= 0.9:
                return np.ceil(monthly_need)
            else:
                return np.floor(monthly_need)
                
        except Exception as e:
            # For debugging - can be uncommented if needed
            # print(f"Error calculating pedido for {row.get('SKU', 'unknown')}: {str(e)}")
            return 0

    def _calculate_pedido_vectorized(
        self,
        compras: pd.DataFrame,
        contemplar_sobre_stock: bool,
        avg_3y_col: str,
        current_year_col: str,
    ) -> np.ndarray:
        """Vectorized purchase order calculation."""
        promedio_total = compras[avg_3y_col]
        ventas_corriente = compras[current_year_col]
        stock_actual = compras['Disponible Teorico']

        cantidad_base = promedio_total - ventas_corriente
        monthly_sales_total = (cantidad_base / 12) * self.meses_compras
        monthly_need = monthly_sales_total - stock_actual

        decimal_part = np.abs(np.mod(monthly_need, 1))
        rounded_need = np.where(decimal_part >= 0.9, np.ceil(monthly_need), np.floor(monthly_need))

        should_zero = (
            (stock_actual >= monthly_sales_total)
            | ((monthly_need < 0) & (not contemplar_sobre_stock))
            | (~compras['COMPRAR'])
        )

        return np.where(should_zero, 0, rounded_need)
    
    def calculate_clientes(self) -> pd.DataFrame:
        """
        Calculate customer analysis (CLIENTES sheet).
        
        Returns:
            DataFrame with customer trends
        """
        if self.ventas_df is None:
            raise ValueError("Sales data must be loaded first")
        
        # Get unique customers
        customers = self.ventas_df['Cliente'].unique()
        
        clientes = pd.DataFrame({
            'Cod': customers
        })
        
        # Add customer name
        name_map = self.ventas_df.groupby('Cliente')['Nombre Cliente'].first()
        clientes['Cliente'] = clientes['Cod'].map(name_map)
        
        # Calculate sales by year
        for year_offset in [2, 1, 0]:
            year = self.current_year - year_offset
            year_sales = self.ventas_df[
                self.ventas_df['Año Factura'] == year
            ].groupby('Cliente')['Importe Neto'].sum()
            
            clientes[f'Año {year}'] = clientes['Cod'].map(year_sales).fillna(0)
        
        # Calculate year-over-year differences
        year_cols = [f'Año {self.current_year - i}' for i in [2, 1, 0]]
        
        clientes[f'Dif {self.current_year - 2} - {self.current_year - 1}'] = clientes.apply(
            lambda row: (row[year_cols[1]] - row[year_cols[0]]) / row[year_cols[0]]
            if row[year_cols[0]] != 0 else 1,
            axis=1
        )
        
        clientes[f'Dif {self.current_year - 1} - {self.current_year}'] = clientes.apply(
            lambda row: (row[year_cols[2]] - row[year_cols[1]]) / row[year_cols[1]]
            if row[year_cols[1]] != 0 else 1,
            axis=1
        )
        
        self.clientes_df = clientes
        return clientes
    
    def get_summary_stats(self) -> Dict:
        """Get summary statistics for the dashboard."""
        stats = {}
        
        if self.compras_df is not None:
            stats['total_stock_value'] = self.compras_df['Stock Valor'].sum()
            stats['total_pedido_value'] = self.compras_df['VALOR PEDIDO'].sum()
            stats['total_pedido_margin'] = self.compras_df['MARGEN PEDIDO'].sum()
            stats['items_to_order'] = (self.compras_df['PEDIDO'] > 0).sum()
            stats['total_stock_units'] = self.compras_df['Stock Unidades'].sum()
        
        if self.clientes_df is not None:
            current_year_col = f'Año {self.current_year}'
            prev_year_col = f'Año {self.current_year - 1}'
            
            if current_year_col in self.clientes_df.columns:
                stats['total_sales_current_year'] = self.clientes_df[current_year_col].sum()
            if prev_year_col in self.clientes_df.columns:
                stats['total_sales_prev_year'] = self.clientes_df[prev_year_col].sum()
        
        return stats
    
    def export_results(self, output_file: str = 'inventory_results.xlsx'):
        """Export results to Excel file."""
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            if self.compras_df is not None:
                self.compras_df.to_excel(writer, sheet_name='COMPRAS', index=False)
            if self.clientes_df is not None:
                self.clientes_df.to_excel(writer, sheet_name='CLIENTES', index=False)
            if self.stock_df is not None:
                self.stock_df.to_excel(writer, sheet_name='Stock Input', index=False)
        
        print(f"Results exported to {output_file}")


if __name__ == "__main__":
    # Example usage
    print("Inventory Manager Module - Ready to use")
    print("Import this module and use InventoryManager class")
