"""
Simple command-line script to process inventory data
Usage: python run_analysis.py <excel_file> [--months N] [--over-stock]
"""

import sys
import argparse
from inventory_manager import InventoryManager


def main():
    parser = argparse.ArgumentParser(
        description='Process inventory data and generate purchase recommendations'
    )
    parser.add_argument(
        'excel_file',
        help='Path to Excel file with inventory data'
    )
    parser.add_argument(
        '--months',
        type=int,
        default=2,
        help='Number of months for purchase calculations (default: 2)'
    )
    parser.add_argument(
        '--over-stock',
        action='store_true',
        help='Consider items that are over-stocked'
    )
    parser.add_argument(
        '--output',
        default='inventory_results.xlsx',
        help='Output file name (default: inventory_results.xlsx)'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("INVENTORY MANAGEMENT SYSTEM")
    print("=" * 80)
    print(f"\nProcessing: {args.excel_file}")
    print(f"Purchase months: {args.months}")
    print(f"Consider over-stock: {args.over_stock}")
    print("\nLoading data...")
    
    try:
        # Initialize manager
        manager = InventoryManager(meses_compras=args.months)
        
        # Load data
        manager.load_data(excel_file=args.excel_file)
        print("✓ Data loaded successfully")
        
        # Calculate purchases
        print("\nCalculating purchase recommendations...")
        compras_df = manager.calculate_compras(contemplar_sobre_stock=args.over_stock)
        print(f"✓ Processed {len(compras_df)} items")
        
        # Calculate customer analysis
        print("\nAnalyzing customer trends...")
        clientes_df = manager.calculate_clientes()
        print(f"✓ Analyzed {len(clientes_df)} customers")
        
        # Get summary statistics
        stats = manager.get_summary_stats()
        
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"Total Stock Value:       €{stats.get('total_stock_value', 0):,.2f}")
        print(f"Total Stock Units:       {stats.get('total_stock_units', 0):,.0f}")
        print(f"Items to Order:          {stats.get('items_to_order', 0)}")
        print(f"Purchase Order Value:    €{stats.get('total_pedido_value', 0):,.2f}")
        print(f"Expected Margin:         €{stats.get('total_pedido_margin', 0):,.2f}")
        
        if 'total_sales_current_year' in stats:
            print(f"Current Year Sales:      €{stats['total_sales_current_year']:,.2f}")
        if 'total_sales_prev_year' in stats:
            print(f"Previous Year Sales:     €{stats['total_sales_prev_year']:,.2f}")
        
        # Export results
        print(f"\nExporting results to: {args.output}")
        manager.export_results(args.output)
        
        print("\n✓ Analysis complete!")
        print("=" * 80)
        
        # Show top 10 items to order
        top_orders = compras_df[compras_df['PEDIDO'] > 0].nlargest(10, 'VALOR PEDIDO')
        
        if len(top_orders) > 0:
            print("\nTOP 10 PURCHASE RECOMMENDATIONS:")
            print("-" * 80)
            print(f"{'SKU':<15} {'Description':<30} {'Order Qty':<12} {'Value':<15}")
            print("-" * 80)
            
            for _, row in top_orders.iterrows():
                sku = str(row['SKU'])[:15]
                desc = str(row['Descripción'])[:30] if pd.notna(row['Descripción']) else 'N/A'
                qty = f"{row['PEDIDO']:.0f}"
                value = f"€{row['VALOR PEDIDO']:,.2f}"
                print(f"{sku:<15} {desc:<30} {qty:<12} {value:<15}")
        
        print("\n" + "=" * 80)
        return 0
        
    except FileNotFoundError:
        print(f"\n❌ Error: File not found: {args.excel_file}")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
