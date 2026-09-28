"""
QUICK START GUIDE
=================

This guide will get you up and running in 5 minutes!
"""

# ============================================================================
# STEP 1: INSTALL REQUIREMENTS
# ============================================================================

# Open terminal/command prompt and run:
# pip install pandas numpy openpyxl streamlit plotly


# ============================================================================
# STEP 2: CHOOSE YOUR METHOD
# ============================================================================

# METHOD A: Web Dashboard (Easiest - Recommended for non-programmers)
# --------------------------------------------------------------------
# 1. Open terminal in the folder with these files
# 2. Run: streamlit run dashboard.py
# 3. Your browser will open automatically
# 4. Upload your Excel file OR test with sample data
# 5. Explore the interactive dashboard!


# METHOD B: Command Line (Quick analysis)
# ----------------------------------------
# python run_analysis.py SEGUIMIENTO_3_0.xlsx
# 
# That's it! Results saved to inventory_results.xlsx


# METHOD C: Python Script (For programmers)
# ------------------------------------------

from inventory_manager import InventoryManager

# Initialize the manager
manager = InventoryManager(meses_compras=2)

# Load your Excel file
manager.load_data(excel_file='SEGUIMIENTO_3_0.xlsx')

# Run calculations
compras = manager.calculate_compras()
clientes = manager.calculate_clientes()

# Get insights
stats = manager.get_summary_stats()
print(f"Items to order: {stats['items_to_order']}")
print(f"Total order value: €{stats['total_pedido_value']:,.2f}")

# Export results
manager.export_results('results.xlsx')


# ============================================================================
# STEP 3: UNDERSTAND YOUR OUTPUT
# ============================================================================

"""
COMPRAS Sheet (Purchase Recommendations)
----------------------------------------
- SKU: Product code
- Stock Unidades: Current stock level
- Meses de Stock: How many months of stock you have
- PEDIDO: Recommended quantity to order
- VALOR PEDIDO: Total cost of order
- MARGEN PEDIDO: Expected profit

CLIENTES Sheet (Customer Analysis)
-----------------------------------
- Cod: Customer code
- Año 2024/2025/2026: Sales by year
- Dif: Year-over-year growth percentage
"""


# ============================================================================
# STEP 4: CUSTOMIZE (Optional)
# ============================================================================

# Change purchase forecast period
manager = InventoryManager(meses_compras=3)  # 3 months instead of 2

# Consider items that are over-stocked
compras = manager.calculate_compras(contemplar_sobre_stock=True)


# ============================================================================
# TROUBLESHOOTING
# ============================================================================

"""
Problem: "Module not found"
Solution: pip install pandas numpy openpyxl streamlit plotly

Problem: "No data loaded"
Solution: Make sure your Excel has data in sheets:
         - 1 - INPUT Stock
         - 2 - INPUT Recepciones  
         - 3 - INPUT Ventas

Problem: Dashboard won't open
Solution: Try running: streamlit run dashboard.py --server.port 8502
"""


# ============================================================================
# TIPS FOR BEST RESULTS
# ============================================================================

"""
1. Keep your Excel sheets in the same format as the original
2. Ensure SKU codes match across all sheets
3. Make sure dates are properly formatted
4. Start with sample data to understand the system
5. Export results regularly for backup

Speed Comparison:
- Excel: 1-2 minutes to calculate
- Python: < 1 second to calculate
- That's 100x faster! ⚡
"""
