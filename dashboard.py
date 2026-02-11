"""
Inventory Management Dashboard - Streamlit App
Interactive web interface for inventory analysis
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from inventory_manager import InventoryManager
import io
import hmac

# Page configuration
st.set_page_config(
    page_title="Inventory Management System",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
    }
    h1 {
        color: #1f77b4;
    }
    h2 {
        color: #2c3e50;
    }
    </style>
    """, unsafe_allow_html=True)


def require_auth() -> bool:
    """Simple password gate using Streamlit secrets."""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return True

    st.title("🔒 Inventory Management System")
    st.markdown("Please enter the app password to continue.")

    if "APP_PASSWORD" not in st.secrets:
        st.error(
            "Missing APP_PASSWORD secret. Set it in Streamlit Cloud "
            "Secrets or in a local `.streamlit/secrets.toml` file."
        )
        return False

    with st.form("auth_form", clear_on_submit=True):
        password = st.text_input("Password", type="password", key="password_input")
        sign_in = st.form_submit_button("Sign in", type="primary")

    if sign_in:
        if hmac.compare_digest(password, str(st.secrets["APP_PASSWORD"])):
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password. Please try again.")

    return False


@st.cache_data
def load_sample_data():
    """Generate sample data for demonstration."""
    import numpy as np
    from datetime import datetime, timedelta
    
    # Sample products
    skus = [f'SKU{str(i).zfill(4)}' for i in range(1, 51)]
    marcas = ['Brand A', 'Brand B', 'Brand C', 'Brand D', 'Brand E']
    
    # Stock data
    stock_data = {
        'Artículo': skus,
        'Descripción': [f'Product {i}' for i in range(1, 51)],
        'Referencia': [f'REF{i}' for i in range(1, 51)],
        'Almacén': 'Main Warehouse',
        'Stock': np.random.randint(0, 500, 50),
        'Cartera': np.random.randint(0, 100, 50),
        'Reservas': np.random.randint(0, 50, 50),
        'Total Pendiente Recibir': np.random.randint(0, 200, 50),
        'Pendiente Recibir Compra': np.random.randint(0, 150, 50),
        'Pendiente Entrar Fabricación': np.random.randint(0, 100, 50),
        'En Tránsito': np.random.randint(0, 50, 50),
        'Disponible': np.random.randint(0, 600, 50),
        'Disponible Teórico': np.random.randint(0, 700, 50),
        'Situación': np.random.choice(['Active', None], 50, p=[0.7, 0.3]),
        'Ubicación': 'A-01',
        'Ubicación 2': '',
        'Precio Tarifa': np.random.uniform(10, 500, 50),
        'Dto. Tarifa': 0,
        'Precio Neto': np.random.uniform(10, 500, 50),
    }
    
    # Sales data
    sales_records = []
    customers = [f'CUST{str(i).zfill(3)}' for i in range(1, 21)]
    
    for _ in range(500):
        sales_records.append({
            'Artículo': np.random.choice(skus),
            'Cliente': np.random.choice(customers),
            'Clave 1': np.random.choice(marcas),
            'Año Factura': np.random.choice([2024, 2025, 2026]),
            'Nombre Cliente': np.random.choice([f'Customer {i}' for i in range(1, 21)]),
            'Mes Factura': np.random.randint(1, 13),
            'Fecha Factura': datetime.now() - timedelta(days=np.random.randint(1, 730)),
            'Descripción Artículo': f'Product Description',
            'Stock Disponible': np.random.randint(0, 500),
            'Precio Coste': np.random.uniform(5, 250),
            'Precio Medio Venta': np.random.uniform(10, 500),
            'CR2: %Margen s/Venta sin Transporte Athena': np.random.uniform(0.1, 0.5),
            'Importe Neto': np.random.uniform(50, 5000),
            'Unidades Venta': np.random.randint(1, 50)
        })
    
    # Receptions data
    receptions_records = []
    for _ in range(100):
        receptions_records.append({
            'Artículo': np.random.choice(skus),
            'Fecha Recepción': datetime.now() - timedelta(days=np.random.randint(1, 365)),
            'Unidades Stock': np.random.randint(10, 200),
            'Precio': np.random.uniform(5, 250)
        })
    
    return pd.DataFrame(stock_data), pd.DataFrame(sales_records), pd.DataFrame(receptions_records)


def main():
    if not require_auth():
        st.stop()

    st.title("📦 Inventory Management System")
    st.markdown("### Advanced Purchase Planning & Customer Analysis")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Data source selection
        data_source = st.radio(
            "Data Source",
            ["Upload Files", "Use Sample Data"],
            help="Choose to upload your own data or use sample data for testing"
        )
        
        st.divider()
        
        # Parameters
        st.subheader("Parameters")
        meses_compras = st.slider(
            "Purchase Months",
            min_value=1,
            max_value=6,
            value=2,
            help="Number of months to calculate purchase needs"
        )
        
        contemplar_sobre_stock = st.checkbox(
            "Consider Over-Stock",
            value=False,
            help="Include items that are over-stocked in recommendations"
        )
        
        st.divider()
        
        # About
        st.subheader("About")
        st.info("""
        **This system replaces your Excel with:**
        - ⚡ 100x faster calculations
        - 📊 Interactive visualizations
        - 📈 Real-time analytics
        - 💾 Export capabilities
        """)
    
    # Initialize session state
    if 'manager' not in st.session_state:
        st.session_state.manager = None
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    
    # Data loading section
    if data_source == "Upload Files":
        st.header("📤 Upload Data Files")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            stock_file = st.file_uploader(
                "Stock Data (Excel/CSV)",
                type=['xlsx', 'csv'],
                help="Upload your stock/inventory file"
            )
        
        with col2:
            ventas_file = st.file_uploader(
                "Sales Data (Excel/CSV)",
                type=['xlsx', 'csv'],
                help="Upload your sales history file"
            )
        
        with col3:
            recepciones_file = st.file_uploader(
                "Receptions Data (Excel/CSV)",
                type=['xlsx', 'csv'],
                help="Upload your receptions file"
            )
        
        if st.button("🚀 Process Data", type="primary"):
            if stock_file and ventas_file:
                with st.spinner("Processing data..."):
                    try:
                        # Load uploaded files
                        manager = InventoryManager(meses_compras=meses_compras)
                        
                        # Read files based on type
                        if stock_file.name.endswith('.xlsx'):
                            manager.stock_df = pd.read_excel(stock_file)
                        else:
                            manager.stock_df = pd.read_csv(stock_file)
                        
                        if ventas_file.name.endswith('.xlsx'):
                            manager.ventas_df = pd.read_excel(ventas_file)
                        else:
                            manager.ventas_df = pd.read_csv(ventas_file)
                        
                        if recepciones_file:
                            if recepciones_file.name.endswith('.xlsx'):
                                manager.recepciones_df = pd.read_excel(recepciones_file)
                            else:
                                manager.recepciones_df = pd.read_csv(recepciones_file)
                        
                        # Clean column names
                        manager.stock_df.columns = manager.stock_df.columns.str.strip().str.replace('\n', ' ')
                        manager.ventas_df.columns = manager.ventas_df.columns.str.strip()
                        if manager.recepciones_df is not None:
                            manager.recepciones_df.columns = manager.recepciones_df.columns.str.strip()
                        
                        st.session_state.manager = manager
                        st.session_state.data_loaded = True
                        st.success("✅ Data loaded successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error loading data: {str(e)}")
            else:
                st.warning("⚠️ Please upload at least Stock and Sales files")
    
    else:  # Use sample data
        st.header("📊 Sample Data Mode")
        st.info("Using generated sample data for demonstration purposes")
        
        if st.button("🚀 Load Sample Data", type="primary"):
            with st.spinner("Generating sample data..."):
                try:
                    stock_df, ventas_df, recepciones_df = load_sample_data()
                    
                    manager = InventoryManager(meses_compras=meses_compras)
                    manager.stock_df = stock_df
                    manager.ventas_df = ventas_df
                    manager.recepciones_df = recepciones_df
                    
                    st.session_state.manager = manager
                    st.session_state.data_loaded = True
                    st.success("✅ Sample data loaded!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    # Main analysis section
    if st.session_state.data_loaded and st.session_state.manager:
        manager = st.session_state.manager
        manager.meses_compras = meses_compras
        
        # Calculate analysis
        with st.spinner("Calculating..."):
            try:
                compras_df = manager.calculate_compras(contemplar_sobre_stock)
                clientes_df = manager.calculate_clientes()
                stats = manager.get_summary_stats()
            except Exception as e:
                st.error(f"Error in calculations: {str(e)}")
                return
        
        # Tabs for different sections
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🛒 Purchase Orders", "👥 Customers", "📁 Export"])
        
        with tab1:
            st.header("📊 Dashboard Overview")
            
            # KPI Metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Total Stock Value",
                    f"€{stats.get('total_stock_value', 0):,.0f}",
                    help="Total value of current inventory"
                )
            
            with col2:
                st.metric(
                    "Purchase Order Value",
                    f"€{stats.get('total_pedido_value', 0):,.0f}",
                    help="Total value of recommended purchases"
                )
            
            with col3:
                st.metric(
                    "Items to Order",
                    f"{stats.get('items_to_order', 0)}",
                    help="Number of items that need to be ordered"
                )
            
            with col4:
                st.metric(
                    "Expected Margin",
                    f"€{stats.get('total_pedido_margin', 0):,.0f}",
                    help="Expected profit margin from orders"
                )
            
            st.divider()
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Top 10 Items by Stock Value")
                top_stock = compras_df.nlargest(10, 'Stock Valor')[['SKU', 'Stock Valor']]
                fig = px.bar(
                    top_stock,
                    x='SKU',
                    y='Stock Valor',
                    color='Stock Valor',
                    color_continuous_scale='Blues'
                )
                fig.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Purchase Orders by Brand")
                orders_by_brand = compras_df[compras_df['PEDIDO'] > 0].groupby('Marca')['VALOR PEDIDO'].sum().reset_index()
                fig = px.pie(
                    orders_by_brand,
                    values='VALOR PEDIDO',
                    names='Marca',
                    hole=0.4
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            # Stock status
            st.subheader("Stock Status Distribution")
            stock_status = compras_df.copy()
            stock_status['Status'] = stock_status.apply(
                lambda x: 'Critical' if x['Meses de Stock'] < 1 else
                         'Low' if x['Meses de Stock'] < 2 else
                         'Normal' if x['Meses de Stock'] < 4 else 'High',
                axis=1
            )
            status_count = stock_status['Status'].value_counts().reset_index()
            status_count.columns = ['Status', 'Count']
            
            fig = px.bar(
                status_count,
                x='Status',
                y='Count',
                color='Status',
                color_discrete_map={
                    'Critical': '#e74c3c',
                    'Low': '#f39c12',
                    'Normal': '#2ecc71',
                    'High': '#3498db'
                }
            )
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.header("🛒 Purchase Recommendations")
            
            # Filters
            col1, col2, col3 = st.columns(3)
            with col1:
                selected_brand = st.multiselect(
                    "Filter by Brand",
                    options=compras_df['Marca'].dropna().unique(),
                    default=None
                )
            
            with col2:
                min_order = st.number_input("Min Order Quantity", value=0, step=1)
            
            with col3:
                show_all = st.checkbox("Show All Items", value=False)
            
            # Filter data
            filtered_df = compras_df.copy()
            
            if selected_brand:
                filtered_df = filtered_df[filtered_df['Marca'].isin(selected_brand)]
            
            if not show_all:
                filtered_df = filtered_df[filtered_df['PEDIDO'] > min_order]
            
            # Display table
            st.dataframe(
                filtered_df[[
                    'SKU', 'Marca', 'Descripción', 'Stock Unidades', 
                    'Meses de Stock', 'PEDIDO', 'VALOR PEDIDO', 'MARGEN PEDIDO'
                ]].style.format({
                    'Stock Unidades': '{:.0f}',
                    'Meses de Stock': '{:.1f}',
                    'PEDIDO': '{:.0f}',
                    'VALOR PEDIDO': '€{:,.2f}',
                    'MARGEN PEDIDO': '€{:,.2f}'
                }),
                use_container_width=True,
                height=600
            )
            
            # Summary
            st.divider()
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Filtered Items", len(filtered_df))
            with col2:
                st.metric("Total Order Value", f"€{filtered_df['VALOR PEDIDO'].sum():,.0f}")
            with col3:
                st.metric("Total Margin", f"€{filtered_df['MARGEN PEDIDO'].sum():,.0f}")
        
        with tab3:
            st.header("👥 Customer Analysis")
            
            # Display customer table
            st.dataframe(
                clientes_df.style.format({
                    col: '€{:,.0f}' for col in clientes_df.columns if 'Año' in col
                }).format({
                    col: '{:.1%}' for col in clientes_df.columns if 'Dif' in col
                }),
                use_container_width=True,
                height=600
            )
            
            # Top customers chart
            st.subheader("Top 10 Customers by Current Year Sales")
            current_year_col = f'Año {manager.current_year}'
            if current_year_col in clientes_df.columns:
                top_customers = clientes_df.nlargest(10, current_year_col)
                fig = px.bar(
                    top_customers,
                    x='Cliente',
                    y=current_year_col,
                    color=current_year_col,
                    color_continuous_scale='Greens'
                )
                fig.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            st.header("📁 Export Results")
            
            st.write("Download your analysis results in Excel format")
            
            # Create Excel file in memory
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                compras_df.to_excel(writer, sheet_name='COMPRAS', index=False)
                clientes_df.to_excel(writer, sheet_name='CLIENTES', index=False)
            
            output.seek(0)
            
            st.download_button(
                label="📥 Download Excel Report",
                data=output,
                file_name=f"inventory_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary"
            )
            
            st.divider()
            
            # Individual exports
            col1, col2 = st.columns(2)
            
            with col1:
                csv_compras = compras_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📄 Download Purchases (CSV)",
                    csv_compras,
                    f"compras_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                    "text/csv"
                )
            
            with col2:
                csv_clientes = clientes_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📄 Download Customers (CSV)",
                    csv_clientes,
                    f"clientes_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                    "text/csv"
                )


if __name__ == "__main__":
    main()
