"""
Inventory Management Dashboard - Streamlit App
Interactive web interface for inventory analysis
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

# Asegurar que Python encuentre el módulo
sys.path.insert(0, os.path.dirname(__file__))

from inventory_manager import InventoryManager
import io
import hmac
import json
import gzip
import base64
from datetime import datetime
from pathlib import Path


HISTORY_FILE = Path(".upload_history_local.json")
MAX_CHUNK_SIZE = 700_000
SNAPSHOT_COLUMNS = {
    "stock": [
        "Artículo", "Descripción", "Situación", "Stock", "Cartera", "Reservas",
        "Pendiente Recibir Compra", "Pendiente Entrar Fabricación", "En Tránsito"
    ],
    "ventas": [
        "Artículo", "Cliente", "Clave 1", "Año Factura", "Nombre Cliente", "Mes Factura",
        "Descripción Artículo", "Precio Coste", "CR2: %Margen s/Venta sin Transporte Athena",
        "Importe Neto", "Unidades Venta"
    ],
    "recepciones": ["Artículo", "Fecha Recepción", "Unidades Stock", "Precio"],
}

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
    /* Keep KPI labels legible on light backgrounds */
    [data-testid="stMetricLabel"],
    [data-testid="stMetricValue"],
    [data-testid="stMetricDelta"] {
        color: #000000;
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


@st.cache_resource
def get_firestore_client():
    """Initialize Firebase Admin SDK and return Firestore client.

    Expects `FIREBASE_SERVICE_ACCOUNT_JSON` in Streamlit secrets.
    """
    if "FIREBASE_SERVICE_ACCOUNT_JSON" not in st.secrets:
        return None

    service_account = st.secrets["FIREBASE_SERVICE_ACCOUNT_JSON"]
    if isinstance(service_account, str):
        service_account = json.loads(service_account)

    if not firebase_admin._apps:
        cred = credentials.Certificate(dict(service_account))
        firebase_admin.initialize_app(cred)

    return firestore.client()


def log_inventory_upload(source: str, stock_file, ventas_file, recepciones_file):
    """Persist upload metadata in Firestore collection `inventory_uploads`."""
    db = get_firestore_client()
    if db is None:
        return

    upload_doc = {
        "source": source,
        "stock_filename": stock_file.name if stock_file else None,
        "ventas_filename": ventas_file.name if ventas_file else None,
        "recepciones_filename": recepciones_file.name if recepciones_file else None,
        "created_at": firestore.SERVER_TIMESTAMP,
    }
    db.collection("inventory_uploads").add(upload_doc)


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


def _get_firebase_collection():
    """Return Firestore collection when configured, else None."""
    try:
        import firebase_admin
        from firebase_admin import credentials, firestore
    except Exception:
        return None

    if not firebase_admin._apps:
        service_account = st.secrets.get("FIREBASE_SERVICE_ACCOUNT")
        if not service_account:
            return None
        if isinstance(service_account, str):
            service_account = json.loads(service_account)
        firebase_admin.initialize_app(credentials.Certificate(service_account))

    return firestore.client().collection("upload_history")


def _serialize_df(df: pd.DataFrame | None, kind: str):
    if df is None:
        return []
    keep_cols = [c for c in SNAPSHOT_COLUMNS[kind] if c in df.columns]
    data = df[keep_cols] if keep_cols else df
    return json.loads(data.to_json(orient="records", date_format="iso"))


def _deserialize_df(records):
    if not records:
        return None
    return pd.DataFrame(records)


def _encode_chunks(records):
    payload = json.dumps(records, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    encoded = base64.b64encode(gzip.compress(payload)).decode("utf-8")
    return [encoded[i:i + MAX_CHUNK_SIZE] for i in range(0, len(encoded), MAX_CHUNK_SIZE)]


def _decode_chunks(chunks):
    if not chunks:
        return []
    encoded = "".join(chunks)
    raw = gzip.decompress(base64.b64decode(encoded.encode("utf-8")))
    return json.loads(raw.decode("utf-8"))


def _load_local_history():
    if not HISTORY_FILE.exists():
        return []
    try:
        return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []


def _save_local_history(history):
    HISTORY_FILE.write_text(json.dumps(history, ensure_ascii=False), encoding="utf-8")


def save_upload_snapshot(stock_df, ventas_df, recepciones_df, source="upload", snapshot_name: str | None = None):
    stock_records = _serialize_df(stock_df, "stock")
    ventas_records = _serialize_df(ventas_df, "ventas")
    recepciones_records = _serialize_df(recepciones_df, "recepciones")

    doc_payload = {
        "uploaded_at": datetime.utcnow().isoformat(),
        "source": source,
        "snapshot_name": (snapshot_name or "").strip() or None,
        "file_count": 2 + int(recepciones_df is not None),
        "stock_chunks": _encode_chunks(stock_records),
        "ventas_chunks": _encode_chunks(ventas_records),
        "recepciones_chunks": _encode_chunks(recepciones_records),
    }

    collection = _get_firebase_collection()
    if collection is not None:
        ref = collection.document()
        doc_payload["id"] = ref.id
        ref.set(doc_payload)
        return

    history = _load_local_history()
    snapshot_id = f"local-{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}"
    doc_payload["id"] = snapshot_id
    history.append(doc_payload)
    _save_local_history(history)


def calculate_clientes_from_ventas(ventas_df: pd.DataFrame, current_year: int) -> pd.DataFrame:
    """Build customer analysis from a (possibly filtered) sales dataframe."""
    if ventas_df is None or ventas_df.empty:
        return pd.DataFrame(columns=["Cod", "Cliente"])

    customers = ventas_df['Cliente'].dropna().unique()
    clientes = pd.DataFrame({'Cod': customers})

    name_map = ventas_df.groupby('Cliente')['Nombre Cliente'].first()
    clientes['Cliente'] = clientes['Cod'].map(name_map)

    for year_offset in [2, 1, 0]:
        year = current_year - year_offset
        year_sales = ventas_df[
            ventas_df['Año Factura'] == year
        ].groupby('Cliente')['Importe Neto'].sum()
        clientes[f'Año {year}'] = clientes['Cod'].map(year_sales).fillna(0)

    year_cols = [f'Año {current_year - i}' for i in [2, 1, 0]]
    clientes[f'Dif {current_year - 2} - {current_year - 1}'] = clientes.apply(
        lambda row: (row[year_cols[1]] - row[year_cols[0]]) / row[year_cols[0]]
        if row[year_cols[0]] != 0 else 1,
        axis=1
    )
    clientes[f'Dif {current_year - 1} - {current_year}'] = clientes.apply(
        lambda row: (row[year_cols[2]] - row[year_cols[1]]) / row[year_cols[1]]
        if row[year_cols[1]] != 0 else 1,
        axis=1
    )

    return clientes


def dataframe_to_excel_bytes(df: pd.DataFrame, sheet_name: str) -> bytes:
    """Convert a dataframe to a single-sheet Excel file in memory."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
    output.seek(0)
    return output.getvalue()


def list_upload_dates():
    collection = _get_firebase_collection()
    if collection is not None:
        docs = collection.order_by("uploaded_at", direction="DESCENDING").limit(50).stream()
        return [doc.to_dict() for doc in docs]

    history = sorted(_load_local_history(), key=lambda x: x.get("uploaded_at", ""), reverse=True)
    return history[:50]


def get_upload_by_id(upload_id: str):
    collection = _get_firebase_collection()
    if collection is not None:
        doc = collection.document(upload_id).get()
        if doc.exists:
            return doc.to_dict()
        return None

    history = _load_local_history()
    for item in history:
        if item.get("id") == upload_id:
            return item
    return None


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

        snapshot_name = st.text_input(
            "Nombre para esta carga histórica (opcional)",
            placeholder="Ej: Stock Febrero 2026"
        )

        st.subheader("🕓 Histórico Firebase")
        try:
            history_items = list_upload_dates()
        except Exception:
            history_items = _load_local_history()
            st.error("No se pudo conectar a Firebase...")

        if history_items:
            history_options = {}
            for item in history_items:
                uploaded_at = item.get("uploaded_at", "")
                try:
                    formatted_date = datetime.fromisoformat(uploaded_at.replace("Z", "")).strftime("%Y-%m-%d %H:%M:%S")
                except Exception:
                    formatted_date = uploaded_at
                source = item.get("source", "upload")
                file_count = item.get("file_count", 0)
                custom_name = item.get("snapshot_name")
                name_part = f"{custom_name} · " if custom_name else ""
                label = f"{name_part}{formatted_date} · {source} · {file_count} archivos"
                history_options[label] = item.get("id")

            selected_label = st.selectbox("Seleccionar carga histórica", options=list(history_options.keys()))
            selected_id = history_options[selected_label]

            if st.button("Cargar histórico"):
                try:
                    doc = get_upload_by_id(selected_id)
                    if not doc:
                        st.error("No se encontró el histórico seleccionado")
                    else:
                        manager = InventoryManager(meses_compras=meses_compras)
                        manager.stock_df = _deserialize_df(_decode_chunks(doc.get("stock_chunks", [])))
                        manager.ventas_df = _deserialize_df(_decode_chunks(doc.get("ventas_chunks", [])))
                        manager.recepciones_df = _deserialize_df(_decode_chunks(doc.get("recepciones_chunks", [])))

                        st.session_state.manager = manager
                        st.session_state.data_loaded = True
                        st.success("✅ Histórico cargado")
                        st.rerun()
                except Exception:
                    st.error("No se pudo conectar a Firebase...")
        else:
            st.caption("Sin cargas históricas disponibles")
        
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

                        try:
                            save_upload_snapshot(
                                manager.stock_df,
                                manager.ventas_df,
                                manager.recepciones_df,
                                source="upload",
                                snapshot_name=snapshot_name
                            )
                        except Exception:
                            st.error("No se pudo conectar a Firebase...")

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
            except Exception as e:
                st.error(f"Error in calculations: {str(e)}")
                return

        # Global filters (apply to all tabs)
        st.subheader("🌐 Filtros globales")
        available_brands = sorted(compras_df['Marca'].dropna().unique()) if 'Marca' in compras_df.columns else []
        selected_brand = st.multiselect(
            "Filtrar por marca (aplica a Dashboard, Purchase Orders, Customers y Export)",
            options=available_brands,
            default=[]
        )

        compras_filtered = compras_df.copy()
        ventas_filtered = manager.ventas_df.copy() if manager.ventas_df is not None else pd.DataFrame()

        if selected_brand:
            compras_filtered = compras_filtered[compras_filtered['Marca'].isin(selected_brand)]
            if not ventas_filtered.empty and 'Clave 1' in ventas_filtered.columns:
                ventas_filtered = ventas_filtered[ventas_filtered['Clave 1'].isin(selected_brand)]

        clientes_df = calculate_clientes_from_ventas(ventas_filtered, manager.current_year)
        stats = {
            'total_stock_value': compras_filtered['Stock Valor'].sum() if 'Stock Valor' in compras_filtered.columns else 0,
            'total_pedido_value': compras_filtered['VALOR PEDIDO'].sum() if 'VALOR PEDIDO' in compras_filtered.columns else 0,
            'total_pedido_margin': compras_filtered['MARGEN PEDIDO'].sum() if 'MARGEN PEDIDO' in compras_filtered.columns else 0,
            'items_to_order': int((compras_filtered['PEDIDO'] > 0).sum()) if 'PEDIDO' in compras_filtered.columns else 0,
        }

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
                top_stock = compras_filtered.nlargest(10, 'Stock Valor')[['SKU', 'Stock Valor']]
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
                orders_by_brand = compras_filtered[compras_filtered['PEDIDO'] > 0].groupby('Marca')['VALOR PEDIDO'].sum().reset_index()
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
            stock_status = compras_filtered.copy()
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
            
            # Tab-specific filters
            col1, col2 = st.columns(2)
            with col1:
                min_order = st.number_input("Min Order Quantity", value=0, step=1)
            
            with col2:
                show_all = st.checkbox("Show All Items", value=False)
            
            # Filter data (already brand-filtered globally)
            filtered_df = compras_filtered.copy()
            
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
                compras_filtered.to_excel(writer, sheet_name='COMPRAS', index=False)
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
                compras_excel = dataframe_to_excel_bytes(compras_filtered, 'COMPRAS')
                st.download_button(
                    "📄 Download Purchases (Excel)",
                    compras_excel,
                    f"compras_{pd.Timestamp.now().strftime('%Y%m%d')}.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            
            with col2:
                clientes_excel = dataframe_to_excel_bytes(clientes_df, 'CLIENTES')
                st.download_button(
                    "📄 Download Customers (Excel)",
                    clientes_excel,
                    f"clientes_{pd.Timestamp.now().strftime('%Y%m%d')}.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )


if __name__ == "__main__":
    main()
