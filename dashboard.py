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
import re
from datetime import datetime, timedelta
from pathlib import Path


HISTORY_FILE = Path(".upload_history_local.json")
MAX_CHUNK_SIZE = 700_000
PRIMARY_UPLOAD_COLLECTION = "upload_history"
HISTORY_RETENTION_DAYS = 7
MARGIN_COLUMN = "CR3: % Margen s/Venta + Transport"
SNAPSHOT_COLUMNS = {
    "stock": [
        "Artículo", "Descripción", "Situación", "Stock", "Cartera", "Reservas",
        "Pendiente Recibir Compra", "Pendiente Entrar Fabricación", "En Tránsito"
    ],
    "ventas": [
        "Artículo", "Cliente", "Clave 1", "Año Factura", "Nombre Cliente", "Mes Factura",
        "Descripción Artículo", "Precio Coste", MARGIN_COLUMN,
        "Importe Neto", "Unidades Venta"
    ],
    "recepciones": ["Artículo", "Fecha Recepción", "Unidades Stock", "Precio"],
    "stock_value": ["Clave 1", "Código Artículo", "Unidades", "Importe"],
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
    """Initialize Firebase Admin SDK and return Firestore client."""
    service_account, options, _ = _extract_firebase_config()
    if not service_account:
        return None

    if not firebase_admin._apps:
        cred = credentials.Certificate(service_account)
        firebase_admin.initialize_app(cred, options=options or None)

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
            MARGIN_COLUMN: np.random.uniform(0.1, 0.5),
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


def _format_growth_badge(label: str, growth: float | None) -> str:
    """Return an HTML line with color and icon according to growth sign."""
    if growth is None:
        return f"<div><strong>{label}</strong>: <span style='color:#6b7280;'>⚪ N/A</span></div>"

    is_positive = growth >= 0
    icon = "🟢 ▲" if is_positive else "🔴 ▼"
    color = "#16a34a" if is_positive else "#dc2626"
    return (
        f"<div><strong>{label}</strong>: "
        f"<span style='color:{color};font-weight:700;'>{icon} {growth:+.1%}</span></div>"
    )


def _build_last_12_months_top_items(ventas_filtered: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build top 20 items by units and by revenue in the latest 12 months available."""
    required_cols = {'Artículo', 'Importe Neto', 'Unidades Venta', 'Año Factura', 'Mes Factura'}
    if ventas_filtered.empty or not required_cols.issubset(ventas_filtered.columns):
        return pd.DataFrame(), pd.DataFrame()

    sales = ventas_filtered.copy()
    sales = sales.dropna(subset=['Artículo', 'Importe Neto', 'Unidades Venta', 'Año Factura', 'Mes Factura'])
    if sales.empty:
        return pd.DataFrame(), pd.DataFrame()

    sales['Marca'] = sales['Clave 1'] if 'Clave 1' in sales.columns else ''
    sales['Descripción'] = sales['Descripción Artículo'] if 'Descripción Artículo' in sales.columns else ''

    sales['invoice_month'] = pd.to_datetime(
        {
            'year': sales['Año Factura'].astype(int),
            'month': sales['Mes Factura'].astype(int),
            'day': 1,
        },
        errors='coerce'
    )
    sales = sales.dropna(subset=['invoice_month'])
    if sales.empty:
        return pd.DataFrame(), pd.DataFrame()

    latest_month = sales['invoice_month'].max()
    period_start = latest_month - pd.DateOffset(months=11)

    sales_last_12m = sales[(sales['invoice_month'] >= period_start) & (sales['invoice_month'] <= latest_month)]
    if sales_last_12m.empty:
        return pd.DataFrame(), pd.DataFrame()

    grouped_sales = (
        sales_last_12m
        .groupby('Artículo', as_index=False)
        .agg(
            {
                'Marca': lambda x: x.dropna().iloc[0] if not x.dropna().empty else '',
                'Descripción': lambda x: x.dropna().iloc[0] if not x.dropna().empty else '',
                'Unidades Venta': 'sum',
                'Importe Neto': 'sum'
            }
        )
        .rename(columns={'Unidades Venta': 'Unidades 12M', 'Importe Neto': 'Ventas 12M'})
    )

    ordered_columns = ['Artículo', 'Marca', 'Descripción', 'Unidades 12M', 'Ventas 12M']
    grouped_sales = grouped_sales[ordered_columns]

    top_units = grouped_sales.nlargest(20, 'Unidades 12M').reset_index(drop=True)
    top_revenue = grouped_sales.nlargest(20, 'Ventas 12M').reset_index(drop=True)
    return top_units, top_revenue


def _build_top_items_excel(top_units: pd.DataFrame, top_revenue: pd.DataFrame) -> bytes:
    """Create an Excel file with top 20 units and top 20 revenue for last 12 months."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        top_units.to_excel(writer, index=False, sheet_name='Top 20 Unidades')
        top_revenue.to_excel(writer, index=False, sheet_name='Top 20 Ventas')
    output.seek(0)
    return output.getvalue()


def _set_firebase_status(message: str | None):
    st.session_state["firebase_status"] = message


def _format_exception_message(exc: Exception) -> str:
    return str(exc).strip() or exc.__class__.__name__


def _extract_firebase_config():
    """Resolve Firebase service account and optional app settings from Streamlit secrets."""
    service_account = st.secrets.get("FIREBASE_SERVICE_ACCOUNT") or st.secrets.get(
        "FIREBASE_SERVICE_ACCOUNT_JSON"
    )

    firebase_section = st.secrets.get("firebase")
    if not service_account and firebase_section:
        service_account = firebase_section.get("service_account")

    if isinstance(service_account, str):
        try:
            service_account = json.loads(service_account)
        except json.JSONDecodeError:
            return None, None, "FIREBASE_SERVICE_ACCOUNT_JSON no es JSON válido."

    if service_account:
        service_account = dict(service_account)

    options = {}
    if firebase_section:
        database_url = firebase_section.get("databaseURL")
        if database_url:
            options["databaseURL"] = database_url

    return service_account, options, None


def _get_firebase_collection(collection_name: str = PRIMARY_UPLOAD_COLLECTION):
    """Return Firestore collection when configured, else None."""
    try:
        import firebase_admin
        from firebase_admin import credentials, firestore
    except Exception:
        _set_firebase_status("No se pudo importar firebase_admin. Revisa dependencias del entorno.")
        return None

    if not firebase_admin._apps:
        service_account, options, parse_error = _extract_firebase_config()
        if parse_error:
            _set_firebase_status(parse_error)
            return None
        if not service_account:
            _set_firebase_status(
                "Falta FIREBASE_SERVICE_ACCOUNT / FIREBASE_SERVICE_ACCOUNT_JSON o [firebase.service_account] en secrets."
            )
            return None

        try:
            firebase_admin.initialize_app(credentials.Certificate(service_account), options=options or None)
        except Exception as exc:
            _set_firebase_status(f"No se pudo inicializar Firebase: {exc}")
            return None

    _set_firebase_status(None)
    return firestore.client().collection(collection_name)


def _normalize_column_name(column_name: str) -> str:
    """Normalize column names from uploaded files."""
    return re.sub(r"\s+", " ", str(column_name).strip())


def _find_existing_column(df: pd.DataFrame, aliases: tuple[str, ...]) -> str | None:
    """Return the first matching column from aliases using normalized names."""
    normalized_to_original = {
        _normalize_column_name(col).casefold(): col
        for col in df.columns
    }
    for alias in aliases:
        match = normalized_to_original.get(_normalize_column_name(alias).casefold())
        if match:
            return match
    return None


def _calculate_total_stock_value(manager: InventoryManager, compras_filtered: pd.DataFrame, selected_brand: list[str]) -> float:
    """Calculate total stock value prioritizing the dedicated stock value input."""
    if manager.stock_value_df is None or manager.stock_value_df.empty:
        return compras_filtered['Stock Valor'].sum() if 'Stock Valor' in compras_filtered.columns else 0.0

    stock_value_data = manager.stock_value_df.copy()
    article_col = _find_existing_column(
        stock_value_data,
        ('Código Artículo', 'Codigo Articulo', 'Artículo', 'Articulo')
    )
    amount_col = _find_existing_column(stock_value_data, ('Importe',))
    if not amount_col or not article_col:
        return compras_filtered['Stock Valor'].sum() if 'Stock Valor' in compras_filtered.columns else 0.0

    stock_value_data = stock_value_data[
        stock_value_data[article_col].notna()
        & stock_value_data[article_col].astype(str).str.strip().ne('')
    ]

    if selected_brand:
        brand_col = _find_existing_column(stock_value_data, ('Clave 1', 'Marca'))
        if brand_col:
            stock_value_data = stock_value_data[stock_value_data[brand_col].isin(selected_brand)]

    return pd.to_numeric(stock_value_data[amount_col], errors='coerce').fillna(0).sum()


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


def _persist_chunks_to_firestore(doc_ref, field_name: str, chunks: list[str]):
    """Persist encoded chunks in a Firestore subcollection to avoid 1MB document limit."""
    if not chunks:
        return

    for index, chunk in enumerate(chunks):
        doc_ref.collection("chunks").document(f"{field_name}_{index:04d}").set(
            {
                "field": field_name,
                "index": index,
                "data": chunk,
            }
        )


def _load_chunks_from_firestore(doc_ref, field_name: str, chunk_count: int):
    """Load encoded chunks persisted in Firestore subcollection."""
    if chunk_count <= 0:
        return []

    chunks = []
    for index in range(chunk_count):
        snapshot = doc_ref.collection("chunks").document(f"{field_name}_{index:04d}").get()
        if not snapshot.exists:
            break
        data = snapshot.to_dict() or {}
        chunk = data.get("data")
        if chunk:
            chunks.append(chunk)
    return chunks


def _load_local_history():
    if not HISTORY_FILE.exists():
        return []
    try:
        return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []


def _save_local_history(history):
    HISTORY_FILE.write_text(json.dumps(history, ensure_ascii=False), encoding="utf-8")


def _prune_expired_history_items(history: list[dict], now_utc: datetime | None = None) -> list[dict]:
    """Keep only entries from the retention window."""
    now_utc = now_utc or datetime.utcnow()
    cutoff = now_utc - timedelta(days=HISTORY_RETENTION_DAYS)
    kept = []
    for item in history:
        uploaded_at = item.get("uploaded_at")
        if not uploaded_at:
            continue
        try:
            uploaded_dt = datetime.fromisoformat(str(uploaded_at).replace("Z", ""))
        except Exception:
            continue
        if uploaded_dt >= cutoff:
            kept.append(item)
    return kept


def _delete_expired_firestore_docs(collection, now_utc: datetime | None = None):
    """Best effort cleanup for snapshots older than retention days."""
    if collection is None:
        return

    now_utc = now_utc or datetime.utcnow()
    cutoff_iso = (now_utc - timedelta(days=HISTORY_RETENTION_DAYS)).isoformat()
    docs = collection.where("uploaded_at", "<", cutoff_iso).limit(500).stream()
    for doc in docs:
        try:
            for chunk_doc in doc.reference.collection("chunks").stream():
                chunk_doc.reference.delete()
            doc.reference.delete()
        except Exception:
            continue


def save_upload_snapshot(
    stock_df,
    ventas_df,
    recepciones_df,
    stock_value_df=None,
    source="upload",
    snapshot_name: str | None = None,
):
    stock_records = _serialize_df(stock_df, "stock")
    ventas_records = _serialize_df(ventas_df, "ventas")
    recepciones_records = _serialize_df(recepciones_df, "recepciones")
    stock_value_records = _serialize_df(stock_value_df, "stock_value")
    stock_chunks = _encode_chunks(stock_records)
    ventas_chunks = _encode_chunks(ventas_records)
    recepciones_chunks = _encode_chunks(recepciones_records)
    stock_value_chunks = _encode_chunks(stock_value_records)

    doc_payload = {
        "uploaded_at": datetime.utcnow().isoformat(),
        "source": source,
        "snapshot_name": (snapshot_name or "").strip() or None,
        "file_count": 2 + int(recepciones_df is not None) + int(stock_value_df is not None),
        "stock_chunk_count": len(stock_chunks),
        "ventas_chunk_count": len(ventas_chunks),
        "recepciones_chunk_count": len(recepciones_chunks),
        "stock_value_chunk_count": len(stock_value_chunks),
    }

    primary_collection = _get_firebase_collection(PRIMARY_UPLOAD_COLLECTION)
    if primary_collection is not None:
        snapshot_id = primary_collection.document().id
        doc_payload["id"] = snapshot_id

        try:
            primary_doc_ref = primary_collection.document(snapshot_id)
            primary_doc_ref.set(doc_payload)
            _persist_chunks_to_firestore(primary_doc_ref, "stock", stock_chunks)
            _persist_chunks_to_firestore(primary_doc_ref, "ventas", ventas_chunks)
            _persist_chunks_to_firestore(primary_doc_ref, "recepciones", recepciones_chunks)
            _persist_chunks_to_firestore(primary_doc_ref, "stock_value", stock_value_chunks)
            _delete_expired_firestore_docs(primary_collection)
            st.caption("💾 Histórico guardado en: temporal (7 días)")
            return
        except Exception as exc:
            _set_firebase_status("No se pudo guardar en Firebase: " + _format_exception_message(exc))

    history = _load_local_history()
    snapshot_id = f"local-{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}"
    doc_payload["stock_chunks"] = stock_chunks
    doc_payload["ventas_chunks"] = ventas_chunks
    doc_payload["recepciones_chunks"] = recepciones_chunks
    doc_payload["stock_value_chunks"] = stock_value_chunks
    doc_payload["id"] = snapshot_id
    history.append(doc_payload)
    _save_local_history(_prune_expired_history_items(history))
    firebase_status = st.session_state.get("firebase_status")
    if firebase_status:
        st.warning(f"⚠️ Guardado local. Firebase no disponible: {firebase_status}")



def format_eur(value: float | int | None) -> str:
    """Format numeric values as EUR with Spanish separators: € 1.234,56."""
    if value is None or pd.isna(value):
        value = 0
    formatted = f"{float(value):,.2f}"
    formatted = formatted.replace(",", "_").replace(".", ",").replace("_", ".")
    return f"€ {formatted}"


def _resolve_current_year_cutoff(ventas_df: pd.DataFrame, current_year: int) -> datetime | None:
    """Get latest invoice date in current year to build YoY comparisons to date."""
    current_year_sales = ventas_df[ventas_df['Año Factura'] == current_year].copy()
    if current_year_sales.empty:
        return None

    cutoff_date = None
    if 'Fecha Factura' in current_year_sales.columns:
        parsed_dates = pd.to_datetime(current_year_sales['Fecha Factura'], errors='coerce')
        parsed_dates = parsed_dates.dropna()
        if not parsed_dates.empty:
            cutoff_date = parsed_dates.max().to_pydatetime()

    if cutoff_date is None and 'Mes Factura' in current_year_sales.columns:
        month_series = pd.to_numeric(current_year_sales['Mes Factura'], errors='coerce').dropna()
        if not month_series.empty:
            cutoff_month = int(month_series.max())
            cutoff_date = datetime(current_year, cutoff_month, 1) + pd.offsets.MonthEnd(0)
            cutoff_date = cutoff_date.to_pydatetime()

    return cutoff_date


def _sum_sales_to_cutoff_previous_year(ventas_df: pd.DataFrame, previous_year: int, cutoff_date: datetime | None) -> pd.Series:
    """Aggregate previous-year sales per customer up to the same month/day as current-year cutoff."""
    previous_year_sales = ventas_df[ventas_df['Año Factura'] == previous_year].copy()
    if previous_year_sales.empty:
        return pd.Series(dtype='float64')

    if cutoff_date is not None and 'Fecha Factura' in previous_year_sales.columns:
        prev_dates = pd.to_datetime(previous_year_sales['Fecha Factura'], errors='coerce')
        cutoff_prev = cutoff_date.replace(year=previous_year)
        valid_mask = prev_dates.notna() & (prev_dates.dt.date <= cutoff_prev.date())
        previous_year_sales = previous_year_sales[valid_mask]
    elif cutoff_date is not None and 'Mes Factura' in previous_year_sales.columns:
        previous_year_sales['Mes Factura'] = pd.to_numeric(previous_year_sales['Mes Factura'], errors='coerce')
        previous_year_sales = previous_year_sales[previous_year_sales['Mes Factura'] <= cutoff_date.month]

    if previous_year_sales.empty:
        return pd.Series(dtype='float64')

    return previous_year_sales.groupby('Cliente')['Importe Neto'].sum()


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
    cutoff_date = _resolve_current_year_cutoff(ventas_df, current_year)
    previous_year_to_date_sales = _sum_sales_to_cutoff_previous_year(
        ventas_df,
        current_year - 1,
        cutoff_date,
    )
    clientes[f'Año {current_year - 1} (to date)'] = clientes['Cod'].map(previous_year_to_date_sales).fillna(0)

    previous_year_to_date_col = f'Año {current_year - 1} (to date)'
    clientes[f'Dif {current_year - 1} - {current_year}'] = clientes.apply(
        lambda row: (row[year_cols[2]] - row[previous_year_to_date_col]) / row[previous_year_to_date_col]
        if row[previous_year_to_date_col] != 0 else 1,
        axis=1
    )

    return clientes


def calculate_clientes_monthly_from_ventas(ventas_df: pd.DataFrame, current_year: int, current_month: int) -> pd.DataFrame:
    """Build customer analysis comparing the latest 3 months."""
    if ventas_df is None or ventas_df.empty:
        return pd.DataFrame(columns=["Cod", "Cliente"])

    customers = ventas_df['Cliente'].dropna().unique()
    clientes = pd.DataFrame({'Cod': customers})

    name_map = ventas_df.groupby('Cliente')['Nombre Cliente'].first()
    clientes['Cliente'] = clientes['Cod'].map(name_map)

    month_points = []
    for offset in [2, 1, 0]:
        month_value = current_month - offset
        year_value = current_year
        if month_value <= 0:
            month_value += 12
            year_value -= 1
        month_points.append((year_value, month_value))

    month_cols = []
    for year_value, month_value in month_points:
        col_name = f"Mes {year_value}-{month_value:02d}"
        month_sales = ventas_df[
            (ventas_df['Año Factura'] == year_value) & (ventas_df['Mes Factura'] == month_value)
        ].groupby('Cliente')['Importe Neto'].sum()
        clientes[col_name] = clientes['Cod'].map(month_sales).fillna(0)
        month_cols.append(col_name)

    clientes[f'Dif {month_cols[0]} - {month_cols[1]}'] = clientes.apply(
        lambda row: (row[month_cols[1]] - row[month_cols[0]]) / row[month_cols[0]]
        if row[month_cols[0]] != 0 else 1,
        axis=1
    )
    clientes[f'Dif {month_cols[1]} - {month_cols[2]}'] = clientes.apply(
        lambda row: (row[month_cols[2]] - row[month_cols[1]]) / row[month_cols[1]]
        if row[month_cols[1]] != 0 else 1,
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
    collection = _get_firebase_collection(PRIMARY_UPLOAD_COLLECTION)
    if collection is not None:
        _delete_expired_firestore_docs(collection)
        docs = collection.order_by("uploaded_at", direction="DESCENDING").limit(500).stream()
        history = []
        for doc in docs:
            data = doc.to_dict() or {}
            data["id"] = data.get("id") or doc.id
            data["storage_scope"] = "temporal"
            history.append(data)
        if history:
            return history

    local_history = _prune_expired_history_items(_load_local_history())
    _save_local_history(local_history)
    history = sorted(local_history, key=lambda x: x.get("uploaded_at", ""), reverse=True)
    return history[:50]


def get_upload_by_id(upload_id: str):
    def _hydrate_firestore_doc(doc_snapshot):
        if not doc_snapshot.exists:
            return None

        data = doc_snapshot.to_dict() or {}

        if "stock_chunks" not in data:
            stock_count = int(data.get("stock_chunk_count", 0) or 0)
            ventas_count = int(data.get("ventas_chunk_count", 0) or 0)
            recepciones_count = int(data.get("recepciones_chunk_count", 0) or 0)
            stock_value_count = int(data.get("stock_value_chunk_count", 0) or 0)

            data["stock_chunks"] = _load_chunks_from_firestore(doc_snapshot.reference, "stock", stock_count)
            data["ventas_chunks"] = _load_chunks_from_firestore(doc_snapshot.reference, "ventas", ventas_count)
            data["recepciones_chunks"] = _load_chunks_from_firestore(
                doc_snapshot.reference,
                "recepciones",
                recepciones_count,
            )
            data["stock_value_chunks"] = _load_chunks_from_firestore(
                doc_snapshot.reference,
                "stock_value",
                stock_value_count,
            )

        return data

    primary_collection = _get_firebase_collection(PRIMARY_UPLOAD_COLLECTION)

    if primary_collection is not None:
        doc = primary_collection.document(upload_id).get()
        hydrated = _hydrate_firestore_doc(doc)
        if hydrated is not None:
            return hydrated
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
            min_value=1.0,
            max_value=6.0,
            value=2.0,
            step=0.1,
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
        except Exception as exc:
            history_items = _load_local_history()
            _set_firebase_status(f"Error consultando históricos: {_format_exception_message(exc)}")
            st.error("No se pudo conectar a Firebase...")

        firebase_status = st.session_state.get("firebase_status")
        if firebase_status:
            st.caption(f"Estado Firebase: {firebase_status}")

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
                storage_scope = item.get("storage_scope", "local")
                storage_label = {
                    "temporal": "temporal",
                    "permanente": "permanente",
                    "permanente+temporal": "temporal + permanente",
                    "local": "local",
                }.get(storage_scope, storage_scope)
                name_part = f"{custom_name} · " if custom_name else ""
                label = f"{name_part}{formatted_date} · {source} · {file_count} archivos · {storage_label}"
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
                        manager.stock_value_df = _deserialize_df(_decode_chunks(doc.get("stock_value_chunks", [])))

                        st.session_state.manager = manager
                        st.session_state.data_loaded = True
                        st.success("✅ Histórico cargado")
                        st.rerun()
                except Exception as exc:
                    _set_firebase_status(f"Error cargando histórico: {_format_exception_message(exc)}")
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
        
        col1, col2, col3, col4 = st.columns(4)
        
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

        with col4:
            stock_value_file = st.file_uploader(
                "Stock Value Data (Excel/CSV)",
                type=['xlsx', 'csv'],
                help="Optional: Clave 1, Código Artículo, Unidades, Importe"
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
                        
                        if stock_value_file:
                            if stock_value_file.name.endswith('.xlsx'):
                                manager.stock_value_df = pd.read_excel(stock_value_file)
                            else:
                                manager.stock_value_df = pd.read_csv(stock_value_file)
                        
                        # Clean column names
                        manager.stock_df.columns = [_normalize_column_name(col) for col in manager.stock_df.columns]
                        manager.ventas_df.columns = [_normalize_column_name(col) for col in manager.ventas_df.columns]
                        if manager.recepciones_df is not None:
                            manager.recepciones_df.columns = [_normalize_column_name(col) for col in manager.recepciones_df.columns]
                        if manager.stock_value_df is not None:
                            manager.stock_value_df.columns = [_normalize_column_name(col) for col in manager.stock_value_df.columns]
                        
                        st.session_state.manager = manager
                        st.session_state.data_loaded = True

                        try:
                            save_upload_snapshot(
                                manager.stock_df,
                                manager.ventas_df,
                                manager.recepciones_df,
                                manager.stock_value_df,
                                source="upload",
                                snapshot_name=snapshot_name
                            )
                        except Exception as exc:
                            _set_firebase_status(f"Error guardando histórico: {_format_exception_message(exc)}")
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
        manager.meses_compras = float(meses_compras)
        
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
        stock_filtered = manager.stock_df.copy() if manager.stock_df is not None else pd.DataFrame()

        if selected_brand:
            compras_filtered = compras_filtered[compras_filtered['Marca'].isin(selected_brand)]
            if not ventas_filtered.empty and 'Clave 1' in ventas_filtered.columns:
                ventas_filtered = ventas_filtered[ventas_filtered['Clave 1'].isin(selected_brand)]
            if not stock_filtered.empty and not ventas_filtered.empty and 'Artículo' in ventas_filtered.columns:
                selected_articles = set(ventas_filtered['Artículo'].dropna().unique())
                stock_filtered = stock_filtered[stock_filtered['Artículo'].isin(selected_articles)]

        clientes_df = calculate_clientes_from_ventas(ventas_filtered, manager.current_year)
        clientes_monthly_df = calculate_clientes_monthly_from_ventas(
            ventas_filtered,
            manager.current_year,
            manager.current_month,
        )

        month_m2 = manager.current_month - 2 if manager.current_month > 2 else manager.current_month - 2 + 12
        year_m2 = manager.current_year if manager.current_month > 2 else manager.current_year - 1
        month_m1 = manager.current_month - 1 if manager.current_month > 1 else 12
        year_m1 = manager.current_year if manager.current_month > 1 else manager.current_year - 1
        sales_m2_total = ventas_filtered[
            (ventas_filtered['Año Factura'] == year_m2) & (ventas_filtered['Mes Factura'] == month_m2)
        ]['Importe Neto'].sum() if not ventas_filtered.empty else 0
        sales_m1_total = ventas_filtered[
            (ventas_filtered['Año Factura'] == year_m1) & (ventas_filtered['Mes Factura'] == month_m1)
        ]['Importe Neto'].sum() if not ventas_filtered.empty else 0
        sales_m1_last_year_total = ventas_filtered[
            (ventas_filtered['Año Factura'] == year_m1 - 1) & (ventas_filtered['Mes Factura'] == month_m1)
        ]['Importe Neto'].sum() if not ventas_filtered.empty else 0
        monthly_growth = (
            (sales_m1_total - sales_m2_total) / sales_m2_total
            if sales_m2_total != 0 else None
        )
        yearly_growth = (
            (sales_m1_total - sales_m1_last_year_total) / sales_m1_last_year_total
            if sales_m1_last_year_total != 0 else None
        )

        unit_cost_map = pd.Series(dtype=float)
        if not ventas_filtered.empty and {'Artículo', 'Precio Coste'}.issubset(ventas_filtered.columns):
            unit_cost_map = (
                ventas_filtered[['Artículo', 'Precio Coste']]
                .dropna(subset=['Artículo'])
                .assign(**{'Precio Coste': lambda df: pd.to_numeric(df['Precio Coste'], errors='coerce').fillna(0)})
                .groupby('Artículo')['Precio Coste']
                .mean()
            )

        unit_sale_price_map = pd.Series(dtype=float)
        if not ventas_filtered.empty and {'Artículo', 'Importe Neto', 'Unidades Venta'}.issubset(ventas_filtered.columns):
            sales_price_df = (
                ventas_filtered[['Artículo', 'Importe Neto', 'Unidades Venta']]
                .dropna(subset=['Artículo'])
                .assign(
                    **{
                        'Importe Neto': lambda df: pd.to_numeric(df['Importe Neto'], errors='coerce').fillna(0),
                        'Unidades Venta': lambda df: pd.to_numeric(df['Unidades Venta'], errors='coerce').fillna(0),
                    }
                )
            )
            sales_totals_by_article = sales_price_df.groupby('Artículo')[['Importe Neto', 'Unidades Venta']].sum()
            unit_sale_price_map = (
                sales_totals_by_article['Importe Neto']
                .div(sales_totals_by_article['Unidades Venta'].replace(0, pd.NA))
                .fillna(0)
            )

        pending_receive_units = 0.0
        pending_receive_value = 0.0
        pending_send_units = 0.0
        pending_send_value = 0.0
        pending_send_no_stock_units = 0.0
        pending_send_no_stock_value = 0.0
        pending_send_with_stock_units = 0.0
        pending_send_with_stock_value = 0.0

        if not stock_filtered.empty and 'Artículo' in stock_filtered.columns:
            stock_metrics = stock_filtered.copy()
            pending_receive_column = next(
                (
                    col for col in stock_metrics.columns
                    if _normalize_column_name(col).casefold() == 'total pendiente recibir'
                ),
                None,
            )
            if pending_receive_column is None:
                pending_receive_column = '__pending_receive__'
                stock_metrics[pending_receive_column] = (
                    pd.to_numeric(stock_metrics.get('Pendiente Recibir Compra', 0), errors='coerce').fillna(0)
                    + pd.to_numeric(stock_metrics.get('Pendiente Entrar Fabricación', 0), errors='coerce').fillna(0)
                    + pd.to_numeric(stock_metrics.get('En Tránsito', 0), errors='coerce').fillna(0)
                )

            numeric_columns = [pending_receive_column, 'Cartera', 'Stock']
            for column in numeric_columns:
                if column not in stock_metrics.columns:
                    stock_metrics[column] = 0
                stock_metrics[column] = pd.to_numeric(stock_metrics[column], errors='coerce').fillna(0)

            stock_metrics['Precio Coste'] = stock_metrics['Artículo'].map(unit_cost_map).fillna(0)
            stock_metrics['Precio Venta'] = stock_metrics['Artículo'].map(unit_sale_price_map).fillna(0)

            pending_receive_units = stock_metrics[pending_receive_column].sum()
            pending_receive_value = (stock_metrics[pending_receive_column] * stock_metrics['Precio Coste']).sum()

            stock_metrics['Cartera Sin Stock'] = (stock_metrics['Cartera'] - stock_metrics['Stock']).clip(lower=0)
            stock_metrics['Cartera Con Stock'] = stock_metrics['Cartera'] - stock_metrics['Cartera Sin Stock']

            pending_send_units = stock_metrics['Cartera'].sum()
            pending_send_value = (stock_metrics['Cartera'] * stock_metrics['Precio Venta']).sum()
            pending_send_no_stock_units = stock_metrics['Cartera Sin Stock'].sum()
            pending_send_no_stock_value = (stock_metrics['Cartera Sin Stock'] * stock_metrics['Precio Venta']).sum()
            pending_send_with_stock_units = stock_metrics['Cartera Con Stock'].sum()
            pending_send_with_stock_value = (stock_metrics['Cartera Con Stock'] * stock_metrics['Precio Venta']).sum()

        current_year_sales_total = 0.0
        if not ventas_filtered.empty and {'Año Factura', 'Importe Neto'}.issubset(ventas_filtered.columns):
            current_year_sales_total = ventas_filtered.loc[
                ventas_filtered['Año Factura'] == manager.current_year,
                'Importe Neto'
            ].sum()

        stats = {
            'total_stock_value': _calculate_total_stock_value(manager, compras_filtered, selected_brand),
            'total_pedido_value': compras_filtered['VALOR PEDIDO'].sum() if 'VALOR PEDIDO' in compras_filtered.columns else 0,
            'total_pedido_margin': compras_filtered['MARGEN PEDIDO'].sum() if 'MARGEN PEDIDO' in compras_filtered.columns else 0,
            'items_to_order': int((compras_filtered['PEDIDO'] > 0).sum()) if 'PEDIDO' in compras_filtered.columns else 0,
            'pending_receive_units': pending_receive_units,
            'pending_receive_value': pending_receive_value,
            'pending_send_units': pending_send_units,
            'pending_send_value': pending_send_value,
            'pending_send_no_stock_units': pending_send_no_stock_units,
            'pending_send_no_stock_value': pending_send_no_stock_value,
            'pending_send_with_stock_units': pending_send_with_stock_units,
            'pending_send_with_stock_value': pending_send_with_stock_value,
            'current_year_sales_total': current_year_sales_total,
        }

        # Tabs for different sections
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Dashboard",
            "🛒 Purchase Orders",
            "👥 Customers",
            "👥 Customers Monthly",
            "📁 Export"
        ])
        
        with tab1:
            st.header("📊 Dashboard Overview")
            
            # KPI Metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Total Stock Value",
                    format_eur(stats.get('total_stock_value', 0)),
                    help="Total value of current inventory"
                )
            
            with col2:
                st.metric(
                    "Purchase Order Value",
                    format_eur(stats.get('total_pedido_value', 0)),
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
                    format_eur(stats.get('total_pedido_margin', 0)),
                    help="Expected profit margin from orders"
                )

            st.markdown("### Indicadores operativos")
            col8, col9, col10 = st.columns(3)

            with col8:
                st.metric(
                    "Pendiente de recibir",
                    format_eur(stats.get('pending_receive_value', 0)),
                    help="Suma de 'Total Pendiente Recibir' * 'Precio Coste' por artículo"
                )
                st.caption(f"{stats.get('pending_receive_units', 0):,.0f} unidades")

            with col9:
                st.metric(
                    "Pendiente de enviar",
                    format_eur(stats.get('pending_send_value', 0)),
                    help="Suma de 'Cartera' * 'Precio Venta medio' por artículo"
                )
                st.caption(
                    f"{stats.get('pending_send_units', 0):,.0f} uds · "
                    f"Con stock: {stats.get('pending_send_with_stock_units', 0):,.0f} uds "
                    f"({format_eur(stats.get('pending_send_with_stock_value', 0))}) · "
                    f"Sin stock: {stats.get('pending_send_no_stock_units', 0):,.0f} uds "
                    f"({format_eur(stats.get('pending_send_no_stock_value', 0))})"
                )

            with col10:
                st.metric(
                    f"Ventas acumuladas {manager.current_year}",
                    format_eur(stats.get('current_year_sales_total', 0)),
                    help="Importe neto acumulado del año actual"
                )

            col5, col6, col7 = st.columns(3)
            with col5:
                st.metric(
                    f"Ventas {year_m2}-{month_m2:02d}",
                    format_eur(sales_m2_total),
                    help="Net sales from 2 months ago"
                )
            with col6:
                st.metric(
                    f"Ventas {year_m1 - 1}-{month_m1:02d}",
                    format_eur(sales_m1_last_year_total),
                    help="Net sales from the same month last year"
                )
            with col7:
                st.metric(
                    f"Ventas {year_m1}-{month_m1:02d}",
                    format_eur(sales_m1_total),
                    help="Net sales from 1 month ago compared to month -2 and same month last year"
                )
                st.markdown(
                    _format_growth_badge("vs m-2", monthly_growth) + _format_growth_badge("vs a-1", yearly_growth),
                    unsafe_allow_html=True,
                )
            
            st.divider()
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Top 20 Items últimos 12 meses")
                st.caption("Listado de los 20 artículos con más unidades vendidas y de los 20 con mayor venta neta en los últimos 12 meses disponibles.")
                top_units_12m, top_revenue_12m = _build_last_12_months_top_items(ventas_filtered)
                if top_units_12m.empty and top_revenue_12m.empty:
                    st.info("No hay datos con los filtros actuales.")
                else:
                    list_col1, list_col2 = st.columns(2)
                    with list_col1:
                        st.markdown("**Top 20 por unidades vendidas**")
                        top_units_display = top_units_12m.copy()
                        top_units_display['Ventas 12M'] = top_units_display['Ventas 12M'].map(format_eur)
                        st.dataframe(
                            top_units_display,
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                'Unidades 12M': st.column_config.NumberColumn(format="%.0f"),
                            }
                        )

                    with list_col2:
                        st.markdown("**Top 20 por ventas netas**")
                        top_revenue_display = top_revenue_12m.copy()
                        top_revenue_display['Ventas 12M'] = top_revenue_display['Ventas 12M'].map(format_eur)
                        st.dataframe(
                            top_revenue_display,
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                'Unidades 12M': st.column_config.NumberColumn(format="%.0f"),
                            }
                        )

                    excel_data = _build_top_items_excel(top_units_12m, top_revenue_12m)
                    st.download_button(
                        label="📥 Descargar Top 20 (Excel)",
                        data=excel_data,
                        file_name="top_20_items_ultimos_12_meses.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
            
            with col2:
                st.subheader("Purchase Orders by Brand")
                st.caption("Distribución del valor total de compra recomendado por marca para identificar concentración de pedidos.")
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
            st.caption("Clasifica los artículos por cobertura de stock según meses disponibles: Critical < 1, Low 1-<2, Normal 2-<4, High ≥ 4.")
            stock_status = compras_filtered.copy()
            stock_status['Status'] = stock_status.apply(
                lambda x: 'Critical' if x['Meses de Stock'] < 1 else
                         'Low' if x['Meses de Stock'] < 2 else
                         'Normal' if x['Meses de Stock'] < 4 else 'High',
                axis=1
            )
            st.markdown(
                "- 🔴 **Critical**: menos de 1 mes de stock\n"
                "- 🟠 **Low**: entre 1 y menos de 2 meses\n"
                "- 🟢 **Normal**: entre 2 y menos de 4 meses\n"
                "- 🔵 **High**: 4 meses o más"
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
            purchase_columns = [
                'SKU', 'Marca', 'Descripción', 'Stock Unidades',
                'Meses de Stock', 'PEDIDO', 'VALOR PEDIDO', 'MARGEN PEDIDO'
            ]
            purchase_display_df = filtered_df[purchase_columns].copy()
            purchase_display_df['Stock Unidades'] = purchase_display_df['Stock Unidades'].map(lambda v: f"{v:.0f}")
            purchase_display_df['Meses de Stock'] = purchase_display_df['Meses de Stock'].map(lambda v: f"{v:.1f}")
            purchase_display_df['PEDIDO'] = purchase_display_df['PEDIDO'].map(lambda v: f"{v:.0f}")
            purchase_display_df['VALOR PEDIDO'] = purchase_display_df['VALOR PEDIDO'].map(format_eur)
            purchase_display_df['MARGEN PEDIDO'] = purchase_display_df['MARGEN PEDIDO'].map(format_eur)
            st.dataframe(purchase_display_df, use_container_width=True, height=600)
            
            # Summary
            st.divider()
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Filtered Items", len(filtered_df))
            with col2:
                st.metric("Total Order Value", format_eur(filtered_df['VALOR PEDIDO'].sum()))
            with col3:
                st.metric("Total Margin", format_eur(filtered_df['MARGEN PEDIDO'].sum()))
        
        with tab3:
            st.header("👥 Customer Analysis")
            
            # Display customer table
            clientes_display_df = clientes_df.copy()
            for col in clientes_display_df.columns:
                if 'Año' in col:
                    clientes_display_df[col] = clientes_display_df[col].map(format_eur)
                if 'Dif' in col:
                    clientes_display_df[col] = clientes_display_df[col].map(lambda v: f"{v:.1%}")
            st.dataframe(clientes_display_df, use_container_width=True, height=600)
            
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
            st.header("👥 Customer Monthly Analysis")

            clientes_monthly_display_df = clientes_monthly_df.copy()
            for col in clientes_monthly_display_df.columns:
                if col.startswith('Mes '):
                    clientes_monthly_display_df[col] = clientes_monthly_display_df[col].map(format_eur)
                if 'Dif' in col:
                    clientes_monthly_display_df[col] = clientes_monthly_display_df[col].map(lambda v: f"{v:.1%}")
            st.dataframe(clientes_monthly_display_df, use_container_width=True, height=600)

            st.subheader("Top 10 Customers by Last Month Sales")
            latest_month_cols = [col for col in clientes_monthly_df.columns if col.startswith('Mes ')]
            if latest_month_cols:
                top_col = latest_month_cols[-1]
                top_customers = clientes_monthly_df.nlargest(10, top_col)
                fig = px.bar(
                    top_customers,
                    x='Cliente',
                    y=top_col,
                    color=top_col,
                    color_continuous_scale='Teal'
                )
                fig.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig, use_container_width=True)

        with tab5:
            st.header("📁 Export Results")
            
            st.write("Download your analysis results in Excel format")
            
            # Create Excel file in memory
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                compras_filtered.to_excel(writer, sheet_name='COMPRAS', index=False)
                clientes_df.to_excel(writer, sheet_name='CLIENTES', index=False)
                clientes_monthly_df.to_excel(writer, sheet_name='CLIENTES_MENSUAL', index=False)
            
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
            col1, col2, col3 = st.columns(3)
            
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

            with col3:
                clientes_monthly_excel = dataframe_to_excel_bytes(clientes_monthly_df, 'CLIENTES_MENSUAL')
                st.download_button(
                    "📄 Download Customers Monthly (Excel)",
                    clientes_monthly_excel,
                    f"clientes_mensual_{pd.Timestamp.now().strftime('%Y%m%d')}.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )


if __name__ == "__main__":
    main()
