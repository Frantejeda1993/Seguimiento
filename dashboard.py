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

import inventory_manager as inventory_manager_module

InventoryManager = inventory_manager_module.InventoryManager
ColumnMappingError = inventory_manager_module.ColumnMappingError
MultiColumnMappingError = getattr(
    inventory_manager_module,
    "MultiColumnMappingError",
    type("MultiColumnMappingError", (Exception,), {}),
)
import io
import hmac
import json
import gzip
import base64
import re
import calendar
from datetime import date, datetime, timedelta
from pathlib import Path

MESES_ES = ['', 'Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']


HISTORY_FILE = Path(".upload_history_local.json")
MAX_CHUNK_SIZE = 700_000
PRIMARY_UPLOAD_COLLECTION = "upload_history"
HISTORY_RETENTION_DAYS = 7
MARGIN_COLUMN = "CR3: % Margen s/Venta + Transport"
MARGIN_ALIASES_FILE = Path(".margin_column_aliases.json")
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


def _load_margin_aliases() -> list[str]:
    if not MARGIN_ALIASES_FILE.exists():
        return []
    try:
        data = json.loads(MARGIN_ALIASES_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    return [str(item).strip() for item in data if str(item).strip()]


def _save_margin_aliases(aliases: list[str]) -> None:
    deduped = list(dict.fromkeys([str(item).strip() for item in aliases if str(item).strip()]))
    MARGIN_ALIASES_FILE.write_text(
        json.dumps(deduped, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def render_column_mapping_form(
    errors: list[ColumnMappingError],
    manager,
    dataframes: dict[str, pd.DataFrame],
) -> bool:
    """
    Render the column-mapping resolution form.

    Shows one section per ColumnMappingError:
      - Already-resolved columns as read-only info (✅)
      - Missing columns as selectboxes
      - Available columns come from the live DataFrame for each erring input,
        falling back to the columns captured in the error.

    Returns True if the form was submitted successfully and st.rerun() should
    be called by the caller.
    """
    st.warning(
        "⚠️ Algunas columnas no pudieron mapearse automáticamente. "
        "Revisa la asignación y haz clic en **Guardar y recalcular**."
    )

    with st.form("column_mapping_form"):
        all_selections: dict[str, dict[str, str]] = {}

        for cme in errors:
            st.subheader(f"📋 Input: {cme.input_name}")
            live_df = dataframes.get(cme.input_name, pd.DataFrame())
            available_options = list(live_df.columns) or list(cme.available)

            if cme.resolved:
                st.markdown("**Columnas mapeadas automáticamente ✅**")
                for logical_name, actual_col in cme.resolved.items():
                    st.info(f"**{logical_name}** → `{actual_col}`")

            input_selections: dict[str, str] = dict(cme.resolved)

            if cme.missing:
                st.markdown("**Columnas que necesitan asignación manual ⚠️**")
                for logical_name in cme.missing:
                    chosen = st.selectbox(
                        f"{logical_name}",
                        options=["— seleccionar —", *available_options],
                        index=0,
                        key=f"mapping_{cme.input_name}_{logical_name}",
                    )
                    input_selections[logical_name] = chosen

            all_selections[cme.input_name] = input_selections
            st.divider()

        submitted = st.form_submit_button("💾 Guardar y recalcular", type="primary")

    if not submitted:
        return False

    still_missing = []
    errors_by_input = {cme.input_name: cme for cme in errors}
    for input_name, selections in all_selections.items():
        cme = errors_by_input.get(input_name)
        if not cme:
            continue
        for logical_name in cme.missing:
            if selections.get(logical_name, "— seleccionar —") == "— seleccionar —":
                still_missing.append(f"{input_name} → {logical_name}")

    if still_missing:
        st.error("Debes seleccionar una columna para: " + ", ".join(still_missing))
        return False

    for input_name, selections in all_selections.items():
        overrides = st.session_state.setdefault("column_overrides", {}).setdefault(input_name, {})
        overrides.update({k: v for k, v in selections.items() if v != "— seleccionar —"})

        if input_name == "Ventas" and "Margen" in selections:
            margin_col = selections["Margen"]
            if margin_col and margin_col != "— seleccionar —":
                aliases = _load_margin_aliases()
                if margin_col not in aliases:
                    aliases.append(margin_col)
                    _save_margin_aliases(aliases)
                manager.set_extra_margin_aliases(_load_margin_aliases())

    return True


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

    try:
        app_password = st.secrets.get("APP_PASSWORD")
    except Exception:
        app_password = None

    if not app_password:
        st.error(
            "Missing APP_PASSWORD secret. Set it in Streamlit Cloud "
            "Secrets or in a local `.streamlit/secrets.toml` file."
        )
        return False

    with st.form("auth_form", clear_on_submit=True):
        password = st.text_input("Password", type="password", key="password_input")
        sign_in = st.form_submit_button("Sign in", type="primary")

    if sign_in:
        if hmac.compare_digest(password, str(app_password)):
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password. Please try again.")

    return False


@st.cache_data
def load_sample_data():
    """Generate sample data for demonstration."""
    import numpy as np
    np.random.seed(42)
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
        'Disponible Teorico': np.random.randint(0, 700, 50),
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
                'Marca': 'first',
                'Descripción': 'first',
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
    priority_cols = [c for c in SNAPSHOT_COLUMNS[kind] if c in df.columns]
    remaining_cols = [c for c in df.columns if c not in priority_cols]
    data = df[priority_cols + remaining_cols]
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
            help="Horizonte de cobertura en meses (plazo de entrega + periodo de revisión). Filtra por marca para ajustarlo al proveedor"
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
                        manager.set_extra_margin_aliases(_load_margin_aliases())
                        manager.stock_df = _deserialize_df(_decode_chunks(doc.get("stock_chunks", [])))
                        manager.ventas_df = _deserialize_df(_decode_chunks(doc.get("ventas_chunks", [])))
                        manager.recepciones_df = _deserialize_df(_decode_chunks(doc.get("recepciones_chunks", [])))
                        manager.stock_value_df = _deserialize_df(_decode_chunks(doc.get("stock_value_chunks", [])))

                        st.session_state.manager = manager
                        st.session_state.data_loaded = True
                        st.session_state.column_overrides = {}
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
    if 'column_overrides' not in st.session_state:
        st.session_state.column_overrides = {}
    
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
                        manager.set_extra_margin_aliases(_load_margin_aliases())
                        
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
                        st.session_state.column_overrides = {}

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
                    manager.set_extra_margin_aliases(_load_margin_aliases())
                    manager.stock_df = stock_df
                    manager.ventas_df = ventas_df
                    manager.recepciones_df = recepciones_df
                    
                    st.session_state.manager = manager
                    st.session_state.data_loaded = True
                    st.session_state.column_overrides = {}
                    st.success("✅ Sample data loaded!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    # Main analysis section
    if st.session_state.data_loaded and st.session_state.manager:
        manager = st.session_state.manager
        manager.meses_compras = float(meses_compras)
        
        # Recalcular sólo al cargar datos nuevos o modificar los parámetros. Así
        # los filtros y widgets no vuelven a ejecutar el forecast completo.
        overrides = st.session_state.get("column_overrides") or {}
        cache_key = (
            id(manager.stock_df), id(manager.ventas_df), id(manager.recepciones_df),
            float(meses_compras), json.dumps(overrides, sort_keys=True, default=str),
        )
        if st.session_state.get("compras_cache_key") == cache_key:
            compras_df = st.session_state["compras_cache"]
        else:
            # Calculate analysis
            with st.spinner("Calculating..."):
                try:
                    compras_df = manager.calculate_compras(
                        column_overrides=overrides,
                    )
                    st.session_state["compras_cache_key"] = cache_key
                    st.session_state["compras_cache"] = compras_df
                except MultiColumnMappingError as multi_err:
                    dataframes = {
                        "Ventas": manager.ventas_df if manager.ventas_df is not None else pd.DataFrame(),
                        "Stock": manager.stock_df if manager.stock_df is not None else pd.DataFrame(),
                    }
                    if render_column_mapping_form(multi_err.errors, manager, dataframes=dataframes):
                        st.rerun()
                    return
                except ColumnMappingError as cme:
                    dataframes = {
                        "Ventas": manager.ventas_df if manager.ventas_df is not None else pd.DataFrame(),
                        "Stock": manager.stock_df if manager.stock_df is not None else pd.DataFrame(),
                    }
                    if render_column_mapping_form([cme], manager, dataframes=dataframes):
                        st.rerun()
                    return
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
            brand_col_v = next((c for c in ['Clave 1', 'Marca'] if c in ventas_filtered.columns), None)
            if not ventas_filtered.empty and brand_col_v:
                ventas_filtered = ventas_filtered[ventas_filtered[brand_col_v].isin(selected_brand)]
            if not stock_filtered.empty and not ventas_filtered.empty and 'Artículo' in ventas_filtered.columns:
                selected_articles = set(ventas_filtered['Artículo'].dropna().unique())
                stock_filtered = stock_filtered[stock_filtered['Artículo'].isin(selected_articles)]

        from clientes import build_clientes_table, build_zonas_table
        # Managers retained in Streamlit session state can predate the `today`
        # attribute introduced in the forecasting release.  Keep those sessions
        # usable instead of failing while rendering the customer tab.
        as_of = getattr(manager, "today", date.today())
        clientes_df = build_clientes_table(ventas_filtered, as_of)
        zonas_df = build_zonas_table(ventas_filtered, as_of)

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

        # Day extraction for Month-to-Date (Month_YTD) calculations
        day_series = None
        day_col = next((c for c in ['Día Factura', 'Dia Factura', 'Día', 'Dia', 'Dia_Factura', 'Day'] if c in ventas_filtered.columns), None)
        if day_col:
            day_series = pd.to_numeric(ventas_filtered[day_col], errors='coerce')
        else:
            date_col = next((c for c in ['Fecha Factura', 'Fecha_Factura', 'Fecha Facturacion', 'Fecha Facturación', 'Fecha', 'Date', 'Fecha Albarán', 'Fecha Albaran'] if c in ventas_filtered.columns), None)
            if date_col:
                day_series = pd.to_datetime(ventas_filtered[date_col], errors='coerce', dayfirst=True).dt.day

        if isinstance(as_of, str):
            today_date = pd.to_datetime(as_of).date()
        elif hasattr(as_of, 'date') and callable(as_of.date):
            today_date = as_of.date()
        elif isinstance(as_of, date):
            today_date = as_of
        else:
            today_date = date.today()

        today_day = today_date.day
        cur_year = manager.current_year
        cur_month = manager.current_month
        py_year = cur_year - 1

        pm_month = cur_month - 1 if cur_month > 1 else 12
        pm_year = cur_year if cur_month > 1 else cur_year - 1
        pm_max_day = calendar.monthrange(pm_year, pm_month)[1]
        pm_cutoff_day = min(today_day, pm_max_day)

        month_ytd_total = 0.0
        month_ytd_py_total = 0.0
        month_ytd_pm_total = 0.0
        has_day_granularity = day_series is not None and day_series.notna().any()

        if not ventas_filtered.empty and {'Año Factura', 'Mes Factura', 'Importe Neto'}.issubset(ventas_filtered.columns):
            if has_day_granularity:
                mask_cur = (ventas_filtered['Año Factura'] == cur_year) & (ventas_filtered['Mes Factura'] == cur_month) & (day_series <= today_day)
                mask_py = (ventas_filtered['Año Factura'] == py_year) & (ventas_filtered['Mes Factura'] == cur_month) & (day_series <= today_day)
                mask_pm = (ventas_filtered['Año Factura'] == pm_year) & (ventas_filtered['Mes Factura'] == pm_month) & (day_series <= pm_cutoff_day)
            else:
                mask_cur = (ventas_filtered['Año Factura'] == cur_year) & (ventas_filtered['Mes Factura'] == cur_month)
                mask_py = (ventas_filtered['Año Factura'] == py_year) & (ventas_filtered['Mes Factura'] == cur_month)
                mask_pm = (ventas_filtered['Año Factura'] == pm_year) & (ventas_filtered['Mes Factura'] == pm_month)

            month_ytd_total = pd.to_numeric(ventas_filtered.loc[mask_cur, 'Importe Neto'], errors='coerce').fillna(0).sum()
            month_ytd_py_total = pd.to_numeric(ventas_filtered.loc[mask_py, 'Importe Neto'], errors='coerce').fillna(0).sum()
            month_ytd_pm_total = pd.to_numeric(ventas_filtered.loc[mask_pm, 'Importe Neto'], errors='coerce').fillna(0).sum()

        growth_cur_vs_py = (
            (month_ytd_total - month_ytd_py_total) / month_ytd_py_total
            if month_ytd_py_total != 0 else None
        )
        growth_cur_vs_pm = (
            (month_ytd_total - month_ytd_pm_total) / month_ytd_pm_total
            if month_ytd_pm_total != 0 else None
        )

        label_m1_vs_m2 = f"{MESES_ES[month_m1]} {str(year_m1)[-2:]} vs {MESES_ES[month_m2]} {str(year_m2)[-2:]}"
        label_m1_vs_py = f"{MESES_ES[month_m1]} {str(year_m1)[-2:]} vs {MESES_ES[month_m1]} {str(year_m1 - 1)[-2:]}"
        label_cur_vs_py = f"{MESES_ES[cur_month]} {str(cur_year)[-2:]} vs {MESES_ES[cur_month]} {str(py_year)[-2:]}"
        label_cur_vs_pm = f"{MESES_ES[cur_month]} {str(cur_year)[-2:]} vs {MESES_ES[pm_month]} {str(pm_year)[-2:]}"

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
            "👥 Seguimiento clientes",
            "📍 Seguimiento zona",
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
                    f"Ventas {MESES_ES[month_m2]} {str(year_m2)[-2:]}",
                    format_eur(sales_m2_total),
                    help=f"Ventas netas cerradas de {MESES_ES[month_m2]} {year_m2}"
                )
            with col6:
                st.metric(
                    f"Ventas {MESES_ES[month_m1]} {str(year_m1 - 1)[-2:]}",
                    format_eur(sales_m1_last_year_total),
                    help=f"Ventas netas cerradas de {MESES_ES[month_m1]} {year_m1 - 1}"
                )
            with col7:
                st.metric(
                    f"Ventas {MESES_ES[month_m1]} {str(year_m1)[-2:]}",
                    format_eur(sales_m1_total),
                    help=f"Ventas netas cerradas de {MESES_ES[month_m1]} {year_m1}"
                )
                st.markdown(
                    _format_growth_badge(label_m1_vs_m2, monthly_growth)
                    + _format_growth_badge(label_m1_vs_py, yearly_growth),
                    unsafe_allow_html=True,
                )

            col_mytd, col_mytd_py = st.columns(2)
            with col_mytd:
                st.metric(
                    f"Month_YTD ({MESES_ES[cur_month]} {str(cur_year)[-2:]})",
                    format_eur(month_ytd_total),
                    help=f"Ventas acumuladas al día de hoy ({today_day} de {MESES_ES[cur_month]} {cur_year})"
                )
                st.caption(f"Acumulado al día {today_day} de {MESES_ES[cur_month]} {cur_year}" if has_day_granularity else f"Ventas mes {MESES_ES[cur_month]} {cur_year}")
                st.markdown(
                    _format_growth_badge(label_cur_vs_py, growth_cur_vs_py)
                    + _format_growth_badge(label_cur_vs_pm, growth_cur_vs_pm),
                    unsafe_allow_html=True,
                )
            with col_mytd_py:
                st.metric(
                    f"Month_YTD_PY ({MESES_ES[cur_month]} {str(py_year)[-2:]})",
                    format_eur(month_ytd_py_total),
                    help=f"Ventas acumuladas hasta el mismo día ({today_day} de {MESES_ES[cur_month]} {py_year})"
                )
                st.caption(f"Acumulado al día {today_day} de {MESES_ES[cur_month]} {py_year}" if has_day_granularity else f"Ventas mes {MESES_ES[cur_month]} {py_year}")
            
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
                'SKU', 'Marca', 'Descripción', 'Stock Unidades', 'Pendiente Servir',
                'Demanda prevista meses de compra', 'Meses de Stock', 'PEDIDO', 'VALOR PEDIDO', 'MARGEN PEDIDO'
            ]
            cols_to_use = [c for c in purchase_columns if c in filtered_df.columns]
            purchase_display_df = filtered_df[cols_to_use].copy()
            if 'Stock Unidades' in purchase_display_df: purchase_display_df['Stock Unidades'] = purchase_display_df['Stock Unidades'].map(lambda v: f"{v:.0f}")
            if 'Pendiente Servir' in purchase_display_df: purchase_display_df['Pendiente Servir'] = purchase_display_df['Pendiente Servir'].map(lambda v: f"{v:.0f}")
            if 'Demanda prevista meses de compra' in purchase_display_df: purchase_display_df['Demanda prevista meses de compra'] = purchase_display_df['Demanda prevista meses de compra'].map(lambda v: f"{v:.0f}")
            if 'Meses de Stock' in purchase_display_df: purchase_display_df['Meses de Stock'] = purchase_display_df['Meses de Stock'].map(lambda v: f"{v:.1f}")
            if 'PEDIDO' in purchase_display_df: purchase_display_df['PEDIDO'] = purchase_display_df['PEDIDO'].map(lambda v: f"{v:.0f}")
            if 'VALOR PEDIDO' in purchase_display_df: purchase_display_df['VALOR PEDIDO'] = purchase_display_df['VALOR PEDIDO'].map(format_eur)
            if 'MARGEN PEDIDO' in purchase_display_df: purchase_display_df['MARGEN PEDIDO'] = purchase_display_df['MARGEN PEDIDO'].map(format_eur)
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
            from clientes import MESES, client_sku_drivers
            st.header("👥 Seguimiento clientes")
            f1, f2, f3, f4 = st.columns(4)
            with f1:
                zona_options = sorted([z for z in clientes_df['Zona'].dropna().unique() if str(z).strip() and str(z) != 'Sin Zona']) if 'Zona' in clientes_df.columns else []
                if not zona_options and 'Zona' in clientes_df.columns and 'Sin Zona' in clientes_df['Zona'].values:
                    zona_options = ['Sin Zona']
                selected_zona = st.multiselect("Filtrar por Zona", options=zona_options, default=[], key="cli_filter_zona")
            with f2:
                tendencia_options = sorted(clientes_df['Tendencia'].dropna().unique()) if 'Tendencia' in clientes_df.columns else []
                selected_tendencia = st.multiselect("Filtrar por Tendencia", options=tendencia_options, default=[])
            with f3:
                abc_options = sorted(clientes_df['ABC'].dropna().unique()) if 'ABC' in clientes_df.columns else []
                selected_abc = st.multiselect("Filtrar por ABC", options=abc_options, default=[])
            with f4:
                search_term = st.text_input("🔍 Buscar cliente (cód, nombre o zona)", "")
            clientes_display = clientes_df.copy()
            if selected_zona and 'Zona' in clientes_display: clientes_display = clientes_display[clientes_display.Zona.isin(selected_zona)]
            if selected_tendencia and 'Tendencia' in clientes_display: clientes_display = clientes_display[clientes_display.Tendencia.isin(selected_tendencia)]
            if selected_abc and 'ABC' in clientes_display: clientes_display = clientes_display[clientes_display.ABC.isin(selected_abc)]
            if search_term.strip():
                clientes_display = clientes_display[
                    clientes_display.Cod.astype(str).str.contains(search_term, case=False, na=False)
                    | clientes_display.Cliente.astype(str).str.contains(search_term, case=False, na=False)
                    | (clientes_display.Zona.astype(str).str.contains(search_term, case=False, na=False) if 'Zona' in clientes_display.columns else False)
                ]
            activos = int(clientes_df.L3M.gt(0).sum()) if 'L3M' in clientes_df.columns else 0
            inactivos = int(clientes_df.Tendencia.isin(['Inactivo', 'Sin compra reciente']).sum()) if 'Tendencia' in clientes_df.columns else 0
            ytd_total = clientes_df.YTD.sum() if 'YTD' in clientes_df.columns else 0
            ytd_py_total = clientes_df.YTD_PY.sum() if 'YTD_PY' in clientes_df.columns else 0
            k1,k2,k3,k4=st.columns(4)
            k1.metric("Clientes activos (L3M)", activos)
            k2.metric("Clientes sin compra / inactivos", inactivos)
            k3.metric("YTD Total", format_eur(ytd_total))
            k4.metric("Variación YTD vs PY", f"{(ytd_total-ytd_py_total)/ytd_py_total:+.1%}" if ytd_py_total else "N/A")
            if not clientes_df.empty:
                st.markdown("🏆 Top 5: " + " · ".join(f"**{r.Cliente}** ({format_eur(r.YTD)})" for _,r in clientes_df.nlargest(5,'YTD').iterrows()))
            priority = ['Cliente', 'Zona', 'Ranking YTD', 'ABC', 'Cod', 'YTD', 'YTD_PY', 'Var_YTD_%', 'Var_YTD_abs', 'L3M', 'L3M_PY', 'Var_L3M_YoY_%', 'Tendencia', 'Cambio Ranking', 'Ticket Medio', 'Recurrencia %', 'Cuota_%', 'Meses_sin_compra', 'Mes en curso (parcial)']
            ordered = [c for c in priority if c in clientes_display] + [c for c in clientes_display if c not in priority]
            config = {c: st.column_config.NumberColumn(c, format="€ %.2f") for c in clientes_display if c in {'YTD','YTD_PY','Var_YTD_abs','L3M','L3M_PY','Mes en curso (parcial)','Ticket Medio'} or c.startswith('Año ') or c in MESES}
            config['Cliente'] = st.column_config.TextColumn('Cliente', pinned=True)
            if 'Zona' in clientes_display: config['Zona'] = st.column_config.TextColumn('Zona')
            if 'Cuota_%' in clientes_display: config['Cuota_%'] = st.column_config.NumberColumn('Cuota_%', format="%.2f%%")
            if 'Var_YTD_%' in clientes_display: config['Var_YTD_%'] = st.column_config.NumberColumn('Var_YTD_%', format="%.1f%%")
            if 'Var_L3M_YoY_%' in clientes_display: config['Var_L3M_YoY_%'] = st.column_config.NumberColumn('Var_L3M_YoY_%', format="%.1f%%")
            if 'Recurrencia %' in clientes_display: config['Recurrencia %'] = st.column_config.NumberColumn('Recurrencia %', format="%.1f%%")
            if 'Meses_sin_compra' in clientes_display: config['Meses_sin_compra'] = st.column_config.NumberColumn('Meses_sin_compra', format="%.0f")
            if 'Ranking YTD' in clientes_display: config['Ranking YTD'] = st.column_config.NumberColumn('Ranking YTD', format="%.0f")
            if 'Ranking PY' in clientes_display: config['Ranking PY'] = st.column_config.NumberColumn('Ranking PY', format="%.0f")
            if 'Cambio Ranking' in clientes_display: config['Cambio Ranking'] = st.column_config.NumberColumn('Cambio Ranking', format="%+d")
            st.dataframe(clientes_display[ordered], use_container_width=True, height=700, column_config=config, hide_index=True)
            
            # Comparativa directa entre clientes seleccionados
            with st.expander("⚖️ Comparativa directa entre clientes", expanded=False):
                st.markdown("Selecciona dos o más clientes para comparar sus ventas, métricas y comportamiento mensual frente a frente.")
                compare_choices = clientes_df.Cod.tolist()
                selected_compare = st.multiselect(
                    "Seleccionar clientes para comparar",
                    options=compare_choices,
                    default=compare_choices[:2] if len(compare_choices) >= 2 else compare_choices,
                    format_func=lambda cod: f"{cod} — {clientes_df.loc[clientes_df.Cod==cod,'Cliente'].iloc[0]}"
                )
                if selected_compare:
                    compare_df = clientes_df[clientes_df.Cod.isin(selected_compare)].copy()
                    cols_compare = [
                        'Cliente', 'Zona', 'Ranking YTD', 'ABC', 'Cod', 'Tendencia',
                        'YTD', 'YTD_PY', 'Var_YTD_%', 'Var_YTD_abs',
                        'L3M', 'Ticket Medio', 'Recurrencia %', 'Meses_sin_compra'
                    ]
                    compare_display = compare_df[[c for c in cols_compare if c in compare_df]].copy()
                    comp_cfg = {
                        c: st.column_config.NumberColumn(c, format="€ %.2f")
                        for c in compare_display if c in {'YTD', 'YTD_PY', 'Var_YTD_abs', 'L3M', 'Ticket Medio'}
                    }
                    comp_cfg['Cliente'] = st.column_config.TextColumn('Cliente', pinned=True)
                    if 'Zona' in compare_display: comp_cfg['Zona'] = st.column_config.TextColumn('Zona')
                    if 'Var_YTD_%' in compare_display: comp_cfg['Var_YTD_%'] = st.column_config.NumberColumn('Var_YTD_%', format="%.1f%%")
                    if 'Recurrencia %' in compare_display: comp_cfg['Recurrencia %'] = st.column_config.NumberColumn('Recurrencia %', format="%.1f%%")
                    if 'Meses_sin_compra' in compare_display: comp_cfg['Meses_sin_compra'] = st.column_config.NumberColumn('Meses_sin_compra', format="%.0f")
                    if 'Ranking YTD' in compare_display: comp_cfg['Ranking YTD'] = st.column_config.NumberColumn('Ranking YTD', format="%.0f")
                    st.dataframe(compare_display, use_container_width=True, hide_index=True, column_config=comp_cfg)
                    
                    months_available = [m for m in MESES if m in compare_df.columns]
                    if months_available:
                        st.markdown("**Evolución mensual comparativa (Año en curso)**")
                        monthly_comp = pd.melt(
                            compare_df[['Cliente'] + months_available],
                            id_vars=['Cliente'],
                            value_vars=months_available,
                            var_name='Mes',
                            value_name='Importe'
                        )
                        fig_comp = px.line(
                            monthly_comp,
                            x='Mes',
                            y='Importe',
                            color='Cliente',
                            markers=True
                        )
                        fig_comp.update_layout(xaxis_title="Mes", yaxis_title="Importe Neto (€)", hovermode="x unified", height=400)
                        st.plotly_chart(fig_comp, use_container_width=True)
                else:
                    st.info("Selecciona al menos un cliente para comparar.")

            c1,c2=st.columns(2)
            with c1:
                chart = pd.melt(clientes_df.nlargest(15,'YTD')[['Cliente','YTD','YTD_PY']], id_vars='Cliente', var_name='Periodo', value_name='Importe')
                st.plotly_chart(px.bar(chart,x='Cliente',y='Importe',color='Periodo',barmode='group',color_discrete_map={'YTD':'#2ecc71','YTD_PY':'#95a5a6'}), use_container_width=True)
            with c2:
                st.plotly_chart(px.bar(clientes_df.nlargest(10,'YTD'),x='Cliente',y='YTD',color='YTD',color_continuous_scale='Greens'), use_container_width=True)
            with st.expander("🔎 Detalle SKU por cliente — ¿Qué productos suben/bajan?"):
                choices = clientes_df.Cod.tolist(); selected = st.multiselect("Seleccionar cliente(s) para analizar", choices, max_selections=5, format_func=lambda cod: f"{cod} — {clientes_df.loc[clientes_df.Cod==cod,'Cliente'].iloc[0]}")
                if selected:
                    up, down = client_sku_drivers(ventas_filtered, selected, getattr(manager,'today',date.today()), top=10); x,y=st.columns(2); x.dataframe(up,use_container_width=True,hide_index=True); y.dataframe(down,use_container_width=True,hide_index=True)

        with tab4:
            from clientes import MESES, zona_sku_drivers
            st.header("📍 Seguimiento zona")
            if zonas_df.empty:
                st.info("No hay información de zonas disponible en las ventas cargadas.")
            else:
                zf1, zf2, zf3 = st.columns(3)
                with zf1:
                    z_tendencia_options = sorted(zonas_df['Tendencia'].dropna().unique()) if 'Tendencia' in zonas_df.columns else []
                    selected_z_tendencia = st.multiselect("Filtrar por Tendencia de Zona", options=z_tendencia_options, default=[], key="z_multisel_tend")
                with zf2:
                    z_abc_options = sorted(zonas_df['ABC'].dropna().unique()) if 'ABC' in zonas_df.columns else []
                    selected_z_abc = st.multiselect("Filtrar por ABC de Zona", options=z_abc_options, default=[], key="z_multisel_abc")
                with zf3:
                    z_search_term = st.text_input("🔍 Buscar zona", "", key="z_search_txt")

                zonas_display = zonas_df.copy()
                if selected_z_tendencia and 'Tendencia' in zonas_display:
                    zonas_display = zonas_display[zonas_display.Tendencia.isin(selected_z_tendencia)]
                if selected_z_abc and 'ABC' in zonas_display:
                    zonas_display = zonas_display[zonas_display.ABC.isin(selected_z_abc)]
                if z_search_term.strip():
                    zonas_display = zonas_display[zonas_display.Zona.astype(str).str.contains(z_search_term, case=False, na=False)]

                z_activas = int(zonas_df.L3M.gt(0).sum()) if 'L3M' in zonas_df.columns else len(zonas_df)
                z_total_clientes = int(zonas_df['Clientes Totales'].sum()) if 'Clientes Totales' in zonas_df.columns else 0
                z_ytd_total = zonas_df.YTD.sum() if 'YTD' in zonas_df.columns else 0
                z_ytd_py_total = zonas_df.YTD_PY.sum() if 'YTD_PY' in zonas_df.columns else 0

                zk1, zk2, zk3, zk4 = st.columns(4)
                zk1.metric("Zonas activas (L3M)", z_activas)
                zk2.metric("Clientes totales en zonas", z_total_clientes)
                zk3.metric("YTD Total", format_eur(z_ytd_total))
                zk4.metric("Variación YTD vs PY", f"{(z_ytd_total - z_ytd_py_total) / z_ytd_py_total:+.1%}" if z_ytd_py_total else "N/A")

                st.markdown("🏆 Top Zonas: " + " · ".join(f"**{r.Zona}** ({format_eur(r.YTD)})" for _, r in zonas_df.nlargest(5, 'YTD').iterrows()))

                priority_z = [
                    'Zona', 'Ranking YTD', 'ABC', 'Clientes Totales', 'Clientes Activos',
                    'YTD', 'YTD_PY', 'Var_YTD_%', 'Var_YTD_abs', 'L3M', 'L3M_PY',
                    'Var_L3M_YoY_%', 'Tendencia', 'Cambio Ranking', 'Ticket Medio',
                    'Venta Media por Cliente', 'Recurrencia %', 'Cuota_%',
                    'Meses_sin_compra', 'Mes en curso (parcial)'
                ]
                ordered_z = [c for c in priority_z if c in zonas_display] + [c for c in zonas_display if c not in priority_z]
                config_z = {
                    c: st.column_config.NumberColumn(c, format="€ %.2f")
                    for c in zonas_display
                    if c in {'YTD', 'YTD_PY', 'Var_YTD_abs', 'L3M', 'L3M_PY', 'Mes en curso (parcial)', 'Ticket Medio', 'Venta Media por Cliente'}
                    or c.startswith('Año ') or c in MESES
                }
                config_z['Zona'] = st.column_config.TextColumn('Zona', pinned=True)
                if 'Cuota_%' in zonas_display: config_z['Cuota_%'] = st.column_config.NumberColumn('Cuota_%', format="%.2f%%")
                if 'Var_YTD_%' in zonas_display: config_z['Var_YTD_%'] = st.column_config.NumberColumn('Var_YTD_%', format="%.1f%%")
                if 'Var_L3M_YoY_%' in zonas_display: config_z['Var_L3M_YoY_%'] = st.column_config.NumberColumn('Var_L3M_YoY_%', format="%.1f%%")
                if 'Recurrencia %' in zonas_display: config_z['Recurrencia %'] = st.column_config.NumberColumn('Recurrencia %', format="%.1f%%")
                if 'Meses_sin_compra' in zonas_display: config_z['Meses_sin_compra'] = st.column_config.NumberColumn('Meses_sin_compra', format="%.0f")
                if 'Ranking YTD' in zonas_display: config_z['Ranking YTD'] = st.column_config.NumberColumn('Ranking YTD', format="%.0f")
                if 'Ranking PY' in zonas_display: config_z['Ranking PY'] = st.column_config.NumberColumn('Ranking PY', format="%.0f")
                if 'Cambio Ranking' in zonas_display: config_z['Cambio Ranking'] = st.column_config.NumberColumn('Cambio Ranking', format="%+d")
                if 'Clientes Totales' in zonas_display: config_z['Clientes Totales'] = st.column_config.NumberColumn('Clientes Totales', format="%.0f")
                if 'Clientes Activos' in zonas_display: config_z['Clientes Activos'] = st.column_config.NumberColumn('Clientes Activos', format="%.0f")

                st.dataframe(zonas_display[ordered_z], use_container_width=True, height=600, column_config=config_z, hide_index=True)

                # Comparativa directa entre zonas
                with st.expander("⚖️ Comparativa directa entre zonas", expanded=False):
                    st.markdown("Selecciona dos o más zonas para comparar sus ventas, métricas y comportamiento mensual frente a frente.")
                    z_compare_choices = zonas_df.Zona.tolist()
                    selected_z_compare = st.multiselect(
                        "Seleccionar zonas para comparar",
                        options=z_compare_choices,
                        default=z_compare_choices[:2] if len(z_compare_choices) >= 2 else z_compare_choices,
                        key="z_compare_multisel"
                    )
                    if selected_z_compare:
                        z_comp_df = zonas_df[zonas_df.Zona.isin(selected_z_compare)].copy()
                        cols_z_comp = [
                            'Zona', 'Ranking YTD', 'ABC', 'Clientes Totales', 'Clientes Activos',
                            'Tendencia', 'YTD', 'YTD_PY', 'Var_YTD_%', 'Var_YTD_abs',
                            'L3M', 'Ticket Medio', 'Venta Media por Cliente', 'Recurrencia %'
                        ]
                        z_comp_display = z_comp_df[[c for c in cols_z_comp if c in z_comp_df]].copy()
                        z_comp_cfg = {
                            c: st.column_config.NumberColumn(c, format="€ %.2f")
                            for c in z_comp_display
                            if c in {'YTD', 'YTD_PY', 'Var_YTD_abs', 'L3M', 'Ticket Medio', 'Venta Media por Cliente'}
                        }
                        z_comp_cfg['Zona'] = st.column_config.TextColumn('Zona', pinned=True)
                        if 'Var_YTD_%' in z_comp_display: z_comp_cfg['Var_YTD_%'] = st.column_config.NumberColumn('Var_YTD_%', format="%.1f%%")
                        if 'Recurrencia %' in z_comp_display: z_comp_cfg['Recurrencia %'] = st.column_config.NumberColumn('Recurrencia %', format="%.1f%%")
                        if 'Ranking YTD' in z_comp_display: z_comp_cfg['Ranking YTD'] = st.column_config.NumberColumn('Ranking YTD', format="%.0f")
                        if 'Clientes Totales' in z_comp_display: z_comp_cfg['Clientes Totales'] = st.column_config.NumberColumn('Clientes Totales', format="%.0f")
                        if 'Clientes Activos' in z_comp_display: z_comp_cfg['Clientes Activos'] = st.column_config.NumberColumn('Clientes Activos', format="%.0f")
                        st.dataframe(z_comp_display, use_container_width=True, hide_index=True, column_config=z_comp_cfg)

                        z_months_avail = [m for m in MESES if m in z_comp_df.columns]
                        if z_months_avail:
                            st.markdown("**Evolución mensual comparativa por zona (Año en curso)**")
                            z_monthly_comp = pd.melt(
                                z_comp_df[['Zona'] + z_months_avail],
                                id_vars=['Zona'],
                                value_vars=z_months_avail,
                                var_name='Mes',
                                value_name='Importe'
                            )
                            fig_z_comp = px.line(
                                z_monthly_comp,
                                x='Mes',
                                y='Importe',
                                color='Zona',
                                markers=True
                            )
                            fig_z_comp.update_layout(xaxis_title="Mes", yaxis_title="Importe Neto (€)", hovermode="x unified", height=400)
                            st.plotly_chart(fig_z_comp, use_container_width=True)
                    else:
                        st.info("Selecciona al menos una zona para comparar.")

                # Gráficos de Zonas
                zc1, zc2 = st.columns(2)
                with zc1:
                    top_z_count = min(15, len(zonas_df))
                    z_chart = pd.melt(
                        zonas_df.nlargest(top_z_count, 'YTD')[['Zona', 'YTD', 'YTD_PY']],
                        id_vars='Zona',
                        var_name='Periodo',
                        value_name='Importe'
                    )
                    st.plotly_chart(
                        px.bar(
                            z_chart,
                            x='Zona',
                            y='Importe',
                            color='Periodo',
                            barmode='group',
                            color_discrete_map={'YTD': '#2ecc71', 'YTD_PY': '#95a5a6'}
                        ),
                        use_container_width=True
                    )
                with zc2:
                    st.plotly_chart(
                        px.pie(
                            zonas_df.nlargest(10, 'YTD'),
                            names='Zona',
                            values='YTD',
                            hole=0.4
                        ),
                        use_container_width=True
                    )

                # Desglose de Clientes por Zona
                with st.expander("👥 Clientes por zona — ¿Qué clientes componen cada zona?"):
                    zona_choice_list = zonas_df.Zona.tolist()
                    chosen_zona = st.selectbox("Seleccionar zona para ver sus clientes", options=zona_choice_list, key="chosen_zona_clients")
                    if chosen_zona and not clientes_df.empty:
                        zone_clients = clientes_df[clientes_df.Zona == chosen_zona].copy()
                        st.write(f"**Total clientes en {chosen_zona}:** {len(zone_clients)}")
                        z_cli_priority = ['Cliente', 'Ranking YTD', 'ABC', 'Cod', 'YTD', 'YTD_PY', 'Var_YTD_%', 'L3M', 'Tendencia', 'Ticket Medio', 'Recurrencia %']
                        z_cli_cols = [c for c in z_cli_priority if c in zone_clients]
                        z_cli_cfg = {
                            c: st.column_config.NumberColumn(c, format="€ %.2f")
                            for c in z_cli_cols if c in {'YTD', 'YTD_PY', 'L3M', 'Ticket Medio'}
                        }
                        z_cli_cfg['Cliente'] = st.column_config.TextColumn('Cliente', pinned=True)
                        if 'Var_YTD_%' in z_cli_cols: z_cli_cfg['Var_YTD_%'] = st.column_config.NumberColumn('Var_YTD_%', format="%.1f%%")
                        if 'Recurrencia %' in z_cli_cols: z_cli_cfg['Recurrencia %'] = st.column_config.NumberColumn('Recurrencia %', format="%.1f%%")
                        if 'Ranking YTD' in z_cli_cols: z_cli_cfg['Ranking YTD'] = st.column_config.NumberColumn('Ranking YTD', format="%.0f")
                        st.dataframe(zone_clients[z_cli_cols], use_container_width=True, hide_index=True, column_config=z_cli_cfg)

                # Detalle SKU por zona
                with st.expander("🔎 Detalle SKU por zona — ¿Qué productos suben/bajan?"):
                    selected_z_skus = st.multiselect("Seleccionar zona(s) para analizar SKUs", zona_choice_list, max_selections=5, key="z_skus_multisel")
                    if selected_z_skus:
                        up_z, down_z = zona_sku_drivers(ventas_filtered, selected_z_skus, as_of, top=10)
                        zx, zy = st.columns(2)
                        with zx:
                            st.markdown("**Top SKUs con mayor incremento YTD**")
                            st.dataframe(up_z, use_container_width=True, hide_index=True)
                        with zy:
                            st.markdown("**Top SKUs con mayor caída YTD**")
                            st.dataframe(down_z, use_container_width=True, hide_index=True)

        with tab5:
            from inventory_manager import get_export_compras_columns
            st.header("📁 Export Results")
            st.write("Descarga los resultados del análisis en Excel")
            available_cols = [c for c in get_export_compras_columns(manager.current_year) if c in compras_filtered]
            compras_export = compras_filtered[available_cols].copy()
            clean = io.BytesIO()
            with pd.ExcelWriter(clean, engine='openpyxl') as writer:
                compras_export.to_excel(writer, sheet_name='COMPRAS', index=False)
                clientes_df.to_excel(writer, sheet_name='CLIENTES', index=False)
                zonas_df.to_excel(writer, sheet_name='ZONAS', index=False)
            st.download_button("📥 Descargar Pedido (columnas esenciales)", clean.getvalue(), f"pedido_{pd.Timestamp.now():%Y%m%d_%H%M%S}.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", type="primary")
            st.caption(f"Incluye {len(available_cols)} columnas: SKU, Marca, histórico ventas, stock, pedido, motivo y alertas.")
            st.divider(); st.subheader("Export completo (debug / auditoría)")
            full = io.BytesIO()
            with pd.ExcelWriter(full, engine='openpyxl') as writer:
                compras_filtered.to_excel(writer, sheet_name='COMPRAS_FULL', index=False)
                clientes_df.to_excel(writer, sheet_name='CLIENTES', index=False)
                zonas_df.to_excel(writer, sheet_name='ZONAS', index=False)
            st.download_button("📥 Descargar Export completo (todas las columnas)", full.getvalue(), f"full_export_{pd.Timestamp.now():%Y%m%d_%H%M%S}.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            col1,col2,col3=st.columns(3)
            with col1: st.download_button("📄 Solo Pedido (Excel)", dataframe_to_excel_bytes(compras_export,'COMPRAS'), f"compras_{pd.Timestamp.now():%Y%m%d}.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            with col2: st.download_button("📄 Solo Clientes (Excel)", dataframe_to_excel_bytes(clientes_df,'CLIENTES'), f"clientes_{pd.Timestamp.now():%Y%m%d}.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            with col3: st.download_button("📄 Solo Zonas (Excel)", dataframe_to_excel_bytes(zonas_df,'ZONAS'), f"zonas_{pd.Timestamp.now():%Y%m%d}.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


if __name__ == "__main__":
    main()
