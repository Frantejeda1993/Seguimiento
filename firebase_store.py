"""Persistencia de snapshots de cargas en Firebase Firestore."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import firebase_admin
import pandas as pd
import streamlit as st
from firebase_admin import credentials, firestore

COLLECTION_NAME = "inventory_uploads"


def init_firebase() -> firestore.Client:
    """Inicializa Firebase solo una vez y retorna el cliente de Firestore."""
    if not firebase_admin._apps:
        service_account_info = st.secrets["FIREBASE_SERVICE_ACCOUNT"]
        cred = credentials.Certificate(dict(service_account_info))
        firebase_admin.initialize_app(cred)

    return firestore.client()


def _serialize_records(value: Any) -> list[dict[str, Any]]:
    """Convierte DataFrame o listas de dicts en un formato serializable para Firestore."""
    if value is None:
        return []
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, list):
        return value
    return []


def save_upload_snapshot(payload: dict) -> str:
    """Guarda un snapshot de la carga actual y devuelve el id del documento creado."""
    db = init_firebase()
    now = datetime.now()

    document = {
        "created_at": firestore.SERVER_TIMESTAMP,
        "created_date": now.strftime("%Y-%m-%d"),
        "created_time": now.strftime("%H:%M:%S"),
        "source": payload.get("source", "upload"),
        "params": {
            "meses_compras": payload.get("meses_compras"),
            "contemplar_sobre_stock": payload.get("contemplar_sobre_stock", False),
        },
        "stock_records": _serialize_records(payload.get("stock_df") or payload.get("stock_records")),
        "ventas_records": _serialize_records(payload.get("ventas_df") or payload.get("ventas_records")),
        "recepciones_records": _serialize_records(
            payload.get("recepciones_df") or payload.get("recepciones_records")
        ),
    }

    doc_ref = db.collection(COLLECTION_NAME).document()
    doc_ref.set(document)
    return doc_ref.id


def list_upload_dates(limit: int = 100) -> list[dict]:
    """Lista snapshots recientes con metadatos de fecha y hora."""
    db = init_firebase()
    query = (
        db.collection(COLLECTION_NAME)
        .order_by("created_at", direction=firestore.Query.DESCENDING)
        .limit(limit)
    )

    uploads: list[dict[str, Any]] = []
    for doc in query.stream():
        data = doc.to_dict() or {}
        uploads.append(
            {
                "id": doc.id,
                "created_at": data.get("created_at"),
                "created_date": data.get("created_date"),
                "created_time": data.get("created_time"),
                "source": data.get("source"),
                "params": data.get("params", {}),
            }
        )

    return uploads


def get_upload_by_id(doc_id: str) -> dict | None:
    """Recupera un snapshot por id y rehidrata los records en DataFrames."""
    db = init_firebase()
    snapshot = db.collection(COLLECTION_NAME).document(doc_id).get()

    if not snapshot.exists:
        return None

    data = snapshot.to_dict() or {}
    stock_records = data.get("stock_records", [])
    ventas_records = data.get("ventas_records", [])
    recepciones_records = data.get("recepciones_records", [])

    data["id"] = snapshot.id
    data["stock_df"] = pd.DataFrame(stock_records)
    data["ventas_df"] = pd.DataFrame(ventas_records)
    data["recepciones_df"] = pd.DataFrame(recepciones_records)

    return data
