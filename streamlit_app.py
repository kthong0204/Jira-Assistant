"""Streamlit UI for DemoLangGraph."""

from __future__ import annotations

import json
from itertools import zip_longest
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import sys
import os

import streamlit as st
from langchain_core.documents import Document

# Ensure local package imports work even without `pip install -e .`
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from demolanggraph import ingest as ingest_module
from demolanggraph.embeddings import get_embeddings
from demolanggraph.graph import build_rag_graph
from demolanggraph.graph_advanced import build_advanced_graph
from demolanggraph.jira_sync import JiraSyncError, sync_single_issue
from demolanggraph.llm import get_llm
from demolanggraph.settings import Settings, with_overrides
from demolanggraph.vectorstore import build_retriever


# --------------------------------------------------------------------------- helpers
def _default_settings() -> Settings:
    cfg = Settings()
    cfg.ensure_directories()
    return cfg


def _load_settings(data_dir: str, vector_dir: str) -> Settings:
    overrides = {
        "data_dir": Path(data_dir).expanduser().resolve(),
        "vector_store_dir": Path(vector_dir).expanduser().resolve(),
    }
    cfg = with_overrides(**overrides)
    cfg.ensure_directories()
    return cfg


@st.cache_resource(show_spinner=False)
def _build_graph(
    graph_type: str,
    data_dir: str,
    vector_dir: str,
) -> Tuple[Any, Settings]:
    cfg = _load_settings(data_dir, vector_dir)
    embeddings = get_embeddings(cfg)
    retriever = build_retriever(embeddings, cfg)
    llm = get_llm(cfg)
    builder = build_advanced_graph if graph_type.lower() == "advanced" else build_rag_graph
    graph = builder(retriever, llm, cfg)
    return graph, cfg


def _format_node_id(node_id: Any) -> str:
    if node_id == "__root__":
        return "__root__"
    if isinstance(node_id, tuple):
        parts = [str(part) for part in node_id if part != "__root__"]
        return " -> ".join(parts) or "__root__"
    return str(node_id)


def _summarize_value(value: Any) -> Any:
    if isinstance(value, Document):
        source = value.metadata.get("source", "unknown")
        preview = value.page_content.strip().replace("\n", " ")
        if len(preview) > 110:
            preview = preview[:107] + "..."
        return f"{source}: {preview}"
    if isinstance(value, list):
        if not value:
            return "[]"
        if isinstance(value[0], Document):
            extra = len(value) - 1
            suffix = f" (+{extra} more)" if extra > 0 else ""
            return f"[Document { _summarize_value(value[0]) }{suffix}]"
        if len(value) > 5:
            return f"[list len={len(value)}]"
        return [_summarize_value(item) for item in value]
    if isinstance(value, dict):
        return {k: _summarize_value(v) for k, v in value.items()}
    if isinstance(value, str):
        stripped = value.strip()
        return stripped if len(stripped) <= 120 else stripped[:117] + "..."
    return value


def _stream_graph(graph, question: str) -> Tuple[Dict[str, Any], List[str]]:
    trace: List[str] = []
    snapshot: Dict[str, Any] = {}
    for event in graph.stream({"question": question}):
        for node_id, payload in event.items():
            label = _format_node_id(node_id)
            trace.append(f"[{label}] {_summarize_value(payload)}")
            if isinstance(payload, dict):
                snapshot.update(payload)
            else:
                snapshot[label] = payload
    return snapshot, trace


def _render_sources(documents: Sequence[Any], scores: Sequence[Any]) -> None:
    if not documents:
        return
    rows = []
    for idx, (doc, score) in enumerate(zip_longest(documents, scores), start=1):
        if isinstance(doc, Document):
            source = doc.metadata.get("source", "unknown")
            text = doc.page_content
        elif isinstance(doc, dict):
            source = doc.get("source", "unknown")
            text = doc.get("page_content") or doc.get("text") or str(doc)
        else:
            source = "unknown"
            text = str(doc)
        rows.append(
            {
                "STT": idx,
                "Source": source,
                "Score/dist": f"{score:.4f}" if isinstance(score, (int, float)) else "n/a",
                "Preview": (text or "").replace("\n", " ")[:200],
            }
        )
    st.dataframe(rows, width="stretch")


def _render_history(history: List[Tuple[str, str]]) -> None:
    for role, message in history:
        with st.chat_message(role):
            st.markdown(message)


# --------------------------------------------------------------------------- AUTO-INGEST
def auto_ingest_if_needed(cfg: Settings):
    """Tự động ingest nếu vectorstore chưa tồn tại."""
    vector_dir = cfg.vector_store_dir

    # Nếu thư mục rỗng → chưa có vectorstore
    has_vectorstore = any(vector_dir.glob("**/*"))

    if not has_vectorstore:
        st.warning("⚠️ Vector store chưa được tạo — tiến hành ingest lần đầu...")
        with st.spinner("Đang ingest dữ liệu..."):
            msg = ingest_module.ingest(cfg)
        st.success(f"✅ Tạo vector store xong: {msg}")

        _build_graph.clear()
        st.experimental_rerun()


# --------------------------------------------------------------------------- tabs
def render_chat_tab(graph, cfg: Settings, inspect: bool) -> None:
    history = st.session_state.setdefault("chat_history", [])
    _render_history(history)

    question = st.chat_input("Đặt câu hỏi về tài liệu đã ingest...")
    if not question:
        return

    with st.chat_message("user"):
        st.markdown(question)
    history.append(("user", question))

    with st.spinner("LangGraph đang xử lý..."):
        if inspect:
            result, trace = _stream_graph(graph, question)
        else:
            result = graph.invoke({"question": question})
            trace = []

    answer = result.get("answer") or result.get("fallback_message") or "Không có câu trả lời."
    with st.chat_message("assistant"):
        st.markdown(answer)
    history.append(("assistant", answer))

    documents = result.get("documents") or []
    scores = result.get("scores") or []
    _render_sources(documents, scores)

    if trace:
        with st.expander("Trace LangGraph", expanded=False):
            st.code("\n".join(trace), language="text")


def render_ingest_tab(cfg: Settings) -> None:
    st.subheader("Ingest tài liệu")
    st.caption(f"Thư mục dữ liệu hiện tại: `{cfg.data_dir}`")

    uploads = st.file_uploader(
        "Tải lên tệp (.txt/.md)",
        type=["txt", "md"],
        accept_multiple_files=True,
    )
    if uploads:
        for uploaded in uploads:
            target = cfg.data_dir / uploaded.name
            target.write_bytes(uploaded.getbuffer())
        st.success(f"Đã lưu {len(uploads)} tệp vào {cfg.data_dir}")

    if st.button("Chạy ingest"):
        with st.spinner("Chunk + build Chroma..."):
            message = ingest_module.ingest(cfg)
        st.success(message)
        _build_graph.clear()


def render_jira_tab(cfg: Settings) -> None:
    st.subheader("Sync Jira dataset")
    with st.form("jira_sync_form"):
        issue_key = st.text_input("Issue key", placeholder="DEV-123")
        submitted = st.form_submit_button("Sync ngay")
    if not submitted or not issue_key.strip():
        if submitted:
            st.warning("Nhập issue key trước khi sync.")
        return
    try:
        dataset_path, payload = sync_single_issue(issue_key.strip(), cfg)
    except JiraSyncError as exc:
        st.error(f"Jira sync failed: {exc}")
        return
    st.success(f"Đã sync {payload.get('key') or issue_key} vào {dataset_path}")
    st.code(json.dumps(payload, indent=2, ensure_ascii=False), language="json")


# --------------------------------------------------------------------------- main
def main() -> None:
    st.set_page_config(page_title="DemoLangGraph UI", page_icon="🤖", layout="wide")
    st.title("DemoLangGraph · Streamlit UI")

    defaults = _default_settings()

    with st.sidebar:
        st.subheader("Cấu hình")
        data_dir = st.text_input("Data directory", value=str(defaults.data_dir))
        vector_dir = st.text_input("Vector store dir", value=str(defaults.vector_store_dir))
        graph_choice = st.selectbox("Workflow", ["basic", "advanced"])
        inspect = st.checkbox("Hiển thị trace LangGraph", value=False)
        if st.button("Reload graph"):
            _build_graph.clear()
        st.caption("Các biến khác (LLM, embeddings, Jira, ...) lấy từ `.env` giống CLI `demo-rag`.")

    # Build graph lần đầu
    graph, cfg = _build_graph(graph_choice, data_dir, vector_dir)

    # ⛔ THÊM DÒNG NÀY: auto-ingest nếu vectorstore chưa có
    auto_ingest_if_needed(cfg)

    chat_tab, ingest_tab, jira_tab = st.tabs(["Chat", "Ingest", "Jira Sync"])
    with chat_tab:
        render_chat_tab(graph, cfg, inspect)
    with ingest_tab:
        render_ingest_tab(cfg)
    with jira_tab:
        render_jira_tab(cfg)


if __name__ == "__main__":
    main()
