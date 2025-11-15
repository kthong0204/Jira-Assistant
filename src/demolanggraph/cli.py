from __future__ import annotations

from pathlib import Path
from pprint import pformat
from typing import Any, Optional

import typer
from langchain_core.documents import Document

from . import ingest as ingest_module
from .embeddings import get_embeddings
from .graph import build_rag_graph
from .graph_advanced import build_advanced_graph
from .jira_sync import JiraSyncError, sync_single_issue
from .llm import get_llm
from .settings import Settings, with_overrides
from .vectorstore import build_retriever

cli_app = typer.Typer(help="LangGraph based RAG demo")


def _resolve_settings(
    data_dir: Optional[Path],
    vector_store_dir: Optional[Path],
) -> Settings:
    return with_overrides(
        data_dir=data_dir.resolve() if data_dir else None,
        vector_store_dir=vector_store_dir.resolve() if vector_store_dir else None,
    )


def _truncate_text(text: str, limit: int = 160) -> str:
    stripped = text.strip()
    return stripped if len(stripped) <= limit else f"{stripped[: limit - 3]}..."


def _summarize_document(doc: Document) -> str:
    source = doc.metadata.get("source", "unknown")
    preview = _truncate_text(doc.page_content.replace("\n", " "))
    return f"{source}: {preview}"


def _summarize_value(value):
    if isinstance(value, Document):
        return _summarize_document(value)
    if isinstance(value, list):
        if not value:
            return "[]"
        first = value[0]
        if isinstance(first, Document):
            summary = _summarize_document(first)
            extra = len(value) - 1
            suffix = f" (+{extra} more)" if extra > 0 else ""
            return f"[Documents {summary}{suffix}]"
        if len(value) > 5:
            return f"[list len={len(value)}]"
        return [_summarize_value(item) for item in value]
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        return _truncate_text(value)
    if isinstance(value, dict):
        return {k: _summarize_value(v) for k, v in value.items()}
    return repr(value)


def _format_trace_payload(payload) -> str:
    summary = (
        {k: _summarize_value(v) for k, v in payload.items()}
        if isinstance(payload, dict)
        else _summarize_value(payload)
    )
    return pformat(summary, width=80, compact=True)


def _is_root_node(node_id) -> bool:
    if node_id == "__root__":
        return True
    if isinstance(node_id, tuple):
        return any(part == "__root__" for part in node_id)
    return False


def _format_node_label(node_id) -> str:
    if _is_root_node(node_id):
        return "__root__"
    if isinstance(node_id, tuple):
        return " -> ".join(str(part) for part in node_id if part != "__root__")
    return str(node_id)


@cli_app.command()
def ingest(
    data_dir: Optional[Path] = typer.Option(
        None, help="Directory that contains raw documents."
    ),
    vector_store_dir: Optional[Path] = typer.Option(
        None, help="Where to persist the Chroma index."
    ),
):
    """Chunk documents and build the Chroma vector store."""

    settings = _resolve_settings(data_dir, vector_store_dir)
    typer.echo(ingest_module.ingest(settings))


@cli_app.command()
def sync_jira(
    issue_key: str = typer.Argument(..., help="Jira issue key to fetch (e.g., DEV-123)."),
):
    """Fetch a single Jira issue and append it to the local dataset."""

    settings = _resolve_settings(None, None)
    try:
        dataset_path, _ = sync_single_issue(issue_key, settings)
    except JiraSyncError as exc:
        typer.secho(f"Jira sync failed: {exc}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1)

    typer.echo(f"Synced {issue_key} into {dataset_path}")


@cli_app.command()
def chat(
    question: Optional[str] = typer.Option(
        None, "--question", "-q", help="Ask a single question instead of interactive chat."
    ),
    show_sources: bool = typer.Option(
        True, help="Print metadata for the retrieved documents."
    ),
    vector_store_dir: Optional[Path] = typer.Option(
        None, help="Override the vector store directory that will be used."
    ),
    graph_type: str = typer.Option(
        "basic",
        "--graph",
        "-g",
        help="Which LangGraph pipeline to run: 'basic' or 'advanced'.",
    ),
    inspect: bool = typer.Option(
        False,
        "--inspect/--no-inspect",
        help="Stream intermediate LangGraph states before printing the answer.",
    ),
):
    """Chat with the LangGraph RAG assistant."""

    settings = _resolve_settings(None, vector_store_dir)
    embeddings = get_embeddings(settings)
    vector_store = build_retriever(embeddings, settings)
    llm = get_llm(settings)
    builders = {
        "basic": build_rag_graph,
        "advanced": build_advanced_graph,
    }
    graph_key = graph_type.lower()
    if graph_key not in builders:
        raise typer.BadParameter(
            "Graph must be one of: basic, advanced.",
            param_hint="--graph",
        )
    graph = builders[graph_key](vector_store, llm, settings)

    def _execute(query: str):
        if not inspect:
            return graph.invoke({"question": query})

        typer.echo("\nTrace:\n------")
        final_state = None
        state_snapshot: dict[str, Any] = {}
        for event in graph.stream({"question": query}):
            for node_id, payload in event.items():
                label = _format_node_label(node_id) or "__root__"
                typer.echo(f"[{label}] {_format_trace_payload(payload)}")
                if isinstance(payload, dict):
                    state_snapshot.update(payload)
                else:
                    state_snapshot[label] = payload
                if _is_root_node(node_id):
                    final_state = payload if isinstance(payload, dict) else state_snapshot
        typer.echo("------")
        if final_state is None:
            return state_snapshot
        return final_state

    def _answer(query: str):
        result = _execute(query)
        typer.echo("\nAnswer:\n--------")
        typer.echo(result.get("answer", "No answer produced."))
        if show_sources:
            docs = result.get("documents") or []
            scores = result.get("scores") or []
            if docs:
                typer.echo("\nSources:")
            for idx, (doc, score) in enumerate(zip(docs, scores), start=1):
                source = doc.metadata.get("source", "unknown")
                score_str = f"{score:.4f}" if isinstance(score, (int, float)) else "n/a"
                typer.echo(f"  {idx}. {source} (distance={score_str})")

    if question:
        _answer(question)
        return

    typer.echo("Enter questions (type 'exit' to quit):")
    while True:
        query = typer.prompt(">> ").strip()
        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            break
        _answer(query)
        typer.echo("\n")


def main():
    """Entry point for `python -m demolanggraph.cli`."""

    cli_app()


if __name__ == "__main__":
    main()
