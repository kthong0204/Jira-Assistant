from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Literal, Sequence, TypedDict

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph

from .jira_sync import JiraSyncError, sync_single_issue


class AdvancedState(TypedDict, total=False):
    """State shared across the multi-intent graph."""

    question: str
    intent: Literal["DOCS", "JIRA", "GIT", "UNKNOWN"]
    context: str
    documents: List[Document]
    scores: List[float]
    jira_data: List[Dict[str, Any]]
    git_data: str
    fallback_message: str
    tool_error: str
    answer: str


PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Bạn là trợ lý nội bộ. Chỉ sử dụng thông tin trong CONTEXT để trả lời. "
            "Nếu không tìm thấy thông tin phù hợp, hãy nói rõ ràng rằng bạn không biết.",
        ),
        (
            "human",
            "Question: {question}\n\n"
            "CONTEXT:\n{context}\n\n"
            "Nếu context trống, hãy xin lỗi và đề nghị cung cấp thêm dữ liệu.",
        ),
    ]
)


DOC_KEYWORDS = ("doc", "document", "docs", "hướng dẫn", "readme")
JIRA_KEYWORDS = ("jira", "ticket", "issue", "story")
GIT_KEYWORDS = ("git", "commit", "branch", "diff", "pull request")
COMMIT_HASH_PATTERN = re.compile(r"\b[0-9a-f]{7,40}\b", re.IGNORECASE)
ISSUE_KEY_PATTERN = re.compile(r"\b[A-Z][A-Z0-9]+-\d+\b", re.IGNORECASE)


def _join_docs(documents: Sequence[Document]) -> str:
    return "\n\n".join(f"[{idx + 1}] {doc.page_content}" for idx, doc in enumerate(documents))


def _normalize_question(question: str) -> str:
    return question.strip().lower()


def _classify_intent(question: str) -> Literal["DOCS", "JIRA", "GIT", "UNKNOWN"]:
    normalized = _normalize_question(question)
    if not normalized:
        return "UNKNOWN"

    def _matches(keywords: Sequence[str]) -> bool:
        return any(keyword in normalized for keyword in keywords)

    if _matches(JIRA_KEYWORDS):
        return "JIRA"
    if _matches(GIT_KEYWORDS):
        return "GIT"
    if _matches(DOC_KEYWORDS) or len(normalized.split()) > 3:
        return "DOCS"
    return "UNKNOWN"


def _load_jira_knowledge(dataset_path: Path) -> List[Dict[str, Any]]:
    jira_path = dataset_path
    if jira_path.is_dir():
        jira_path = jira_path / "jira_samples.json"
    if not jira_path.exists():
        return []

    try:
        payload = json.loads(jira_path.read_text(encoding="utf-8"))
    except Exception:
        return []

    if isinstance(payload, list):
        return [issue for issue in payload if isinstance(issue, dict)]

    if isinstance(payload, dict):
        tickets = payload.get("tickets")
        if isinstance(tickets, dict):
            return [issue for issue in tickets.values() if isinstance(issue, dict)]
        if isinstance(tickets, list):
            return [issue for issue in tickets if isinstance(issue, dict)]
        # fallback: treat all dict values as candidate issues
        return [issue for issue in payload.values() if isinstance(issue, dict)]

    return []


def _search_jira_data(question: str, issues: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not issues:
        return []
    normalized = _normalize_question(question)
    tokens = [tok for tok in normalized.split() if tok]
    matches: List[Dict[str, Any]] = []
    for issue in issues:
        haystack = " ".join(
            str(issue.get(field, "")).lower()
            for field in ("key", "title", "summary", "description")
        )
        if any(token in haystack for token in tokens):
            matches.append(issue)
        if len(matches) >= 3:
            break
    return matches or list(issues[:1])


def _extract_issue_keys(question: str) -> List[str]:
    if not question:
        return []
    return [match.upper() for match in ISSUE_KEY_PATTERN.findall(question)]


def _filter_issues_by_keys(
    issues: Sequence[Dict[str, Any]], keys: Sequence[str]
) -> List[Dict[str, Any]]:
    if not issues or not keys:
        return []
    normalized = {key.upper() for key in keys}
    matches: List[Dict[str, Any]] = []
    for issue in issues:
        key = (issue.get("key") or issue.get("id") or "").upper()
        if key and key in normalized:
            matches.append(issue)
    return matches


def _format_jira_context(issues: Sequence[Dict[str, Any]]) -> str:
    chunks = []
    for issue in issues:
        key = issue.get("key") or issue.get("id") or "UNKNOWN"
        title = issue.get("title") or issue.get("summary") or "No title"
        status = issue.get("status", "UNKNOWN")
        assignee = issue.get("assignee", "Unassigned")
        summary = issue.get("description") or issue.get("summary") or ""
        chunks.append(
            f"[Jira:{key}] {title}\nStatus: {status} | Assignee: {assignee}\nSummary: {summary}"
        )
    return "\n\n".join(chunks)


def _collect_git_info(question: str, project_root: Path) -> str:
    normalized = _normalize_question(question)
    base_cmd = ["git", "-C", str(project_root)]
    commit_match = COMMIT_HASH_PATTERN.search(question or "")
    if commit_match:
        commit = commit_match.group(0)
        cmd = base_cmd + ["show", "--stat", "--unified=3", commit]
    elif any(keyword in normalized for keyword in ("diff", "change", "changes", "compare")):
        cmd = base_cmd + ["show", "--stat", "-1"]
    else:
        cmd = base_cmd + ["log", "-5", "--oneline"]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError("Git CLI is not installed or not available in PATH.") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() or "Unknown git error."
        raise RuntimeError(stderr) from exc
    output = (result.stdout or "").strip()
    if not output:
        raise RuntimeError("Git command did not return any data.")
    return output


def build_advanced_graph(vector_store, llm, settings):
    """LangGraph pipeline với nhiều intent (DOCS/JIRA/GIT/UNKNOWN)."""

    graph = StateGraph(AdvancedState)
    chain = PROMPT | llm

    def classify(state: AdvancedState):
        return {"intent": _classify_intent(state.get("question", ""))}

    def retrieve_docs(state: AdvancedState):
        question = state["question"]
        results = vector_store.similarity_search_with_score(question, k=settings.retriever_k)

        if settings.retriever_score_threshold is not None:
            results = [
                (doc, score)
                for doc, score in results
                if score <= settings.retriever_score_threshold
            ]

        documents = [doc for doc, _ in results]
        scores = [score for _, score in results]
        context = _join_docs(documents) if documents else ""
        return {"documents": documents, "scores": scores, "context": context}

    def handle_jira(state: AdvancedState):
        question = state.get("question", "")
        dataset = _load_jira_knowledge(settings.jira_dataset_path)
        candidate_keys = _extract_issue_keys(question)

        if candidate_keys:
            matches = _filter_issues_by_keys(dataset, candidate_keys)
            if matches:
                return {"jira_data": matches, "tool_error": ""}

            last_error: str | None = None
            for key in candidate_keys:
                try:
                    sync_single_issue(key, settings)
                except JiraSyncError as exc:
                    last_error = f"Không thể sync ticket {key}: {exc}"
                    continue

                dataset = _load_jira_knowledge(settings.jira_dataset_path)
                matches = _filter_issues_by_keys(dataset, [key])
                if matches:
                    return {"jira_data": matches, "tool_error": ""}

            message = last_error or "Không có ticket nào khớp với câu hỏi."
            return {"jira_data": [], "tool_error": message}

        matches = _search_jira_data(question, dataset)
        if matches:
            return {"jira_data": matches, "tool_error": ""}

        message = (
            "Không tìm thấy data Jira nội bộ (data/jira_samples.json)."
            if not dataset
            else "Không có ticket nào khớp với câu hỏi."
        )
        return {"jira_data": [], "tool_error": message}

    def handle_git(state: AdvancedState):
        try:
            git_info = _collect_git_info(state["question"], settings.git_repo_root)
        except RuntimeError as exc:
            return {"git_data": "", "tool_error": str(exc)}
        return {"git_data": git_info, "tool_error": ""}

    def fallback(state: AdvancedState):
        message = state.get("fallback_message")
        if not message:
            intent = state.get("intent", "UNKNOWN")
            if state.get("tool_error"):
                message = f"Không thể xử lý yêu cầu {intent}: {state['tool_error']}"
            elif intent == "DOCS":
                message = "Không tìm thấy tài liệu phù hợp trong vector store. Hãy ingest thêm dữ liệu."
            elif intent == "JIRA":
                message = "Chưa cấu hình dữ liệu Jira mẫu nên không thể trả lời."
            elif intent == "GIT":
                message = "Không thể đọc lịch sử Git của dự án hiện tại."
            else:
                message = "Chưa xác định được loại câu hỏi. Vui lòng mô tả chi tiết hơn."
        return {"fallback_message": message}

    def generate_answer(state: AdvancedState):
        fallback_message = state.get("fallback_message")
        jira_data = state.get("jira_data") or []
        git_data = state.get("git_data")

        if fallback_message and not (state.get("context") or jira_data or git_data):
            return {"answer": fallback_message}

        if state.get("context"):
            context = state["context"]
        elif jira_data:
            context = _format_jira_context(jira_data)
        elif git_data:
            context = git_data
        else:
            context = fallback_message or ""

        response = chain.invoke(
            {
                "question": state["question"],
                "context": context or "No context available.",
            }
        )
        content = getattr(response, "content", str(response))
        return {"answer": content}

    graph.add_node("classify_intent", classify)
    graph.add_node("retrieve_docs", retrieve_docs)
    graph.add_node("handle_jira", handle_jira)
    graph.add_node("handle_git", handle_git)
    graph.add_node("fallback", fallback)
    graph.add_node("generate_answer", generate_answer)

    graph.set_entry_point("classify_intent")

    graph.add_conditional_edges(
        "classify_intent",
        lambda state: state.get("intent", "UNKNOWN"),
        {
            "DOCS": "retrieve_docs",
            "JIRA": "handle_jira",
            "GIT": "handle_git",
            "UNKNOWN": "fallback",
        },
    )

    graph.add_conditional_edges(
        "retrieve_docs",
        lambda state: "generate" if state.get("context") else "fallback",
        {"generate": "generate_answer", "fallback": "fallback"},
    )

    graph.add_conditional_edges(
        "handle_jira",
        lambda state: "generate" if state.get("jira_data") else "fallback",
        {"generate": "generate_answer", "fallback": "fallback"},
    )

    graph.add_conditional_edges(
        "handle_git",
        lambda state: "generate" if state.get("git_data") else "fallback",
        {"generate": "generate_answer", "fallback": "fallback"},
    )

    graph.add_edge("fallback", "generate_answer")
    graph.add_edge("generate_answer", END)

    return graph.compile()
