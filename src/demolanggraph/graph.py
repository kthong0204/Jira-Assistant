from __future__ import annotations

from typing import List, TypedDict, Tuple

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph


class RAGState(TypedDict, total=False):
    question: str
    context: str
    answer: str
    documents: List[Document]
    scores: List[float]


PROMPT = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant that answers questions about the ingested knowledge base. "
                "Use ONLY the provided context. If you are unsure, say you do not know.",
            ),
            ("human", "Question: {question}\n\nContext:\n{context}"),
        ]
)


def _join_docs(documents: List[Document]) -> str:
    return "\n\n".join(f"[{idx+1}] {doc.page_content}" for idx, doc in enumerate(documents))


def build_rag_graph(vector_store, llm, settings):
    graph = StateGraph(RAGState)
    chain = PROMPT | llm

    def retrieve(state: RAGState):
        question = state["question"]
        results: List[Tuple[Document, float]] = vector_store.similarity_search_with_score(
            question, k=settings.retriever_k
        )

        if settings.retriever_score_threshold is not None:
            results = [
                (doc, score)
                for doc, score in results
                if score <= settings.retriever_score_threshold
            ]

        docs = [doc for doc, _ in results]
        scores = [score for _, score in results]

        return {
            "documents": docs,
            "scores": scores,
            "context": _join_docs(docs) if docs else "",
        }

    def generate(state: RAGState):
        response = chain.invoke(
            {
                "question": state["question"],
                "context": state.get("context", "No context available."),
            }
        )

        content = getattr(response, "content", str(response))
        return {"answer": content}

    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile()
