# DemoLangGraph Flow Overview

Tài liệu này mô tả luồng hoạt động chính của hệ thống RAG trong DemoLangGraph, từ ingest tài liệu đến trả lời câu hỏi bằng LangGraph.

## 1. Ingest Flow (`demo-rag ingest`)

```
┌────────────┐    ┌──────────────┐    ┌──────────────────────┐    ┌───────────────────────┐    ┌──────────────────────┐
│ CLI ingest │ -> │ settings.py  │ -> │ ingest.load_documents │ -> │ ingest.split_documents │ -> │ embeddings factory   │
└────────────┘    └──────────────┘    └──────────────────────┘    └───────────────────────┘    └──────────┬───────────┘
                                                                                                          │
                                                                                             ┌────────────┴─────────────┐
                                                                                             │ vectorstore.save_vector… │
                                                                                             └────────────┬─────────────┘
                                                                                                          │
                                                                                             ┌────────────┴─────────────┐
                                                                                             │  Chroma index persisted  │
                                                                                             └──────────────────────────┘
```

1. Typer CLI nhận tham số và gọi `ingest(settings)`.
2. `settings.py` xác định đường dẫn tài liệu, chunk config và provider embeddings.
3. `ingest.load_documents` đọc toàn bộ `.txt/.md`.
4. `ingest.split_documents` chia thành chunk theo `CHUNK_SIZE/CHUNK_OVERLAP`.
5. `embeddings.get_embeddings` tạo encoder (Azure OpenAI hoặc BGE-M3).
6. `vectorstore.save_vectorstore` sinh embeddings cho từng chunk và lưu vào Chroma (`artifacts/vectorstore/chroma_index`).

## 2. Chat Flow (`demo-rag chat`)

```
┌────────────┐    ┌──────────────┐    ┌────────────────────┐    ┌────────────────┐    ┌────────────────┐    ┌───────────────┐
│ CLI chat   │ -> │ settings.py  │ -> │ embeddings factory │ -> │ load Chroma    │ -> │ build_retriever│ -> │ LangGraph RAG │
└────────────┘    └──────────────┘    └────────────────────┘    └────────────────┘    └────────────────┘    └──────┬────────┘
                                                                                                                    │
                                                                          question ────────────────────────────────┘│
                                                                                                                    │
                                                                 ┌─────────────────────────────┐                    │
                                                                 │ retrieve node (retriever)   │  top-k documents  │
                                                                 └──────────────┬──────────────┘                    │
                                                                                │                                   │
                                                                 ┌──────────────┴──────────────┐                    │
                                                                 │ generate node (LLM prompt) │── answer + sources─┘
                                                                 └─────────────────────────────┘
```

1. CLI khởi tạo settings (có thể override `--vector-store-dir`), tạo embeddings, retriever và LLM (`get_llm`).
2. `vectorstore.build_retriever` nạp Chroma index và tạo retriever với `k=RETRIEVER_K`.
3. `graph.build_rag_graph` biên dịch `StateGraph` gồm hai node:
   - `retrieve`: gọi retriever với câu hỏi để lấy top-k documents, nối context.
   - `generate`: inject `{question, context}` vào prompt và gọi LLM để sinh câu trả lời.
4. CLI hiển thị `answer` và (nếu bật) danh sách nguồn từ metadata.

## 3. Thông tin cấu hình nhanh

| Thành phần               | File                         | Biến cấu hình chính                                  |
|--------------------------|------------------------------|------------------------------------------------------|
| Settings + env           | `src/demolanggraph/settings.py` | Azure endpoints, embeddings, retriever thresholds    |
| Embeddings factory       | `src/demolanggraph/embeddings.py` | Azure OpenAI (`AZURE_*`) hoặc BGE-M3 (SentenceTransformer) |
| Vector store (Chroma)    | `src/demolanggraph/vectorstore.py` | `RETRIEVER_K`, `vector_store_dir`                    |
| LangGraph định nghĩa RAG | `src/demolanggraph/graph.py` | Prompt cố định `retrieve -> generate`                |
| CLI                      | `src/demolanggraph/cli.py`   | Lệnh `ingest`, `chat`, các tùy chọn Typer            |

> Sau mỗi lần đổi embeddings/LLM hoặc cập nhật tài liệu, hãy chạy lại `demo-rag ingest` để index đồng bộ trước khi `demo-rag chat`.
