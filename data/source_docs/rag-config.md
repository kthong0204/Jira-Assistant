# Hướng Dẫn Cấu Hình & Kiểm thử RAG cho PrivateGPT-API

Tài liệu này giúp bạn kích hoạt pipeline RAG để trả lời dựa trên dữ liệu ticket đã đồng bộ.

---

## 1. Biến môi trường cần thiết (`privateGPT-API/.env`)

| Biến | Giá trị gợi ý | Mô tả |
| --- | --- | --- |
| `RAG_ENABLED` | `true` | Bật RAG pipeline |
| `RAG_VECTOR_STORE` | `vector_store` | Thư mục chứa Chroma DB |
| `RAG_EMBEDDING_MODEL` | `BAAI/bge-m3` | Model embedding mặc định |
| `RAG_TRANSLATE` | `false` | Tắt dịch sang tiếng Anh (dữ liệu thuần Việt) |
| `RAG_TOP_K` | `4` | Số chunk tốt nhất trả về cho LLM |
| `RAG_MIN_SCORE` | `0.35` | Ngưỡng cosine tối thiểu để chấp nhận chunk |

> Nếu cần chuyển qua mô hình khác (VD: `sentence-transformers/all-MiniLM-L6-v2`) chỉ cần đổi giá trị `RAG_EMBEDDING_MODEL`.

---

## 2. Chuẩn bị dữ liệu ticket cho vector store

1. Xuất dữ liệu ticket Jira thành các file văn bản (Markdown, JSON hoặc TXT). Mỗi ticket nên là một file riêng nằm trong thư mục, ví dụ `docs/tickets/`.
2. Chạy script ingest (tự viết hoặc tái sử dụng script của privateGPT gốc). Pseudocode:

   ```bash
   python tools/ingest_ticket_docs.py \
     --input docs/tickets \
     --vector-store vector_store \
     --embedding-model "BAAI/bge-m3"
   ```

   Script nên:
   - Đọc từng file, cắt thành chunk 500–800 token
   - Ghi metadata (`ticket_key`, `updated_at`, …)
   - Gọi model embedding và lưu vào Chroma DB (`vector_store/`)

---

## 3. Kiểm thử endpoint RAG

Gửi request mẫu tới proxy:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
        "model": "gpt-4o",
        "messages": [
          {"role": "system", "content": "Bạn là trợ lý Jira."},
          {"role": "user", "content": "Tóm tắt tiến độ ticket JIRA2I-3 dựa trên tài liệu đã đồng bộ."}
        ],
        "use_rag": true
      }'
```

Khi `use_rag=true`, response sẽ kèm `privategpt_metadata.rag` chứa danh sách chunk được sử dụng. Kiểm tra trường này để chắc chắn RAG hoạt động và hiểu chunk nào đã được lấy ra.

---

## 4. Lưu ý & mẹo

- **Không dịch khi dữ liệu tiếng Việt**: đặt `RAG_TRANSLATE=false` để giữ ngữ cảnh chuẩn.
- **Tối ưu độ chính xác**: nếu dữ liệu nhiễu, tăng `RAG_MIN_SCORE` hoặc giảm `RAG_TOP_K`.
- **Cập nhật theo ticket**: mỗi khi có ticket mới hoặc description thay đổi nhiều, chạy lại script ingest để Chroma DB được làm mới.
- **Giám sát**: nên log lại `rag_metadata` trong response để biết câu trả lời dựa trên những chunk nào, phòng trường hợp “hallucination”.

---

## 5. Checklist nhanh

1. [ ] `.env` đã bật `RAG_ENABLED=true` và cấu hình đúng model/chroma.
2. [ ] Dữ liệu ticket được lưu tại `docs/tickets/` (hoặc thư mục bạn chỉ định).
3. [ ] Chạy script ingest, sinh ra thư mục `vector_store/`.
4. [ ] `uvicorn privategpt_api.presentation.api.main:app --reload`.
5. [ ] Gửi request `use_rag=true` và kiểm tra `privategpt_metadata.rag`.

Hoàn tất 5 bước trên là bạn có thể test đầy đủ tính năng RAG dựa trên dữ liệu ticket nội bộ.
