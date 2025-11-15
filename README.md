# DemoLangGraph

DemoLangGraph là dự án mẫu RAG sử dụng [LangGraph](https://python.langchain.com/docs/langgraph) để nối pipeline ingest → retrieve → generate. Dự án cung cấp CLI, workflow nâng cao (DOCS/JIRA/GIT) và UI Streamlit, phù hợp cho hackathon hoặc làm khung tham chiếu.

## Tính năng nổi bật
- `demo-rag ingest`: chunk tài liệu `.txt/.md`, tạo embedding và build Chroma vector store.
- `demo-rag chat`: hỏi nhanh với graph cơ bản hoặc advanced (tự phát hiện intent, auto-sync Jira ticket theo mã `ABC-123`).
- `demo-rag sync-jira <ISSUE>`: đồng bộ từng ticket vào `data/jira_samples.json`; graph advanced sẽ sử dụng dataset này.
- Streamlit UI (`streamlit_app.py`): ingest, chat và sync ticket trực quan.

## Chuẩn bị môi trường
```bash
git clone <repo-url> DemoLangGraph
cd DemoLangGraph
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -e .
cp .env.example .env   # cập nhật AZURE_OPENAI_*, JIRA_*, ...
```
Yêu cầu: Python 3.10–3.12, Git CLI (cho node GIT), tài khoản Jira Cloud với API token hoặc bearer token.

## Cấu hình quan trọng (`.env`)
| Biến | Ý nghĩa | Ví dụ |
| --- | --- | --- |
| `LLM_PROVIDER` | `azure_openai` hoặc `openai` | `azure_openai` |
| `EMBEDDINGS_PROVIDER` | `azure_openai` / `openai` / `bge` | `azure_openai` |
| `AZURE_OPENAI_*` | Endpoint/key/deployment cho LLM & embedding | phụ thuộc resource |
| `EMBEDDINGS_MODEL` | Model embedding | `text-embedding-3-small` hoặc `BAAI/bge-m3` |
| `JIRA_URL`, `JIRA_EMAIL`, `JIRA_API_TOKEN` | Credential Jira | `https://xxx.atlassian.net`, ... |
| `JIRA_BEARER_TOKEN` | Thay cho email+token (nếu có) | tùy chọn |
| `JIRA_DATASET_PATH` | File JSON lưu ticket | `data/jira_samples.json` |

Các biến khác (chunk size, retriever_k, ...) xem `src/demolanggraph/settings.py`.

## Luồng CLI mẫu
```bash
# 1. Ingest tài liệu
 demo-rag ingest

# 2. Chat với graph advanced
 demo-rag chat --graph advanced -q "tiến độ ticket JIRA2I-3"

# 3. Sync ticket mới khi cần
 demo-rag sync-jira JIRA2I-4
```
Nếu câu hỏi chứa mã ticket chưa có trong dataset, graph advanced sẽ tự gọi `sync_single_issue`. Nếu Jira trả lỗi (401/404), thông báo lỗi được hiển thị ở fallback.

## Chạy UI Streamlit
```bash
.\.venv\Scripts\Activate.ps1
python -m streamlit run streamlit_app.py
```
Sidebar cho phép chọn workflow (`basic`/`advanced`), reload graph và chỉnh đường dẫn data/vectorstore. Tab “Sync Jira ticket” nhận issue key và hiển thị payload vừa sync.

## Khắc phục nhanh
- **UI báo “không biết ticket X”**: chắc chắn ticket đã sync (`demo-rag sync-jira X`). Kiểm tra log Streamlit xem auto-sync có bị 401/404.
- **Git node lỗi**: xác nhận đã cài Git CLI và `GIT_REPO_ROOT` trỏ tới repo cần phân tích.
- **LLM/embedding 401**: kiểm tra lại key hoặc quota từ nhà cung cấp.

## Sử dụng Git
1. Khởi tạo repo (nếu chưa):
   ```bash
   git init
   git add .
   git commit -m "feat: init DemoLangGraph"
   ```
2. Tạo remote và push:
   ```bash
   git remote add origin <remote-url>
   git push -u origin main
   ```
`.gitignore` đã được chuẩn hoá để bỏ qua `.venv`, artifacts, `.env`, dữ liệu sinh ra.

## Cấu trúc thư mục
```
src/demolanggraph/
  cli.py            # Typer CLI
  graph.py          # Graph cơ bản
  graph_advanced.py # Graph đa intent + auto-sync Jira
  jira_sync.py      # sync_single_issue -> data/jira_samples.json
  embeddings.py     # factory embeddings
  llm.py            # factory LLM
  vectorstore.py    # quản lý Chroma
  settings.py       # đọc cấu hình từ .env
```
Chúc bạn hack vui vẻ!
