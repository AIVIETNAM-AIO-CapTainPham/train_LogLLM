# Log Anomaly Demo (Streamlit, 2 mode)

Demo phát hiện log bất thường trên **dữ liệu held-out BGL (dòng 1.5M+)** — vùng
**cả 2 model đều chưa train**, có sẵn ground truth để so sánh.

| Mode | Model | Cách chia data | Nguồn |
|------|-------|----------------|-------|
| **Mode 1** | LogLLM (BERT + Llama-3-8B) | 100 dòng / session | `heldout_sessions.csv` |
| **Mode 2** | XGBoost / CatBoost | sliding time-window 5 phút | `heldout_boosting.parquet` |

Vì sao dòng 1.5M+: pipeline train (cả LogLLM lẫn boosting) chỉ dùng dòng
`0–1,500,000`. Từ 1.5M trở đi là unseen hoàn toàn.

## Cài đặt (uv)

```bash
uv sync
```

## Chuẩn bị data (chạy 1 lần mỗi mode)

```bash
# Mode 1 — LogLLM: gom held-out thành session 100 dòng
uv run python demo_app/prepare_heldout.py

# Mode 2 — Boosting: regen template train + Brain-parse toàn bộ held-out 1.5M+
uv run python demo_app/boosting/prepare_boosting.py
```

> Mode 2 map mỗi dòng held-out về đúng 227 template train **theo text template**
> (Brain đánh EventId khác nhau giữa các corpus nên không map theo EventId được).
> Coverage ~62% trên toàn bộ 1.5M+ (drift theo thời gian); dòng không khớp → bỏ
> qua (EventIdx = -1), giống cách NB4 xử lý EventId lạ.

## Chạy app

```bash
uv run streamlit run demo_app/app.py
```

Chọn mode ở sidebar.

## Cấu trúc

| File | Vai trò |
|------|---------|
| `app.py` | Streamlit UI, router 2 mode |
| `inference.py` | Mode 1: load LogLLM + predict |
| `prepare_heldout.py` | Mode 1: tạo `heldout_sessions.csv` |
| `boosting/infer_boosting.py` | Mode 2: load XGB/Cat + dựng time-window + predict + metrics |
| `boosting/prepare_boosting.py` | Mode 2: tạo `heldout_boosting.parquet` (+ regen `model/templates.csv`) |

## Phụ thuộc weight (Mode 2)

Đọc weight từ `train_boosting/BGL_log_error_classification/model/`:
`xgboost_model.json`, `catboost_model.cbm`, `tfidf.joblib`,
`template_event_ids.json`, `templates.csv`. **Cần đủ cả 5 file.**

## Lưu ý

- Mode 1 cần GPU CUDA (Llama-3-8B 4-bit), cache bằng `@st.cache_resource`.
- Mode 2 chạy CPU, nhanh (~2s build window + predict cho 62k window).
