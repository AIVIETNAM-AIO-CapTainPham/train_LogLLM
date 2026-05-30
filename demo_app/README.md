# Demo phát hiện log bất thường (BGL) — 2 mode

Ứng dụng Streamlit so sánh **2 hướng phát hiện anomaly** trên cùng tập log
**held-out** (dòng 1.5M+ của `BGL.log` — phần model **chưa từng train**, có sẵn
ground truth để chấm điểm):

| | Mode 1 — **LogLLM** | Mode 2 — **Boosting** |
|---|---|---|
| Model | BERT + Llama-3-8B (LoRA) | XGBoost / CatBoost |
| Cách chia log | **100 dòng / session** | **cửa sổ thời gian 5 phút** |
| Đặc trưng | hiểu ngữ nghĩa câu log | TF-IDF trên 227 template (Brain) |
| Phần cứng | cần **GPU** (4-bit) | chạy **CPU**, nhanh |

> Cả 2 chỉ train trên dòng `0–1.5M` của BGL → vùng `1.5M+` là unseen, dùng để
> demo công bằng.

---

## 1. Sơ đồ thư mục

```
LogLLM/                              # gốc repo
├── data/
│   └── BGL.log                      # log gốc 4.7M dòng (tải về, gitignored)
├── models/                          # base model (tải về, gitignored)
│   ├── bert-base-uncased
│   └── Meta-Llama-3-8B              # ~16GB, gated  → chỉ Mode 1 cần
├── checkpoints/
│   └── ft_model_BGL_.../            # LoRA fine-tune LogLLM → Mode 1 cần
├── train_boosting/BGL_log_error_classification/
│   └── model/                       # WEIGHTS boosting → Mode 2 cần
│       ├── xgboost_model.json · catboost_model.cbm
│       ├── tfidf.joblib             # transformer đã fit (bắt buộc)
│       ├── template_event_ids.json · templates.csv
│       └── ...
└── demo_app/                        # ◀ THƯ MỤC DEMO
    ├── app.py                       # Streamlit UI — router 2 mode
    ├── run_demo.sh                  # chạy Streamlit + mở Cloudflare tunnel
    ├── inference.py                 # Mode 1: load LogLLM + predict()
    ├── prepare_heldout.py           # Mode 1: sinh heldout_sessions.csv
    ├── boosting/
    │   ├── infer_boosting.py        # Mode 2: load XGB/Cat, chia time-window, predict
    │   └── prepare_boosting.py      # Mode 2: sinh heldout_boosting.parquet
    ├── heldout_sessions.csv         # (sinh ra, gitignored) dữ liệu Mode 1
    ├── heldout_boosting.parquet     # (sinh ra, gitignored) dữ liệu Mode 2
    └── README.md
```

---

## 2. Cài thư viện (uv)

Toàn bộ phụ thuộc khai báo trong `pyproject.toml` ở gốc repo. Chỉ cần:

```bash
cd LogLLM
uv sync          # tạo .venv + cài streamlit, xgboost, catboost, torch, transformers...
```

Mọi lệnh bên dưới chạy qua `uv run` để dùng đúng `.venv`.

---

## 3. Tải dữ liệu & model

### 3.1. Log BGL (cả 2 mode đều cần)
```bash
uv run python scripts/download_bgl.py        # → data/BGL.log
```

### 3.2. Base model + checkpoint (CHỈ Mode 1 — LogLLM)
```bash
# cần HF_TOKEN (Llama-3-8B là gated) — xem .env.example
uv run python scripts/download_models.py     # → models/bert-base-uncased, models/Meta-Llama-3-8B
```
Checkpoint LoRA (`checkpoints/ft_model_BGL_...`) đã có sẵn trong repo (branch
`feature/train_boosting`). Mode 2 **không cần** base model — chỉ cần `train_boosting/.../model/`.

### 3.3. Sinh dữ liệu demo held-out (chạy 1 lần mỗi mode)
```bash
# Mode 1: gom dòng 1.5M+ thành session 100 dòng
uv run python demo_app/prepare_heldout.py            # → demo_app/heldout_sessions.csv

# Mode 2: Brain-parse toàn bộ 1.5M+, map về 227 template train
uv run python demo_app/boosting/prepare_boosting.py  # → demo_app/heldout_boosting.parquet
```

> **Mode 2 map template theo TEXT** (Brain đánh EventId khác nhau giữa các corpus).
> Coverage ~62% trên toàn bộ held-out (template drift theo thời gian); dòng không
> khớp → bỏ qua. Đây là giới hạn của boosting + template cố định — đúng chỗ LogLLM
> (hiểu ngữ nghĩa) có lợi thế.

---

## 4. Chạy app

### Cách A — kèm Cloudflare tunnel (URL public để demo)
```bash
bash demo_app/run_demo.sh
```
Script tự chọn **port trống**, chờ Streamlit sẵn sàng, rồi in dòng
`https://....trycloudflare.com`. **Ctrl+C** để tắt cả hai. (URL random mỗi lần chạy.)

### Cách B — chỉ chạy local
```bash
uv run streamlit run demo_app/app.py
```

---

## 5. Dùng app

Chọn **mode** ở sidebar:

- **Mode 1 — LogLLM**: lấy mẫu / chọn index các session 100 dòng → model dự đoán
  normal/anomaly → bảng so vs ground truth + xem chi tiết log từng session.
- **Mode 2 — Boosting**: chọn XGBoost/CatBoost → "Đánh giá toàn bộ held-out" →
  metrics (Accuracy/Precision/Recall/F1) + bảng từng cửa sổ 5 phút + ô lọc "chỉ
  xem đoán sai".

---

## 6. Phụ thuộc tóm tắt

| Mode | Cần gì để chạy |
|------|----------------|
| 1 — LogLLM | `data/BGL.log` · `models/*` (base) · `checkpoints/ft_model_BGL_*` · GPU · `heldout_sessions.csv` |
| 2 — Boosting | `train_boosting/.../model/*` (5 file) · `heldout_boosting.parquet` |

File nặng (`BGL.log`, base models, `heldout_*`) đều **gitignored** — phải tự tải/sinh theo các bước trên.
