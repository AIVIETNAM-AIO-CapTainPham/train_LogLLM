# LogLLM Demo App

Streamlit UI để demo phân loại log **normal / anomalous** bằng LogLLM, chạy trên
**dữ liệu chưa từng được train**.

## Vì sao dùng dòng 1.5M+?

Pipeline train (`prepareData/sliding_window.py`) chỉ structure dòng `0 → 1,500,000`
của `BGL.log`, rồi split 80/20 thành `train.csv` / `test.csv`. Do đó **từ dòng
1,500,000 trở đi (~3.25M dòng) là hoàn toàn unseen**. BGL có sẵn ground truth
(cột `Label`: `-` = normal, còn lại = anomalous), nên demo so sánh được prediction
với GT thật.

## Cài đặt

```bash
# trong thư mục LogLLM/
uv sync          # cài thêm streamlit (đã thêm vào pyproject.toml)
```

## Bước 1 — Tạo dữ liệu held-out (chạy 1 lần)

```bash
uv run python demo_app/prepare_heldout.py
```

Sinh ra `demo_app/heldout_sessions.csv` (cùng định dạng `test.csv`: cột
`Content`, `Label`, `item_Label`, `session_length`). Có thể chỉnh `START_LINE` /
`END_LINE` trong script nếu muốn vùng khác.

## Bước 2 — Chạy app

```bash
uv run streamlit run demo_app/app.py
```

## Tính năng

- **Lấy mẫu ngẫu nhiên**: lọc theo nhãn (normal/anomalous/tất cả), chọn số lượng + seed.
- **Chọn theo index**: nhập index cụ thể từ pool held-out.
- Hiển thị **Accuracy / Precision / Recall / F1** trên batch đã chọn.
- Bảng so sánh GT vs Pred (✅/❌) + xem chi tiết từng dòng log của mỗi session.

## Lưu ý

- Model nặng (BERT + Llama-3-8B 4-bit) → cần GPU CUDA. Model được cache bằng
  `@st.cache_resource` nên chỉ load **1 lần**.
- Đường dẫn model/checkpoint khớp với `eval.py`. Nếu đổi checkpoint, sửa `FT_PATH`
  trong `inference.py`.

## Cấu trúc

| File | Vai trò |
|------|---------|
| `prepare_heldout.py` | One-time: trích sessions unseen từ `BGL.log` kèm GT |
| `inference.py` | `load_model()` + `predict()` (tái dùng `model.py`, `customDataset.py`) |
| `app.py` | Streamlit UI |
