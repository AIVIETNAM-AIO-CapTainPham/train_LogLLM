# Mode 2 — Boosting (chi tiết kỹ thuật)

Phần inference XGBoost/CatBoost cho demo (chia log theo **cửa sổ thời gian 5 phút**).
Xem README tổng ở [`../README.md`](../README.md) để biết cách cài/chạy toàn bộ.

## 2 file

| File | Vai trò |
|------|---------|
| `prepare_boosting.py` | Chạy **1 lần**: Brain-parse held-out (BGL 1.5M+), map mỗi dòng về 227 template train, lưu `../heldout_boosting.parquet` (cột `Time, Label, EventIdx`). Đồng thời regen `model/templates.csv` nếu thiếu. |
| `infer_boosting.py` | Dùng trong app: load weights, chia time-window → vector đếm 227 chiều → `tfidf.transform` → `predict_proba`, kèm hàm `evaluate()`. |

## Luồng feature (giống pipeline đã train)

```
log → Brain template (EventId) → đếm số lần mỗi template trong window (227 chiều)
    → TF-IDF (transformer đã fit lúc train) → XGBoost/CatBoost → prob anomaly
```

## Gotcha quan trọng: map template theo TEXT

Brain là parser tự suy template từ corpus → chạy lại trên held-out sẽ đánh
`E0, E1...` **lệch nghĩa** so với lúc train. Vì vậy KHÔNG map theo EventId mà map
theo **text template** (`"generating <*>"`...). Cần chuẩn hóa vì:
- `model/templates.csv` (train) lưu dạng tuple `('generating', '<*>')`
- `structured.csv` (held-out) lưu dạng plain `generating <*>`

→ chuẩn hóa tuple → space-joined rồi mới khớp. Dòng không khớp template train →
`EventIdx = -1` (bỏ qua khi đếm, giống cách xử lý EventId lạ lúc train).
Coverage ~62% trên toàn bộ held-out (drift theo thời gian).

## Weights cần (ở `train_boosting/BGL_log_error_classification/model/`)

`xgboost_model.json` · `catboost_model.cbm` · `tfidf.joblib` ·
`template_event_ids.json` (thứ tự 227 EventId) · `templates.csv` (text template).
**Thiếu bất kỳ file nào → infer sai hoặc lỗi.**
