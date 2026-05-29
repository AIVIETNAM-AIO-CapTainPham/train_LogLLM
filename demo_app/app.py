"""Streamlit demo: phân loại log normal/anomalous bằng LogLLM trên dữ liệu UNSEEN.

Chọn/sample các session từ vùng held-out (dòng 1.5M+ của BGL.log, model CHƯA train)
rồi so sánh dự đoán của model với ground truth.

Chuẩn bị 1 lần:
    uv run python demo_app/prepare_heldout.py     # tạo heldout_sessions.csv

Chạy app:
    uv run streamlit run demo_app/app.py
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))
from inference import SPLITER, load_model, predict  # noqa: E402

HELDOUT_CSV = APP_DIR / "heldout_sessions.csv"

st.set_page_config(page_title="LogLLM — Anomaly Demo", page_icon="🪵", layout="wide")


# --------------------------------------------------------------------------- #
# Cache: model load 1 lần duy nhất (nặng + tốn VRAM), data cache theo file.
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Đang tải LogLLM (BERT + Llama-3-8B)... lần đầu lâu")
def get_model(device: str):
    return load_model(device)


@st.cache_data(show_spinner="Đang đọc held-out sessions...")
def load_sessions(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.reset_index(drop=True)
    return df


def run_predict_batched(model, sequences, batch_size, progress=None):
    """Chạy predict theo batch nhỏ để tránh OOM VRAM."""
    preds = []
    total = len(sequences)
    for i in range(0, total, batch_size):
        chunk = sequences[i : i + batch_size]
        preds.extend(predict(model, chunk))
        if progress is not None:
            progress.progress(min((i + len(chunk)) / total, 1.0))
    return preds


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
st.sidebar.title("⚙️ Cấu hình")
device = st.sidebar.text_input("Device", value="cuda:0")
batch_size = st.sidebar.slider("Batch size (inference)", 1, 32, 8)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Dữ liệu demo lấy từ **dòng 1.5M+ của BGL.log** — phần model **chưa từng "
    "train**, có sẵn ground truth."
)

if not HELDOUT_CSV.exists():
    st.error(
        f"Chưa có `{HELDOUT_CSV.name}`.\n\n"
        "Hãy chạy trước:\n\n"
        "```bash\nuv run python demo_app/prepare_heldout.py\n```"
    )
    st.stop()

df = load_sessions(str(HELDOUT_CSV))

# --------------------------------------------------------------------------- #
# Header + thống kê pool
# --------------------------------------------------------------------------- #
st.title("🪵 LogLLM — Demo phát hiện log bất thường")
st.caption("BGL dataset · dữ liệu held-out (unseen) · so sánh prediction vs ground truth")

n_total = len(df)
n_anom = int((df["Label"] == 1).sum())
n_norm = int((df["Label"] == 0).sum())
c1, c2, c3 = st.columns(3)
c1.metric("Tổng session (held-out)", f"{n_total:,}")
c2.metric("Normal", f"{n_norm:,}")
c3.metric("Anomalous", f"{n_anom:,}")

st.markdown("---")

# --------------------------------------------------------------------------- #
# Chọn cách lấy mẫu
# --------------------------------------------------------------------------- #
tab_sample, tab_index = st.tabs(["🎲 Lấy mẫu ngẫu nhiên", "🔢 Chọn theo index"])

selected = None

with tab_sample:
    colf = st.columns(3)
    label_filter = colf[0].selectbox("Lọc theo nhãn", ["Tất cả", "Chỉ normal", "Chỉ anomalous"])
    n_samples = colf[1].number_input("Số session", min_value=1, max_value=200, value=10, step=1)
    seed = colf[2].number_input("Random seed", min_value=0, value=42, step=1)

    if st.button("🎲 Lấy mẫu & dự đoán", type="primary"):
        pool = df
        if label_filter == "Chỉ normal":
            pool = df[df["Label"] == 0]
        elif label_filter == "Chỉ anomalous":
            pool = df[df["Label"] == 1]
        if len(pool) == 0:
            st.warning("Không có session nào khớp bộ lọc.")
        else:
            selected = pool.sample(n=min(n_samples, len(pool)), random_state=int(seed))

with tab_index:
    idx_text = st.text_input(
        "Nhập các index (cách nhau bởi dấu phẩy)",
        placeholder="vd: 0, 5, 123, 4567",
    )
    if st.button("🔢 Dự đoán theo index", type="primary"):
        try:
            idxs = [int(x.strip()) for x in idx_text.split(",") if x.strip() != ""]
            idxs = [i for i in idxs if 0 <= i < n_total]
            if not idxs:
                st.warning("Không có index hợp lệ.")
            else:
                selected = df.loc[idxs]
        except ValueError:
            st.error("Index không hợp lệ — chỉ nhập số nguyên cách nhau bởi dấu phẩy.")


# --------------------------------------------------------------------------- #
# Chạy dự đoán + hiển thị
# --------------------------------------------------------------------------- #
if selected is not None and len(selected) > 0:
    model = get_model(device)

    sequences = [str(c).split(SPLITER) for c in selected["Content"].tolist()]
    gts = ["anomalous" if lbl == 1 else "normal" for lbl in selected["Label"].tolist()]

    prog = st.progress(0.0, text="Đang dự đoán...")
    preds = run_predict_batched(model, sequences, batch_size, prog)
    prog.empty()

    # Metrics
    y_true = [1 if g == "anomalous" else 0 for g in gts]
    y_pred = [1 if p == "anomalous" else 0 for p in preds]
    acc = accuracy_score(y_true, y_pred)
    has_pos = any(y_true) or any(y_pred)
    prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0) if has_pos else 0.0
    rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0) if has_pos else 0.0
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0) if has_pos else 0.0
    n_correct = sum(int(p == g) for p, g in zip(preds, gts))

    st.subheader(f"📊 Kết quả ({n_correct}/{len(preds)} đúng)")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{acc:.3f}")
    m2.metric("Precision", f"{prec:.3f}")
    m3.metric("Recall", f"{rec:.3f}")
    m4.metric("F1", f"{f1:.3f}")

    st.markdown("---")

    # Bảng tổng hợp
    rows = []
    for idx, gt, pred in zip(selected.index.tolist(), gts, preds):
        rows.append(
            {
                "index": idx,
                "ground_truth": gt,
                "prediction": pred,
                "match": "✅" if pred == gt else "❌",
            }
        )
    result_df = pd.DataFrame(rows)
    st.dataframe(result_df, use_container_width=True, hide_index=True)

    # Chi tiết từng session
    st.subheader("🔍 Chi tiết session")
    for idx, seq, gt, pred in zip(selected.index.tolist(), sequences, gts, preds):
        ok = pred == gt
        icon = "✅" if ok else "❌"
        gt_badge = "🔴" if gt == "anomalous" else "🟢"
        pr_badge = "🔴" if pred == "anomalous" else ("🟢" if pred == "normal" else "⚪")
        with st.expander(
            f"{icon} #{idx} · GT {gt_badge} {gt} · Pred {pr_badge} {pred} · {len(seq)} dòng"
        ):
            st.code("\n".join(seq), language="text")
else:
    st.info("👈 Chọn cách lấy mẫu ở trên rồi bấm nút để chạy dự đoán.")
