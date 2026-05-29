"""Streamlit demo phát hiện log bất thường trên dữ liệu held-out (BGL dòng 1.5M+).

2 MODE (cùng vùng unseen, đều có ground truth để so sánh):
  • Mode 1 — LogLLM:   chia theo 100 dòng/session (kiểu LLM).
  • Mode 2 — Boosting: chia theo chuỗi thời gian (sliding 5 phút), model XGBoost/CatBoost.

Chuẩn bị 1 lần:
    uv run python demo_app/prepare_heldout.py            # Mode 1: heldout_sessions.csv
    uv run python demo_app/boosting/prepare_boosting.py  # Mode 2: heldout_boosting.parquet

Chạy app:
    uv run streamlit run demo_app/app.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))
sys.path.insert(0, str(APP_DIR / "boosting"))

st.set_page_config(page_title="Log Anomaly Demo", page_icon="🪵", layout="wide")

HELDOUT_CSV = APP_DIR / "heldout_sessions.csv"
HELDOUT_PARQUET = APP_DIR / "heldout_boosting.parquet"


# =========================================================================== #
# MODE 1 — LogLLM (100 dòng/session)
# =========================================================================== #
@st.cache_resource(show_spinner="Đang tải LogLLM (BERT + Llama-3-8B)... lần đầu lâu")
def get_llm(device: str):
    from inference import load_model
    return load_model(device)


@st.cache_data(show_spinner="Đang đọc held-out sessions...")
def load_llm_sessions(path: str) -> pd.DataFrame:
    return pd.read_csv(path).reset_index(drop=True)


def render_llm_mode():
    from inference import SPLITER, predict

    st.title("🪵 Mode 1 — LogLLM")

    if not HELDOUT_CSV.exists():
        st.error(
            f"Chưa có `{HELDOUT_CSV.name}`. Chạy trước:\n\n"
            "```bash\nuv run python demo_app/prepare_heldout.py\n```"
        )
        return

    device = st.sidebar.text_input("Device", value="cuda:0")
    batch_size = st.sidebar.slider("Batch size", 1, 32, 8)

    df = load_llm_sessions(str(HELDOUT_CSV))
    n_total, n_anom, n_norm = len(df), int((df.Label == 1).sum()), int((df.Label == 0).sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Tổng session", f"{n_total:,}")
    c2.metric("Normal", f"{n_norm:,}")
    c3.metric("Anomalous", f"{n_anom:,}")
    st.markdown("---")

    tab_sample, tab_index = st.tabs(["🎲 Lấy mẫu ngẫu nhiên", "🔢 Chọn theo index"])
    selected = None
    with tab_sample:
        col = st.columns(3)
        label_filter = col[0].selectbox("Lọc nhãn", ["Tất cả", "Chỉ normal", "Chỉ anomalous"])
        n_samples = col[1].number_input("Số session", 1, 200, 10)
        seed = col[2].number_input("Seed", 0, value=42)
        if st.button("🎲 Lấy mẫu & dự đoán", type="primary"):
            pool = df
            if label_filter == "Chỉ normal":
                pool = df[df.Label == 0]
            elif label_filter == "Chỉ anomalous":
                pool = df[df.Label == 1]
            if len(pool) == 0:
                st.warning("Không có session khớp bộ lọc.")
            else:
                selected = pool.sample(n=min(n_samples, len(pool)), random_state=int(seed))
    with tab_index:
        idx_text = st.text_input("Index (cách nhau dấu phẩy)", placeholder="vd: 0, 5, 123")
        if st.button("🔢 Dự đoán theo index", type="primary"):
            try:
                idxs = [int(x) for x in idx_text.split(",") if x.strip()]
                idxs = [i for i in idxs if 0 <= i < n_total]
                selected = df.loc[idxs] if idxs else None
                if not idxs:
                    st.warning("Không có index hợp lệ.")
            except ValueError:
                st.error("Index không hợp lệ.")

    if selected is None or len(selected) == 0:
        st.info("👆 Chọn cách lấy mẫu rồi bấm nút để dự đoán.")
        return

    model = get_llm(device)
    sequences = [str(c).split(SPLITER) for c in selected["Content"].tolist()]
    gts = ["anomalous" if v == 1 else "normal" for v in selected["Label"].tolist()]

    prog = st.progress(0.0, text="Đang dự đoán...")
    preds = []
    for i in range(0, len(sequences), batch_size):
        preds.extend(predict(model, sequences[i : i + batch_size]))
        prog.progress(min((i + batch_size) / len(sequences), 1.0))
    prog.empty()

    y_true = [1 if g == "anomalous" else 0 for g in gts]
    y_pred = [1 if p == "anomalous" else 0 for p in preds]
    _show_metrics(y_true, y_pred)

    rows = [
        {"index": i, "ground_truth": g, "prediction": p, "match": "✅" if p == g else "❌"}
        for i, g, p in zip(selected.index, gts, preds)
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.subheader("🔍 Chi tiết session")
    for i, seq, g, p in zip(selected.index, sequences, gts, preds):
        icon = "✅" if p == g else "❌"
        gb = "🔴" if g == "anomalous" else "🟢"
        pb = "🔴" if p == "anomalous" else ("🟢" if p == "normal" else "⚪")
        with st.expander(f"{icon} #{i} · GT {gb} {g} · Pred {pb} {p} · {len(seq)} dòng"):
            st.code("\n".join(seq), language="text")


# =========================================================================== #
# MODE 2 — Boosting (chuỗi thời gian)
# =========================================================================== #
@st.cache_resource(show_spinner="Đang tải XGBoost + CatBoost...")
def get_boosting():
    import infer_boosting as ib
    return ib.load_artifacts()


@st.cache_data(show_spinner="Đang chia chuỗi thời gian...")
def build_time_windows_cached(path: str, n_templates: int, win: int, step: int):
    import infer_boosting as ib
    df = ib.load_heldout()
    X, y, meta = ib.build_time_windows(df, n_templates, win, step)
    return X, y, meta


def render_boosting_mode():
    import infer_boosting as ib

    st.title("📊 Mode 2 — Boosting")

    if not HELDOUT_PARQUET.exists():
        st.error(
            f"Chưa có `{HELDOUT_PARQUET.name}`. Chạy trước:\n\n"
            "```bash\nuv run python demo_app/boosting/prepare_boosting.py\n```"
        )
        return

    model_name = st.selectbox(
        "Model", ["xgb", "cat"],
        format_func=lambda k: {"xgb": "XGBoost", "cat": "CatBoost"}[k],
    )

    # session_state: kết quả không biến mất khi tick checkbox (st.button chỉ True 1 lần).
    if st.button("📊 Đánh giá toàn bộ held-out", type="primary"):
        st.session_state["boost_evaluated"] = True
    if not st.session_state.get("boost_evaluated"):
        st.info("👆 Bấm nút để infer toàn bộ held-out (dòng 1.5M → hết) theo window 5 phút.")
        return

    WIN, STEP, THR = 5, 1, 0.5
    art = get_boosting()
    X, y, meta = build_time_windows_cached(str(HELDOUT_PARQUET), art["n_templates"], WIN, STEP)
    prob = ib.predict(art, model_name, X)
    pred = (prob >= THR).astype(int)
    res = ib.evaluate(y, prob, THR)
    name = {"xgb": "XGBoost", "cat": "CatBoost"}[model_name]
    n_wrong = int((pred != y).sum())

    st.subheader(f"📈 Kết quả — {name}")
    cols = st.columns(5)
    cols[0].metric("Accuracy", f"{res['accuracy']:.3f}")
    cols[1].metric("Precision", f"{res['precision']:.3f}")
    cols[2].metric("Recall", f"{res['recall']:.3f}")
    cols[3].metric("F1", f"{res['f1']:.3f}")
    cols[4].metric("Window đoán sai", f"{n_wrong:,}/{len(y):,}")

    tbl = meta.copy()
    tbl.insert(0, "window", range(len(tbl)))
    tbl["Sự thật"] = np.where(y == 1, "🔴 anomaly", "🟢 normal")
    tbl["Model đoán"] = np.where(pred == 1, "🔴 anomaly", "🟢 normal")
    tbl["Xác suất"] = prob.round(3)
    tbl["Kết quả"] = np.where(pred == y, "✅ đúng", "❌ SAI")
    tbl = tbl.rename(columns={"window_start": "Bắt đầu", "window_end": "Kết thúc", "num_logs": "Số log"})

    only_wrong = st.checkbox(f"🔴 Chỉ xem các window đoán SAI ({n_wrong:,})", value=False)
    view = tbl[tbl["Kết quả"] == "❌ SAI"] if only_wrong else tbl
    st.dataframe(view, use_container_width=True, hide_index=True, height=420)


# =========================================================================== #
# Helpers + Router
# =========================================================================== #
def _show_metrics(y_true, y_pred):
    n_correct = sum(int(a == b) for a, b in zip(y_true, y_pred))
    st.subheader(f"📊 Kết quả ({n_correct}/{len(y_true)} đúng)")
    m = st.columns(4)
    m[0].metric("Accuracy", f"{accuracy_score(y_true, y_pred):.3f}")
    m[1].metric("Precision", f"{precision_score(y_true, y_pred, zero_division=0):.3f}")
    m[2].metric("Recall", f"{recall_score(y_true, y_pred, zero_division=0):.3f}")
    m[3].metric("F1", f"{f1_score(y_true, y_pred, zero_division=0):.3f}")


st.sidebar.title("⚙️ Demo mode")
mode = st.sidebar.radio(
    "Chọn mode",
    ["Mode 1 — LogLLM", "Mode 2 — Boosting"],
)
st.sidebar.markdown("---")
st.sidebar.caption("Dữ liệu: BGL.log **dòng 1.5M+** (vùng model chưa train), có ground truth.")

if mode.startswith("Mode 1"):
    render_llm_mode()
else:
    render_boosting_mode()
