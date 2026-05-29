"""Chuẩn bị dữ liệu cho demo boosting (XGBoost / CatBoost).

Hai việc:
  1. Regen TEXT của 227 template train (Brain trên dòng 0-1.5M) -> lưu cố định
     vào model/templates.csv  (vì templates.csv gốc đã bị xóa). Chỉ chạy 1 lần.
  2. Parse TOÀN BỘ held-out (dòng 1.5M -> hết file) bằng Brain, map EventTemplate
     (text) về đúng index template train, lưu demo_app/heldout_boosting.parquet
     gồm [Time, Label, EventIdx]  -> dùng chung cho cả 2 kiểu window.

Vì sao map theo TEXT chứ không theo EventId: Brain tự đánh E0,E1,... theo từng
corpus, nên chạy lại trên held-out sẽ ra EventId lệch nghĩa. Text template
('generating core <*>') mới so sánh được giữa 2 lần chạy.

Chạy:
    python demo_app/boosting/prepare_boosting.py
"""

import ast
import json
import re
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path("/home/angle/jupyterlab/DEV/LogLLM_train/LogLLM")
BGL = ROOT / "data" / "BGL.log"
BOOST_REPO = ROOT / "train_boosting" / "BGL_log_error_classification"
BRAIN_DIR = BOOST_REPO / "Brain"
MODEL_DIR = BOOST_REPO / "model"
OUT_PARQUET = ROOT / "demo_app" / "heldout_boosting.parquet"

TRAIN_END = 1_500_000          # train dùng 0..1.5M
HELDOUT_START = 1_500_000      # held-out 1.5M..hết file
HELDOUT_END = None

sys.path.insert(0, str(BRAIN_DIR))
from Brain import LogParser  # noqa: E402

# Regex parse 10 trường BGL thô (giống NB2)
LOG_PATTERN = re.compile(
    r"^(\S+)\s+(\d+)\s+(\d{4}\.\d{2}\.\d{2})\s+(\S+)\s+"
    r"(\d{4}-\d{2}-\d{2}-\d{2}\.\d{2}\.\d{2}.\d{6})\s+(\S+)\s+"
    r"(\S+)\s+(\S+)\s+(\S+)\s+(.*)$"
)
BRAIN_CONFIG = {
    "log_format": "<Label> <Node> <Time> <Type> <Component> <Level> <Content>",
    "threshold": 6,
    "delimeter": [],
    "rex": [
        r"core\.\d+",
        r"0x[0-9A-Fa-f]+",
        r"\d+\.\d+\.\d+\.\d+",
        r"(?<=[^A-Za-z0-9])(\-?\+?\d+)(?=[^A-Za-z0-9])|[0-9]+$",
    ],
}


def make_processed_log(start, end, out_path):
    """BGL.log[start:end) -> processed.log format Brain (Label Node Time Type Comp Level Content)."""
    kept = skipped = 0
    with open(BGL, "r", encoding="utf-8", errors="ignore") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:
        for pos, line in enumerate(fin):
            if pos < start:
                continue
            if end is not None and pos >= end:
                break
            m = LOG_PATTERN.search(line.strip())
            if not m:
                skipped += 1
                continue
            label, _ts, _date, node, time, _rep, typ, comp, level, content = m.groups()
            fout.write(f"{label} {node} {time} {typ} {comp} {level} {content}\n")
            kept += 1
    print(f"  processed.log: giữ {kept:,} | bỏ {skipped:,}")
    return kept


def run_brain(processed_log, workdir):
    """Chạy Brain -> trả về (structured_csv, templates_csv)."""
    indir = workdir / "in"
    outdir = workdir / "out"
    indir.mkdir(parents=True, exist_ok=True)
    outdir.mkdir(parents=True, exist_ok=True)
    dest = indir / processed_log.name
    shutil.copy(processed_log, dest)
    t0 = datetime.now()
    parser = LogParser(
        logname=processed_log.name,
        log_format=BRAIN_CONFIG["log_format"],
        indir=str(indir) + "/",
        outdir=str(outdir) + "/",
        threshold=BRAIN_CONFIG["threshold"],
        delimeter=BRAIN_CONFIG["delimeter"],
        rex=BRAIN_CONFIG["rex"],
    )
    parser.parse(processed_log.name)
    print(f"  Brain xong {(datetime.now()-t0).total_seconds():.0f}s")
    return (
        outdir / (processed_log.name + "_structured.csv"),
        outdir / (processed_log.name + "_templates.csv"),
    )


def ensure_train_templates():
    """Regen model/templates.csv (text 227 template train) nếu chưa có."""
    dst = MODEL_DIR / "templates.csv"
    if dst.exists():
        print(f"[train templates] đã có: {dst}")
        return dst
    print("[train templates] regen Brain trên 0-1.5M ...")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        plog = td / "train.log"
        make_processed_log(0, TRAIN_END, plog)
        _structured, templates = run_brain(plog, td)
        df = pd.read_csv(templates)
        df.to_csv(dst, index=False)
    print(f"[train templates] lưu: {dst} ({len(df)} templates)")
    return dst


def normalize_template(t):
    """Chuẩn hóa template về dạng plain space-joined.

    templates.csv (train) lưu tuple "('generating', '<*>')";
    structured.csv (held-out) lưu plain "generating <*>". Đưa cả 2 về plain để map."""
    t = str(t).strip()
    if t.startswith("(") and t.endswith(")"):
        try:
            return " ".join(str(x) for x in ast.literal_eval(t))
        except (ValueError, SyntaxError):
            pass
    return t


def build_text_to_idx():
    """text EventTemplate train (đã chuẩn hóa) -> idx (theo thứ tự template_event_ids.json)."""
    order = json.loads((MODEL_DIR / "template_event_ids.json").read_text())  # [E0,E1,..]
    tdf = pd.read_csv(MODEL_DIR / "templates.csv")  # EventId, EventTemplate, ...
    eid_to_text = {
        str(e): normalize_template(t)
        for e, t in zip(tdf["EventId"], tdf["EventTemplate"])
    }
    text_to_idx = {}
    for idx, eid in enumerate(order):
        txt = eid_to_text.get(str(eid))
        if txt is not None:
            text_to_idx[txt] = idx
    return text_to_idx


def prepare_heldout():
    text_to_idx = build_text_to_idx()
    print(f"[held-out] parse {HELDOUT_START:,}..hết file")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        plog = td / "heldout.log"
        make_processed_log(HELDOUT_START, HELDOUT_END, plog)
        structured, _templates = run_brain(plog, td)
        df = pd.read_csv(structured, usecols=["Label", "Time", "EventTemplate"])

    # map text -> idx train (không khớp -> -1)
    df["EventIdx"] = (
        df["EventTemplate"].map(normalize_template).map(text_to_idx).fillna(-1).astype(int)
    )
    n_match = int((df["EventIdx"] >= 0).sum())
    print(f"[held-out] coverage template: {n_match:,}/{len(df):,} = {n_match/len(df)*100:.1f}%")

    # Label thô '-'/text -> 0/1
    normal = {"-", "0", "normal", "false", "n"}
    df["Label"] = (~df["Label"].astype(str).str.strip().str.lower().isin(normal)).astype("int8")
    df = df[["Time", "Label", "EventIdx"]].copy()
    df["EventIdx"] = df["EventIdx"].astype("int32")
    df.to_parquet(OUT_PARQUET, index=False)
    print(f"[held-out] lưu: {OUT_PARQUET} ({len(df):,} dòng)")


if __name__ == "__main__":
    ensure_train_templates()
    prepare_heldout()
    print("XONG.")
