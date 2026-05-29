"""Chuẩn bị dữ liệu held-out (CHƯA từng được train) để demo.

Pipeline train chỉ structure dòng 0 -> 1,500,000 của BGL.log rồi split 80/20
thành train.csv / test.csv. Vì vậy phần từ dòng 1,500,000 trở đi là HOÀN TOÀN
unseen, và BGL có sẵn ground truth (cột Label: '-' = normal, còn lại = anomalous).

Script này:
  1. Đọc BGL.log từ START_LINE -> END_LINE.
  2. Structure theo đúng log_format của BGL (tái dùng helper.py của repo).
  3. Gom thành session 100 dòng (fixedSize_window) — giống hệt cách tạo test.csv.
  4. Ghi ra demo_app/heldout_sessions.csv (cùng định dạng test.csv).

Chạy 1 lần:
    uv run python demo_app/prepare_heldout.py
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent          # .../LogLLM
sys.path.insert(0, str(ROOT / "prepareData"))
from helper import fixedSize_window, generate_logformat_regex, log_to_dataframe  # noqa: E402

DATA_DIR = ROOT / "data"
LOG_NAME = "BGL.log"
OUT_CSV = Path(__file__).resolve().parent / "heldout_sessions.csv"

# Vùng unseen: từ dòng cuối của tập train trở đi tới hết file.
START_LINE = 1_500_000
END_LINE = None          # None = tới hết file (~4,747,963)
WINDOW_SIZE = 100        # giống prepareData/sliding_window.py
STEP_SIZE = 100
SPLITER = " ;-; "

# BGL log format (xem prepareData/sliding_window.py)
LOG_FORMAT = (
    "<Label> <Id> <Date> <Code1> <Time> <Code2> "
    "<Component1> <Component2> <Level> <Content>"
)


def main():
    log_file = DATA_DIR / LOG_NAME
    if not log_file.exists():
        raise FileNotFoundError(f"Không tìm thấy {log_file}")

    print(f"Đọc {LOG_NAME} từ dòng {START_LINE} -> {END_LINE or 'hết file'} ...")
    headers, regex = generate_logformat_regex(LOG_FORMAT)
    df = log_to_dataframe(str(log_file), regex, headers, START_LINE, END_LINE)
    print(f"Số dòng log đã parse: {len(df)}")

    # GT: '-' => normal (0), khác => anomalous (1)
    df["Label"] = df["Label"].apply(lambda x: int(x != "-"))

    print(f"Gom session (window={WINDOW_SIZE}, step={STEP_SIZE}) ...")
    sessions = fixedSize_window(df[["Content", "Label"]], WINDOW_SIZE, STEP_SIZE)

    sessions["session_length"] = sessions["Content"].apply(len)
    sessions["Content"] = sessions["Content"].apply(lambda x: SPLITER.join(x))
    sessions = sessions[["Content", "Label", "item_Label", "session_length"]]

    n_anom = int((sessions["Label"] == 1).sum())
    n_norm = int((sessions["Label"] == 0).sum())
    print(f"Tổng session: {len(sessions)} | normal: {n_norm} | anomalous: {n_anom}")

    sessions.to_csv(OUT_CSV, index=False)
    print(f"Đã ghi: {OUT_CSV}")


if __name__ == "__main__":
    main()
