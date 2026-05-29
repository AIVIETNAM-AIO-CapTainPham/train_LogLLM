"""Tải model LogLLM (1 lần) và chạy dự đoán cho 1 hoặc nhiều log sequence.

Tái dùng:
  - LogLLM            (model.py)        : kiến trúc BERT + projector + Llama
  - replace_patterns  (customDataset.py): tiền xử lý (mask số/IP/path -> <*>)
  - merge_data        (customDataset.py): nối các sequence + tính seq_positions

Luồng giống eval.py nhưng gói lại thành hàm dùng được cho UI.
"""

import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent          # .../LogLLM
sys.path.insert(0, str(ROOT))
from customDataset import merge_data, replace_patterns  # noqa: E402
from model import LogLLM  # noqa: E402

# Đường dẫn khớp với eval.py
BERT_PATH = str(ROOT / "models" / "bert-base-uncased")
LLAMA_PATH = str(ROOT / "models" / "Meta-Llama-3-8B")
FT_PATH = str(ROOT / "checkpoints" / "ft_model_BGL_20260525_1414")

MAX_CONTENT_LEN = 100
MAX_SEQ_LEN = 128
SPLITER = " ;-; "


def load_model(device: str = "cuda:0") -> LogLLM:
    """Tải LogLLM ở chế độ inference. Gọi 1 lần rồi cache lại (nặng + tốn VRAM)."""
    dev = torch.device(device)
    model = LogLLM(
        BERT_PATH,
        LLAMA_PATH,
        ft_path=FT_PATH,
        is_train_mode=False,
        device=dev,
        max_content_len=MAX_CONTENT_LEN,
        max_seq_len=MAX_SEQ_LEN,
    )
    model.eval()
    return model


def _preprocess(seq):
    """1 sequence (list các dòng log thô) -> list dòng đã mask, cắt MAX_SEQ_LEN.

    Mirror chính xác CustomDataset: replace_patterns chạy trên chuỗi đã nối
    bằng ' ;-; ', sau đó split lại."""
    joined = replace_patterns(SPLITER.join(seq))
    return joined.split(SPLITER)[:MAX_SEQ_LEN]


@torch.no_grad()
def predict(model: LogLLM, sequences):
    """sequences: list[list[str]] (mỗi phần tử là 1 session gồm nhiều dòng log).

    Trả về list[str] gồm 'normal' / 'anomalous' / 'unknown' theo thứ tự đầu vào.
    """
    if not sequences:
        return []

    device = model.device
    tokenizer = model.Bert_tokenizer

    processed = [_preprocess(seq) for seq in sequences]
    data, seq_positions = merge_data(processed)
    seq_positions = seq_positions[1:]  # bỏ vị trí 0 đầu tiên (xem collator)

    inputs = tokenizer(
        data,
        return_tensors="pt",
        max_length=MAX_CONTENT_LEN,
        padding=True,
        truncation=True,
    ).to(device)
    seq_positions_t = torch.tensor(seq_positions, dtype=torch.long)

    output_ids = model(inputs, seq_positions_t)
    texts = model.Llama_tokenizer.batch_decode(output_ids)

    preds = []
    for text in texts:
        match = re.search(r"normal|anomalous", text, re.IGNORECASE)
        preds.append(match.group().lower() if match else "unknown")
    return preds
