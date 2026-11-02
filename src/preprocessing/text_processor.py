import torch
"""Xử lý Text CMU-MOSI: làm sạch tối thiểu và tokenize."""

import unicodedata
import numpy as np


def kiem_tra_ky_tu(text):
    if not isinstance(text, str):
        raise TypeError("Transcript phải là chuỗi")

    canh_bao = []

    for vi_tri, ky_tu in enumerate(text):
        nhom = unicodedata.category(ky_tu)

        la_ky_tu_thay_the = ky_tu == "\ufffd"

        la_ky_tu_dieu_khien = (
            nhom in {"Cc", "Cf", "Cs"}
            and not ky_tu.isspace()
        )

        if la_ky_tu_thay_the or la_ky_tu_dieu_khien:
            canh_bao.append({
                "vi_tri": vi_tri,
                "ky_tu": repr(ky_tu),
                "ma_unicode": f"U+{ord(ky_tu):04X}",
                "nhom": nhom,
            })

    return canh_bao


def lam_sach_text(text):
    if not isinstance(text, str):
        raise TypeError("Transcript phải là chuỗi")

    clean_text = " ".join(text.split())

    if clean_text == "":
        raise ValueError("Transcript trống sau khi chuẩn hóa khoảng trắng")

    return clean_text


def process_text(text, config):
   

    bo_tokenize = config["tokenizer"]

    if not bo_tokenize.is_fast:
        raise ValueError("Cần fast tokenizer để lấy word_ids")

    canh_bao = kiem_tra_ky_tu(text)
    clean_text = lam_sach_text(text)
    words = clean_text.split()

    ma_hoa = bo_tokenize(
        words,
        is_split_into_words=True,
        add_special_tokens=True,
        padding=False,
        truncation=False,
        return_attention_mask=True,
        return_token_type_ids=True,
    )

    input_ids = np.asarray(
        ma_hoa["input_ids"],
        dtype=np.int64,
    )

    attention_mask = np.asarray(
        ma_hoa["attention_mask"],
        dtype=np.int64,
    )

    token_type_ids = np.asarray(
        ma_hoa["token_type_ids"],
        dtype=np.int64,
    )

    word_ids = np.asarray(
        [
            -1 if chi_so is None else chi_so
            for chi_so in ma_hoa.word_ids()
        ],
        dtype=np.int64,
    )

    so_token = len(input_ids)

    if not (
        so_token
        == len(attention_mask)
        == len(token_type_ids)
        == len(word_ids)
    ):
        raise ValueError("Các chuỗi đầu ra không cùng chiều dài")

    tu_da_anh_xa = set(word_ids[word_ids >= 0].tolist())

    if tu_da_anh_xa != set(range(len(words))):
        raise ValueError("Có từ chưa được ánh xạ sang token")
    embeddings = None

    if config.get("extract_embeddings", False):
        mo_hinh = config.get("bert_model")

        if mo_hinh is None:
            raise ValueError(
                "Cần khởi tạo bert_model trước khi trích xuất embedding"
            )

        if so_token > mo_hinh.config.max_position_embeddings:
            raise ValueError(
                "Số token vượt giới hạn BERT; chưa tự cắt transcript"
            )

        thiet_bi = next(mo_hinh.parameters()).device

        dau_vao = {
            "input_ids": torch.tensor(
                input_ids, dtype=torch.long, device=thiet_bi
            ).unsqueeze(0),
            "attention_mask": torch.tensor(
                attention_mask, dtype=torch.long, device=thiet_bi
            ).unsqueeze(0),
            "token_type_ids": torch.tensor(
                token_type_ids, dtype=torch.long, device=thiet_bi
            ).unsqueeze(0),
        }

        mo_hinh.eval()

        with torch.inference_mode():
            dau_ra = mo_hinh(**dau_vao, return_dict=True)

        embeddings = (
            dau_ra.last_hidden_state[0]
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        if embeddings.shape != (
            so_token, mo_hinh.config.hidden_size
        ):
            raise ValueError("Embedding không đúng kích thước")

        if not np.isfinite(embeddings).all():
            raise ValueError("Embedding chứa NaN hoặc vô cực")

    return {
        "raw_text": text,
        "clean_text": clean_text,
        "words": words,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "word_ids": word_ids,
        "embeddings": embeddings,
        "warnings": canh_bao,
    }
