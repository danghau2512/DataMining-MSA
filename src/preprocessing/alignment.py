
"""Kiểm tra khoảng từ MFA và ánh xạ timestamp sang token."""

import math
from numbers import Integral


def process_alignment(word_intervals, words, word_ids, duration=None):
    """Trả timestamp khi từ MFA khớp words; báo lỗi nếu không khớp."""

    words = list(words)
    word_ids = list(word_ids)

    if not words:
        raise ValueError("Danh sách words trống")

    if any(not isinstance(w, str) or not w.strip() for w in words):
        raise ValueError("words phải chứa các từ không trống")

    if duration is not None:
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("Thời lượng clip không hợp lệ")

    # Bỏ khoảng im lặng, giữ nguyên các nhãn từ
    cac_khoang = []

    for khoang in word_intervals:
        tu = khoang["word"]

        if not isinstance(tu, str):
            raise ValueError("Nhãn từ MFA phải là chuỗi")

        if not tu.strip():
            continue

        cac_khoang.append({
            "word": tu,
            "start": float(khoang["start"]),
            "end": float(khoang["end"]),
        })

    words_mfa = [x["word"] for x in cac_khoang]

    if [w.casefold() for w in words] != [
        w.casefold() for w in words_mfa
    ]:
        raise ValueError(
            "Từ Text và MFA không khớp; không ghép timestamp. "
            f"Text: {words!r}; MFA: {words_mfa!r}"
        )

    end_truoc = 0.0

    for vi_tri, khoang in enumerate(cac_khoang):
        start = khoang["start"]
        end = khoang["end"]

        if not (math.isfinite(start) and math.isfinite(end)):
            raise ValueError(f"Từ {vi_tri}: thời gian không hữu hạn")

        if not 0 <= start < end:
            raise ValueError(f"Từ {vi_tri}: khoảng thời gian không hợp lệ")

        if duration is not None and end > duration + 0.001:
            raise ValueError(f"Từ {vi_tri}: vượt thời lượng clip")

        if start < end_truoc - 0.000001:
            raise ValueError(f"Từ {vi_tri}: chồng lấn hoặc sai thứ tự")

        end_truoc = end

    for chi_so in word_ids:
        if isinstance(chi_so, bool) or not isinstance(chi_so, Integral):
            raise ValueError("word_ids phải chứa chỉ số nguyên")

        if chi_so != -1 and not 0 <= chi_so < len(words):
            raise ValueError(f"word_id ngoài phạm vi: {chi_so}")

    chi_so_tu = [int(i) for i in word_ids if i != -1]

    if set(chi_so_tu) != set(range(len(words))):
        raise ValueError("word_ids chưa ánh xạ đủ các từ")

    if chi_so_tu != sorted(chi_so_tu):
        raise ValueError("word_ids không đúng thứ tự từ")

    word_start = [x["start"] for x in cac_khoang]
    word_end = [x["end"] for x in cac_khoang]

    token_start = []
    token_end = []
    token_time_mask = []

    for chi_so in word_ids:
        if chi_so == -1:
            token_start.append(None)
            token_end.append(None)
            token_time_mask.append(False)
        else:
            token_start.append(word_start[chi_so])
            token_end.append(word_end[chi_so])
            token_time_mask.append(True)

    return {
        "words": words,
        "word_ids": [int(i) for i in word_ids],
        "word_intervals": cac_khoang,
        "word_sequence_match": True,
        "timestamp_structure_pass": True,
        "word_start": word_start,
        "word_end": word_end,
        "token_start": token_start,
        "token_end": token_end,
        "token_time_mask": token_time_mask,
    }


def read_textgrid(textgrid_path):
    """Đọc các khoảng có từ trong tier words của MFA."""
    from pathlib import Path
    from praatio import textgrid

    tep = Path(textgrid_path)

    if not tep.is_file():
        raise FileNotFoundError(f"Không tìm thấy TextGrid: {tep}")

    tg = textgrid.openTextgrid(
        str(tep),
        includeEmptyIntervals=True,
    )

    if "words" not in tg.tierNames:
        raise ValueError(f"Không thấy tier words: {tep}")

    return [
        {"word": w, "start": float(s), "end": float(e)}
        for s, e, w in tg.getTier("words").entries
        if w.strip()
    ]


def align_textgrid(textgrid_path, words, word_ids, duration=None):
    """Đọc TextGrid, kiểm tra và ánh xạ thời gian sang token."""
    cac_khoang = read_textgrid(textgrid_path)

    return process_alignment(
        word_intervals=cac_khoang,
        words=words,
        word_ids=word_ids,
        duration=duration,
    )
