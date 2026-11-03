# Module alignment CMU-MOSI

## Phụ thuộc
pip install praatio

## Giao diện
- read_textgrid(textgrid_path): đọc tier words, bỏ khoảng im lặng.
- process_alignment(word_intervals, words, word_ids, duration=None):
  kiểm tra khoảng từ đã đọc và ánh xạ timestamp sang token.
- align_textgrid(textgrid_path, words, word_ids, duration=None):
  đọc TextGrid và gọi process_alignment.

## Đầu vào
- words và word_ids lấy từ process_text.
- duration là thời lượng WAV tính bằng giây, nếu có.

## Đầu ra
words, word_ids, word_intervals,
word_sequence_match, timestamp_structure_pass,
word_start, word_end, token_start, token_end, token_time_mask.

## Quy tắc
- Chỉ bỏ qua khác biệt hoa/thường khi đối chiếu từ.
- Từ thiếu, khác hoặc tách khác phải báo ValueError.
- Không ghép timestamp theo thứ tự khi danh sách từ không khớp.
- Token đặc biệt word_id=-1 nhận thời gian None và mask False.
- Các subword cùng từ nhận cùng khoảng thời gian của từ.
- Kiểm tra thời gian hữu hạn, dương và không chồng lấn.
- Kiểm tra giới hạn thời lượng khi truyền duration.
- Không tự sửa transcript hoặc trạng thái nghe kiểm tra.

## Kiểm chứng
Đã đối chiếu module với kết quả notebook trên 10 mẫu pilot.
Chưa chạy alignment toàn bộ 2.199 mẫu.

## Kết quả Text Full trên Drive
preprocessing/text/full/20261102T152904_636429Z/
Text và BERT embedding: 2.199 mẫu thành công.
Embedding hiện theo token, tầng cuối BERT; chưa gộp subword.
Cấu hình đầu vào DeepCU cần được nhóm thống nhất.
