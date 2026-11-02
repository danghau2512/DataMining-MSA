# Bàn giao Text pilot CMU-MOSI

## Giao diện
process_text(text, config) -> dict

## Xử lý Text
- Giữ raw_text; clean_text chỉ chuẩn hóa khoảng trắng, tab, xuống dòng.
- Giữ phủ định, từ đệm, từ lặp và dấu câu.
- Không stemming, xóa stopwords hoặc tự sửa transcript.
- Kiểm tra ký tự đáng ngờ và trả về warnings.

## Đầu ra
- raw_text, clean_text, words, warnings.
- input_ids, attention_mask, token_type_ids, word_ids:
  mảng int64 một chiều, cùng chiều dài.
- word_ids ánh xạ token về từ; token đặc biệt dùng -1.
- embeddings=None khi không yêu cầu trích xuất.

## Embedding pilot
- Checkpoint: bert-base-uncased.
- Lấy tầng cuối last_hidden_state theo token.
- Shape: (số token, 768), dtype float32.
- Gồm CLS và SEP; chưa gộp subword về từ.
- Khởi tạo BERT một lần ngoài vòng lặp.
- Dùng requires_grad_(False), eval() và inference_mode().
- Không padding hoặc truncation.
- Cấu hình đầu vào cuối cùng cho DeepCU chưa được nhóm xác nhận.

## File kết quả
- 10 file NPZ theo ID: embedding và các mảng token.
- embedding_config.json: cấu hình và phiên bản thư viện.
- embedding_log.csv: nhật ký xử lý.
- text_processor.py: bản module kèm kết quả.
- Text JSON gốc:
  /content/drive/MyDrive/CMU-MOSI/preprocessing/text/pilot/20261102T092142_188135Z/text_pilot_results.json
- Các đường dẫn trên tính từ thư mục CMU-MOSI.

## Kiểm chứng
- 10/10 mẫu pilot thành công.
- Kết quả tokenize khớp bản Text trước đó.
- File NPZ đọc lại khớp dữ liệu trước khi lưu.
- Một mẫu chạy lại cho kết quả ổn định.
- Chưa xử lý toàn bộ 2.199 mẫu.

## Alignment
Quy trình MFA nằm trong notebook MOSI_Alignment_Pilot.ipynb.
Clip 7JsX8y1ysxY$_$12 còn needs_review do chưa rõ cụm đầu I DONT KNOW.
