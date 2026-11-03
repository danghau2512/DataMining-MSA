# Bàn giao Visual CMU-MOSI — Bảo Hoàng

## Kết quả
- 2.199 clip: train 1.284, valid 229, test 686.
- Giữ nguyên ID, nhãn và split.
- 22 chiều: 17 AU intensity, 3 góc đầu, 2 góc nhìn.
- 2.139 ok; 60 needs_review: train 28, valid 6, test 26.
- Đã xem trực quan hai clip train có mức thiếu cao nhất.
- 58 clip needs_review còn lại chưa được xác minh trực quan.
- Giữ nguyên clip và mask; chưa loại hay điền thiếu.
- ok không chứng minh luôn theo dõi đúng người nói.
- OpenFace thay Facet của paper MAG-BERT; cần công khai khác biệt.

## Hậu dùng lại
1. Pull nhánh feat/preprocess-visual.
2. Truy cập thư mục CMU-MOSI được chia sẻ trên Drive.
3. Trong Colab, mount đúng tài khoản và có lối tắt CMU-MOSI trong MyDrive.
4. Đọc NPZ sẵn có; không cần cài OpenFace hay trích xuất lại.

Đường dẫn tương đối từ CMU-MOSI:
- Manifest: preprocessing/shared/manifest.csv
- Features: preprocessing/visual/full_visual_22/3c908e7632a1/features/<id>.npz
- Report: preprocessing/visual/full_visual_22/3c908e7632a1/runs/20261102T140801_205739Z/visual_manifest.csv
- Provenance: preprocessing/visual/full_visual_22/3c908e7632a1/extraction_config.json

Git pull lấy mã và báo cáo. Quyền truy cập Drive là riêng.
Gói GitHub không chứa NPZ, video hoặc model.

## Đọc bằng Python trong checkout repository
python src/preprocessing/02_preprocessing_Visual_BaoHoang.py --mode reuse --data-root "/content/drive/MyDrive/CMU-MOSI"

Python >=3.11; cần numpy và pandas để đọc.
Thêm --verify nếu cần kiểm tra lại từng NPZ.
Chức năng xử lý video mới cần OpenCV/OpenFace.

## NPZ
Mở bằng numpy.load(path, allow_pickle=False).
- features: float32 (T,22), giữ thứ tự frame.
- timestamps: float64 (T,), đơn vị giây.
- feature_names: đúng tên/thứ tự theo cấu hình.
- success: uint8 (T,).
- confidence: float32 (T,).
- observed_mask: bool (T,22).
- observed_mask = isfinite(features) AND success=1.
- frame_numbers: 1..T.
- Có ID, signature, nguồn ZIP, CRC và kích thước nguồn.

Không coi mask=False là quan sát hợp lệ.
Clip thiếu toàn bộ visual vẫn được giữ.
Chưa điền thiếu, chuẩn hóa hoặc căn chỉnh theo từ.
Thông số chuẩn hóa/điền thiếu nếu cần học chỉ tính từ train.

## Báo cáo
- visual_manifest.csv: kết quả từng clip.
- source_manifest.csv: manifest đầu vào.
- quality_summary.csv: chất lượng theo split.
- needs_review_detail.csv: 60 clip cần xem lại.
- manual_review_notes.csv: hai quan sát của Bảo Hoàng.
- extraction_config.json: recipe và signature.
- handoff.json: thông tin và đường dẫn bàn giao.
- package_files_sha256.csv: hash các file trong gói.

Thư viện mặc định đọc lại kết quả.
--mode extract mới chạy vòng trích xuất có kiểm tra cache.
Không ép dùng signature cũ khi module/runtime/cấu hình đã đổi.
