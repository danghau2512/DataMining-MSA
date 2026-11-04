# Visual Bảo Hoàng — notebook báo cáo có code đầy đủ

Notebook: [02_Visual_BaoHoang_BaoCao.ipynb](../src/notebooks/02_Visual_BaoHoang_BaoCao.ipynb).

Đọc từ trên xuống. Các hàm có tên theo chức năng và được định nghĩa trước khi dùng.
Hai hàm read_visual_csv() và extract_visual_features() nằm đầy đủ trong notebook,
tại Task 2.12A–B; các task xử lý gọi đúng hai hàm này.

## Phần cần mở khi báo cáo

| Nội dung | Task mới | Hàm hoặc đoạn code |
| --- | --- | --- |
| Manifest, ID, nhãn, split | 2.3 | read_manifest() |
| Tên/thứ tự 22 chiều | 2.11 | AU_NAMES, POSE_NAMES, GAZE_NAMES, FEATURE_NAMES |
| Chọn cột, float32, thời gian, mask | 2.12A | numeric[names], dtype=np.float32, observed_mask |
| Chạy công cụ OpenFace | 2.12B | command, -aus/-pose/-gaze, subprocess.run() |
| Điều phối một clip và 10 clip | 2.14–2.15 | extract_one_clip(), extract_pilot_clips() |
| Quan sát và ghi nhận thủ công | 2.16–2.17 | show_sample_frames(), create_visual_review_form() |
| Xem dữ liệu đã lưu | 2.18 | save_pilot_features(), show_visual_values() |
| Đọc báo cáo đầy đủ | 2.21 | read_saved_results() |
| Dùng lại cache và tiếp tục trích | 2.22 | process_full_dataset(), extract_full_dataset() |

File/thư mục báo cáo trên Drive giữ tên cũ. Task 2.2 liệt kê tên thực tế.
Thư mục: CMU-MOSI/preprocessing/visual/Task_2_BaoCao_Visual.
Cấu hình cũ 2.18 chuyển sang 2.11; đọc CSV cũ 2.19 chuyển sang 2.18.
Đây là thứ tự mới của notebook cá nhân, chưa thay số task trong notebook chung.

## Chạy để đọc kết quả

Mount tài khoản có CMU-MOSI trong MyDrive, hoặc sửa VisualPaths.data_root cho phù hợp.
Giữ SETUP_OPENFACE, RUN_PILOT, RUN_EXTRACTION và VERIFY_SAVED_FEATURES đều False.
Mặc định không cài, biên dịch hoặc chạy lại OpenFace. Task 2.18 đọc một CSV pilot nếu
còn đường dẫn hợp lệ; Task 2.21 đọc báo cáo cùng một NPZ để minh họa shape/dtype/mask.
Không cần trích lại 2.199 clip để báo cáo. Quyền truy cập Drive cần được cấp riêng.

Đường dẫn từ CMU-MOSI:
- preprocessing/shared/manifest.csv
- preprocessing/visual/config/visual_feature_config.json
- preprocessing/visual/pilot_batches/20261102T070035_074030Z/pilot_quality_summary.csv
- preprocessing/visual/full_visual_22/3c908e7632a1/features/<id>.npz
- preprocessing/visual/full_visual_22/3c908e7632a1/runs/20261102T140801_205739Z/visual_manifest.csv

## Tích hợp vào notebook chung

Dùng tên hàm và thứ tự mới trong bản này; cập nhật các chỗ gọi tên task_* cũ khi tích hợp.
Giữ phần audio/text/alignment của các thành viên khác. Notebook cũ và module
vision_processor.py vẫn được giữ để tương thích với code và báo cáo đã merge.
Hai hàm lõi inline đã được đối chiếu AST với module gốc; không đổi cách trích đặc trưng.

## Phạm vi và kiểm chứng

22 chiều là lựa chọn nhóm (17 AU intensity + 3 góc đầu + 2 góc nhìn), khác Facet trong paper.
float32 là kiểu số thực 32 bit, không phải chuẩn hóa.
Mask = isfinite(features) AND success=1; không dùng ngưỡng confidence.
Giữ cả frame theo dõi thất bại và giá trị không hợp lệ, chưa tự loại mẫu/điền thiếu.
Số liệu đã bàn giao: 2.199 clip; 2.139 ok và 60 needs_review.
ok chỉ là đạt kiểm tra kỹ thuật, chưa chứng minh đúng khuôn mặt người nói.

Đã kiểm tra cú pháp, thứ tự phụ thuộc, lựa chọn cột/dtype/mask bằng dữ liệu giả,
và lệnh gọi OpenFace bằng mô phỏng. Chưa chạy lại toàn bộ dữ liệu thật với bản notebook này.
Chưa chuẩn hóa, căn chỉnh theo từ/token hoặc huấn luyện MAG-BERT.
