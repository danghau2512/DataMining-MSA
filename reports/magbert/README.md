# MAG-BERT paper outputs

Nguồn: multiple_seeds/seed_runs.csv (trial_dir, best_epoch), training_history.csv của 6 lượt,
multiple_seeds/test_results_each_seed.csv và 6 test_predictions.csv đã lưu.
Không train, tải BERT, nạp checkpoint hoặc chạy đánh giá test bằng mô hình.

- magbert_validation_mae_by_seed.pdf/.png: 3 ô, chung trục y; sao là min validation MAE.
  Frozen seed 42 chỉ có 7 epoch và best epoch 2; không nối tới epoch 10.
- magbert_test_mae_by_seed.pdf/.png: 2 điểm cạnh nhau mỗi seed; số từ bảng test đã lưu.
- magbert_aligned_test.csv: chỉ 3 Fine-tune, dùng cho bảng so sánh ba mô hình.
  model=MAG-BERT, variant=aligned_p99; thứ tự cột theo DeepCU; các cột chung như Self-MM.
  Acc-2 theo phần trăm; Macro-F1 theo 0–1; không làm tròn giá trị trước khi xuất CSV.
- magbert_strategy_test_each_seed.csv: cả 6 lượt cho đối chiếu Frozen/Fine-tune.
- magbert_strategy_mean_std.csv: mean và sample std (ddof=1), giữ đủ chữ số.
- magbert_prediction_class_counts.csv: đếm lớp trực tiếp trên CSV dự đoán đã lưu.
- magbert_report_sources.csv: ánh xạ trial_dir gốc sang file log dùng thực tế.
- magbert_plot_style.json: nhãn/màu dùng chung cho hai hình; có thể dùng cho Self-MM.

Test MAE Fine-tune 0.7988 ± 0.0285; Frozen 1.4664 ± 0.0528.
Biểu đồ dùng PDF vector với font nhúng, PNG 300 dpi.
Frozen ở đây là các lượt đã có trong log, không tự sửa chế độ dropout BERT.

Notebook Self-MM ở GitHub main chưa có cell vẽ nên không thể xác nhận màu của một hình
Self-MM khác. Quy ước xuất hiện tại: Frozen BERT #1f77b4; Fine-tune BERT #ff7f0e.
Nếu cập nhật bảng hyperparameter, chỉ định format LR riêng như {:.1e}; không round toàn
bảng làm 1e-5 và 2e-5 hiển thị thành 0.0.

Frozen seed 21 và 42 dự đoán lớp tích cực cho toàn bộ 656 mẫu test có nhãn thật khác 0. Có 277 mẫu thật dương, nên cả hai đạt 277/656 × 100 = 42.23% Acc-2. Frozen seed 123 dự đoán cả hai lớp; không áp dụng nhận xét sụp về một lớp cho seed này. Nhận xét dựa trên test_predictions.csv đã lưu, không suy luận lại bằng mô hình.
