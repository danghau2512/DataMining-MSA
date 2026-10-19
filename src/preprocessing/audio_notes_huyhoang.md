# Tiền xử lý âm thanh — Huy Hoàng

## Phạm vi và cách gọi

Code: `src/preprocessing/audio.py`.
Gọi `process_audio(video_id, clip_id, video_path)` trên một MP4.
Các hàm phụ: `probe_audio`, `read_audio`, `extract_audio_features`.
Không xử lý văn bản, nhãn, hình ảnh hoặc ghép các phương thức.

## Đầu ra

Kết quả có `video_id`, `clip_id`, `status`, `length`, `features`, `error`, `metadata`.
`features` là NumPy float32 dạng `(T_audio, 40)` với cấu hình mặc định.
`length` là số hàng đặc trưng thực tế, chưa padding hoặc cắt cố định độ dài chuỗi.
`ok`: thành công; `error`: chưa có đặc trưng, length=0, kèm lý do.
`needs_review`: tín hiệu sau giải mã và chuyển mono toàn số 0; vẫn giữ đặc trưng và lý do.
Hậu xem trạng thái trước khi quyết định sử dụng mẫu, không tự loại mẫu trong audio.py.

## Căn cứ

Paper: Attention Is All You Need (Vaswani et al., 2017).
https://papers.nips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf

| Quyết định | Bằng chứng | Áp dụng hoặc điều chỉnh | Cần kiểm tra thêm |
|---|---|---|---|
| Giữ chuỗi có thứ tự | Paper mục 3, 3.5 | Giữ các cửa sổ theo thời gian; positional encoding thuộc phần mô hình | Quy ước vị trí khi tích hợp |
| Trả độ dài thật | Paper mục 3.2.3 nói về masking trong decoder | Vận dụng ý tưởng kiểm soát vị trí được chú ý; padding mask do Hậu tạo, khác causal mask của decoder | Mask batch ở phần tích hợp |
| Kiểm tra và giải mã bằng ffprobe/ffmpeg | EDA mục 13 đã khảo sát 30 mẫu Train | Kế thừa việc kiểm tra track và chọn track audio đầu tiên | Chưa kết luận cho toàn bộ dữ liệu |
| Mono 16 kHz và log-Mel 40 chiều | Đề xuất của nhóm; EDA ghi 30 mẫu nguồn 44,1 kHz, hai kênh | Cấu hình khởi đầu, không phải đặc trưng paper quy định | Nghe mẫu Train; lựa chọn sau này dựa vào Train/Valid |

Không tuyên bố tái hiện kiến trúc hoặc thí nghiệm của paper bằng file này.

## Tham số khởi đầu

- Sample rate đích: 16000 Hz, mono; giữ thông tin sample rate/số kênh nguồn.
- n_fft=win_length=400: cửa sổ 25 ms ở 16 kHz; Hann, power=2.
- hop_length=160: mỗi cửa sổ bắt đầu sau cửa sổ trước 10 ms.
- n_mels=40; dải tần dùng mặc định librosa từ 0 đến sr/2.
- center=False: không đệm biên; chỉ lấy cửa sổ đầy đủ. Audio ngắn hơn một cửa sổ trả lỗi.
- T_audio = 1 + (num_samples - n_fft) // hop_length khi num_samples >= n_fft.
- log-Mel: power_to_db(ref=1.0, amin=1e-10, top_db=None), không phải dB âm thanh đã hiệu chuẩn.
- Probe chờ tối đa 30 giây; tham số timeout của process_audio mặc định 120 giây áp dụng cho giải mã.
- Chưa khử nhiễu, cắt khoảng lặng, cân bằng âm lượng từng đoạn hoặc học thống kê chuẩn hóa.
- Nếu bổ sung chuẩn hóa học từ dữ liệu, chỉ fit trên Train rồi áp dụng cho Valid/Test.
- Chưa giới hạn chiều dài chuỗi; cần khảo sát độ dài Train và bộ nhớ trước khi chốt xử lý chuỗi dài.

## Kiểm tra đã thực hiện

- Tín hiệu nhân tạo đúng một giây ở 16 kHz: (98, 40), float32, hữu hạn.
- MP4 nhân tạo có hình và audio nguồn 44,1 kHz: status=ok, (100, 40), mono 16 kHz; thời lượng giải mã 1.0216875 giây trong lần thử Colab.
- File không tồn tại và video_path=None: trả error, length=0, features=None, có lý do; None cần TypeError trong khối except.
- Kiểm tra bổ sung trên môi trường trợ lý bằng dữ liệu nhân tạo: file rỗng/hỏng/không có audio, tín hiệu toàn số 0, tín hiệu quá ngắn/không hữu hạn/không mono, cấu hình sai.
- Chưa chạy các đoạn CMU-MOSI thật hoặc xử lý toàn bộ dataset.

## Việc còn lại trước bàn giao

- Nhận 3–5 cặp mã Train do Hậu thống nhất, kiểm tra MP4 và chạy thử cùng các phần khác.
- Ghi từng mã, status, length, shape, tính hữu hạn, kiểm tra công thức thời gian và lý do lỗi; nghe âm thanh trước/sau chuyển đổi.
- Kiểm tra start_time, thời lượng nguồn và thời lượng giải mã trước khi kết luận đồng bộ với video; mã hóa có thể tạo chênh lệch nhỏ.
- Đề nghị Hậu quản lý phụ thuộc chung: numpy, librosa và hai công cụ ffmpeg/ffprobe; chưa sửa requirements.txt.
- PR chỉ gồm audio.py và ghi chú riêng; chưa merge main. Cập nhật kết quả MP4 thật trước khi đề nghị review tích hợp.
