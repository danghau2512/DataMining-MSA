# Audio Preprocessing – CMU-MOSI

## 1. Giới thiệu

Module `audio_processor.py` phụ trách tiền xử lý dữ liệu âm thanh cho đồ án **Multimodal Sentiment Analysis trên CMU-MOSI**.

Các công việc chính:
- Tách âm thanh từ video gốc bằng FFmpeg.
- Chuyển đổi sang định dạng WAV thống nhất.
- Trích xuất đặc trưng âm thanh bằng openSMILE.
- Giữ thông tin thời gian của từng cửa sổ đặc trưng.
- Kiểm tra chất lượng dữ liệu và ghi nhận các giá trị cần xử lý.

**Người phụ trách:** Huy Hoàng

**Trạng thái:** Đã trích xuất và kiểm tra 2.199/2.199 mẫu CMU-MOSI.

## 2. Cấu hình tiền xử lý

| Thành phần | Cấu hình |
|---|---|
| Công cụ tách âm thanh | FFmpeg |
| Định dạng đầu ra | WAV PCM 16-bit |
| Số kênh | Mono |
| Tần số lấy mẫu | 16.000 Hz |
| Công cụ trích xuất | openSMILE |
| Bộ đặc trưng | eGeMAPSv02 |
| Feature level | LowLevelDescriptors (LLD) |
| Số chiều đặc trưng | 25 |
| Chuẩn hóa Z-score | Chưa thực hiện |
| Cắt hoặc padding | Chưa thực hiện |

Các chuỗi đặc trưng được giữ nguyên độ dài theo từng clip, không lấy trung bình toàn bộ clip và không ép về 50 bước thời gian.

## 3. Cài đặt môi trường

Môi trường triển khai và kiểm thử: Google Colab.

Các phiên bản đã sử dụng:
- FFmpeg: 6.1.1-3ubuntu5
- openSMILE: 2.6.0
- soundfile: 0.14.0

Cài đặt thư viện Python:

```bash
pip install opensmile==2.6.0 soundfile==0.14.0 numpy
```

FFmpeg cần được cài đặt và có thể gọi bằng lệnh `ffmpeg`.

## 4. Module và cách sử dụng

**Đường dẫn module:**

`src/preprocessing/audio_processor.py`

Module cung cấp hai hàm chính.

### 4.1. prepare_audio()

```python
prepare_audio(video_path, wav_path, config) -> Path
```

Chức năng:
- Kiểm tra video đầu vào.
- Tách âm thanh từ video bằng FFmpeg.
- Chuyển sang WAV mono, 16 kHz, PCM 16-bit theo cấu hình.
- Kiểm tra định dạng và thời lượng.
- Trả về đường dẫn WAV.

### 4.2. extract_audio_features()

```python
extract_audio_features(wav_path, config) -> dict
```

Chức năng:
- Kiểm tra WAV đầu vào.
- Trích xuất chuỗi đặc trưng openSMILE eGeMAPSv02 LLD.
- Lấy timestamp của từng cửa sổ.
- Giữ tên các chiều đặc trưng.
- Tạo `observed_mask` để ghi nhận trạng thái dữ liệu.

### 4.3. Ví dụ sử dụng

Ví dụ dưới đây được chạy từ thư mục gốc repository:

```python
from src.preprocessing.audio_processor import (
    prepare_audio,
    extract_audio_features
)

config = {"sample_rate": 16000}

wav_path = prepare_audio(
    video_path="path/to/video.mp4",
    wav_path="path/to/audio.wav",
    config=config
)

result = extract_audio_features(
    wav_path=wav_path,
    config=config
)

print(result["features"].shape)
print(result["start_times"].shape)
print(result["end_times"].shape)
print(len(result["feature_names"]))
print(result["observed_mask"].shape)
```

Thay các đường dẫn minh họa bằng đường dẫn dữ liệu thực tế trước khi chạy.

## 5. Cấu trúc đầu ra

Hàm `extract_audio_features()` trả về dictionary với năm thành phần:

| Trường | Kiểu dữ liệu | Kích thước |
|---|---|---|
| `features` | float32 | `(T_audio, 25)` |
| `start_times` | float64 | `(T_audio,)` |
| `end_times` | float64 | `(T_audio,)` |
| `feature_names` | Danh sách chuỗi | 25 phần tử |
| `observed_mask` | bool | `(T_audio, 25)` |

Trong đó:
- `T_audio` là số cửa sổ đặc trưng của từng clip.
- `start_times`, `end_times` được tính bằng giây.
- `observed_mask` có cùng kích thước với `features`.

Dữ liệu đầu ra chưa được chuẩn hóa Z-score, alignment hoặc padding.

## 6. Kết quả thực nghiệm

Module được kiểm thử ban đầu trên 10 clip pilot, sau đó chạy trên toàn bộ 2.199 mẫu CMU-MOSI.

### 6.1. Kết quả xử lý

| Chỉ số | Kết quả |
|---|---:|
| Tổng số clip | 2.199 |
| WAV xử lý thành công | 2.199 |
| File đặc trưng hợp lệ | 2.199 |
| Số lỗi kiểm chứng | 0 |
| Số lỗi đọc WAV | 0 |
| Clip đạt kiểm tra thời lượng | 2.199 |

### 6.2. Thống kê chuỗi đặc trưng

| Chỉ số | Giá trị |
|---|---:|
| Số chiều đặc trưng | 25 |
| Số frame trung bình | 421,65 |
| Số frame nhỏ nhất | 17 |
| Số frame lớn nhất | 5.240 |
| Tổng giá trị NaN | 0 |
| Tổng giá trị Inf | 0 |
| Tổng giá trị -201 | 1.172.521 |
| Tổng vị trí mask=False | 1.172.518 |

Thời điểm kết thúc đặc trưng khớp với thời lượng WAV trong sai số tính toán rất nhỏ. Kiểm tra này chưa chứng minh đồng bộ audio/video tuyệt đối.

## 7. Lưu ý về observed_mask

Trong quá trình trích xuất, openSMILE xuất hiện giá trị `-201` ở một số đặc trưng.

Quy tắc hiện tại:
- Giữ nguyên các giá trị đặc trưng gốc.
- Đánh dấu `False` đối với NaN/Inf.
- Đánh dấu `False` cho giá trị `-201` ở ba đặc trưng:
  - `F1amplitudeLogRelF0_sma3nz`
  - `F2amplitudeLogRelF0_sma3nz`
  - `F3amplitudeLogRelF0_sma3nz`

Kiểm tra toàn bộ dataset còn phát hiện ba vị trí có giá trị `-201` nhưng mask vẫn là `True`, thuộc các đặc trưng `logRelF0-H1-H2_sma3nz` và `logRelF0-H1-A3_sma3nz`.

**Quy tắc mask này chưa được người tổng hợp xác nhận chính thức.** Các vị trí trên được giữ nguyên và ghi nhận trong báo cáo, chưa tự ý sửa dữ liệu.

## 8. Lưu trữ và bàn giao

Dữ liệu đầu ra được lưu trên Google Drive theo cấu trúc:

```text
CMU-MOSI/
└── preprocessing/
    └── audio/
        ├── wav/
        ├── features_pilot/
        ├── reports/
        └── audio_full/
            ├── wav/
            ├── features/
            └── reports/
```

Thư mục `audio_full/features/` chứa các file `.npz` theo từng ID.

Các báo cáo chính:
- `audio_full_status.csv`
- `audio_full_quality_report.csv`
- `audio_full_validation_errors.csv`
- `audio_full_mask_review.csv`
- `audio_full_duration_report.csv`
- `audio_full_duration_errors.csv`
- `audio_full_handover.txt`

Các file WAV, NPZ và dữ liệu gốc không được đưa trực tiếp lên GitHub do kích thước lớn. Người tổng hợp sử dụng dữ liệu từ Google Drive hoặc chạy lại module với nguồn video gốc.
