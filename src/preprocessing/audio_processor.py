from pathlib import Path
import subprocess
import soundfile as sf


def prepare_audio(video_path, wav_path, config) -> Path:
    video_path = Path(video_path)
    wav_path = Path(wav_path)

    # 1. Kiểm tra video đầu vào
    if not video_path.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy video: {video_path}"
        )

    if video_path.stat().st_size == 0:
        raise ValueError("Video đầu vào rỗng")

    # 2. Tạo thư mục lưu WAV
    wav_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # 3. Tách audio bằng FFmpeg
    command = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-vn",
        "-ac", "1",
        "-ar", str(config["sample_rate"]),
        "-c:a", "pcm_s16le",
        str(wav_path)
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg thất bại: {result.stderr}"
        )

    # 4. Kiểm tra WAV đầu ra
    info = sf.info(str(wav_path))

    if info.samplerate != config["sample_rate"]:
        raise ValueError("Sample rate không đúng")

    if info.channels != 1:
        raise ValueError("WAV không phải mono")

    if info.subtype != "PCM_16":
        raise ValueError("WAV không phải PCM 16-bit")

    if info.duration <= 0:
        raise ValueError("WAV có thời lượng không hợp lệ")

    # 5. Trả về đường dẫn WAV
    return wav_path



import numpy as np
import opensmile

# Khởi tạo một lần và tái sử dụng
_SMILE = None


def get_smile():
    global _SMILE

    if _SMILE is None:
        _SMILE = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.LowLevelDescriptors
        )

    return _SMILE


def extract_audio_features(wav_path, config) -> dict:
    wav_path = Path(wav_path)

    if not wav_path.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy WAV: {wav_path}"
        )

    info = sf.info(str(wav_path))

    if info.duration <= 0:
        raise ValueError("WAV rỗng hoặc không hợp lệ")

    if info.samplerate != config["sample_rate"]:
        raise ValueError("Sample rate không đúng")

    if info.channels != 1 or info.subtype != "PCM_16":
        raise ValueError("WAV không đúng định dạng chung")

    # Trích xuất đặc trưng
    smile = get_smile()
    df = smile.process_file(str(wav_path))

    features = df.to_numpy(dtype=np.float32)
    feature_names = df.columns.tolist()

    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError("Không trích xuất được chuỗi đặc trưng")

    # Timestamp tính bằng giây
    start_times = (
        df.index.get_level_values("start")
        .total_seconds()
        .to_numpy(dtype=np.float64)
    )

    end_times = (
        df.index.get_level_values("end")
        .total_seconds()
        .to_numpy(dtype=np.float64)
    )

    # Kiểm tra timestamp
    if (
        len(start_times) != len(features)
        or len(end_times) != len(features)
        or not np.all(np.isfinite(start_times))
        or not np.all(np.isfinite(end_times))
        or not np.all(start_times >= 0)
        or not np.all(end_times > start_times)
        or not np.all(np.diff(start_times) >= 0)
        or not np.all(np.diff(end_times) >= 0)
        or not np.all(end_times <= info.duration + 1e-4)
    ):
        raise ValueError("Timestamp audio không hợp lệ")

    # Đánh dấu các giá trị hữu hạn
    observed_mask = np.isfinite(features)

    # Quy tắc thử nghiệm trên pilot:
    # -201 là giá trị sàn ở ba đặc trưng biên độ
    # tương đối F0, chưa tự thay đổi dữ liệu gốc.
    floor_features = [
        "F1amplitudeLogRelF0_sma3nz",
        "F2amplitudeLogRelF0_sma3nz",
        "F3amplitudeLogRelF0_sma3nz"
    ]

    for name in floor_features:
        if name in feature_names:
            col_idx = feature_names.index(name)
            observed_mask[:, col_idx] &= (
                features[:, col_idx] != -201.0
            )

    return {
        "features": features,
        "start_times": start_times,
        "end_times": end_times,
        "feature_names": feature_names,
        "observed_mask": observed_mask
    }
