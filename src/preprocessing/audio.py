import numpy as np
import librosa
import json
import subprocess
from pathlib import Path


def probe_audio(video_path, timeout=30):
    """Kiểm tra thông tin luồng âm thanh, chưa giải mã tín hiệu."""
    video_path = Path(video_path)

    if not video_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file: {video_path}")

    if video_path.stat().st_size == 0:
        raise ValueError("File có kích thước 0 byte.")

    command = [
        "ffprobe", "-v", "error",
        "-select_streams", "a",
        "-show_entries", "stream=index,sample_rate,channels,start_time,duration",
        "-of", "json",
        str(video_path),
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True,
        timeout=timeout,
    )

    streams = json.loads(result.stdout).get("streams", [])

    if not streams:
        raise ValueError("MP4 không có luồng âm thanh.")

    first_stream = streams[0]
    sample_rate = int(first_stream.get("sample_rate", 0))
    channels = int(first_stream.get("channels", 0))

    if sample_rate <= 0 or channels <= 0:
        raise ValueError("Sample rate hoặc số kênh không hợp lệ.")

    return {
        "audio_tracks": len(streams),
        "source_sample_rate": sample_rate,
        "source_channels": channels,
        "stream_index": int(first_stream["index"]),
        "start_time": first_stream.get("start_time"),
        "duration": first_stream.get("duration"),
    }

    
def read_audio(video_path, sample_rate=16000, timeout=120):
    """Đọc luồng âm thanh đầu tiên thành tín hiệu mono float32."""
    if (
        not isinstance(sample_rate, int)
        or isinstance(sample_rate, bool)
        or sample_rate <= 0
    ):
        raise ValueError("sample_rate phải là số nguyên dương.")

    metadata = probe_audio(video_path)

    command = [
        "ffmpeg", "-nostdin", "-v", "error", "-xerror",
        "-i", str(video_path),
        "-map", "0:a:0", "-vn",
        "-ac", "1", "-ar", str(sample_rate),
        "-c:a", "pcm_f32le",
        "-f", "f32le", "pipe:1",
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        check=True,
        timeout=timeout,
    )

    samples = np.frombuffer(result.stdout, dtype="<f4").copy()

    if samples.size == 0:
        raise ValueError("Âm thanh sau giải mã không có mẫu tín hiệu.")

    if not np.isfinite(samples).all():
        raise ValueError("Âm thanh chứa NaN hoặc giá trị vô cực.")

    metadata.update({
        "sample_rate": sample_rate,
        "channels": 1,
        "num_samples": int(samples.size),
        "decoded_duration_sec": samples.size / sample_rate,
        "all_zero": bool((samples == 0).all()),
    })

    return samples, metadata



def extract_audio_features(samples, sample_rate=16000,
                           n_fft=400, hop_length=160, n_mels=40):
    """Tạo chuỗi log-Mel float32 dạng (thời gian, dải Mel)."""
    parameters = [
        ("sample_rate", sample_rate),
        ("n_fft", n_fft),
        ("hop_length", hop_length),
        ("n_mels", n_mels),
    ]

    for name, value in parameters:
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value <= 0
        ):
            raise ValueError(f"{name} phải là số nguyên dương.")

    if n_fft < 2 or hop_length > n_fft or n_mels > n_fft // 2 + 1:
        raise ValueError("Cấu hình cửa sổ, bước nhảy hoặc số dải Mel không hợp lệ.")

    samples = np.asarray(samples, dtype=np.float32)

    if samples.ndim != 1:
        raise ValueError("Tín hiệu phải là mảng mono một chiều.")

    if samples.size < n_fft:
        raise ValueError("Âm thanh ngắn hơn một cửa sổ phân tích.")

    if not np.isfinite(samples).all():
        raise ValueError("Tín hiệu chứa NaN hoặc giá trị vô cực.")

    mel = librosa.feature.melspectrogram(
        y=samples,
        sr=sample_rate,
        n_fft=n_fft,
        win_length=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        window="hann",
        center=False,
        power=2.0,
    )

    log_mel = librosa.power_to_db(
        mel,
        ref=1.0,
        amin=1e-10,
        top_db=None,
    )

    features = np.ascontiguousarray(log_mel.T, dtype=np.float32)

    if not np.isfinite(features).all():
        raise ValueError("Đặc trưng chứa NaN hoặc giá trị vô cực.")

    return features



def process_audio(video_id, clip_id, video_path, sample_rate=16000,
                  n_fft=400, hop_length=160, n_mels=40, timeout=120):
    """Xử lý một đoạn âm thanh và trả kết quả theo quy ước nhóm."""
    result = {
        "video_id": video_id,
        "clip_id": clip_id,
        "status": "error",
        "length": 0,
        "features": None,
        "error": "",
        "metadata": {},
    }

    try:
        samples, metadata = read_audio(
            video_path,
            sample_rate=sample_rate,
            timeout=timeout,
        )

        result["metadata"] = metadata

        features = extract_audio_features(
            samples,
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )

        metadata.update({
            "feature_type": "log_mel",
            "n_fft": n_fft,
            "hop_length": hop_length,
            "n_mels": n_mels,
            "frame_duration_sec": n_fft / sample_rate,
            "frame_step_sec": hop_length / sample_rate,
            "center": False,
            "log_ref": 1.0,
            "log_amin": 1e-10,
            "log_top_db": None,
        })

        result["features"] = features
        result["length"] = int(features.shape[0])
        result["status"] = "ok"

        if metadata["all_zero"]:
            result["status"] = "needs_review"
            result["error"] = (
                "Tín hiệu sau giải mã và chuyển mono toàn số 0."
            )

    except (
        OSError,
        ValueError,
        TypeError,
        subprocess.SubprocessError,
        librosa.util.exceptions.ParameterError,
    ) as error:
        details = getattr(error, "stderr", None)

        if isinstance(details, bytes):
            details = details.decode("utf-8", errors="replace")

        message = details.strip() if details else str(error)
        result["error"] = f"{type(error).__name__}: {message}"

    return result
