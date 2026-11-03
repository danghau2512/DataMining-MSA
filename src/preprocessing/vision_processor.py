
"""OpenFace temporal visual features. Importing has no side effects."""

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


def read_visual_csv(csv_path, feature_names):
    """Read OpenFace CSV; preserve frame order and missing values."""
    csv_path = Path(csv_path)

    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Missing OpenFace CSV: {csv_path}"
        )

    names = list(feature_names)

    if not names or any(
        not isinstance(name, str) for name in names
    ):
        raise ValueError(
            "feature_names must be a nonempty list of strings."
        )

    if len(names) != len(set(names)):
        raise ValueError("feature_names contains duplicates.")

    table = pd.read_csv(csv_path)
    table.columns = table.columns.str.strip()

    if table.columns.duplicated().any():
        raise ValueError(
            "OpenFace CSV contains duplicate column names."
        )

    if table.empty:
        raise ValueError(
            f"OpenFace CSV has no frames: {csv_path}"
        )

    required = [
        "frame", "timestamp", "success", "confidence"
    ] + names

    missing = [
        name for name in required
        if name not in table.columns
    ]

    if missing:
        raise ValueError(
            f"Missing OpenFace columns: {missing}"
        )

    numeric = table[required].apply(
        pd.to_numeric,
        errors="raise",
    )

    frame_values = numeric["frame"].to_numpy(
        dtype=np.float64
    )

    expected_frames = np.arange(
        1, len(table) + 1,
        dtype=np.int64,
    )

    if not np.array_equal(
        frame_values, expected_frames
    ):
        raise ValueError(
            "Frame numbers must be consecutive, from 1 to T."
        )

    timestamps = numeric["timestamp"].to_numpy(
        dtype=np.float64
    )

    if (
        not np.isfinite(timestamps).all()
        or (timestamps < 0).any()
        or (np.diff(timestamps) <= 0).any()
    ):
        raise ValueError(
            "Timestamps must be finite, nonnegative "
            "and increasing."
        )

    success_values = numeric["success"].to_numpy(
        dtype=np.float64
    )

    if not np.isin(success_values, [0, 1]).all():
        raise ValueError(
            "OpenFace success must be 0 or 1."
        )

    success = success_values.astype(np.uint8)

    confidence = numeric["confidence"].to_numpy(
        dtype=np.float32
    )

    if (
        not np.isfinite(confidence).all()
        or (confidence < 0).any()
        or (confidence > 1).any()
    ):
        raise ValueError(
            "OpenFace confidence must be finite "
            "and in [0, 1]."
        )

    features = numeric[names].to_numpy(
        dtype=np.float32,
        copy=True,
    )

    observed_mask = (
        np.isfinite(features)
        & (success[:, None] == 1)
    )

    return {
        "features": features,
        "timestamps": timestamps,
        "feature_names": names,
        "success": success,
        "confidence": confidence,
        "observed_mask": observed_mask,
        "frame_numbers": expected_frames,
        "csv_path": str(csv_path),
    }


def extract_visual_features(video_path, config):
    """Run OpenFace on one clip and return temporal features.

    Required config:
        openface_bin, model_path, output_dir, feature_names.

    Each call creates a separate folder inside output_dir.
    """
    video_path = Path(
        video_path
    ).expanduser().resolve()

    binary = Path(
        config["openface_bin"]
    ).expanduser().resolve()

    model_path = Path(
        config["model_path"]
    ).expanduser().resolve()

    for path in (video_path, binary, model_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing input/runtime file: {path}"
            )

    names = list(config["feature_names"])

    if (
        not names
        or any(
            not isinstance(name, str)
            for name in names
        )
        or len(names) != len(set(names))
    ):
        raise ValueError(
            "feature_names must contain unique string names."
        )

    if int(
        config.get("feature_dim", len(names))
    ) != len(names):
        raise ValueError(
            "feature_dim does not match feature_names."
        )

    output_root = Path(
        config["output_dir"]
    ).expanduser().resolve()

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    run_dir = Path(
        tempfile.mkdtemp(
            prefix="openface_",
            dir=output_root,
        )
    )

    log_path = run_dir / "openface.log"

    command = [
        str(binary),
        "-f", str(video_path),
        "-mloc", str(model_path),
        "-out_dir", str(run_dir),
        "-of", "features",
        "-aus", "-pose", "-gaze",
    ]

    (run_dir / "command.json").write_text(
        json.dumps(
            {
                "video_path": str(video_path),
                "command": command,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    try:
        with log_path.open(
            "w", encoding="utf-8"
        ) as log:
            process = subprocess.run(
                command,
                cwd=str(binary.parent),
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=config.get(
                    "timeout_seconds"
                ),
                check=False,
            )

    except (
        OSError,
        subprocess.TimeoutExpired,
    ) as error:
        raise RuntimeError(
            "OpenFace could not finish; "
            f"inspect {log_path}"
        ) from error

    if process.returncode != 0:
        raise RuntimeError(
            f"OpenFace exit code {process.returncode}; "
            f"inspect {log_path}"
        )

    result = read_visual_csv(
        run_dir / "features.csv",
        names,
    )

    result.update({
        "video_path": str(video_path),
        "log_path": str(log_path),
        "run_dir": str(run_dir),
    })

    return result
