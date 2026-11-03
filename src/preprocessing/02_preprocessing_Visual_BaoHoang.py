
# %% Imports — tập trung tại đầu file
import argparse
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import traceback
import types
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

try:
    import cv2
except ImportError:
    cv2 = None
try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(items, **kwargs):
        class Progress:
            def __iter__(self):
                return iter(items)
            def set_postfix(self, **values):
                print(values, flush=True)
        return Progress()
try:
    from IPython.display import display, Video
except ImportError:
    display = print
    Video = None
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
try:
    from google.colab import drive
except ImportError:
    drive = None

SOURCE_COMMIT = '3d4b5cf8d96138be42bed229447f36cbb09a5a29'
SOURCE_URL = 'https://github.com/TadasBaltrusaitis/OpenFace.git'
PROCESSOR_SHA256 = '300b86b4a706c0a246ef614e0f67d7074c99aff90e97f241e52354ff554919ad'
SAVED_SIGNATURE = '3c908e7632a119a9bf6a88d0b4c96d7f21465ed35a57619483811ff9addc17a0'
COMPLETED_RUN = '20261102T140801_205739Z'
AU_NAMES = [f'AU{i:02d}_r' for i in [1,2,4,5,6,7,9,10,12,14,15,17,20,23,25,26,45]]
POSE_NAMES = ['pose_Rx', 'pose_Ry', 'pose_Rz']
GAZE_NAMES = ['gaze_angle_x', 'gaze_angle_y']
FEATURE_NAMES = AU_NAMES + POSE_NAMES + GAZE_NAMES
MODEL_URLS = {
    'cen_patches_0.25_of.dat': 'https://www.dropbox.com/s/7na5qsjzz8yfoer/cen_patches_0.25_of.dat?dl=1',
    'cen_patches_0.35_of.dat': 'https://www.dropbox.com/s/k7bj804cyiu474t/cen_patches_0.35_of.dat?dl=1',
    'cen_patches_0.50_of.dat': 'https://www.dropbox.com/s/ixt4vkbmxgab1iu/cen_patches_0.50_of.dat?dl=1',
    'cen_patches_1.00_of.dat': 'https://www.dropbox.com/s/2t5t1sdpshzfhpj/cen_patches_1.00_of.dat?dl=1',
}

@dataclass(frozen=True)
class VisualPaths:
    data_root: Path = Path('/content/drive/MyDrive/CMU-MOSI')
    openface_root: Path = Path('/content/OpenFace')
    local_root: Path = Path('/content/mosi_visual')

    @property
    def visual(self):
        return self.data_root / 'preprocessing/visual'
    @property
    def shared(self):
        return self.data_root / 'preprocessing/shared'
    @property
    def manifest(self):
        return self.shared / 'manifest.csv'
    @property
    def raw_zip(self):
        return self.data_root / 'Raw.zip'
    @property
    def module(self):
        return self.data_root / 'src/preprocessing/vision_processor.py'
    @property
    def config(self):
        return self.visual / 'config/visual_feature_config.json'
    @property
    def binary(self):
        return self.openface_root / 'build/bin/FeatureExtraction'
    @property
    def model(self):
        return self.binary.parent / 'model/main_ceclm_general.txt'
    @property
    def backup(self):
        return self.visual / 'tools/OpenFace_colab.tar.gz'
    @property
    def full_root(self):
        return self.visual / 'full_visual_22' / SAVED_SIGNATURE[:12]


def file_sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.part')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def safe_child(root, relative):
    root = Path(root).resolve()
    candidate = (root / str(relative)).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(f'Đường dẫn nằm ngoài thư mục: {relative}')
    return candidate


def require_cv2():
    if cv2 is None:
        raise RuntimeError('Cần opencv-python cho chức năng đọc video; chế độ reuse không cần.')




# %% Task 2.1 — Kết nối Drive khi người dùng yêu cầu
# Mục đích: truy cập dữ liệu chung. Import file không tự mount Drive.
def task_2_1(mount=False):
    if mount:
        if drive is None:
            raise RuntimeError('Mount Drive chỉ dùng trong Google Colab.')
        drive.mount('/content/drive')
    print('Python:', platform.python_version(), '| NumPy:', np.__version__, '| pandas:', pd.__version__)




# %% Task 2.2 — Đường dẫn dữ liệu và công cụ
# Không tạo bản sao dữ liệu. Có thể thay data_root khi Hậu dùng tài khoản khác.
def task_2_2(paths=None):
    paths = paths or VisualPaths()
    display(pd.DataFrame([{'file': p.name, 'exists': p.is_file(), 'path': str(p)}
                          for p in [paths.manifest, paths.raw_zip, paths.config, paths.module, paths.backup]]))
    return paths




# %% Task 2.3 — Kiểm tra manifest, pilot và cảnh báo metadata
# Manifest ghép đúng văn bản, audio, visual và nhãn bằng ID phát ngôn.
# Giữ label hồi quy, nhãn 0, split và cả transcript giống nhau.
def task_2_3(paths, check_shared=True):
    manifest = pd.read_csv(paths.manifest, dtype={'id': str, 'video_id': str, 'clip_id': str}, keep_default_na=False)
    required = ['id','video_id','clip_id','split','raw_text','label','source_archive','video_relpath','video_size_bytes']
    if set(required) - set(manifest.columns):
        raise ValueError(f'Manifest thiếu cột: {set(required)-set(manifest.columns)}')
    label = pd.to_numeric(manifest.label, errors='coerce')
    size = pd.to_numeric(manifest.video_size_bytes, errors='coerce')
    strings = ['id','video_id','clip_id','split','raw_text','source_archive','video_relpath']
    checks = pd.Series({
        'empty_manifest': int(manifest.empty),
        'blank_required': int(manifest[strings].apply(lambda s: s.str.strip().eq('')).any(axis=1).sum()),
        'duplicate_id': int(manifest.id.duplicated().sum()),
        'wrong_id': int(manifest.id.ne(manifest.video_id + '$_$' + manifest.clip_id).sum()),
        'invalid_label': int((~np.isfinite(label) | ~label.between(-3,3)).sum()),
        'invalid_size': int((~np.isfinite(size) | size.le(0) | size.mod(1).ne(0)).sum()),
        'invalid_split': int((~manifest.split.isin(['train','valid','test'])).sum()),
        'video_cross_split': int(manifest.groupby('video_id').split.nunique().gt(1).sum()),
    }, name='count')
    for identifier in manifest.id:
        if identifier in {'', '.', '..'} or any(c in identifier for c in ['/', '\\', '\x00']):
            raise ValueError(f'ID không an toàn: {identifier!r}')
    if checks.gt(0).any():
        raise ValueError(f'Manifest chưa hợp lệ:\n{checks}')
    expected = {'train':1284, 'valid':229, 'test':686}
    actual = manifest.split.value_counts().to_dict()
    if actual != expected:
        raise ValueError(f'Split khác bản đã chốt: {actual}; không tự chia lại.')
    if not manifest.source_archive.eq('Raw.zip').all():
        raise ValueError('Manifest có nguồn khác Raw.zip.')
    if check_shared:
        pilot = task_2_4(paths, manifest)
        issues = pd.read_csv(paths.shared / 'metadata_issues.csv', dtype=str, keep_default_na=False)
        if not {'id','issue','detail'}.issubset(issues):
            raise ValueError('metadata_issues thiếu cột.')
        if (~issues.id.isin(manifest.id)).any() or issues[['id','issue','detail']].apply(lambda s: s.str.strip().eq('')).any().any():
            raise ValueError('metadata_issues có ID/thông tin không hợp lệ.')
        display(issues)
    display(checks.to_frame())
    display(manifest.groupby('split').agg(samples=('id','size'), videos=('video_id','nunique')))
    return manifest




# %% Task 2.4 — Đọc pilot chung, không tự bốc mẫu mới
# Giữ thứ tự và nội dung pilot do nhóm đã thống nhất.
def task_2_4(paths, manifest, clip_id=None):
    source = pd.read_csv(paths.shared / 'pilot_manifest.csv', dtype=str, keep_default_na=False)
    if 'id' not in source or source.empty or source.id.duplicated().any() or (~source.id.isin(manifest.id)).any():
        raise ValueError('Pilot rỗng, trùng ID hoặc có ID ngoài manifest.')
    pilot = manifest.set_index('id').loc[source.id].reset_index()
    for column in set(source.columns) & set(manifest.columns):
        a, b = source[column].reset_index(drop=True), pilot[column].reset_index(drop=True)
        equal = (pd.to_numeric(a, errors='coerce').eq(pd.to_numeric(b, errors='coerce'))
                 if column in {'label','video_size_bytes'} else a.astype(str).eq(b.astype(str)))
        if not equal.all():
            raise ValueError(f'Pilot khác manifest ở cột {column}')
    if clip_id is None:
        return pilot
    found = pilot[pilot.id.eq(clip_id)]
    if len(found) != 1:
        raise ValueError('ID không thuộc pilot chung.')
    return found.iloc[0]




# %% Task 2.5 — Trích đúng video từ ZIP, kiểm tra giải mã
# Giữ nguyên video và thời gian; không lấy mẫu frame để trích đặc trưng.
def task_2_5(paths, row):
    require_cv2()
    destination = safe_child(paths.local_root / 'media', f'{row.id}.mp4')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(paths.raw_zip) as archive:
        matches = [m for m in archive.infolist() if m.filename == row.video_relpath]
        if len(matches) != 1:
            raise ValueError('Video phải xuất hiện đúng một lần trong ZIP.')
        member = matches[0]
        if member.is_dir() or member.file_size != int(row.video_size_bytes):
            raise ValueError('Kích thước video không khớp manifest.')
        # Ghi lại từ ZIP để không dùng nhầm file local cũ cùng kích thước.
        temporary = destination.with_suffix('.mp4.part')
        with archive.open(member) as src, temporary.open('wb') as dst:
            shutil.copyfileobj(src, dst)
        temporary.replace(destination)
    cap = cv2.VideoCapture(str(destination))
    count = 0
    try:
        if not cap.isOpened():
            raise ValueError('Không mở được video.')
        while cap.read()[0]:
            count += 1
    finally:
        cap.release()
    if not count:
        raise ValueError('Không có frame đọc được.')
    return destination, count




# %% Task 2.6 — Kiểm tra môi trường; chưa chạy OpenFace
# Binary tồn tại chưa đủ để kết luận model/thư viện hoạt động.
def task_2_6(paths):
    report = {'platform':platform.platform(), 'binary_exists':paths.binary.is_file(),
              'model_exists':paths.model.is_file(), 'ffmpeg':shutil.which('ffmpeg')}
    display(report)
    return report




# %% Task 2.7 — Kiểm tra các công cụ Linux
# Đây là công cụ cài đặt/biên dịch, không phải mô hình sentiment.
def task_2_7():
    if platform.system() != 'Linux':
        raise RuntimeError('Chức năng cài OpenFace dành cho Colab/Linux.')
    report = {name: shutil.which(name) for name in ['apt-get','cmake','git','g++','wget','ffmpeg']}
    display(report)
    return report




# %% Task 2.8 — Cài thư viện hệ thống (chỉ gọi ở mode setup)
def task_2_8():
    task_2_7()
    packages = ['build-essential','cmake','git','wget','pkg-config','libopencv-dev',
                'libdlib-dev','libopenblas-dev','libboost-filesystem-dev','libboost-system-dev','ffmpeg']
    for command in [['apt-get','update','-qq'],
                    ['apt-get','install','-y','--no-install-recommends',*packages]]:
        print('Đang chạy:', ' '.join(command), flush=True)
        subprocess.run(command, check=True, env={**os.environ, 'DEBIAN_FRONTEND':'noninteractive'})




# %% Task 2.9 — Build đúng commit; tối đa 2 tiến trình để hạn chế RAM
# Build lại có thể đổi binary/hash; Task 2.22 sẽ không trộn cache khác cấu hình.
def task_2_9(paths):
    root = paths.openface_root
    if not root.exists():
        subprocess.run(['git','init',str(root)], check=True)
        subprocess.run(['git','-C',str(root),'remote','add','origin',SOURCE_URL], check=True)
    if not (root / '.git').is_dir():
        raise RuntimeError('OpenFace root có sẵn nhưng không phải source Git; ưu tiên restore.')
    subprocess.run(['git','-C',str(root),'fetch','--depth','1','origin',SOURCE_COMMIT], check=True)
    subprocess.run(['git','-C',str(root),'checkout','--detach',SOURCE_COMMIT], check=True)
    for command in [['cmake','-S',str(root),'-B',str(root/'build'),'-DCMAKE_BUILD_TYPE=Release'],
                    ['cmake','--build',str(root/'build'),'--target','FeatureExtraction','--parallel','2']]:
        subprocess.run(command, check=True)
    if not paths.binary.is_file():
        raise RuntimeError('Build chưa tạo FeatureExtraction.')
    return paths.binary




# %% Task 2.10 — Khôi phục hoặc lưu bộ công cụ đã có model
# Không ghi đè bản sao đã có. Restore ưu tiên byte gốc để tiếp tục đúng cache.
def task_2_10(paths, action='restore'):
    if action == 'restore':
        if paths.binary.is_file() and paths.model.is_file():
            return paths.binary
        if not paths.backup.is_file():
            raise FileNotFoundError(paths.backup)
        if paths.openface_root.name != 'OpenFace':
            raise ValueError('Archive cần openface_root có tên OpenFace.')
        with tarfile.open(paths.backup, 'r:gz') as archive:
            for member in archive.getmembers():
                relative = PurePosixPath(member.name)
                if relative.is_absolute() or '..' in relative.parts or not relative.parts or relative.parts[0] != 'OpenFace':
                    raise ValueError(f'Đường dẫn archive không hợp lệ: {member.name}')
            archive.extractall(paths.openface_root.parent, filter='data')
    elif action == 'backup':
        task_2_11(paths, download=False)
        if paths.backup.exists():
            print('Giữ bản sao đã có:', paths.backup)
            return paths.backup
        paths.backup.parent.mkdir(parents=True, exist_ok=True)
        temporary = paths.backup.with_suffix('.gz.part')
        with tarfile.open(temporary, 'w:gz') as archive:
            archive.add(paths.binary.parent, arcname='OpenFace/build/bin')
        temporary.replace(paths.backup)
        return paths.backup
    else:
        raise ValueError('action phải là restore hoặc backup.')
    task_2_11(paths, download=False)
    return paths.binary




# %% Task 2.11 — Bốn model CE-CLM chính thức
# Đây là bốn mức scale của detector, không phải bốn frame.
# SHA của toàn bộ runtime được đối chiếu với provenance cũ trước Task 2.22.
def task_2_11(paths, download=False):
    if not paths.binary.is_file() or not paths.model.is_file():
        raise FileNotFoundError('Thiếu FeatureExtraction hoặc main_ceclm_general.txt.')
    folder = paths.binary.parent / 'model/patch_experts'
    for name, url in MODEL_URLS.items():
        target = folder / name
        if not target.is_file() or target.stat().st_size == 0:
            if not download:
                raise FileNotFoundError(target)
            folder.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix('.dat.part')
            subprocess.run(['wget','--timeout=60','--tries=2','--no-verbose','-O',str(temporary),url], check=True)
            with temporary.open('rb') as stream:
                header = stream.read(512).lstrip().lower()
            if not temporary.stat().st_size or header.startswith((b'<!doctype html',b'<html',b'<?xml')):
                raise ValueError(f'Model không hợp lệ: {name}')
            temporary.replace(target)
    check = subprocess.run(['ldd',str(paths.binary)], capture_output=True, text=True, check=False)
    if check.returncode or 'not found' in check.stdout:
        raise RuntimeError(f'OpenFace thiếu thư viện: {check.stdout}\n{check.stderr}')
    return paths.model



# %% Task 2.12 — Chạy OpenFace trên một clip pilot
# Dùng toàn bộ frame và giữ success/confidence/timestamp; chưa loại hay điền frame.
def task_2_12(paths, row, output_dir=None):
    video_path, decoded = task_2_5(paths, row)
    task_2_11(paths)
    processor = task_2_20(paths)
    config = dict(task_2_18(paths))
    config.update(openface_bin=str(paths.binary.resolve()), model_path=str(paths.model.resolve()),
                  output_dir=str(output_dir or (paths.visual / 'pilot' / row.id)))
    result = processor.extract_visual_features(video_path, config)
    if len(result['features']) != decoded:
        raise ValueError('Số frame CSV không khớp số frame giải mã.')
    display(task_2_14(result))
    return result




# %% Task 2.13 — Gợi ý Git để đưa mã lên nhánh riêng
# Không chứa token, không tự commit/push và không đưa dữ liệu lớn lên GitHub.
def task_2_13():
    print('Trong checkout DataMining-MSA, đặt file này tại src/preprocessing/.')
    print('git switch feat/preprocess-visual')
    print('git add src/preprocessing/02_preprocessing_Visual_BaoHoang.py')
    print('git diff --cached --stat')
    print('git commit -m "Consolidate visual preprocessing tasks 2.1-2.22"')
    print('git push origin feat/preprocess-visual')
    print('Hậu pull nhánh này; Drive cần được chia sẻ riêng để đọc NPZ và báo cáo.')




# %% Task 2.14 — Thống kê chất lượng frame
# success=1 là trạng thái detector; không chứng minh đó là đúng người nói.
# confidence được giữ để phân tích; không tự đặt ngưỡng loại frame mới.
def task_2_14(result):
    x = result['features']
    mask = result['observed_mask']
    failed = int((result['success'] == 0).sum())
    missing = int((~mask).sum())
    return {'T_vision':len(x), 'D_vision':x.shape[1], 'failed_frames':failed,
            'unobserved_values':missing, 'min_confidence':float(result['confidence'].min()),
            'quality_flag':'needs_review' if failed or missing else 'ok'}




# %% Task 2.15 — Chạy các clip trong pilot chung
# Số clip do pilot_manifest quyết định, không tự quy định lại là 10.
def task_2_15(paths, manifest):
    pilot = task_2_4(paths, manifest)
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    root = paths.visual / 'pilot_batches' / run_id
    root.mkdir(parents=True, exist_ok=False)
    records = []
    try:
        for row in pilot.itertuples(index=False):
            try:
                result = task_2_12(paths, row, root / 'openface' / row.id)
                record = {'id':row.id, 'status':'extracted', 'csv_file':str(result['csv_path']),
                          **task_2_14(result)}
            except Exception as error:
                record = {'id':row.id, 'status':'error', 'error':f'{type(error).__name__}: {error}'}
            records.append(record)
            with (root / 'progress.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + '\n')
    finally:
        pd.DataFrame(records).to_csv(root / 'pilot_quality_summary.csv', index=False)
    return pd.DataFrame(records)




# %% Task 2.16 — Xem frame để kiểm chứng trực quan
# Đầu/giữa/cuối chỉ giúp xem độ bao phủ thời gian, không đại diện mọi frame.
# Có thể truyền chỉ số frame 1-based cụ thể; không thay dữ liệu trích xuất.
def task_2_16(video_path, frame_numbers=None):
    require_cv2()
    cap = cv2.VideoCapture(str(video_path))
    count = 0
    try:
        if not cap.isOpened():
            raise ValueError('Không mở được video.')
        while cap.read()[0]:
            count += 1
    finally:
        cap.release()
    if not count:
        raise ValueError('Video không có frame.')
    selected = frame_numbers or sorted(set([1,(count+1)//2,count]))
    if any(not isinstance(i, int) or i < 1 or i > count for i in selected):
        raise ValueError('Chỉ số frame phải là số nguyên trong [1, T].')
    selected = sorted(set(selected))
    result = []
    cap = cv2.VideoCapture(str(video_path))
    try:
        for index in range(1, max(selected)+1):
            ok, frame = cap.read()
            if not ok:
                raise ValueError('Video lỗi khi đọc lại các frame minh họa.')
            if index in selected:
                result.append({'frame':index, 'rgb':cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)})
    finally:
        cap.release()
    if plt is not None:
        fig, axes = plt.subplots(1, len(result), figsize=(5*len(result), 4), squeeze=False)
        for axis, item in zip(axes[0], result):
            axis.imshow(item['rgb'])
            axis.set_title(f"Frame {item['frame']}", parse_math=False)
            axis.axis('off')
        fig.tight_layout()
        plt.show()
        plt.close(fig)
    return result




# %% Task 2.17 — Ghi nhận kiểm tra bằng mắt, không tự gán kết luận
# Quan sát: mặt đúng người nói, che khuất, nhiều mặt, chuyển cảnh hoặc lệch điểm.
# Chỉ tạo bảng nếu người dùng gọi hàm; không tự loại clip needs_review.
def task_2_17(ids, output_path=None):
    review = pd.DataFrame({'id':list(ids), 'speaker_visible':'', 'occlusion_or_multiple_faces':'',
                           'tracking_notes':'', 'review_decision':'not_reviewed'})
    if output_path is not None:
        path = Path(output_path)
        if path.exists():
            raise FileExistsError(f'Giữ ghi chú đã có: {path}')
        path.parent.mkdir(parents=True, exist_ok=True)
        review.to_csv(path, index=False)
    return review




# %% Task 2.18 — Chốt 22 chiều và thứ tự cột
# 17 AU intensity: cường độ hoạt động cơ mặt; 3 pose_R*: góc xoay đầu;
# 2 gaze_angle_*: góc nhìn. Không dùng tọa độ đầu tịnh tiến trong cấu hình này.
# 22 = 17+3+2 là lựa chọn của nhóm dựa trên các đầu ra OpenFace được tài liệu hóa.
# Không gọi đây là bộ 22 chiều bắt buộc của tác giả OpenFace hay MAG-BERT.
# Nguồn: https://github.com/TadasBaltrusaitis/OpenFace/wiki/Output-Format
# MAG-BERT gốc dùng Facet; bộ OpenFace này là thay đổi của đồ án.
def task_2_18(paths, create=False):
    default = {'schema_version':'1.0', 'extractor':'OpenFace', 'feature_dim':22,
               'feature_names':FEATURE_NAMES,
               'feature_groups':{'au_intensity':AU_NAMES,'head_rotation':POSE_NAMES,'gaze_angles':GAZE_NAMES},
               'dtype':'float32', 'timestamp_unit':'seconds',
               'metadata_columns':['frame','timestamp','success','confidence'],
               'selection_basis':'Project configuration using documented OpenFace outputs: 17 AU intensities, 3 head rotation angles, 2 gaze angles.',
               'reference_url':'https://github.com/TadasBaltrusaitis/OpenFace/wiki/Output-Format'}
    if paths.config.is_file():
        config = json.loads(paths.config.read_text(encoding='utf-8'))
    elif create:
        write_json(paths.config, default)
        config = default
    else:
        raise FileNotFoundError(paths.config)
    if config.get('feature_dim') != 22 or config.get('feature_names') != FEATURE_NAMES:
        raise ValueError('Cấu hình không khớp tên/thứ tự 22 chiều đã chốt.')
    return config




# %% Task 2.19 — Đọc CSV và lưu pilot 22 chiều theo thời gian
# observed_mask = finite(features) AND success=1, tính riêng cho từng giá trị.
# Giữ frame hỏng/giá trị thiếu và mask; chưa impute/normalize/align theo từ.
def task_2_19(paths, csv_path, output_path=None):
    config = task_2_18(paths)
    result = task_2_20(paths).read_visual_csv(csv_path, config['feature_names'])
    if output_path is not None:
        output = Path(output_path)
        if output.exists():
            raise FileExistsError(f'Không ghi đè pilot: {output}')
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.with_suffix(output.suffix + '.part').open('wb') as stream:
            np.savez_compressed(stream, **{k:np.asarray(result[k]) for k in
                ['features','timestamps','feature_names','success','confidence','observed_mask','frame_numbers']})
        output.with_suffix(output.suffix + '.part').replace(output)
    return result




# %% Task 2.20 — Module dùng chung có hợp đồng đầu ra cố định
# Mã module được nhúng nguyên byte vào cuối file để chỉ cần bàn giao một .py.
# Không sửa module đã có trên Drive. install=True chỉ tạo khi chưa có.
def task_2_20(paths=None, install=False):
    if hashlib.sha256(PROCESSOR_SOURCE.encode('utf-8')).hexdigest() != PROCESSOR_SHA256:
        raise ValueError('Mã module nhúng không khớp SHA256 đã xác nhận.')
    if paths is not None and install:
        if paths.module.exists():
            if file_sha256(paths.module) != PROCESSOR_SHA256:
                raise ValueError('Module trên Drive khác phiên bản; không ghi đè.')
        else:
            paths.module.parent.mkdir(parents=True, exist_ok=True)
            paths.module.write_bytes(PROCESSOR_SOURCE.encode('utf-8'))
    module = types.ModuleType('vision_processor_embedded')
    exec(compile(PROCESSOR_SOURCE, '<verified_vision_processor>', 'exec'), module.__dict__)
    return module




# %% Task 2.21 — Đối chiếu module, cấu hình và toàn bộ runtime
# Signature là hash của recipe, không phải số ngẫu nhiên.
# Chỉ tiếp tục cache cũ khi nội dung bộ trích xuất thực sự khớp.
def task_2_21(paths):
    task_2_18(paths)
    if not paths.module.is_file() or file_sha256(paths.module) != PROCESSOR_SHA256:
        raise ValueError('Module Drive thiếu/khác bản đã xác nhận. Chạy setup hoặc kiểm tra nguồn.')
    task_2_11(paths)
    runtime = {str(p.relative_to(paths.binary.parent)):file_sha256(p)
               for p in sorted(paths.binary.parent.rglob('*')) if p.is_file()}
    recipe = {'schema_version':'visual22-full-v1', 'feature_names':FEATURE_NAMES,
              'processor_sha256':PROCESSOR_SHA256, 'openface_runtime_sha256':runtime}
    signature = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode('utf-8')).hexdigest()
    saved = paths.full_root / 'extraction_config.json'
    if saved.is_file():
        provenance = json.loads(saved.read_text(encoding='utf-8'))
        if provenance.get('signature') != signature or provenance.get('recipe') != recipe:
            raise ValueError('Runtime không khớp provenance cũ; không trộn hoặc chạy lại toàn bộ.')
    if signature != SAVED_SIGNATURE:
        raise ValueError(f'Signature khác kết quả 2.199 clip: {signature}. Cần đối chiếu trước.')
    print('Khớp module/runtime:', signature, '| runtime files:', len(runtime))
    return recipe




# %% Task 2.22 — Đọc kết quả hoàn tất hoặc tiếp tục xử lý có kiểm tra cache
# Mặc định action=reuse: không cần OpenFace, không gọi lại bộ trích xuất.
# action=extract: kiểm tra từng NPZ; file hợp lệ cached, file thiếu/hỏng mới trích.
# Dùng journal từng clip và ghi report trong finally; không chia lại train/valid/test.
def task_2_22(paths=None, action='reuse', verify=False, run_id=COMPLETED_RUN):
    paths = paths or VisualPaths()
    if action == 'reuse':
        return read_saved_run(paths, run_id=run_id, verify=verify)
    if action != 'extract':
        raise ValueError('action phải là reuse hoặc extract.')
    require_cv2()
    task_2_3(paths)
    task_2_20(paths, install=True)
    task_2_21(paths)
    return _run_full_extraction(paths)


def load_saved_visual(clip_id, paths=None):
    """Đọc một NPZ có kiểm tra schema/hash/mask, không chạy OpenFace."""
    paths = paths or VisualPaths()
    manifest = pd.read_csv(paths.manifest, dtype={'id':str}, keep_default_na=False)
    rows = manifest[manifest.id.eq(clip_id)]
    if len(rows) != 1:
        raise ValueError('ID phải xuất hiện đúng một lần trong manifest.')
    row = next(rows.itertuples(index=False))
    file = safe_child(paths.full_root / 'features', f'{clip_id}.npz')
    # Không cần Raw.zip khi đọc lại; CRC lấy từ NPZ, source size/member đối chiếu manifest.
    # Muốn đối chiếu CRC với ZIP, dùng kiểm tra cache trong mode extract.
    with np.load(file, allow_pickle=False) as saved:
        member = types.SimpleNamespace(CRC=saved['source_crc32'].item(), file_size=int(row.video_size_bytes))
    inspect_npz(file, row, member)
    with np.load(file, allow_pickle=False) as saved:
        return {key:saved[key].copy() for key in saved.files}


def read_saved_run(paths, run_id=COMPLETED_RUN, verify=False):
    """Đọc provenance/report, đối chiếu ID và file; --verify mở từng NPZ."""
    root = paths.full_root
    provenance = json.loads((root / 'extraction_config.json').read_text(encoding='utf-8'))
    if provenance.get('signature') != SAVED_SIGNATURE:
        raise ValueError('Provenance khác signature đã chốt.')
    recipe = provenance.get('recipe', {})
    calculated = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode('utf-8')).hexdigest()
    if calculated != SAVED_SIGNATURE or recipe.get('feature_names') != FEATURE_NAMES or recipe.get('processor_sha256') != PROCESSOR_SHA256:
        raise ValueError('Nội dung provenance không khớp hash/đặc trưng/module.')
    if run_id in {'', '.', '..'} or any(c in run_id for c in ['/', '\\']):
        raise ValueError('run_id không hợp lệ.')
    report_path = root / 'runs' / run_id / 'visual_manifest.csv'
    manifest = task_2_3(paths, check_shared=False)
    report = pd.read_csv(report_path, dtype={'id':str}, keep_default_na=False)
    if not {'id','split','status','quality_flag','feature_file'}.issubset(report):
        raise ValueError('Báo cáo thiếu cột bắt buộc.')
    if report.id.duplicated().any() or set(report.id) != set(manifest.id):
        raise ValueError('Report không khớp toàn bộ ID manifest.')
    check_columns = [c for c in ['split','video_id','clip_id','raw_text','label','source_archive','video_relpath','video_size_bytes'] if c in report]
    joined = manifest.merge(report, on='id', suffixes=('_source','_report'), validate='one_to_one')
    for column in check_columns:
        a, b = joined[column+'_source'], joined[column+'_report']
        same = pd.to_numeric(a).eq(pd.to_numeric(b)) if column in {'label','video_size_bytes'} else a.astype(str).eq(b.astype(str))
        if not same.all():
            raise ValueError(f'Report khác manifest ở {column}.')
    if not report.status.isin(['cached','extracted']).all():
        raise ValueError('Report còn error/pending; chưa bàn giao như bộ hoàn tất.')
    feature_paths = {}
    for row in report.itertuples(index=False):
        candidate = safe_child(root, row.feature_file)
        expected = safe_child(root / 'features', f'{row.id}.npz')
        if candidate != expected or not candidate.is_file():
            raise FileNotFoundError(f'Đường dẫn/file đặc trưng không hợp lệ: {row.id}')
        feature_paths[row.id] = candidate
    if verify:
        source_rows = {row.id:row for row in manifest.itertuples(index=False)}
        actual_stats = []
        report_by_id = report.set_index('id')
        for identifier, candidate in tqdm(feature_paths.items(), total=len(feature_paths), desc='Verify saved NPZ'):
            row = source_rows[identifier]
            with np.load(candidate, allow_pickle=False) as saved:
                member = types.SimpleNamespace(CRC=saved['source_crc32'].item(), file_size=int(row.video_size_bytes))
            stats = inspect_npz(candidate, row, member)
            reported = report_by_id.loc[identifier]
            for key in ['T_vision','D_vision','failed_frames','unobserved_values','quality_flag']:
                if key in report and stats[key] != reported[key]:
                    raise ValueError(f'NPZ khác report: {identifier}, {key}')
            actual_stats.append({'id':identifier, **stats})
        print('Đã kiểm tra payload của', len(actual_stats), 'NPZ. Chưa đối chiếu CRC lại với ZIP gốc.')
    display(report.groupby(['split','status']).size().unstack(fill_value=0))
    display(report.quality_flag.value_counts())
    print('Report:', report_path)
    print('Không chạy lại OpenFace. needs_review vẫn được giữ, chưa loại hoặc điền thiếu.')
    return {'manifest':manifest, 'report':report, 'feature_paths':feature_paths,
            'feature_names':FEATURE_NAMES.copy(), 'full_root':root, 'report_path':report_path}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--mode', choices=['reuse','setup','pilot','extract'], default='reuse')
    parser.add_argument('--data-root', type=Path, default=VisualPaths().data_root)
    parser.add_argument('--openface-root', type=Path, default=VisualPaths().openface_root)
    parser.add_argument('--local-root', type=Path, default=VisualPaths().local_root)
    parser.add_argument('--mount', action='store_true', help='Chỉ dùng trên Colab')
    parser.add_argument('--verify', action='store_true', help='Mở từng NPZ và kiểm tra schema/mask')
    parser.add_argument('--run-id', default=COMPLETED_RUN, help='ID lần chạy report cần đọc')
    parser.add_argument('--build', action='store_true', help='Cho phép build nếu không có backup (mode setup)')
    args = parser.parse_args(argv)
    paths = VisualPaths(args.data_root, args.openface_root, args.local_root)
    task_2_1(mount=args.mount)
    task_2_2(paths)
    if args.mode == 'reuse':
        return task_2_22(paths, verify=args.verify, run_id=args.run_id)
    if args.mode == 'setup':
        task_2_8()
        if paths.backup.is_file() or paths.binary.is_file():
            task_2_10(paths, action='restore')
        elif args.build:
            task_2_9(paths)
            task_2_11(paths, download=True)
        else:
            raise FileNotFoundError('Thiếu backup. Chỉ build khi chủ động thêm --build.')
        task_2_18(paths, create=True)
        task_2_20(paths, install=True)
        task_2_21(paths)
        task_2_10(paths, action='backup')
        return
    if args.mode == 'pilot':
        return task_2_15(paths, task_2_3(paths))
    return task_2_22(paths, action='extract')


# %% Implementation retained from the saved Task 2.22 notebook
def inspect_npz(path, row, member):
    with np.load(path, allow_pickle=False) as saved:
        expected = {
            "id": row.id,
            "signature": SAVED_SIGNATURE,
            "source_member": row.video_relpath,
            "source_crc32": member.CRC,
            "source_size": member.file_size,
        }
        for key, value in expected.items():
            if saved[key].item() != value:
                raise ValueError(f"File đã lưu không khớp {key}.")
        if saved["feature_names"].tolist() != FEATURE_NAMES:
            raise ValueError("Tên/thứ tự đặc trưng không khớp.")
        x = saved["features"]
        if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] != 22:
            raise ValueError("Shape features không hợp lệ.")
        if x.dtype != np.float32:
            raise ValueError("features không phải float32.")
        T = x.shape[0]
        ts = saved["timestamps"]
        success = saved["success"]
        confidence = saved["confidence"]
        mask = saved["observed_mask"]
        for name, array, dtype in [
            ("timestamps", ts, np.float64),
            ("success", success, np.uint8),
            ("confidence", confidence, np.float32),
        ]:
            if array.shape != (T,) or array.dtype != dtype:
                raise ValueError(f"Shape/dtype của {name} không đúng.")
        if (
            not np.isfinite(ts).all()
            or (ts < 0).any()
            or (np.diff(ts) <= 0).any()
        ):
            raise ValueError("Timestamp không hợp lệ.")
        if not np.isin(success, [0, 1]).all():
            raise ValueError("success không hợp lệ.")
        if (
            not np.isfinite(confidence).all()
            or (confidence < 0).any()
            or (confidence > 1).any()
        ):
            raise ValueError("confidence không hợp lệ.")
        if (
            mask.shape != x.shape
            or mask.dtype != np.bool_
            or not np.array_equal(
                mask, np.isfinite(x) & (success[:, None] == 1)
            )
        ):
            raise ValueError("observed_mask không hợp lệ.")
        if not np.array_equal(
            saved["frame_numbers"], np.arange(1, T + 1)
        ):
            raise ValueError("Thứ tự frame không hợp lệ.")
        if saved["decoded_frames"].item() != T:
            raise ValueError("Số frame video không khớp.")

        missing_values = int((~mask).sum())
        failed_frames = int((success == 0).sum())
        return {
            "T_vision": T,
            "D_vision": 22,
            "failed_frames": failed_frames,
            "unobserved_values": missing_values,
            "min_confidence": float(confidence.min()),
            "quality_flag": (
                "needs_review" if failed_frames or missing_values else "ok"
            ),
            "csv_file": saved["csv_file"].item(),
            "log_file": saved["log_file"].item(),
        }

PROCESSOR_SOURCE = r'''
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
'''

TASK22_SETUP_SOURCE = r'''


DATA_ROOT = paths.data_root
VISUAL_ROOT = DATA_ROOT / "preprocessing" / "visual"
MANIFEST_PATH = DATA_ROOT / "preprocessing" / "shared" / "manifest.csv"
RAW_ZIP = DATA_ROOT / "Raw.zip"
MODULE_PATH = DATA_ROOT / "src" / "preprocessing" / "vision_processor.py"
CONFIG_PATH = VISUAL_ROOT / "config" / "visual_feature_config.json"
OPENFACE_BIN = paths.binary
MODEL_PATH = paths.model

for required_path in [
    MANIFEST_PATH, RAW_ZIP, MODULE_PATH, CONFIG_PATH, OPENFACE_BIN, MODEL_PATH,
]:
    if not required_path.is_file():
        raise FileNotFoundError(f"Thiếu file: {required_path}")

full_manifest = pd.read_csv(
    MANIFEST_PATH, dtype={"id": str, "video_id": str, "clip_id": str}, keep_default_na=False
)
required_columns = {
    "id", "split", "source_archive", "video_relpath", "video_size_bytes",
}
if not required_columns.issubset(full_manifest.columns):
    raise ValueError("Manifest thiếu cột cần thiết.")
if len(full_manifest) != 2199:
    raise ValueError("Cần manifest đầy đủ 2.199 clip đã thống nhất.")
if full_manifest[list(required_columns)].isna().any().any():
    raise ValueError("Manifest có thông tin đầu vào bị thiếu.")
if full_manifest["id"].duplicated().any():
    raise ValueError("Manifest có ID trùng.")
if not full_manifest["source_archive"].eq(RAW_ZIP.name).all():
    raise ValueError("Manifest có nguồn archive khác Raw.zip.")
if not full_manifest["split"].isin(["train", "valid", "test"]).all():
    raise ValueError("Manifest có split ngoài train/valid/test.")
for clip_id in full_manifest["id"]:
    if clip_id in {"", ".", ".."} or any(
        char in clip_id for char in ["/", "\\", "\x00"]
    ):
        raise ValueError(f"ID không an toàn để tạo tên file: {clip_id!r}")

feature_config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
feature_names_full = feature_config["feature_names"]
if (
    feature_config["feature_dim"] != 22
    or len(feature_names_full) != 22
    or len(set(feature_names_full)) != 22
):
    raise ValueError("Cấu hình không đúng 22 chiều.")

spec_full = importlib.util.spec_from_file_location(
    "vision_processor_full", MODULE_PATH
)
processor_full = importlib.util.module_from_spec(spec_full)
spec_full.loader.exec_module(processor_full)


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


print("Đang nhận diện lại module và bộ OpenFace...", flush=True)
runtime_files = {
    str(path.relative_to(OPENFACE_BIN.parent)): file_sha256(path)
    for path in sorted(OPENFACE_BIN.parent.rglob("*"))
    if path.is_file()
}
# Giữ đúng công thức tạo mã của Task 2.22 trước đó.
recipe = {
    "schema_version": "visual22-full-v1",
    "feature_names": feature_names_full,
    "processor_sha256": file_sha256(MODULE_PATH),
    "openface_runtime_sha256": runtime_files,
}
recipe_signature = hashlib.sha256(
    json.dumps(recipe, sort_keys=True).encode("utf-8")
).hexdigest()

FULL_ROOT = VISUAL_ROOT / "full_visual_22" / recipe_signature[:12]
FEATURE_DIR = FULL_ROOT / "features"
current_count = len(list(FEATURE_DIR.glob("*.npz")))
other_results = []
parent_root = FULL_ROOT.parent
if parent_root.is_dir():
    for folder in parent_root.iterdir():
        if folder.is_dir() and folder.name != FULL_ROOT.name:
            count = len(list((folder / "features").glob("*.npz")))
            if count:
                other_results.append({"folder": folder.name, "npz_files": count})

print("Mã cấu hình hiện tại:", recipe_signature[:12])
print("Số file cùng cấu hình đang có:", current_count)
if current_count == 0 and other_results:
    display(pd.DataFrame(other_results))
    raise RuntimeError(
        "Đã có kết quả ở mã cấu hình khác. "
        "Dừng để kiểm tra bộ OpenFace/module, không tự chạy lại toàn bộ. "
        "Gửi bảng này để xác định nguyên nhân."
    )

LOCAL_VIDEO_DIR = paths.local_root / "full/videos"
run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
RUN_DIR = FULL_ROOT / "runs" / run_id
for folder in [FEATURE_DIR, LOCAL_VIDEO_DIR, RUN_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

provenance_path = FULL_ROOT / "extraction_config.json"
if provenance_path.is_file():
    previous_config = json.loads(provenance_path.read_text(encoding="utf-8"))
    if previous_config["signature"] != recipe_signature:
        raise ValueError("Mã thư mục trùng nhưng cấu hình khác.")
else:
    provenance_path.write_text(
        json.dumps({
            "signature": recipe_signature,
            "recipe": recipe,
            "feature_config": feature_config,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

(RUN_DIR / "environment.json").write_text(
    json.dumps({
        "python": sys.version,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "opencv": cv2.__version__,
        "manifest_sha256": file_sha256(MANIFEST_PATH),
    }, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
full_manifest.to_csv(
    RUN_DIR / "source_manifest.csv", index=False, encoding="utf-8"
)
print("Số clip:", len(full_manifest))
print(full_manifest["split"].value_counts().to_string())
print("Thư mục kết quả:", FULL_ROOT)
print("Sẵn sàng chạy cell tiếp theo: file hợp lệ sẽ được ghi cached.")
'''

TASK22_BATCH_SOURCE = r'''def decoded_frame_count(video_path):
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise ValueError("Không mở được video.")
        count = 0
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            count += 1
        if count == 0:
            raise ValueError("Video không có frame đọc được.")
        return count
    finally:
        cap.release()


def feature_payload(result, row, member, decoded_frames):
    if result["features"].shape[0] != decoded_frames:
        raise ValueError("Số frame OpenFace khác số frame đọc từ video.")
    return {
        "id": np.asarray(row.id),
        "features": result["features"],
        "timestamps": result["timestamps"],
        "feature_names": np.asarray(result["feature_names"], dtype=np.str_),
        "success": result["success"],
        "confidence": result["confidence"],
        "observed_mask": result["observed_mask"],
        "frame_numbers": result["frame_numbers"],
        "decoded_frames": np.asarray(decoded_frames, dtype=np.int64),
        "signature": np.asarray(recipe_signature),
        "source_member": np.asarray(row.video_relpath),
        "source_crc32": np.asarray(member.CRC, dtype=np.uint32),
        "source_size": np.asarray(member.file_size, dtype=np.int64),
        "csv_file": np.asarray(str(
            Path(result["csv_path"]).resolve().relative_to(FULL_ROOT.resolve())
        )),
        "log_file": np.asarray(str(
            Path(result["log_path"]).resolve().relative_to(FULL_ROOT.resolve())
        )),
    }


def inspect_saved_npz(path, row, member):
    with np.load(path, allow_pickle=False) as saved:
        expected = {
            "id": row.id,
            "signature": recipe_signature,
            "source_member": row.video_relpath,
            "source_crc32": member.CRC,
            "source_size": member.file_size,
        }
        for key, value in expected.items():
            if saved[key].item() != value:
                raise ValueError(f"File đã lưu không khớp {key}.")
        if saved["feature_names"].tolist() != feature_names_full:
            raise ValueError("Tên/thứ tự đặc trưng không khớp.")
        x = saved["features"]
        if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] != 22:
            raise ValueError("Shape features không hợp lệ.")
        if x.dtype != np.float32:
            raise ValueError("features không phải float32.")
        T = x.shape[0]
        ts = saved["timestamps"]
        success = saved["success"]
        confidence = saved["confidence"]
        mask = saved["observed_mask"]
        for name, array, dtype in [
            ("timestamps", ts, np.float64),
            ("success", success, np.uint8),
            ("confidence", confidence, np.float32),
        ]:
            if array.shape != (T,) or array.dtype != dtype:
                raise ValueError(f"Shape/dtype của {name} không đúng.")
        if (
            not np.isfinite(ts).all()
            or (ts < 0).any()
            or (np.diff(ts) <= 0).any()
        ):
            raise ValueError("Timestamp không hợp lệ.")
        if not np.isin(success, [0, 1]).all():
            raise ValueError("success không hợp lệ.")
        if (
            not np.isfinite(confidence).all()
            or (confidence < 0).any()
            or (confidence > 1).any()
        ):
            raise ValueError("confidence không hợp lệ.")
        if (
            mask.shape != x.shape
            or mask.dtype != np.bool_
            or not np.array_equal(
                mask, np.isfinite(x) & (success[:, None] == 1)
            )
        ):
            raise ValueError("observed_mask không hợp lệ.")
        if not np.array_equal(
            saved["frame_numbers"], np.arange(1, T + 1)
        ):
            raise ValueError("Thứ tự frame không hợp lệ.")
        if saved["decoded_frames"].item() != T:
            raise ValueError("Số frame video không khớp.")

        missing_values = int((~mask).sum())
        failed_frames = int((success == 0).sum())
        return {
            "T_vision": T,
            "D_vision": 22,
            "failed_frames": failed_frames,
            "unobserved_values": missing_values,
            "min_confidence": float(confidence.min()),
            "quality_flag": (
                "needs_review" if failed_frames or missing_values else "ok"
            ),
            "csv_file": saved["csv_file"].item(),
            "log_file": saved["log_file"].item(),
        }


records_full = []
JOURNAL_PATH = RUN_DIR / "progress.jsonl"
REPORT_PATH = RUN_DIR / "visual_manifest.csv"

try:
    with zipfile.ZipFile(RAW_ZIP, "r") as archive:
        progress = tqdm(
            full_manifest.itertuples(index=False),
            total=len(full_manifest), desc="Visual MOSI",
        )
        for row in progress:
            started = time.perf_counter()
            record = {
                "id": row.id, "status": "error",
                "quality_flag": "not_checked", "feature_file": "",
                "cache_note": "", "error": "",
            }
            try:
                member = archive.getinfo(row.video_relpath)
                if member.is_dir() or member.file_size != int(row.video_size_bytes):
                    raise ValueError("Thông tin video trong ZIP không khớp.")
                destination = FEATURE_DIR / f"{row.id}.npz"
                cached_stats = None
                if destination.is_file():
                    try:
                        cached_stats = inspect_saved_npz(destination, row, member)
                    except Exception as error:
                        record["cache_note"] = (
                            f"Tạo lại file: {type(error).__name__}: {error}"
                        )
                if cached_stats is not None:
                    record.update(cached_stats)
                    record["status"] = "cached"
                else:
                    local_video = LOCAL_VIDEO_DIR / f"{row.id}.mp4"
                    with archive.open(member) as source:
                        with local_video.open("wb") as target:
                            shutil.copyfileobj(source, target)
                    decoded_frames = decoded_frame_count(local_video)
                    run_config = dict(feature_config)
                    run_config.update({
                        "openface_bin": str(OPENFACE_BIN),
                        "model_path": str(MODEL_PATH),
                        "output_dir": str(RUN_DIR / "openface" / row.id),
                    })
                    result = processor_full.extract_visual_features(
                        local_video, run_config
                    )
                    payload = feature_payload(
                        result, row, member, decoded_frames
                    )
                    temporary = destination.with_name(destination.name + f".{run_id}.part")
                    with temporary.open("wb") as stream:
                        np.savez_compressed(stream, **payload)
                    inspect_saved_npz(temporary, row, member)
                    temporary.replace(destination)
                    record.update(inspect_saved_npz(destination, row, member))
                    record["status"] = "extracted"

                record["feature_file"] = str(destination.relative_to(FULL_ROOT))

            except Exception as error:
                record["error"] = f"{type(error).__name__}: {error}"
                (RUN_DIR / f"{row.id}.error.txt").write_text(
                    traceback.format_exc(), encoding="utf-8"
                )

            record["elapsed_seconds"] = round(time.perf_counter() - started, 3)
            records_full.append(record)
            with JOURNAL_PATH.open("a", encoding="utf-8") as journal:
                journal.write(json.dumps(record, ensure_ascii=False) + "\n")
            progress.set_postfix(
                last_status=record["status"], quality=record["quality_flag"]
            )

finally:
    processed = (
        pd.DataFrame(records_full) if records_full
        else pd.DataFrame(columns=["id", "status", "quality_flag"])
    )
    full_report = full_manifest.merge(
        processed, on="id", how="left", validate="one_to_one"
    )
    full_report["status"] = full_report["status"].fillna("pending")
    full_report["quality_flag"] = full_report["quality_flag"].fillna("not_checked")
    temporary_report = REPORT_PATH.with_suffix(".csv.part")
    full_report.to_csv(temporary_report, index=False, encoding="utf-8")
    temporary_report.replace(REPORT_PATH)
    display(
        full_report.groupby(["split", "status"]).size().unstack(fill_value=0)
    )
    print("Tổng hợp trạng thái:")
    print(full_report["status"].value_counts().to_string())
    print("Chất lượng kỹ thuật:")
    print(full_report["quality_flag"].value_counts().to_string())
    print("Manifest kết quả:", REPORT_PATH)
    print("Nhật ký lần chạy:", JOURNAL_PATH)
'''

def _run_full_extraction(paths):
    """Execute saved notebook logic in an isolated namespace, with configurable paths."""
    namespace = dict(globals())
    namespace['paths'] = paths
    exec(compile(TASK22_SETUP_SOURCE, '<task_2_22_setup>', 'exec'), namespace)
    exec(compile(TASK22_BATCH_SOURCE, '<task_2_22_batch>', 'exec'), namespace)
    return {'report': namespace['full_report'],
            'report_path': namespace['REPORT_PATH'],
            'full_root': namespace['FULL_ROOT']}

if __name__ == "__main__":
    main()
