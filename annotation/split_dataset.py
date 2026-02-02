import os
import shutil
import random
from pathlib import Path

def split_dataset(root_dir: str, output_dir: str, seed: int = 42):
    """
    데이터셋을 8:1:1 비율로 train, valid, test 세트로 분할.
    """
    # 경로 설정 및 시드 고정
    src_root = Path(root_dir)
    dst_root = Path(output_dir)
    random.seed(seed)

    # 분할 명칭 및 비율
    splits = ['train', 'valid', 'test']
    
    # 원본 이벤트 디렉토리 탐색
    event_dirs = [d for d in src_root.iterdir() if d.is_dir()]

    for event_dir in event_dirs:
        event_name = event_dir.name
        # 해당 이벤트 내 모든 .avi 파일 수집
        files = list(event_dir.glob('*.avi'))
        random.shuffle(files)

        n_total = len(files)
        # 인덱스 계산 (8:1:1)
        n_train = int(n_total * 0.8)
        n_val = int(n_total * 0.1)
        # n_test는 나머지 전체

        # 데이터 분할 슬라이싱
        file_mapping = {
            'train': files[:n_train],
            'valid': files[n_train:n_train + n_val],
            'test': files[n_train + n_val:]
        }

        # 파일 복사 실행
        for split_name, split_files in file_mapping.items():
            # 목적지 경로 생성: output_dir/{split}/{event_name}/
            target_path = dst_root / split_name / event_name
            target_path.mkdir(parents=True, exist_ok=True)

            for f in split_files:
                # shutil.copy2는 메타데이터(수정 시간 등)를 보존하며 복사함
                shutil.copy2(f, target_path / f.name)

        print(f"Completed: {event_name} (Total: {n_total} | Tr: {len(file_mapping['train'])}, Val: {len(file_mapping['valid'])}, Te: {len(file_mapping['test'])})")

if __name__ == "__main__":
    input_root = input("root_dir 경로를 입력하세요: ")
    output_root = input("output_dir 경로를 입력하세요: ")
    split_dataset(input_root, output_root)