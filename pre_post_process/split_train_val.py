import os
import shutil
import random
from pathlib import Path

def split_train_val(root_dir: str, output_dir: str, extension: str, seed: int = 42):
    """
    기존 Train 데이터를 train과 valid 세트로 9:1 비율로 분할합니다.
    """
    # 경로 설정 및 시드 고정
    src_root = Path(root_dir)
    dst_root = Path(output_dir)
    random.seed(seed)

    # 분할 명칭 (test 제외)
    splits = ['train', 'valid']
    
    # 원본 이벤트 디렉토리 탐색
    event_dirs = [d for d in src_root.iterdir() if d.is_dir()]

    for event_dir in event_dirs:
        event_name = event_dir.name
        # 해당 이벤트 내 모든 주어진 확장자 파일 수집
        files = list(event_dir.glob(f"*.{extension}"))
        random.shuffle(files)

        n_total = len(files)
        if n_total == 0:
            print(f"Skip: {event_name} (파일 없음)")
            continue

        # 인덱스 계산 (9:1)
        n_train = int(n_total * 0.9)

        # 데이터 분할 슬라이싱 (나머지 전체를 valid로 할당하여 누락 방지)
        file_mapping = {
            'train': files[:n_train],
            'valid': files[n_train:]
        }

        # 파일 복사 실행
        for split_name, split_files in file_mapping.items():
            target_path = dst_root / split_name / event_name
            target_path.mkdir(parents=True, exist_ok=True)

            for f in split_files:
                shutil.copy2(f, target_path / f.name)

        # 결과 출력
        print(f"Completed: {event_name:<15} | Total: {n_total:>4} | Tr: {len(file_mapping['train']):>4} | Val: {len(file_mapping['valid']):>4}")

if __name__ == "__main__":
    input_root = input("분할할 데이터의 root_dir(Train 원본) 경로: ")
    output_root = input("저장할 output_dir 경로: ")
    extension = input("파일 확장자 (e.g., mp4, npy): ").strip('.')
    
    split_train_val(input_root, output_root, extension)