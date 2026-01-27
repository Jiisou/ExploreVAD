import os
import argparse
import torch
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

class VideoFeatureExtractor:
    def __init__(self, model_name="openai/clip-vit-base-patch32", device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

    @torch.no_grad()
    def extract_window_features(self, frames):
        """N개의 프레임을 입력받아 하나의 윈도우 특징 벡터로 변환 (Temporal Mean Pooling)"""
        inputs = self.processor(images=frames, return_tensors="pt", padding=True).to(self.device)
        
        # CLIP Visual Projection까지 수행하여 d_e 차원 임베딩 추출
        vision_outputs = self.model.vision_model(**inputs)
        I_f = vision_outputs.pooler_output 
        I_e_unnorm = self.model.visual_projection(I_f)
        I_e = torch.nn.functional.normalize(I_e_unnorm, p=2, dim=-1)
        
        return I_e.mean(dim=0, keepdim=True).cpu().numpy()

    def process_video(self, video_path, window_time=2, num_frames=16):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = int(total_frames / fps)

        video_features = []
        
        # 0초부터 비디오 끝(초 단위)까지 루프를 돌며 각 1초마다 특징 생성
        for target_sec in range(duration_sec):
            # 윈도우 설정 (기본적으로 target_sec을 중심으로 전후 1초씩, 총 2초)
            # 단, 0초 시점에는 윈도우가 차지 않으므로 0~1초 프레임을 복사/샘플링하여 보정
            if target_sec == 0:
                start_t = 0
                end_t = 1
            else:
                start_t = max(0, target_sec - 1)
                end_t = min(duration_sec, target_sec + 1)
            
            # Uniform Sampling: 정해진 시간 범위 내에서 num_frames개만큼 균등 추출
            sample_times = np.linspace(start_t, end_t, num_frames, endpoint=False)
            batch_frames = []
            
            for t in sample_times:
                frame_idx = int(t * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    batch_frames.append(Image.fromarray(frame))
                else:
                    # 프레임 읽기 실패 시 마지막 정상 프레임이나 빈 프레임 방지 로직
                    if len(batch_frames) > 0:
                        batch_frames.append(batch_frames[-1])
            
            # 윈도우 특징 추출 및 저장
            if len(batch_frames) > 0:
                # 0초 구간일 경우 1초 분량에서 num_frames를 뽑으므로 결과적으로 프레임 밀도가 복사된 효과
                window_feat = self.extract_window_features(batch_frames)
                video_features.append(window_feat)

        cap.release()
        
        # 최종 결과: [비디오 시간(초), d_e] 크기의 행렬
        return np.vstack(video_features)

def main():
    parser = argparse.ArgumentParser(description="Extract CLIP-ViT features with sliding window")
    parser.add_argument("--video", type=str, help="Path to a single video file")
    parser.add_argument("--video_dir", type=str, help="Directory containing videos (recursive)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save .npy features")
    parser.add_argument("--model_name", type=str, default="openai/clip-vit-base-patch32",
                        help="CLIP model name (default: openai/clip-vit-base-patch32)")
    parser.add_argument("--window_time", type=int, default=2, help="Window time in seconds (default: 2)")
    parser.add_argument("--num_frames", type=int, default=16, help="Number of frames per window (default: 16)")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--video_ext", type=str, default=".mp4", help="Video file extension (default: .mp4)")
    args = parser.parse_args()

    # Validate input arguments
    if args.video is None and args.video_dir is None:
        parser.error("Either --video or --video_dir must be provided")
    if args.video is not None and args.video_dir is not None:
        parser.error("Cannot use both --video and --video_dir")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize extractor
    print(f"Loading CLIP model: {args.model_name}")
    extractor = VideoFeatureExtractor(model_name=args.model_name, device=args.device)

    # Get list of (video_path, output_path) pairs to process
    video_output_pairs = []
    if args.video:
        if not os.path.isfile(args.video):
            raise FileNotFoundError(f"Video file not found: {args.video}")
        video_name = os.path.splitext(os.path.basename(args.video))[0]
        output_path = os.path.join(args.output_dir, f"{video_name}.npy")
        video_output_pairs.append((args.video, output_path))
    else:
        # Recursively find all videos and preserve directory structure
        for root, dirs, files in os.walk(args.video_dir):
            for f in files:
                if f.endswith(args.video_ext):
                    video_path = os.path.join(root, f)
                    # Compute relative path from video_dir
                    rel_path = os.path.relpath(video_path, args.video_dir)
                    # Replace extension with .npy
                    rel_npy = os.path.splitext(rel_path)[0] + ".npy"
                    output_path = os.path.join(args.output_dir, rel_npy)
                    video_output_pairs.append((video_path, output_path))
        print(f"Found {len(video_output_pairs)} videos")

    # Process videos
    for video_path, output_path in tqdm(video_output_pairs, desc="Processing videos"):
        try:
            features = extractor.process_video(video_path, args.window_time, args.num_frames)

            # Create output directory if needed
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Save as .npy file
            np.save(output_path, features)
            tqdm.write(f"Saved {output_path} with shape: {features.shape}")
        except Exception as e:
            tqdm.write(f"Error processing {video_path}: {e}")


if __name__ == "__main__":
    main()