import os
import argparse
import numpy as np
import torch
import torch.multiprocessing as mp
import cv2
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

"""
python extract_clip_vit.py --video /mnt/c/JJS/UCF_Crimes/Videos/train/Abuse/Abuse001_x264.mp4 --output_dir /mnt/c/Users/USER/Desktop/ExploreVAD/features/ucf_crimes
"""

def load_clip_model(model_name="openai/clip-vit-base-patch32", device="cuda"):
    """Load CLIP model and processor."""
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    model = model.to(device)
    model.eval()

    # Image Encoder 추출 (Backbone: I_f를 생성)
    image_encoder = model.vision_model

    # Image Projection Matrix 추출 (W_i: d_i -> d_e)
    visual_projection = model.visual_projection

    return image_encoder, visual_projection, processor


def extract_frames_from_video(video_path, snippet_size=16):
    """
    Extract middle frames from 16-frame snippets.

    Args:
        video_path: Path to video file
        snippet_size: Number of frames per snippet (default: 16)

    Returns:
        List of middle frames (as RGB numpy arrays)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    num_snippets = total_frames // snippet_size

    middle_frames = []
    middle_idx = snippet_size // 2  # Middle frame index within snippet (8 for 16-frame snippets)

    for snippet_idx in range(num_snippets):
        # Calculate the global frame index of the middle frame
        frame_idx = snippet_idx * snippet_size + middle_idx
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if ret:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            middle_frames.append(frame_rgb)
        else:
            print(f"Warning: Could not read frame {frame_idx} from {video_path}")

    cap.release()
    return middle_frames


@torch.no_grad()
def extract_clip_features(frames, image_encoder, visual_projection, processor, device="cuda", batch_size=32):
    """
    Extract CLIP features from frames.

    Args:
        frames: List of RGB frames (numpy arrays)
        image_encoder: CLIP vision model
        visual_projection: CLIP visual projection layer
        processor: CLIP processor
        device: Device to run inference on
        batch_size: Batch size for processing

    Returns:
        numpy array of shape [num_frames, d_e]
    """
    all_features = []

    for i in range(0, len(frames), batch_size):
        batch_frames = frames[i:i + batch_size]

        # Process images
        inputs = processor(images=batch_frames, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # [Step 1]: I_f = image_encoder(I) [n, d_i]
        # VisionModel의 pooler_output: [CLS] 토큰의 representation
        I_f = image_encoder(**inputs).pooler_output

        # [Step 2]: I_e = l2_normalize(np.dot(I_f, W_i)) [n, d_e]
        I_e_unnorm = visual_projection(I_f)
        I_e = torch.nn.functional.normalize(I_e_unnorm, p=2, dim=-1)

        all_features.append(I_e.cpu().numpy())

    return np.concatenate(all_features, axis=0)


def process_video(video_path, output_path, image_encoder, visual_projection, processor,
                  device="cuda", snippet_size=16, batch_size=32):
    """
    Process a single video and save features as .npy file.

    Args:
        video_path: Path to video file
        output_path: Full path for the output .npy file
        image_encoder: CLIP vision model
        visual_projection: CLIP visual projection layer
        processor: CLIP processor
        device: Device to run inference on
        snippet_size: Number of frames per snippet
        batch_size: Batch size for feature extraction
    """
    # Extract middle frames from snippets
    frames = extract_frames_from_video(video_path, snippet_size)

    if len(frames) == 0:
        print(f"Warning: No frames extracted from {video_path}")
        return

    # Extract CLIP features
    features = extract_clip_features(frames, image_encoder, visual_projection, processor, device, batch_size)

    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save as .npy file
    np.save(output_path, features)

    return features.shape


def worker(gpu_id, video_output_pairs, model_name, snippet_size, batch_size):
    """
    Worker function for multi-GPU processing.
    Each worker processes a subset of videos on a specific GPU.
    """
    device = f"cuda:{gpu_id}"
    print(f"[GPU {gpu_id}] Loading CLIP model...")
    image_encoder, visual_projection, processor = load_clip_model(model_name, device)
    print(f"[GPU {gpu_id}] Processing {len(video_output_pairs)} videos")

    for video_path, output_path in tqdm(video_output_pairs, desc=f"GPU {gpu_id}", position=gpu_id):
        try:
            shape = process_video(
                video_path, output_path,
                image_encoder, visual_projection, processor,
                device, snippet_size, batch_size
            )
            if shape:
                tqdm.write(f"[GPU {gpu_id}] Saved {output_path} with shape: {shape}")
        except Exception as e:
            tqdm.write(f"[GPU {gpu_id}] Error processing {video_path}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Extract CLIP-ViT features from videos")
    parser.add_argument("--video", type=str, help="Path to a single video file")
    parser.add_argument("--video_dir", type=str, help="Directory containing videos")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save .npy features")
    parser.add_argument("--model_name", type=str, default="openai/clip-vit-base-patch32",
                        help="CLIP model name (default: openai/clip-vit-base-patch32)")
    parser.add_argument("--snippet_size", type=int, default=16, help="Frames per snippet (default: 16)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference (default: 32)")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--video_ext", type=str, default=".mp4", help="Video file extension (default: .mp4)")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use (default: 1)")
    args = parser.parse_args()

    # Validate input arguments
    if args.video is None and args.video_dir is None:
        parser.error("Either --video or --video_dir must be provided")
    if args.video is not None and args.video_dir is not None:
        parser.error("Cannot use both --video and --video_dir")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

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

    # Determine number of GPUs to use
    num_gpus = args.num_gpus
    if args.device == "cpu" or not torch.cuda.is_available():
        num_gpus = 1
        device = "cpu"
    else:
        available_gpus = torch.cuda.device_count()
        if num_gpus > available_gpus:
            print(f"Requested {num_gpus} GPUs but only {available_gpus} available. Using {available_gpus}.")
            num_gpus = available_gpus

    # Single GPU or CPU mode
    if num_gpus == 1:
        device = args.device if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        print(f"Loading CLIP model: {args.model_name}")
        image_encoder, visual_projection, processor = load_clip_model(args.model_name, device)

        for video_path, output_path in tqdm(video_output_pairs, desc="Processing videos"):
            try:
                shape = process_video(
                    video_path, output_path,
                    image_encoder, visual_projection, processor,
                    device, args.snippet_size, args.batch_size
                )
                if shape:
                    tqdm.write(f"Saved {output_path} with shape: {shape}")
            except Exception as e:
                tqdm.write(f"Error processing {video_path}: {e}")
    else:
        # Multi-GPU mode: split videos across GPUs
        print(f"Using {num_gpus} GPUs")
        mp.set_start_method('spawn', force=True)

        # Split video pairs across GPUs
        chunks = [[] for _ in range(num_gpus)]
        for i, pair in enumerate(video_output_pairs):
            chunks[i % num_gpus].append(pair)

        # Launch workers
        processes = []
        for gpu_id in range(num_gpus):
            p = mp.Process(
                target=worker,
                args=(gpu_id, chunks[gpu_id], args.model_name, args.snippet_size, args.batch_size)
            )
            p.start()
            processes.append(p)

        # Wait for all workers to finish
        for p in processes:
            p.join()


if __name__ == "__main__":
    main()