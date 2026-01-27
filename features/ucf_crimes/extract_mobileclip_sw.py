import os
import argparse
import numpy as np
import torch
import torch.multiprocessing as mp
import cv2
from PIL import Image
from tqdm import tqdm

"""
MobileCLIP Video Feature Extraction

Requirements:
    pip install open_clip_torch
    # For MobileCLIP models, follow: https://github.com/apple/ml-mobileclip

Usage:
    # Using OpenCLIP model names (e.g., ViT-B-16, MobileCLIP-S0, etc.)
    python extract_mobileclip.py --video /path/to/video.mp4 --output_dir ./features --model_name ViT-B-16 --pretrained openai

    # Using local checkpoint
    python extract_mobileclip.py --video_dir /path/to/videos --output_dir ./features --model_name MobileCLIP-S2 --pretrained /path/to/mobileclip_s2.pt
"""

# Optional: MobileCLIP reparameterization
try:
    from mobileclip.modules.common.mobileone import reparameterize_model
    MOBILECLIP_AVAILABLE = True
except ImportError:
    MOBILECLIP_AVAILABLE = False


def load_mobileclip_model(model_name, pretrained, device="cuda"):
    """
    Load MobileCLIP or OpenCLIP model.

    Args:
        model_name: Model architecture name (e.g., "MobileCLIP-S2", "ViT-B-16")
        pretrained: Pretrained weights name or path to .pt file
        device: Device to load model on

    Returns:
        model, preprocess function
    """
    import open_clip

    # Set image normalization for MobileCLIP models (except S3, S4)
    model_kwargs = {}
    if "MobileCLIP" in model_name and not (model_name.endswith("S3") or model_name.endswith("S4")):
        model_kwargs = {"image_mean": (0, 0, 0), "image_std": (1, 1, 1)}

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=device, **model_kwargs
    )
    model.eval()

    # Reparameterize MobileCLIP for faster inference (if available)
    if MOBILECLIP_AVAILABLE and "MobileCLIP" in model_name:
        print("Reparameterizing MobileCLIP model for inference...")
        model = reparameterize_model(model)

    return model, preprocess


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
    middle_idx = snippet_size // 2

    for snippet_idx in range(num_snippets):
        frame_idx = snippet_idx * snippet_size + middle_idx
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            middle_frames.append(frame_rgb)
        else:
            print(f"Warning: Could not read frame {frame_idx} from {video_path}")

    cap.release()
    return middle_frames


@torch.no_grad()
def extract_features(frames, model, preprocess, device="cuda", batch_size=32):
    """
    Extract features using MobileCLIP/OpenCLIP model.

    Args:
        frames: List of RGB frames (numpy arrays)
        model: OpenCLIP model
        preprocess: Preprocessing transform
        device: Device for inference
        batch_size: Batch size for processing

    Returns:
        numpy array of shape [num_frames, d_e]
    """
    all_features = []

    for i in range(0, len(frames), batch_size):
        batch_frames = frames[i:i + batch_size]

        # Convert numpy arrays to PIL Images and preprocess
        images = [preprocess(Image.fromarray(frame)) for frame in batch_frames]
        image_input = torch.stack(images).to(device)

        # Extract features with automatic mixed precision
        with torch.cuda.amp.autocast(enabled=device != "cpu"):
            image_features = model.encode_image(image_input)
            image_features = torch.nn.functional.normalize(image_features, p=2, dim=-1)

        all_features.append(image_features.cpu().float().numpy())

    return np.concatenate(all_features, axis=0)


def process_video(video_path, output_path, model, preprocess,
                  device="cuda", snippet_size=16, batch_size=32):
    """
    Process a single video and save features as .npy file.
    """
    frames = extract_frames_from_video(video_path, snippet_size)

    if len(frames) == 0:
        print(f"Warning: No frames extracted from {video_path}")
        return

    features = extract_features(frames, model, preprocess, device, batch_size)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, features)

    return features.shape


def worker(gpu_id, video_output_pairs, model_name, pretrained, snippet_size, batch_size):
    """
    Worker function for multi-GPU processing.
    """
    device = f"cuda:{gpu_id}"
    print(f"[GPU {gpu_id}] Loading MobileCLIP model...")
    model, preprocess = load_mobileclip_model(model_name, pretrained, device)
    print(f"[GPU {gpu_id}] Processing {len(video_output_pairs)} videos")

    for video_path, output_path in tqdm(video_output_pairs, desc=f"GPU {gpu_id}", position=gpu_id):
        try:
            shape = process_video(
                video_path, output_path,
                model, preprocess,
                device, snippet_size, batch_size
            )
            if shape:
                tqdm.write(f"[GPU {gpu_id}] Saved {output_path} with shape: {shape}")
        except Exception as e:
            tqdm.write(f"[GPU {gpu_id}] Error processing {video_path}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Extract MobileCLIP/OpenCLIP features from videos")
    parser.add_argument("--video", type=str, help="Path to a single video file")
    parser.add_argument("--video_dir", type=str, help="Directory containing videos (recursive)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save .npy features")
    parser.add_argument("--model_name", type=str, default="MobileCLIP-S0",
                        help="Model name (e.g., MobileCLIP-S0, MobileCLIP-S2, ViT-B-16)")
    parser.add_argument("--pretrained", type=str, default=None,
                        help="Pretrained weights: 'openai', 'laion2b_s34b_b79k', or path to .pt file")
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
    if args.pretrained is None:
        parser.error("--pretrained is required (e.g., 'openai' or path to .pt file)")

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
                    rel_path = os.path.relpath(video_path, args.video_dir)
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
        print(f"Loading model: {args.model_name} with pretrained: {args.pretrained}")
        model, preprocess = load_mobileclip_model(args.model_name, args.pretrained, device)

        for video_path, output_path in tqdm(video_output_pairs, desc="Processing videos"):
            try:
                shape = process_video(
                    video_path, output_path,
                    model, preprocess,
                    device, args.snippet_size, args.batch_size
                )
                if shape:
                    tqdm.write(f"Saved {output_path} with shape: {shape}")
            except Exception as e:
                tqdm.write(f"Error processing {video_path}: {e}")
    else:
        # Multi-GPU mode
        print(f"Using {num_gpus} GPUs")
        mp.set_start_method('spawn', force=True)

        chunks = [[] for _ in range(num_gpus)]
        for i, pair in enumerate(video_output_pairs):
            chunks[i % num_gpus].append(pair)

        processes = []
        for gpu_id in range(num_gpus):
            p = mp.Process(
                target=worker,
                args=(gpu_id, chunks[gpu_id], args.model_name, args.pretrained, args.snippet_size, args.batch_size)
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()


if __name__ == "__main__":
    main()
