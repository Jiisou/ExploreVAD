#!/usr/bin/env python3
"""
Realtime Video Inference Visualizer for Anomaly Action Spotting.

Uses MobileCLIP for feature extraction and ResidualMLPSpottingModel for classification.
Displays video playback with timeline visualization showing prediction results.

Reference style: qwen3-vl-2b-int4-video_binary_timeline.py

Usage:
    python realtime_video_inference.py \
        --checkpoint model.pth \
        --video video.mp4 \
        --model-name mobileclip_s0

    # With directory of videos:
    python realtime_video_inference.py \
        --checkpoint model.pth \
        --video-dir /path/to/videos \
        --model-name mobileclip_s0
"""

import argparse
import os
import threading
import time
from collections import deque
from typing import List, Dict, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

import mobileclip
import open_clip
from mobileclip.modules.common.mobileone import reparameterize_model

# MobileCLIP imports
try:
    MOBILECLIP_V1_AVAILABLE = True
except ImportError:
    MOBILECLIP_V1_AVAILABLE = False

try:
    OPEN_CLIP_AVAILABLE = True
except ImportError:
    OPEN_CLIP_AVAILABLE = False

import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from model import ResidualMLPSpottingModel
from utils import get_device, load_checkpoint


# --- Color Palette (BGR for OpenCV) ---
CLASS_COLORS = {
    "NORMAL": (255, 178, 102),       # Light blue
    "ABNORMAL": (90, 90, 255),       # Light red
    "SCANNING": (100, 100, 100),     # Gray
    "PROCESSING": (0, 215, 255),     # Gold/Orange
}

# Timeline colors
C_BLUE = (255, 178, 102)
C_RED = (90, 90, 255)
C_WHITE = (245, 245, 245)
C_DARK = (30, 30, 30)
C_ACCENT = (0, 215, 255)


class MobileCLIPExtractor:
    """MobileCLIP feature extractor for realtime inference."""

    def __init__(
        self,
        model_name: str = "MobileCLIP2-S0",
        pretrained_path: Optional[str] = None,
        device: str = "cuda",
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model_name = model_name

        is_mobileclip_v2 = model_name.startswith("MobileCLIP2-")

        if is_mobileclip_v2:
            if not OPEN_CLIP_AVAILABLE:
                raise ImportError("open_clip is required for MobileCLIP2 models")

            print(f"Loading MobileCLIP2 model: {model_name}")
            model_kwargs = {}
            if not (model_name.endswith("S3") or model_name.endswith("S4") or model_name.endswith("L-14")):
                model_kwargs = {"image_mean": (0, 0, 0), "image_std": (1, 1, 1)}

            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name,
                pretrained=pretrained_path if pretrained_path else "dfndr2b",
                **model_kwargs
            )
            self.model.eval()
            self.model = reparameterize_model(self.model)
            self.model = self.model.to(self.device)
        else:
            if not MOBILECLIP_V1_AVAILABLE:
                raise ImportError("mobileclip package is required for MobileCLIP v1 models")

            print(f"Loading MobileCLIP v1 model: {model_name}")
            if pretrained_path is None and model_name == "mobileclip_s0":
                pretrained_path = os.path.expanduser(
                    "~/.cache/huggingface/hub/models--apple--MobileCLIP-S0/"
                    "snapshots/71aa3e13dda93115871afbd017336535ba29886c/mobileclip_s0.pt"
                )

            self.model, _, self.preprocess = mobileclip.create_model_and_transforms(
                model_name,
                pretrained=pretrained_path,
                device=self.device
            )

        print(f"MobileCLIP loaded on {self.device}")

    @torch.no_grad()
    def extract_features(self, frames: List[Image.Image]) -> np.ndarray:
        """
        Extract features from frames and return averaged feature vector.

        Args:
            frames: List of PIL Images.

        Returns:
            Feature vector of shape (1, D).
        """
        if not frames:
            return None

        batch_images = torch.stack([self.preprocess(frame) for frame in frames]).to(self.device)

        with torch.cuda.amp.autocast():
            image_features = self.model.encode_image(batch_images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Temporal mean pooling
        return image_features.mean(dim=0, keepdim=True).cpu().numpy()


class RealtimeVideoInference:
    """Realtime video inference with timeline visualization."""

    def __init__(
        self,
        feature_extractor: MobileCLIPExtractor,
        classifier: ResidualMLPSpottingModel,
        device: torch.device,
        window_time: int = 2,
        num_frames: int = 8,
        stride_time: float = 1.0,
        threshold: float = 0.5,
    ):
        self.extractor = feature_extractor
        self.classifier = classifier
        self.device = device
        self.window_time = window_time
        self.num_frames = num_frames
        self.stride_time = stride_time
        self.threshold = threshold

        # State
        self.latest_label = "SCANNING"
        self.latest_prob = 0.0
        self.latest_color = CLASS_COLORS["SCANNING"]
        self.is_processing = False
        self.last_latency = 0.0

        # Results storage
        self.timeline_data = []
        self.inference_triggers = []

        # [KEY CHANGE] Buffer stores numpy arrays (BGR), not PIL Images
        self.buffer_maxlen = int(30 * 2) # fps * window_time
        self.frame_buffer = deque(maxlen=self.buffer_maxlen)

    def reset(self):
        """Reset state for new video."""
        self.latest_label = "SCANNING"
        self.latest_prob = 0.0
        self.latest_color = CLASS_COLORS["SCANNING"]
        self.is_processing = False
        self.last_latency = 0.0
        self.timeline_data = []
        self.inference_triggers = []
        self.frame_buffer.clear()

    def update_buffer(self, frame: np.ndarray):
        """Store raw numpy frame. Very fast."""
        self.frame_buffer.append(frame)

    def _get_sampled_frames(self) -> List[np.ndarray]:
        """Sample raw frames from buffer."""
        curr_len = len(self.frame_buffer)
        if curr_len == 0:
            return []
        
        # Linear sampling with replication if needed
        indices = np.linspace(0, curr_len - 1, self.num_frames, dtype=int)
        return [self.frame_buffer[i] for i in indices]

    @torch.no_grad()
    def run_inference(
        self,
        raw_frames: List[np.ndarray], # 입력 타입 변경: Image -> np.ndarray
        current_time: float,
    ):
        start_t = time.time()
        
        try:
            # [최적화] 추론 스레드 내부에서, 실제로 필요한 8장만 변환 수행
            pil_frames = []
            for frame in raw_frames:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_frames.append(Image.fromarray(frame_rgb))

            # Extract features
            features = self.extractor.extract_features(pil_frames)            
            if features is None:
                self.is_processing = False
                return

            # Run classifier
            # features shape: (1, D) -> need (1, T, D) for model
            # Since we already did temporal pooling, we expand to (1, 1, D)
            features_tensor = torch.from_numpy(features).float().unsqueeze(0).to(self.device)

            self.classifier.eval()
            logit = self.classifier(features_tensor).squeeze()

            prob = torch.sigmoid(logit).item()
            pred = 1 if logit.item() > 0 else 0

            # Update state
            if pred == 1:
                self.latest_label = "ABNORMAL"
                self.latest_color = CLASS_COLORS["ABNORMAL"]
            else:
                self.latest_label = "NORMAL"
                self.latest_color = CLASS_COLORS["NORMAL"]

            self.latest_prob = prob
            self.last_latency = time.time() - start_t

            # Store result
            self.timeline_data.append({
                'time': current_time,
                'label': self.latest_label,
                'prob': prob,
                'pred': pred,
            })

            print(f"  -> [{current_time:.1f}s] {self.latest_label} (prob={prob:.3f}, latency={self.last_latency:.2f}s)")

        except Exception as e:
            print(f"Inference error: {e}")
            self.latest_label = "SCANNING"
            self.latest_color = CLASS_COLORS["SCANNING"]

        finally:
            self.is_processing = False

    def _sample_frames_from_video(
        self,
        cap: cv2.VideoCapture,
        center_time: float,
        fps: float,
        duration: float,
    ) -> List[Image.Image]:
        """Sample frames from video around center_time."""
        half_window = self.window_time / 2

        if center_time < half_window:
            start_t = 0
            end_t = min(self.window_time, duration)
        elif center_time > duration - half_window:
            start_t = max(0, duration - self.window_time)
            end_t = duration
        else:
            start_t = center_time - half_window
            end_t = center_time + half_window

        sample_times = np.linspace(start_t, end_t, self.num_frames, endpoint=False)
        frames = []

        for t in sample_times:
            frame_idx = int(t * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
            elif frames:
                frames.append(frames[-1])

        return frames


# --- Padding layout ---
# All UI is drawn in padding areas; original video frame is never touched.
HEADER_H = 28   # top bar: prediction label + latency
FOOTER_H = 22   # bottom bar: timeline


def create_padded_frame(frame: np.ndarray) -> np.ndarray:
    """Add dark header/footer padding around the original frame."""
    h, w = frame.shape[:2]
    padded = np.zeros((HEADER_H + h + FOOTER_H, w, 3), dtype=np.uint8)
    padded[:HEADER_H, :] = C_DARK
    padded[HEADER_H + h:, :] = C_DARK
    padded[HEADER_H:HEADER_H + h, :] = frame
    return padded


def draw_header(padded, label, prob, color, latency):
    """Draw prediction + latency inside the header padding."""
    w = padded.shape[1]
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Color accent line at bottom edge of header
    cv2.rectangle(padded, (0, HEADER_H - 3), (w, HEADER_H), color, -1)

    # Prediction (left)
    cv2.putText(padded, f"{label} ({prob:.1%})", (6, 19),
                font, 0.42, color, 1, cv2.LINE_AA)

    # Latency (right)
    lat = f"latency: {latency:.2f}s"
    (tw, _), _ = cv2.getTextSize(lat, font, 0.38, 1)
    cv2.putText(padded, lat, (w - tw - 6, 19),
                font, 0.38, (0, 230, 0), 1, cv2.LINE_AA)


def draw_footer_timeline(padded, timeline_data, current_time, total_duration, orig_h):
    """Draw compact timeline bar inside the footer padding."""
    w = padded.shape[1]
    margin = 6
    bar_h = 12
    tl_w = w - margin * 2
    tl_x = margin
    tl_y = HEADER_H + orig_h + 4  # inside footer area

    # Segments
    if timeline_data and total_duration > 0:
        seg_w = max(2, int(tl_w / max(1, total_duration)))
        for r in timeline_data:
            ratio = min(1.0, r['time'] / total_duration)
            sx = int(tl_x + ratio * tl_w)
            color = C_RED if r['pred'] == 1 else C_BLUE
            cv2.rectangle(padded, (sx, tl_y), (sx + seg_w, tl_y + bar_h), color, -1)

    # Playhead + time label
    if total_duration > 0:
        px = int(tl_x + (current_time / total_duration) * tl_w)
        cv2.line(padded, (px, tl_y - 2), (px, tl_y + bar_h + 2), C_ACCENT, 2, cv2.LINE_AA)

        time_str = f"{current_time:.1f}s/{total_duration:.0f}s"
        (tw, _), _ = cv2.getTextSize(time_str, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)
        cv2.putText(padded, time_str, (w - tw - margin, tl_y + bar_h + 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, C_WHITE, 1, cv2.LINE_AA)


def process_video_realtime(
    video_path: str,
    inference_engine: RealtimeVideoInference,
    output_path: Optional[str] = None,
    show_preview: bool = True,
    playback_speed: float = 1.0,
):
    """
    Process video with realtime inference visualization.

    Args:
        video_path: Path to input video.
        inference_engine: RealtimeVideoInference instance.
        output_path: Optional path to save output video.
        show_preview: Whether to show preview window.
        playback_speed: Playback speed multiplier.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    filename = os.path.basename(video_path)
    out_w = w
    out_h = h + HEADER_H + FOOTER_H
    print(f"\nProcessing: {filename}")
    print(f"  Duration: {duration:.1f}s, FPS: {fps:.1f}, Video: {w}x{h}, Canvas: {out_w}x{out_h}")

    # Reset engine and sync FPS
    inference_engine.fps = fps
    inference_engine.buffer_maxlen = int(fps * inference_engine.window_time)
    inference_engine.frame_buffer = deque(maxlen=inference_engine.buffer_maxlen)
    inference_engine.reset()

    # Video writer (padded size)
    out = None
    if output_path:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (out_w, out_h))

    # Preview window - scale up for small videos
    if show_preview:
        cv2.namedWindow('Anomaly Detection - Realtime', cv2.WINDOW_NORMAL)
        scale = max(1, min(3, 720 // out_h))
        cv2.resizeWindow('Anomaly Detection - Realtime', out_w * scale, out_h * scale)

    wait_ms = max(1, int(1000 / fps / playback_speed))
    last_inference_time = -inference_engine.stride_time
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret: break

        inference_engine.update_buffer(frame)

        current_time = frame_idx / fps
        frame_idx += 1

        # Trigger inference at stride intervals
        if current_time - last_inference_time >= inference_engine.stride_time:
            if not inference_engine.is_processing:
                # 버퍼에 데이터가 있는지 확인
                if len(inference_engine.frame_buffer) > 0:
                    inference_engine.is_processing = True
                    last_inference_time = current_time
                    
                    # 샘플링 (여기서는 Numpy Array 리스트가 반환됨)
                    sampled_raw_frames = inference_engine._get_sampled_frames()

                    # 스레드 시작
                    thread = threading.Thread(
                        target=inference_engine.run_inference,
                        args=(sampled_raw_frames, current_time),
                        daemon=True
                    )
                    thread.start()

        # Build display: header + original frame + footer (no overlay on video)
        display_frame = create_padded_frame(frame)
        draw_header(
            display_frame,
            inference_engine.latest_label,
            inference_engine.latest_prob,
            inference_engine.latest_color,
            inference_engine.last_latency,
        )
        draw_footer_timeline(
            display_frame,
            inference_engine.timeline_data,
            current_time,
            duration,
            h,
        )

        # Write output
        if out:
            out.write(display_frame)

        # Show preview
        if show_preview:
            cv2.imshow('Anomaly Detection - Realtime', display_frame)
            key = cv2.waitKey(wait_ms) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('n'):
                break
            elif key == ord(' '):
                cv2.waitKey(0)

    cap.release()
    if out:
        out.release()

    if show_preview:
        cv2.destroyAllWindows()

    # Print summary
    print_video_summary(inference_engine, filename)

    if output_path:
        print(f"Output saved to: {output_path}")


def print_video_summary(engine: RealtimeVideoInference, filename: str):
    """Print summary of inference results."""
    if not engine.timeline_data:
        print("No inference results")
        return

    total_windows = len(engine.timeline_data)
    abnormal_count = sum(1 for r in engine.timeline_data if r['pred'] == 1)
    normal_count = total_windows - abnormal_count

    probs = [r['prob'] for r in engine.timeline_data]
    max_prob = max(probs)
    mean_prob = np.mean(probs)

    print(f"\n{'='*60}")
    print(f"INFERENCE SUMMARY: {filename}")
    print(f"{'='*60}")
    print(f"Total windows: {total_windows}")
    print(f"Normal: {normal_count} ({100*normal_count/total_windows:.1f}%)")
    print(f"Abnormal: {abnormal_count} ({100*abnormal_count/total_windows:.1f}%)")
    print(f"Max anomaly prob: {max_prob:.4f}")
    print(f"Mean anomaly prob: {mean_prob:.4f}")

    # Final verdict (consecutive abnormal or high ratio)
    consecutive = 0
    max_consecutive = 0
    for r in engine.timeline_data:
        if r['pred'] == 1:
            consecutive += 1
            max_consecutive = max(max_consecutive, consecutive)
        else:
            consecutive = 0

    if max_consecutive >= 2 or abnormal_count / total_windows >= 0.3:
        verdict = "ABNORMAL"
    else:
        verdict = "NORMAL"

    print(f"\nFinal Verdict: {verdict}")
    print(f"{'='*60}\n")


def get_video_files(directory: str) -> List[str]:
    """Get all video files from directory recursively."""
    extensions = ('.mp4', '.avi', '.mkv', '.mov', '.webm')
    files = []
    for root, _, filenames in os.walk(directory):
        for f in sorted(filenames):
            if f.lower().endswith(extensions):
                files.append(os.path.join(root, f))
    return files


def parse_args():
    parser = argparse.ArgumentParser(
        description="Realtime Video Inference Visualizer for Anomaly Action Spotting"
    )

    # Required
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to ResidualMLPSpottingModel checkpoint")

    # Input
    parser.add_argument("--video", type=str, default=None,
                        help="Path to a single video file")
    parser.add_argument("--video-dir", type=str, default=None,
                        help="Directory containing videos (recursive)")

    # Feature extractor
    parser.add_argument("--model-name", type=str, default="MobileCLIP2-S0",
                        help="MobileCLIP model name (default: MobileCLIP2-S0)")
    parser.add_argument("--pretrained-path", type=str, default=None,
                        help="Path to MobileCLIP pretrained weights")

    # Model
    parser.add_argument("--embed-dim", type=int, default=512,
                        help="Feature embedding dimension")
    parser.add_argument("--temporal-agg", type=str, default="mean",
                        choices=["mean", "max", "attention"],
                        help="Temporal aggregation method")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="Dropout rate")

    # Inference
    parser.add_argument("--window-time", type=int, default=2,
                        help="Window size in seconds (default: 2)")
    parser.add_argument("--num-frames", type=int, default=8,
                        help="Number of frames per window (default: 8)")
    parser.add_argument("--stride-time", type=float, default=1.0,
                        help="Stride between windows in seconds (default: 1.0)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Anomaly threshold (default: 0.5)")

    # Output
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save output videos")
    parser.add_argument("--no-preview", action="store_true",
                        help="Disable preview window")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Playback speed multiplier (default: 1.0)")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.video is None and args.video_dir is None:
        print("Error: Either --video or --video-dir must be provided")
        return

    print("=" * 60)
    print("Realtime Video Inference Visualizer")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"MobileCLIP model: {args.model_name}")
    print(f"Window: {args.window_time}s, {args.num_frames} frames, stride={args.stride_time}s")
    print(f"Threshold: {args.threshold}")
    print("=" * 60)

    # Initialize device
    device = get_device()

    # Load feature extractor
    print("\nLoading MobileCLIP feature extractor...")
    extractor = MobileCLIPExtractor(
        model_name=args.model_name,
        pretrained_path=args.pretrained_path,
        device=str(device),
    )

    # Load classifier
    print("\nLoading ResidualMLPSpottingModel...")
    classifier = ResidualMLPSpottingModel(
        embed_dim=args.embed_dim,
        dropout_rate=args.dropout,
        temporal_agg=args.temporal_agg,
    )
    load_checkpoint(args.checkpoint, model=classifier, device=device)
    classifier = classifier.to(device)
    classifier.eval()

    # Create inference engine
    inference_engine = RealtimeVideoInference(
        feature_extractor=extractor,
        classifier=classifier,
        device=device,
        window_time=args.window_time,
        num_frames=args.num_frames,
        stride_time=args.stride_time,
        threshold=args.threshold,
    )

    # Get video files
    if args.video:
        video_files = [args.video]
    else:
        video_files = get_video_files(args.video_dir)
        print(f"\nFound {len(video_files)} videos")

    # Process videos
    for video_path in video_files:
        output_path = None
        if args.output_dir:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            output_path = os.path.join(args.output_dir, f"{video_name}_inference.mp4")

        process_video_realtime(
            video_path=video_path,
            inference_engine=inference_engine,
            output_path=output_path,
            show_preview=not args.no_preview,
            playback_speed=args.speed,
        )

    print("\nAll videos processed!")


if __name__ == "__main__":
    main()