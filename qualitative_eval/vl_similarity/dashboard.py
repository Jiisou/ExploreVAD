"""
Vision-Language Similarity Dashboard

Streamlit application for visualizing cosine similarity between
video frame embeddings (npy) and CLIP text embeddings over time.

Usage:
    streamlit run qualitative_eval/vl_similarity/dashboard.py
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Add parent directory to path for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_registry import get_available_models, get_model_info, load_text_encoder
from similarity import (
    discover_classes,
    list_videos_per_class,
    sample_one_per_class,
    load_video_features,
    compute_similarity_matrix,
    build_text_prompts,
    apply_smoothing,
    parse_annotations,
    lookup_annotation,
)


@st.cache_resource
def cached_load_text_encoder(model_key, device, pretrained_path):
    """Cache model loading to avoid reloading on every interaction."""
    return load_text_encoder(model_key, device, pretrained_path)


@st.cache_data
def cached_encode_text(_text_encoder, prompts_tuple):
    """Cache text encoding results."""
    return _text_encoder.encode_text(list(prompts_tuple))


@st.cache_data
def cached_load_features(npy_path):
    """Cache npy file loading."""
    return load_video_features(npy_path)


def plot_similarity(
    sim_matrix,
    sim_smoothed,
    class_names,
    own_class,
    video_name,
    shape_info,
    show_raw,
    smoothing,
    anomaly_intervals=None,
):
    """
    Plot similarity-over-time chart for a single video.

    Args:
        sim_matrix: [T, C] raw similarity matrix
        sim_smoothed: [T, C] smoothed similarity matrix
        class_names: list of class names (legend labels)
        own_class: the class this video belongs to
        video_name: display name
        shape_info: shape tuple for display
        show_raw: whether to show raw signal behind smoothed
        smoothing: smoothing window size
        anomaly_intervals: list of (start_sec, end_sec) tuples or None
    """
    fig, ax = plt.subplots(figsize=(14, 4))
    time_axis = np.arange(sim_matrix.shape[0])

    # Draw anomaly shading first (behind everything)
    if anomaly_intervals:
        for i, (start, end) in enumerate(anomaly_intervals):
            ax.axvspan(
                start,
                end,
                color="red",
                alpha=0.12,
                zorder=0,
                label="Anomaly interval" if i == 0 else None,
            )

    own_idx = class_names.index(own_class) if own_class in class_names else -1

    for i, cls in enumerate(class_names):
        is_own = i == own_idx
        linewidth = 2.5 if is_own else 1.0
        alpha = 1.0 if is_own else 0.5
        zorder = 10 if is_own else 1

        if show_raw and smoothing > 1:
            ax.plot(
                time_axis,
                sim_matrix[:, i],
                alpha=0.15,
                color=f"C{i}",
                linewidth=0.5,
                zorder=0,
            )

        ax.plot(
            time_axis,
            sim_smoothed[:, i],
            label=cls,
            color=f"C{i}",
            linewidth=linewidth,
            alpha=alpha,
            zorder=zorder,
        )

    ax.set_xlabel("Time (seconds)", fontsize=11)
    ax.set_ylabel("Cosine Similarity", fontsize=11)
    ax.set_title(
        f"{video_name}  |  class: {own_class}  |  shape: {shape_info}",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(
        loc="upper right",
        fontsize=7,
        ncol=min(len(class_names) + (1 if anomaly_intervals else 0), 6),
        framealpha=0.8,
    )
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def main():
    st.set_page_config(
        page_title="VL Similarity Dashboard",
        layout="wide",
    )
    st.title("Vision-Language Similarity Dashboard")
    st.caption(
        "Visualize cosine similarity between video frame embeddings and CLIP text embeddings over time."
    )

    # ========== SIDEBAR ==========
    with st.sidebar:
        st.header("Data Settings")
        feature_root = st.text_input(
            "Feature Root Path",
            value="",
            help="Root directory containing class-name subfolders with .npy feature files.",
        )
        timestamp_dir = st.text_input(
            "Timestamp Annotation Dir (optional)",
            value="",
            help="Directory containing *_timestamps.csv files (e.g., Abuse_timestamps.csv). "
                 "Anomaly intervals will be shaded on the graphs.",
        )

        st.divider()
        st.header("Model Settings")
        model_key = st.selectbox("Text Encoder Model", get_available_models())
        model_info = get_model_info(model_key)
        st.caption(model_info["description"])

        # Conditional pretrained path input for models that need it
        pretrained_path = None
        needs_path = model_info.get("pretrained") is None
        if needs_path:
            pretrained_path = st.text_input(
                "Pretrained weights path (.pt)",
                help="Required for MobileCLIP v1 models. "
                     "For MobileCLIP v2, leave empty to use default 'dfndr2b' weights.",
            )

        device_options = ["cpu"]
        try:
            import torch

            if torch.cuda.is_available():
                device_options.append("cuda")
        except ImportError:
            pass
        device = st.selectbox("Device", device_options, index=0)

        st.divider()
        st.header("Text Prompt Settings")
        template_options = [
            "a photo of {}",
            "a video of {}",
            "{}",
            "Custom",
        ]
        template_choice = st.selectbox("Prompt Template", template_options)
        if template_choice == "Custom":
            template = st.text_input(
                "Custom template (use {} for class name)",
                value="a surveillance video of {}",
            )
        else:
            template = template_choice

        st.divider()
        st.header("Sampling & Visualization")
        seed = st.number_input("Random Seed", value=42, min_value=0, step=1)
        smoothing = st.slider("Smoothing Window (moving average)", 1, 20, 1)
        show_raw = st.checkbox(
            "Show raw signal alongside smoothed",
            value=False,
            disabled=smoothing <= 1,
        )

    # ========== MAIN AREA ==========

    # Validate feature root
    if not feature_root:
        st.info("Enter a feature root path in the sidebar to begin.")
        return

    if not os.path.isdir(feature_root):
        st.error(f"Directory not found: `{feature_root}`")
        return

    # Discover classes
    videos_per_class = list_videos_per_class(feature_root)
    class_names = sorted(videos_per_class.keys())

    if not class_names:
        st.error(
            "No class folders with .npy files found. "
            "Expected structure: `root/{ClassName}/*.npy`"
        )
        return

    # Class filter in sidebar
    with st.sidebar:
        st.divider()
        st.header("Class Filter")
        selected_classes = st.multiselect(
            "Select classes to analyze",
            class_names,
            default=class_names,
        )

    if not selected_classes:
        st.warning("Select at least one class from the sidebar.")
        return

    # Sample one video per class
    filtered = {c: videos_per_class[c] for c in selected_classes if c in videos_per_class}
    sampled = sample_one_per_class(filtered, seed=seed)

    # Show sampled videos
    with st.expander("Sampled Videos", expanded=False):
        for cls, path in sampled.items():
            video_name = os.path.splitext(os.path.basename(path))[0]
            st.text(f"  {cls}: {video_name}")

    # Run button
    run_clicked = st.sidebar.button("Run Analysis", type="primary", use_container_width=True)

    if not run_clicked:
        st.info("Configure settings in the sidebar, then click **Run Analysis**.")
        return

    # ========== ANALYSIS ==========

    # Dimension validation (quick check on first file)
    first_path = list(sampled.values())[0]
    first_feat = np.load(first_path)
    feat_dim = first_feat.shape[-1]
    expected_dim = model_info["embedding_dim"]

    if feat_dim != expected_dim:
        st.error(
            f"**Dimension mismatch!** Features have dim={feat_dim}, "
            f"but `{model_key}` produces dim={expected_dim}. "
            f"Please select the correct text encoder model."
        )
        return

    st.info(
        f"Feature dim: **{feat_dim}** | Model: **{model_key}** | "
        f"Classes: **{len(selected_classes)}** | "
        f"Ensure the text encoder matches the model used for feature extraction."
    )

    # Load text encoder
    try:
        with st.spinner(f"Loading text encoder: {model_key}..."):
            text_encoder = cached_load_text_encoder(model_key, device, pretrained_path)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return

    # Encode text prompts
    prompts = build_text_prompts(selected_classes, template=template)
    with st.expander("Text Prompts", expanded=False):
        for cls, prompt in zip(selected_classes, prompts):
            st.text(f"  {cls} -> \"{prompt}\"")

    try:
        with st.spinner("Encoding text prompts..."):
            text_embeddings = cached_encode_text(text_encoder, tuple(prompts))
    except Exception as e:
        st.error(f"Text encoding failed: {e}")
        return

    # Parse anomaly annotations (if provided)
    anomaly_map = {}
    if timestamp_dir and os.path.isdir(timestamp_dir):
        anomaly_map = parse_annotations(timestamp_dir)
        st.info(f"Loaded annotations for **{len(anomaly_map)}** videos from `{timestamp_dir}`")

    # Compute and plot per-video similarity
    st.divider()
    st.subheader("Similarity over Time")

    progress = st.progress(0, text="Processing videos...")
    n_videos = len(sampled)

    for idx, (class_name, npy_path) in enumerate(sampled.items()):
        video_name = os.path.splitext(os.path.basename(npy_path))[0]

        try:
            video_features = cached_load_features(npy_path)
        except Exception as e:
            st.warning(f"Error loading {npy_path}: {e}")
            continue

        sim_matrix = compute_similarity_matrix(video_features, text_embeddings)
        sim_smoothed = apply_smoothing(sim_matrix, smoothing)

        # Look up anomaly intervals for this video
        intervals = lookup_annotation(video_name, anomaly_map)

        fig = plot_similarity(
            sim_matrix=sim_matrix,
            sim_smoothed=sim_smoothed,
            class_names=selected_classes,
            own_class=class_name,
            video_name=video_name,
            shape_info=video_features.shape,
            show_raw=show_raw,
            smoothing=smoothing,
            anomaly_intervals=intervals,
        )
        st.pyplot(fig)
        plt.close(fig)

        progress.progress((idx + 1) / n_videos, text=f"Processed {idx + 1}/{n_videos}")

    progress.empty()

    # Summary heatmap: mean similarity per (video_class, text_class)
    st.divider()
    st.subheader("Summary: Mean Similarity Heatmap")

    heatmap_data = np.zeros((len(sampled), len(selected_classes)))
    video_labels = []

    for i, (class_name, npy_path) in enumerate(sampled.items()):
        video_name = os.path.splitext(os.path.basename(npy_path))[0]
        video_labels.append(f"{video_name}\n({class_name})")
        try:
            video_features = cached_load_features(npy_path)
            sim = compute_similarity_matrix(video_features, text_embeddings)
            heatmap_data[i] = sim.mean(axis=0)
        except Exception:
            pass

    fig_hm, ax_hm = plt.subplots(
        figsize=(max(8, len(selected_classes) * 0.8), max(4, len(sampled) * 0.6))
    )
    im = ax_hm.imshow(heatmap_data, aspect="auto", cmap="YlOrRd")
    ax_hm.set_xticks(range(len(selected_classes)))
    ax_hm.set_xticklabels(selected_classes, rotation=45, ha="right", fontsize=8)
    ax_hm.set_yticks(range(len(video_labels)))
    ax_hm.set_yticklabels(video_labels, fontsize=8)
    ax_hm.set_xlabel("Text Class")
    ax_hm.set_ylabel("Sampled Video")
    ax_hm.set_title("Mean Cosine Similarity (Video x Text Class)")

    # Annotate cells
    for i in range(heatmap_data.shape[0]):
        for j in range(heatmap_data.shape[1]):
            ax_hm.text(
                j, i, f"{heatmap_data[i, j]:.3f}",
                ha="center", va="center", fontsize=7,
                color="white" if heatmap_data[i, j] > heatmap_data.mean() else "black",
            )

    fig_hm.colorbar(im, ax=ax_hm, shrink=0.8)
    plt.tight_layout()
    st.pyplot(fig_hm)
    plt.close(fig_hm)


if __name__ == "__main__":
    main()
