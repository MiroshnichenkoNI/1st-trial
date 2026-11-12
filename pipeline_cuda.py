"""Decode video once and prepare inputs for DOVER and Qwen."""

import math
import os
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
sys.path[:0] = [str(ROOT / ".vendor/DOVER"), str(ROOT / ".vendor")]

import av
import numpy as np
import torch
import yaml
from dover.models import DOVER
from dover_views import UnifiedFrameSampler, get_single_view
from transformers import AutoProcessor, AutoTokenizer
from transformers.video_utils import VideoMetadata


PROMPT = "Сгенерируй детальное описание всего, что происходит на видео"


def qwen_frame_indices(count, source_fps, target_fps, min_frames=4, max_frames=768, factor=2):
    """Choose an even number of uniformly spaced frames for Qwen."""
    lower = math.ceil(min_frames / factor) * factor
    upper = math.floor(min(max_frames, count) / factor) * factor
    number = min(max(count / source_fps * target_fps, lower), upper, count)
    number = math.floor(number / factor) * factor
    if not factor <= number <= count:
        raise ValueError(f"Need at least {factor} source frames, got {count}")
    return np.linspace(0, count - 1, number).round().astype(int).tolist()


def prepare_video(path, sample_types, args, video_processor):
    """One producer owns CPU RNG; never seed or invoke CUDA from this function."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.random.default_generator.manual_seed(args.seed)
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        count, fps = stream.frames, float(stream.average_rate or 0)
        width, height = stream.width, stream.height
        if not count:
            count = sum(1 for _ in container.decode(stream))
    if count <= 0 or not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"Missing frame count/FPS: {path}")
    metadata = VideoMetadata(
        total_num_frames=count, fps=fps, width=width, height=height, duration=count / fps, video_backend="pyav"
    )
    qwen_indices = qwen_frame_indices(
        count,
        fps,
        args.fps,
        video_processor.min_frames,
        video_processor.max_frames,
        video_processor.temporal_patch_size,
    )
    indices = {}
    for name, spec in sample_types.items():
        sampler = (
            UnifiedFrameSampler(spec["clip_len"], spec["num_clips"], spec["frame_interval"])
            if "t_frag" not in spec
            else UnifiedFrameSampler(
                spec["clip_len"] // spec["t_frag"], spec["t_frag"], spec["frame_interval"], spec["num_clips"]
            )
        )
        indices[name] = sampler(count)
    required = set(qwen_indices) | set(np.concatenate(list(indices.values())).tolist())
    frames = {}
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = args.cpu_threads
        stream.codec_context.thread_type = args.decode_thread_type
        for index, frame in enumerate(container.decode(stream)):
            if index in required:
                frames[index] = torch.from_numpy(frame.to_ndarray(format="rgb24"))
            if len(frames) == len(required):
                break
    mean, std = torch.tensor([123.675, 116.28, 103.53]), torch.tensor([58.395, 57.12, 57.375])
    views = {}
    for name, spec in sample_types.items():
        clip = torch.stack([frames[int(i)] for i in indices[name]]).permute(3, 0, 1, 2)
        view = get_single_view(clip, name, **spec)
        view = ((view.permute(1, 2, 3, 0) - mean) / std).permute(3, 0, 1, 2)
        views[name] = view.reshape(3, spec["num_clips"], -1, *view.shape[2:]).transpose(0, 1)
    metadata.frames_indices = qwen_indices.copy()
    qwen_frames = torch.stack([frames[i] for i in qwen_indices]).permute(0, 3, 1, 2)
    return views, qwen_frames, metadata


class Pipeline:
    def __init__(self, args):
        self.args = args
        torch.cuda.set_device(args.dover_device)
        torch.set_num_threads(args.cpu_threads)
        torch.set_num_interop_threads(1)
        # DOVER keeps its official FP32 computation; no mixed-precision context.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        config = yaml.safe_load((ROOT / ".vendor/DOVER/dover.yml").read_text())
        self.sample_types = config["data"]["val-l1080p"]["args"]["sample_types"]
        self.dover = DOVER(**config["model"]["args"])
        self.dover.load_state_dict(
            torch.load(ROOT / ".models/dover/DOVER.pth", map_location="cpu", weights_only=True), strict=True
        )
        self.dover.to(args.dover_device, dtype=torch.float32).eval()
        self.processor = AutoProcessor.from_pretrained(ROOT / ".models/qwen", local_files_only=True)
        # Preparation and decoding run in different threads; give each its tokenizer.
        self.tokenizer = AutoTokenizer.from_pretrained(ROOT / ".models/qwen", local_files_only=True)
        torch.cuda.synchronize()

    def prepare(self, path):
        views, frames, metadata = prepare_video(path, self.sample_types, self.args, self.processor.video_processor)
        return {"path": path, "views": views, "qwen_inputs": self.build_qwen_inputs(path, frames, metadata)}

    def score_dover(self, prepared):
        views = prepared.pop("views")
        with torch.cuda.device(self.args.dover_device), torch.inference_mode():
            outputs = self.dover({key: value.to(self.args.dover_device) for key, value in views.items()})
            branch_means = torch.stack([value.mean() for value in outputs])
            # tolist() waits for the scores before the CPU publishes the result.
            scores = dict(zip(views, branch_means.tolist(), strict=True))
        return scores

    def build_qwen_inputs(self, path, frames, metadata):
        prompt = self.processor.apply_chat_template(
            [{"role": "user", "content": [
                {"type": "video", "video": str(path)},
                {"type": "text", "text": PROMPT},
            ]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.processor(
            text=[prompt],
            videos=[frames],
            video_metadata=[metadata],
            do_sample_frames=False,
            size={"shortest_edge": 4096, "longest_edge": self.args.video_pixels},
            padding=False,
            return_tensors="pt",
        )
        t, h, w = inputs["video_grid_thw"][0].tolist()
        vp = self.processor.video_processor
        pixels = t * vp.temporal_patch_size * h * w * vp.patch_size**2
        if pixels > self.args.video_pixels:
            raise ValueError(f"Qwen pixels {pixels} exceed budget {self.args.video_pixels}")
        return inputs
