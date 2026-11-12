"""Qwen generation through AsyncLLM; vLLM owns its GPU process and queues."""

import itertools
import os

import torch

from pipeline_cuda import ROOT


def make_request(engine, request_id, inputs, params):
    """Pass existing HF pixels and prompt IDs to the pinned vLLM 0.11.2."""
    from transformers import BatchFeature
    from vllm.multimodal.inputs import (
        MultiModalFeatureSpec, MultiModalFieldConfig, MultiModalKwargsItems, PlaceholderRange,
    )

    ids = inputs["input_ids"][0]
    grid = inputs["video_grid_thw"]
    pixels = inputs["pixel_values_videos"]
    video_token = engine.vllm_config.model_config.hf_config.video_token_id
    positions = (ids == video_token).nonzero().flatten()
    if positions.numel() != int(grid.prod()) // 4 or pixels.shape[0] != int(grid.prod()):
        raise ValueError("Video tokens and patches disagree with the grid")
    offset, last = int(positions[0]), int(positions[-1])
    placeholder = PlaceholderRange(
        offset=offset, length=last - offset + 1, is_embed=(ids[offset:last + 1] == video_token),
    )
    fields = {
        "pixel_values_videos": MultiModalFieldConfig.flat_from_sizes("video", grid.prod(-1)),
        "video_grid_thw": MultiModalFieldConfig.batched("video"),
    }
    items = MultiModalKwargsItems.from_hf_inputs(
        BatchFeature({"pixel_values_videos": pixels, "video_grid_thw": grid}), fields,
    )
    request = engine.processor.process_inputs(request_id, {"prompt_token_ids": ids.tolist()}, params)
    request.mm_features = [MultiModalFeatureSpec(
        data=items["video"][0], modality="video", identifier=request_id, mm_position=placeholder,
    )]
    return request


class Qwen:
    def __init__(self, args):
        import pynvml

        # CUDA is already initialized by DOVER. Keep the same physical GPU in vLLM children.
        uuid = "GPU-" + str(torch.cuda.get_device_properties(0).uuid).removeprefix("GPU-")
        pynvml.nvmlInit()
        try:
            index = pynvml.nvmlDeviceGetIndex(pynvml.nvmlDeviceGetHandleByUUID(uuid))
        finally:
            pynvml.nvmlShutdown()
        os.environ["CUDA_VISIBLE_DEVICES"] = str(index)
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"

        from vllm import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM
        from vllm.usage.usage_lib import UsageContext

        self.args = args
        self.request_ids = itertools.count()
        self.engine = AsyncLLM.from_engine_args(AsyncEngineArgs(
            model=str(ROOT / ".models/qwen"),
            dtype="float16",
            tensor_parallel_size=1,
            max_model_len=8192,
            max_num_seqs=args.vllm_max_active,
            max_num_batched_tokens=8192,
            gpu_memory_utilization=args.vllm_memory_utilization,
            enable_prefix_caching=False,
            mm_processor_cache_gb=0,
            limit_mm_per_prompt={"image": 0, "video": 1},
            enforce_eager=False,
            async_scheduling=False,
            worker_extension_cls="vllm_extensions.SyncWorkerExtension",
            logits_processors=["vllm_extensions:SlidingPresencePenalty"],
            disable_log_stats=True,
            seed=args.seed,
        ), usage_context=UsageContext.LLM_CLASS)
        self.info = {"frontend": "AsyncLLM", "gpu": torch.cuda.get_device_name(), "gpu_uuid": uuid}

    async def generate(self, inputs):
        from vllm import SamplingParams
        from vllm.sampling_params import RequestOutputKind

        request_id = str(next(self.request_ids))
        params = SamplingParams(
            max_tokens=self.args.max_tokens, temperature=0.7, top_p=0.8, top_k=20,
            repetition_penalty=1.0, presence_penalty=0.0, frequency_penalty=0.0,
            seed=self.args.seed, output_kind=RequestOutputKind.FINAL_ONLY, detokenize=False,
        )
        request = make_request(self.engine, request_id, inputs, params)
        result = None
        async for output in self.engine.generate(request, params, request_id):
            if output.finished:
                result = output.outputs[0]
        if result is None:
            raise RuntimeError("Qwen finished without a result")
        return result

    async def synchronize(self):
        await self.engine.collective_rpc("synchronize_dover_handoff")

    def close(self):
        self.engine.shutdown()
