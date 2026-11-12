"""Run DOVER and Qwen on one GPU; optimized scheduling is the default."""

import argparse
import asyncio
from contextlib import aclosing
import json
import os
from pathlib import Path
import re
import time
from types import SimpleNamespace

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
RESULT_FIELDS = (
    "job_id",
    "video_id",
    "technical_score",
    "aesthetic_score",
    "description",
    "generated_tokens",
    "finish_reason",
)


def settings():
    return dict(
        cpu_threads=4,
        prefetch=2,
        decode_thread_type="AUTO",
        seed=42,
        fps=2.0,
        video_pixels=8388608,
        max_tokens=768,
        dover_device="cuda:0",
        vllm_max_active=4,
        vllm_memory_utilization=0.6,
    )


def video_files(folder, names=None):
    if names is None:
        names = sorted(
            (path.name for path in folder.glob("*.mp4")),
            key=lambda name: [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", name)],
        )
    if not names:
        raise ValueError("No input videos")
    return names


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


async def phase_run(pipeline, paths, output, mode, detailed=False):
    rows = []
    started = time.perf_counter()
    with output.open("w", encoding="utf-8") as stream:
        async with aclosing(pipeline.run(paths, mode)) as results:
            async for row in results:
                rows.append(row)
                public = row if detailed else {key: row[key] for key in RESULT_FIELDS}
                stream.write(json.dumps(public, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
    return rows, time.perf_counter() - started


async def execute(folder, files, output, mode="optimized", benchmark=False):
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Output directory must be absent or empty")
    if benchmark and len(files) != 10:
        raise ValueError("Benchmark requires exactly ten videos")
    output.mkdir(parents=True, exist_ok=True)

    from pipeline import Pipeline

    config = settings()
    pipeline = Pipeline(SimpleNamespace(**config))
    try:
        paths = [folder / name for name in files]
        await phase_run(pipeline, paths, output / "warmup.jsonl", mode)
        rows, seconds = await phase_run(
            pipeline,
            paths * (5 if benchmark else 1),
            output / ("stream.jsonl" if benchmark else "videos.jsonl"),
            mode,
            detailed=benchmark,
        )
        report = dict(
            mode=mode,
            settings=config,
            runtime=pipeline.qwen.info,
            input_order=files,
            warmup_videos=len(paths),
            completed_videos=len(rows),
            seconds=seconds,
            generated_tokens=sum(row["generated_tokens"] for row in rows),
            truncated=sum(row["finish_reason"] == "length" for row in rows),
        )
        write_json(output / ("benchmark.json" if benchmark else "run.json"), report)
        print(f"{mode}: {len(rows)} videos, {seconds:.3f} s", flush=True)
        return report
    finally:
        pipeline.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--videos", nargs="+", help="Ordered filenames; default: all *.mp4")
    parser.add_argument("--mode", choices=("baseline", "optimized"), default="optimized")
    parser.add_argument("--benchmark", action="store_true", help="Warm up on ten videos, then repeat them five times")
    args = parser.parse_args()
    try:
        folder = args.input.resolve()
        files = video_files(folder, args.videos)
        asyncio.run(execute(folder, files, args.output.resolve(), args.mode, args.benchmark))
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
