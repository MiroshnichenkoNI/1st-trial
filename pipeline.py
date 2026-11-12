"""Fixed groups and continuous admission over the same video/model inputs."""

import asyncio
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import aclosing
from itertools import batched
import time

import torch

from pipeline_cuda import Pipeline as VideoPipeline
from qwen import Qwen


class Pipeline(VideoPipeline):
    def __init__(self, args):
        super().__init__(args)
        self.qwen = Qwen(args)
        self.batch_id = 0

    def prepare_group(self, jobs):
        prepared = []
        for job_id, path in jobs:
            item = self.prepare(path)
            item["job_id"] = job_id
            prepared.append(item)
        return prepared

    async def prepared_groups(self, paths, batch_size, prefetch):
        groups = iter(batched(enumerate(paths), batch_size))
        loop = asyncio.get_running_loop()
        producer = ThreadPoolExecutor(
            max_workers=1, initializer=torch.set_num_threads, initargs=(self.args.cpu_threads,),
        )
        pending = deque()

        def submit_next():
            group = next(groups, None)
            if group is not None:
                pending.append(loop.run_in_executor(producer, self.prepare_group, group))

        try:
            for _ in range(prefetch):
                submit_next()
            while pending:
                prepared = await pending.popleft()
                submit_next()
                yield prepared
        finally:
            for future in pending:
                future.cancel()
            producer.shutdown(wait=True, cancel_futures=True)

    async def describe(self, item, scores):
        output = await self.qwen.generate(item.pop("qwen_inputs"))
        completed = time.perf_counter()
        description = self.tokenizer.decode(output.token_ids, skip_special_tokens=True).strip()
        if not description:
            raise ValueError(f"Empty Qwen output: {item['path']}")
        return {
            "job_id": item["job_id"],
            "video_id": item["path"].name,
            "technical_score": scores["technical"],
            "aesthetic_score": scores["aesthetic"],
            "description": description,
            "generated_tokens": len(output.token_ids),
            "finish_reason": output.finish_reason,
            "inference_batch": {"id": item["job_id"], "size": 1},
            "completed_monotonic_s": completed,
        }

    async def run_grouped(self, paths):
        async with aclosing(self.prepared_groups(paths, batch_size=4, prefetch=1)) as groups:
            async for group in groups:
                scores = []
                for item in group:
                    scores.append(await asyncio.to_thread(self.score_dover, item))
                tasks = [asyncio.create_task(self.describe(item, score)) for item, score in zip(group, scores)]
                try:
                    rows = await asyncio.gather(*tasks)
                    await self.qwen.synchronize()
                    completed = time.perf_counter()
                    self.batch_id += 1
                    for row in rows:
                        row["inference_batch"] = {"id": self.batch_id, "size": len(rows)}
                        row["completed_monotonic_s"] = completed
                        yield row
                finally:
                    for task in tasks:
                        task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)

    async def run_streaming(self, paths):
        active = set()

        async def take_completed():
            nonlocal active
            done, active = await asyncio.wait(active, return_when=asyncio.FIRST_COMPLETED)
            rows = await asyncio.gather(*done)
            if not active:
                await self.qwen.synchronize()
                # The final completion also includes the executor handoff.
                latest = max(rows, key=lambda row: row["completed_monotonic_s"])
                latest["completed_monotonic_s"] = time.perf_counter()
            return sorted(rows, key=lambda row: row["completed_monotonic_s"])

        try:
            async with aclosing(self.prepared_groups(paths, batch_size=1, prefetch=self.args.prefetch)) as groups:
                while True:
                    if active and (len(active) == self.args.vllm_max_active or any(t.done() for t in active)):
                        for row in await take_completed():
                            yield row
                        continue
                    group = await anext(groups, None)
                    if group is None:
                        break
                    item = group[0]
                    scores = await asyncio.to_thread(self.score_dover, item)
                    active.add(asyncio.create_task(self.describe(item, scores)))
                while active:
                    for row in await take_completed():
                        yield row
        finally:
            for task in active:
                task.cancel()
            await asyncio.gather(*active, return_exceptions=True)

    async def run(self, paths, mode):
        stream = self.run_grouped(paths) if mode == "baseline" else self.run_streaming(paths)
        async with aclosing(stream):
            async for row in stream:
                yield row

    def close(self):
        self.qwen.close()
