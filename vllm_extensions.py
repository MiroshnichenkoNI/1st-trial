"""Sliding presence penalty and a CUDA handoff inside the vLLM worker."""

import torch
from vllm.v1.sample.logits_processor.interface import LogitsProcessor, MoveDirectionality


class SlidingPresencePenalty(LogitsProcessor):
    def __init__(self, vllm_config, device, is_pin_memory):
        self.requests = {}

    def is_argmax_invariant(self):
        return False

    def update_state(self, batch_update):
        if batch_update is None:
            return
        for index in batch_update.removed:
            self.requests.pop(index, None)
        for index, params, prompt, output in batch_update.added:
            self.requests[index] = (prompt, output)
        for source, target, direction in batch_update.moved:
            left = self.requests.pop(source, None)
            right = self.requests.pop(target, None)
            if left is not None:
                self.requests[target] = left
            if direction == MoveDirectionality.SWAP and right is not None:
                self.requests[source] = right

    def apply(self, logits):
        for index, (prompt, output) in self.requests.items():
            previous = (prompt[-256:] + output[-256:])[-256:]
            if previous:
                ids = torch.tensor(previous, dtype=torch.long, device=logits.device)
                row = logits[index]
                row.scatter_(0, ids, row.gather(0, ids) - 1.5)
        return logits


class SyncWorkerExtension:
    def synchronize_dover_handoff(self):
        torch.cuda.synchronize()
