import os

import torch
from transformers import TrainerCallback


class TorchProfilerCallback(TrainerCallback):
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.profiler = None
        self.enabled = int(os.environ.get("RANK", "0")) == 0

    def _trace_handler(self, prof):
        os.makedirs(self.output_dir, exist_ok=True)

        tensorboard_handler = torch.profiler.tensorboard_trace_handler(
            self.output_dir
        )
        tensorboard_handler(prof)

        summary = prof.key_averages().table(
            sort_by="cuda_time_total",
            row_limit=25,
        )

        with open(
            os.path.join(self.output_dir, "profile_summary.txt"),
            "w",
        ) as f:
            f.write(summary)

        print("\n=== TORCH PROFILER SUMMARY ===")
        print(summary)

    def on_train_begin(self, args, state, control, **kwargs):
        if not self.enabled:
            return

        self.profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=1,
                warmup=1,
                active=4,
                repeat=1,
            ),
            on_trace_ready=self._trace_handler,
            record_shapes=True,
            profile_memory=True,
        )

        self.profiler.start()

    def on_step_end(self, args, state, control, **kwargs):
        if self.profiler is not None:
            self.profiler.step()

    def on_train_end(self, args, state, control, **kwargs):
        if self.profiler is not None:
            self.profiler.stop()
