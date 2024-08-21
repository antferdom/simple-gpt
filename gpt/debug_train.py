import subprocess
import os
from pathlib import Path
import sys

from typing import Optional

WD_DIR: str = Path(__file__).parent.absolute()
CLI_PATH: str = os.path.join(WD_DIR, "train.py")

num_warmup_steps: Optional[int] = 0
seed: Optional[int] = 1337
gradient_checkpointing: Optional[bool] = False
gradient_checkpointing_offload: Optional[bool] = False
zero_stage: Optional[int] = 0
output_dir: Optional[str] = "output"
offload: Optional[bool] = False
debug: Optional[bool] = False
run_name: Optional[str] = None
local_rank: Optional[int] = -1
profile: Optional[bool] = False
sequence_length: Optional[int] = 1024
micro_batch_size: Optional[int] = 32
global_batch_size: Optional[int] = 524288
learning_rate: Optional[float] = 6e-4
weight_decay: Optional[float] = 0.1
epochs: Optional[int] = 1
lr_scheduler_type: Optional[str] = "linear"
sparse_ffn: Optional[bool] = False
use_compile: Optional[bool] = False
command: str = f"""
                {sys.executable} {CLI_PATH}
                --num-warmup-steps {num_warmup_steps}
                --seed {seed}
                --no-gradient-checkpointing
                --no-gradient-checkpointing 
                --zero-stage {zero_stage}
                --output-dir {output_dir}
                --no-offload
                --no-debug
                --no-profile
                --sequence-length {sequence_length}
                --micro-batch-size {micro_batch_size}
                --global-batch-size {global_batch_size}
                --learning-rate {learning_rate}
                --weight-decay {weight_decay}
                --epochs {epochs}
                --lr-scheduler-type {lr_scheduler_type}
                --no-sparse-ffn 
                --no-use-compile
                """
print(command)
subprocess.run(command.split())