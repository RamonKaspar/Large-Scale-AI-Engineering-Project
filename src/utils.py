import argparse
import functools
import logging
import os

from contextlib import contextmanager

import torch
import torch.distributed as dist
from torch.optim.lr_scheduler import LambdaLR

logger = logging.getLogger()

PRECISION_STR_TO_DTYPE = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
    "fp64": torch.float64,
}

def init_logger():
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)

def get_num_params(model: torch.nn.Module, exclude_embedding: bool = False) -> int:
    num_params = sum(p.numel() for p in model.parameters())
    if exclude_embedding:
        num_params -= sum(
            sum(p.numel() for p in m.parameters())
            for m in model.children()
            if isinstance(m, torch.nn.Embedding)
        )
    return num_params


def get_num_flop_per_token(num_params: int, model_config) -> int:
    l, h, q, t = (
        model_config.n_layers,
        model_config.n_heads,
        model_config.dim // model_config.n_heads,
        model_config.seq_len,
    )
    # Reasoning behind the factor of 12 for the self-attention part of the formula:
    # 1. each self-attention has 2 matmul in the forward and 4 in the backward (6)
    # 2. the flash attention does 1 more matmul recomputation in the backward
    #    but recomputation should not be counted in calculating MFU           (+0)
    # 3. each matmul performs 1 multiplication and 1 addition                 (*2)
    # 4. we follow the convention and do not account for sparsity in causal attention
    flop_per_token = 6 * num_params + 12 * l * h * q * t

    return flop_per_token


def build_lr_scheduler(optimizer: torch.optim, warmup_steps: int):

    def linear_warmup_constant(
        warmup_steps: int, current_step: int
    ) -> float:
        """Computes linear warmup followed by linear decay.

        Per LambdaLR requirement, this is accomplished by returning
        a multiplicative factor to adjust the learning rate to
        create the desired schedule.
        """
        if current_step < warmup_steps:
            # linear warmup
            # 0-indexed step, hence + 1 adjustments
            current_step += 1
            curr_adjustment = float(current_step / (warmup_steps + 1))

        else:
            # constant
            curr_adjustment = 1

        return curr_adjustment

    lr_lambda = functools.partial(linear_warmup_constant, warmup_steps)
    return LambdaLR(optimizer, lr_lambda)
    
@torch.no_grad()
def clip_grad_norm_(parameters, grad_max_norm):
  grads = [p.grad for p in parameters if p.grad is not None]
  total_norm = torch.nn.utils.get_total_norm(grads, error_if_nonfinite=True)
  torch.nn.utils.clip_grads_with_norm_(parameters, grad_max_norm, total_norm)
  return total_norm

@contextmanager
def set_default_dtype(dtype: torch.dtype):
    """
    Context manager to set torch's default dtype.
    """
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old_dtype)
        
def setup_distributed(backend='nccl'):
    """
    Initialize the distributed environment
    """
    # Set by torchrun
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    node_id = os.environ.get("SLURM_NODEID", "N/A")
    
    torch.cuda.set_device(local_rank)
    
    dist.init_process_group(backend=backend)
    
    logger.info(f"[Distributed Init] Rank {rank} initialized on node {node_id} on GPU {local_rank}.")
    
    # Add barrier to ensure all processes are ready before proceeding
    dist.barrier()
    if rank == 0:
        logger.info(f"[Rank {rank}] All ranks ready!")
    
    return {
        'rank': rank,
        'world_size': world_size,
        'local_rank': local_rank,
        'device': torch.device(f"cuda:{local_rank}"),
        'is_master': rank == 0
    }

def cleanup_distributed():
    """Clean up the distributed environment."""
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="/capstor/store/cscs/ethz/large-sc/datasets/train_data.parquet",
        help="Path to a parquet file containing a 'text' column with documents (`str`)",
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        choices=['padded', 'padding-free', 'token-list'],
        default='padded',
        help="Type of dataset to use: 'padded' (ParquetDataset), 'padding-free' (IterableParquetDataset), 'token-list'",
    )
    parser.add_argument(
        "--pretokenized",
        action='store_true',
        help="Use pretokenized data instead of on-the-fly tokenization"
    )
    parser.add_argument(
        "--tokenizer-name-or-path",
        type=str,
        default="unsloth/Mistral-Nemo-Base-2407-bnb-4bit",
        help="A path to a directory containing vocabulary files required by the tokenizer or the model id of a predefined tokenizer hosted inside a model repo on the Hugging Face Hub.",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=4096,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--fused-optimizer",
        action='store_true',
        help="Set to fuse the optimizer for increased performance or not"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
    )
    parser.add_argument(
        "--lr-warmup-steps",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--training-steps",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--logging-frequency",
        type=int,
        default=5,
        help="Log every `--logging-frequency` steps"
    )
    parser.add_argument(
        "--profile",
        action='store_true',
        help="Profile the run using the NSYS profiler"
    )
    parser.add_argument(
        "--profile-step-start",
        type=int,
        default=10,
        help="Starting step to profile using the NSYS profiler"
    )
    parser.add_argument(
        "--profile-step-end",
        type=int,
        default=10,
        help="Last step to profile using the NSYS profiler"
    )
    parser.add_argument(
        "--grad-max-norm",
        type=float,
        default=1,
    )
    parser.add_argument(
        "--model-dtype",
        type=str,
        default="bf16",
        help="Model dtype for parameters, gradients and optimizer states. Default: bf16",
    )
    parser.add_argument(
        "--compile",
        action='store_true',
        help="Set to compile the model with `torch.compile`"
    )
    parser.add_argument(
        "--distributed",
        action='store_true',
        help="Enable distributed data parallel training"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="nccl",
        choices=["nccl", "gloo"],
        help="Backend for distributed training (nccl for GPU, gloo for CPU)"
    )
    args = parser.parse_args()
    return args