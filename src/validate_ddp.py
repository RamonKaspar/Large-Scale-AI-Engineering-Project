"""
DDP Verification Script: Verifies gradient synchronization
"""
import os
import torch
import torch.distributed as dist
import numpy as np
import argparse
import logging
from pathlib import Path

from model.model import Transformer, TransformerModelArgs
from transformers import AutoTokenizer
from utils import setup_distributed, cleanup_distributed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def verify_ddp_gradient_sync(args):
    """Verify DDP gradient synchronization - the core functionality of DDP."""
    results_dir = Path("verification_results")
    results_dir.mkdir(exist_ok=True)
    
    # Setup distributed environment
    dist_info = setup_distributed(args.backend)
    rank = dist_info['rank']
    world_size = dist_info['world_size'] 
    local_rank = dist_info['local_rank']
    device = dist_info['device']
    is_master = dist_info['is_master']
    
    if is_master:
        logger.info(f"Running verification with world_size={world_size}")
    
    log_file = results_dir / f"ddp_verification_rank{rank}.log"
    file_handler = logging.FileHandler(log_file)
    logger.addHandler(file_handler)
    
    logger.info(f"Process info - Rank: {rank}, World size: {world_size}, Device: {device}")
    
    # Set deterministic behavior for reproducibility
    # Different seed per rank to create different inputs
    torch.manual_seed(42 + rank)
    np.random.seed(42 + rank)
    
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    
    model_args = TransformerModelArgs(
        dim=4096,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        vocab_size=tokenizer.vocab_size,
        seq_len=args.sequence_length,
    )
    
    model = Transformer(model_args).to(device)
    
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank
    )
    logger.info(f"Rank {rank}: Model wrapped with DDP")
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    # Create DIFFERENT inputs for each rank, gradients should be identical after all-reduce
    sample_text = f"This is unique text for rank {rank} to verify DDP gradient synchronization."
    inputs = tokenizer.encode_plus(
        sample_text,
        max_length=args.sequence_length,
        padding='max_length',
        truncation=True,
        return_tensors="pt"
    ).to(device)
    
    model.train()
    optimizer.zero_grad()
    
    logits = model(inputs['input_ids'])
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        inputs['input_ids'].reshape(-1)
    )
    
    loss.backward()
    
    # VERIFICATION: Check gradient synchronization
    grad_hashes = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_hash = torch.sum(param.grad).item()
            grad_hashes.append((name, grad_hash))
    
    # Gather first gradient hash from all processes to verify synchronization
    ref_name, ref_hash = grad_hashes[0]
    local_hash = torch.tensor([ref_hash], device=device)
    gathered_hashes = [torch.zeros(1, device=device) for _ in range(world_size)]
    dist.all_gather(gathered_hashes, local_hash)
    
    # On master process, check if all processes have the same gradient hash
    verification_passed = True
    if is_master:
        logger.info(f"Verification for parameter: {ref_name}")
        logger.info(f"Gradient hashes: {[h.item() for h in gathered_hashes]}")
        
        # Check if all hashes are the same (within numerical tolerance)
        ref_value = gathered_hashes[0].item()
        for i, hash_tensor in enumerate(gathered_hashes):
            hash_value = hash_tensor.item()
            if abs(hash_value - ref_value) > 1e-5:
                logger.error(f"Gradient hash mismatch: {ref_value} vs {hash_value} (rank {i})")
                verification_passed = False
        
        if verification_passed:
            logger.info("VERIFICATION PASSED: DDP gradient synchronization is working correctly")
        else:
            logger.error("VERIFICATION FAILED: DDP gradient synchronization is not working correctly")
        
        with open(results_dir / "verification_summary.txt", "w") as f:
            f.write(f"DDP VERIFICATION: {'PASSED' if verification_passed else 'FAILED'}\n")
            f.write(f"Verified with world_size={world_size}\n")
            f.write(f"All processes have identical gradients: {verification_passed}\n")
    cleanup_distributed()
    return verification_passed

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DDP Gradient Synchronization Verification")
    parser.add_argument("--backend", type=str, default="nccl", choices=["nccl", "gloo"], help="DDP backend")
    parser.add_argument("--sequence-length", type=int, default=2048, help="Sequence length")
    parser.add_argument(
        "--tokenizer-name", 
        type=str, 
        default="unsloth/Mistral-Nemo-Base-2407-bnb-4bit",
        help="Tokenizer name or path"
    )
    args = parser.parse_args()
    
    verify_ddp_gradient_sync(args)