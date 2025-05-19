import os
import time

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from data.dataset import CollatorForCLM, ParquetDataset
from data.iterable_dataset import IterableParquetDataset
from data.pretokenized_dataset import PreTokenizedDataset, IterablePreTokenizedDataset
from model.model import Transformer, TransformerModelArgs
from utils import build_lr_scheduler, clip_grad_norm_, get_args, get_num_params, get_num_flop_per_token, init_logger, logger, PRECISION_STR_TO_DTYPE, set_default_dtype, setup_distributed, cleanup_distributed
from distr.data_utils import create_train_dataloader

def train(args):
  # Initialize distributed training (if requested)
  dist_info = None
  if args.distributed:
    # Set up distributed training
    dist_info = setup_distributed(args.backend)
    device = dist_info['device']
    rank = dist_info['rank']
    world_size = dist_info['world_size']
    local_rank = dist_info['local_rank']
    is_master = dist_info['is_master']
    # Log distributed training configuration from master process
    if is_master:
      logger.info(f"Distributed training enabled: {world_size} processes")
      logger.info(f"Master process: {rank} on {device}")
  else:
    device = torch.device(f"cuda:{int(os.getenv('LOCAL_RANK', 0))}")
    is_master = True
    rank = 0
    world_size = 1
      
  # Log experiment arguments from master process
  if is_master:
    logger.info(f"Experiment args: {args}")
  
  # Initialize model, tokenizer, etc.
  model_dtype = PRECISION_STR_TO_DTYPE[args.model_dtype]

  # Set up tokenizer
  if is_master:
      logger.info("Setting up Tokenizer...")
  tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)
  
  # Setup dataset and dataloader based on configuration
  if is_master:
    logger.info("Setting up DataLoaders...")
  
  if args.pretokenized:
    if is_master:
        logger.info(f"Using pretokenized data: {args.dataset}")
    if args.dataset_type == 'padded':
      # Use pretokenized padded dataset
      train_ds = PreTokenizedDataset(
        args.dataset,
        args.sequence_length,
        args.batch_size*args.training_steps
      )
      train_collator = CollatorForCLM(args.sequence_length, tokenizer.pad_token_id)
      
      # Create dataloader with distributed support if needed
      train_dl = create_train_dataloader(
        train_ds,
        args.batch_size,
        train_collator,
        is_distributed=args.distributed,
        rank=rank,
        world_size=world_size
      )
        
    elif args.dataset_type == 'token-list':
      # Use pretokenized token-list with iterable dataset
      train_ds = IterablePreTokenizedDataset(
        args.dataset,
        args.sequence_length,
        bos_token_id=tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 1
      )
      # Create dataloader with distributed support if needed
      train_dl = create_train_dataloader(
        train_ds,
        args.batch_size,
        None,  # No collator needed for IterableDataset
        is_distributed=args.distributed,
        rank=rank,
        world_size=world_size
      )
    else:
      raise NotImplementedError(f"Pretokenized dataset type '{args.dataset_type}' not implemented")
  else:
    # Original on-the-fly tokenization code
    if args.dataset_type == 'padded':
      # Original padded dataset with collator
      if is_master:
          logger.info("Using padded ParquetDataset with on-the-fly tokenization")
      train_ds = ParquetDataset(args.dataset, tokenizer, args.sequence_length, args.batch_size*args.training_steps)
      train_collator = CollatorForCLM(args.sequence_length, tokenizer.pad_token_id)
      
      # Create dataloader with distributed support if needed
      train_dl = create_train_dataloader(
          train_ds,
          args.batch_size,
          train_collator,
          is_distributed=args.distributed,
          rank=rank,
          world_size=world_size
      )
        
    elif args.dataset_type == 'padding-free':
      # Padding-free IterableParquetDataset
      if is_master:
          logger.info("Using padding-free IterableParquetDataset with on-the-fly tokenization")
      train_ds = IterableParquetDataset(
          args.dataset,
          tokenizer,
          args.sequence_length,
          bos_token_id=tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 1
      )
      
      # Create dataloader with distributed support if needed
      train_dl = create_train_dataloader(
          train_ds,
          args.batch_size,
          None,  # No collator needed for IterableDataset
          is_distributed=args.distributed,
          rank=rank,
          world_size=world_size
      )
  
  train_dl_iterator = iter(train_dl)

  # Set up Model
  if is_master:
    logger.info("Setting up Model...")
  model_config = TransformerModelArgs(
    dim=2048,
    n_layers=16,
    n_heads=16,
    n_kv_heads=4,
    ffn_dim_multiplier=1.3,
    multiple_of=256,
    rope_theta=500000,
    vocab_size=tokenizer.vocab_size,
    seq_len=args.sequence_length,
  )
  with set_default_dtype(model_dtype):
    model = Transformer(model_config).to(device)
  
  # Wrap model in DDP if distributed training is enabled
  if args.distributed:
    # Add a barrier before wrapping model with DDP
    torch.distributed.barrier()

    model = torch.nn.parallel.DistributedDataParallel(
      model,
      device_ids=[local_rank],
      output_device=local_rank,
      broadcast_buffers=False  # Can help with stability
    )
    
    # Log model parallelization info
    if is_master:
      logger.info(f"Model wrapped with DistributedDataParallel")
  
  if args.compile:
    if is_master:
      logger.info("Using `torch.compile`")
    model = torch.compile(model, fullgraph=True)
  
  model.train()

  # Build Optimizers & LR Scheduler
  optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, fused=args.fused_optimizer)
  lr_scheduler = build_lr_scheduler(optimizer, args.lr_warmup_steps)

  # Calculate global batch size for logging
  global_batch_size = args.batch_size * world_size
  if is_master:
      logger.info(f"Global batch size: {global_batch_size} (local: {args.batch_size} × {world_size} processes)")

  # Utils for calculating throughput
  num_flop_per_token = get_num_flop_per_token(
      get_num_params(model, exclude_embedding=True),
      model_config,
  )

  ntokens_since_last_log = 0
  ntraining_tokens_since_last_log = 0
  time_last_log = time.perf_counter()

  if is_master:
    logger.info("Starting training!")
  train_step = 0
  
  try:
    while train_step < args.training_steps:
      train_step += 1

      # Set epoch in sampler for distributed training
      if args.distributed and hasattr(train_dl, 'dist_sampler'):
        train_dl.dist_sampler.set_epoch(train_step)

      # Profiling
      if args.profile and args.profile_step_start == train_step and is_master:
          torch.cuda.cudart().cudaProfilerStart()
          torch.autograd.profiler.emit_nvtx(record_shapes=True).__enter__()

      # Get batch and move to device
      try:
          input_ids, labels = next(train_dl_iterator)
      except StopIteration:
          # Restart iterator if we've reached the end
          train_dl_iterator = iter(train_dl)
          input_ids, labels = next(train_dl_iterator)
          
      ntokens_since_last_log += args.batch_size * args.sequence_length * world_size
      num_items_in_batch = labels.ne(-100).sum()
      ntraining_tokens_since_last_log += num_items_in_batch
      
      input_ids = input_ids.to(device)
      labels = labels.to(device)

      optimizer.zero_grad()

      # Forward pass
      logits = model(input_ids)
      loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1).float(), labels.flatten(0, 1), reduction="sum")
      
      local_num_items = num_items_in_batch  # Store the local count
      
      # Gather counts across all processes if distributed
      if args.distributed:
          # All-reduce the number of -100 tokens to get total across all processes
          num_items_tensor = torch.tensor(num_items_in_batch, device=device)
          torch.distributed.all_reduce(num_items_tensor)
          num_items_in_batch = num_items_tensor.item()
      
      # Normalize by number of non-masked tokens
      loss = loss / local_num_items
      
      del logits
      
      # Backward pass
      loss.backward()

      # Clip gradients
      clip_grad_norm_(model.parameters(), args.grad_max_norm)

      # Optimizer step
      optimizer.step()
      lr_scheduler.step()

      # Logging
      if (train_step == 1 or train_step % args.logging_frequency == 0):
          # Synchronize before logging for accurate timing
          if args.distributed:
              torch.distributed.barrier()
          
          time_delta = time.perf_counter() - time_last_log
          
          # tokens per second per device, abbreviated as tps
          tps = ntokens_since_last_log / time_delta / world_size  # Divide by world_size to get per-device
          total_tps = ntokens_since_last_log / time_delta
          mfu = 100 * num_flop_per_token * tps / 989e12
          tflops = num_flop_per_token * tps / 1e12
          training_tps = ntraining_tokens_since_last_log / time_delta / world_size
          global_mfu = 100 * num_flop_per_token * total_tps / (989e12 * world_size)  # Scale denominator by world_size
          global_tflops = num_flop_per_token * total_tps / 1e12

          # If distributed, synchronize the loss across all processes
          if args.distributed:
              loss_tensor = torch.tensor([loss.item()], device=device)
              torch.distributed.all_reduce(loss_tensor, op=torch.distributed.ReduceOp.AVG)
              loss_item = loss_tensor.item()
          else:
              loss_item = loss.item()
          
          if is_master:
              logger.info(
                  f"Step: {train_step} | "
                  f"Loss: {loss_item:.2f} | "
                  f"Tokens per second: {tps:.2f} | "
                  f"Training tokens per second (%): {100*training_tps/tps:.2f} | "
                  f"MFU (%): {mfu:.2f} | "
                  f"TFLOPs: {tflops:.2f} | "
                  f"Global batch size: {global_batch_size} | "
                  f"Global tokens/sec: {total_tps:.2f} | "
                  f"Global MFU (%): {global_mfu:.2f} | "
                  f"Global TFLOPs: {global_tflops:.2f} | "
              )
          
          ntokens_since_last_log = 0
          ntraining_tokens_since_last_log = 0
          time_last_log = time.perf_counter()
      
      # Profiling
      if args.profile and args.profile_step_end == train_step and is_master:
          torch.cuda.cudart().cudaProfilerStop()

    if is_master:
      logger.info("Training completed")
          
  except Exception as e:
      logger.error(f"Error during training: {e}")
      raise
  finally:
      # Always clean up distributed environment
      if args.distributed:
          cleanup_distributed()

if __name__ == "__main__":
    init_logger()
    args = get_args()
    train(args)