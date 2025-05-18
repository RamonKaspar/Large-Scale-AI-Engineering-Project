import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset
from typing import Optional, Callable


def create_train_dataloader(dataset, batch_size: int, collate_fn: Optional[Callable] = None, is_distributed: bool = False, rank: int = 0, world_size: int = 1, seed: int = 42) -> DataLoader:
    """
    Create a dataloader with proper distributed sampling (if needed).
    """
    # Check if dataset is IterableDataset or regular Dataset
    is_iterable = isinstance(dataset, IterableDataset)
    
    sampler = None
    if is_distributed and not is_iterable:
        from distr.samplers import DistributedSampler
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=seed
        )
        shuffle = False  # We don't need to shuffle as the sampler will handle it
    else:
        shuffle = not is_iterable
    
    # Create dataloader with sampler passed as parameter
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
        drop_last=False
    )
    
    if sampler is not None:
        dataloader.dist_sampler = sampler
        
    return dataloader