"""Optimizer utilities."""

import torch
from torch import optim
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


def create_optimizer(model, optim_config, fsdp_config, rank):
    """Create optimizer."""
    optimizer = optim.AdamW(
        model.parameters(),
        lr=optim_config.lr,
        betas=optim_config.get('betas', (0.9, 0.999)),
        weight_decay=optim_config.get('weight_decay', 1e-2),
    )
    return optimizer


def create_scheduler(optimizer, optim_config, fsdp_config, rank):
    """Create learning rate scheduler."""
    total_steps = optim_config.get('total_training_steps', 0)
    num_warmup_steps_ratio = optim_config.get('lr_warmup_steps_ratio', 0.)
    num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

    if rank == 0:
        print(f'Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}')

    scheduler = get_constant_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
    )
    return scheduler


def get_constant_schedule_with_warmup(optimizer, num_warmup_steps):
    """Create a schedule with a constant learning rate preceded by a warmup period."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return 1.0

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def offload_fsdp_optimizer(optimizer):
    """Offload optimizer state to CPU."""
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cpu()
