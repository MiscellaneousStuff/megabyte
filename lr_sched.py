from typing import Optional, List

def polynomial_decay_lr_schedule(
    lr: List[float],
    warmup_updates: int = 500,
    force_anneal: Optional[int] = None,
    end_learning_rate: float = 0.0,
    zero_lr_warmup_steps: int = 0,
    power: float = 1.0,
    total_num_update: float = None,
    optimizer=None,
    epoch=0,
    num_updates=0,
):
    assert total_num_update > 0

    current_lr = lr[0]
    if warmup_updates > 0:
        warmup_factor = 1.0 / warmup_updates
    else:
        warmup_factor = 1
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = warmup_factor * current_lr

    lrs = lr
    if force_anneal is None or epoch < force_anneal:
        # use fixed LR schedule
        next_lr = lrs[min(epoch, len(lrs) - 1)]
    else:
        # annneal based on lr_shrink
        next_lr = optimizer.param_groups[0]["lr"] # = optimizer.get_lr()

    current_lr = next_lr
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = warmup_factor * current_lr

    if zero_lr_warmup_steps > 0 and num_updates <= zero_lr_warmup_steps:
        lr = 0
    elif (
        warmup_updates > 0
        and num_updates <= warmup_updates + zero_lr_warmup_steps
    ):
        warmup_factor = (num_updates - zero_lr_warmup_steps) / float(
            warmup_updates
        )
        lr = warmup_factor * current_lr
    elif num_updates >= total_num_update:
        lr = end_learning_rate
    else:
        warmup = warmup_updates + zero_lr_warmup_steps
        lr_range = current_lr - end_learning_rate
        pct_remaining = 1 - (num_updates - warmup) / (
            total_num_update - warmup
        )
        lr = lr_range * pct_remaining**power + end_learning_rate
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = warmup_factor * current_lr

    return optimizer.param_groups[0]["lr"] # optimizer.get_lr()