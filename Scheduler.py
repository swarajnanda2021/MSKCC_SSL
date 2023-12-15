import torch

class CustomScheduler: # Need this else NaNs
    def __init__(self, optimizer, warmup_epochs, initial_lr, final_lr, total_epochs):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.total_epochs = total_epochs
        self.after_warmup_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs)

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # Warm-up phase
            lr = self.initial_lr + (self.final_lr - self.initial_lr) * epoch / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        else:
            # After warm-up, use the cosine annealing schedule
            self.after_warmup_scheduler.step(epoch - self.warmup_epochs)


