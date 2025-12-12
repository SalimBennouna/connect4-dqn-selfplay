import torch


class MinimumExponentialLR(torch.optim.lr_scheduler.ExponentialLR):
    def __init__(self, optimizer, lr_decay, min_lr=1e-6):
        self.min_lr = min_lr
        super().__init__(optimizer, gamma=lr_decay)

    def get_lr(self):
        return [
            max(base_lr * self.gamma ** self.last_epoch, self.min_lr)
            for base_lr in self.base_lrs
        ]