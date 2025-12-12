import torch
import torch.nn.init as init


class QNetwork(torch.nn.Module):
    def __init__(self, n_obs, n_actions, l1, l2, l3=0, l4=0, l5=0):
        super().__init__()

        l3 = l3 or l2
        l4 = l4 or l3
        l5 = l5 or l4

        self.layers = torch.nn.Sequential(
            torch.nn.Linear(n_obs, l1),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(l1, l2),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(l2, l3),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(l3, l4),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(l4, l5),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(l5, n_actions),
        )

        for m in self.layers:
            if isinstance(m, torch.nn.Linear):
                init.xavier_uniform_(m.weight)

    def forward(self, x):
        return self.layers(x)