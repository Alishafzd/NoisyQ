import math
import torch
from torch import nn
import torch.nn.functional as F


device = "cuda:0" if torch.cuda.is_available() else "cpu"

class Network(nn.Module):
    def __init__(self, input_size, kernel_size, num_filters, out_features, device=device):
        super(Network, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, num_filters, kernel_size),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(int(num_filters*(input_size[0]-kernel_size+1)*(input_size[1]-kernel_size+1)/4), 32),
            nn.PReLU(),
            nn.Linear(32, out_features)
        )
        self.to(device)

    def forward(self, x):
        return self.net(x)

    def act(self, obs):
        obs_t = torch.as_tensor(obs, dtype=torch.float32).to(device)
        q_values = self.forward(obs_t.unsqueeze(0))
        max_q_index = torch.argmax(q_values, dim=1)[0]
        action = max_q_index.detach().item()
        return action


class NoisyLinear(nn.Module):
    """Noisy linear module for NoisyNet."""
    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        super(NoisyLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.Tensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer("bias_epsilon", torch.Tensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def reset_noise(self):
        epsilon_in = self.scale_noise(self.in_features)
        epsilon_out = self.scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(
            x,
            self.weight_mu + self.weight_sigma * self.weight_epsilon,
            self.bias_mu + self.bias_sigma * self.bias_epsilon,
        )

    @staticmethod
    def scale_noise(size: int) -> torch.Tensor:
        x = torch.randn(size)
        return x.sign().mul(x.abs().sqrt())


class NoisyNetwork(nn.Module):
    def __init__(self, input_size, kernel_size, num_filters, out_features, device=device):
        super(NoisyNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, num_filters, kernel_size),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(int(num_filters*(input_size[0]-kernel_size+1)*(input_size[1]-kernel_size+1)/4), 32),
            nn.PReLU(),
            NoisyLinear(32, out_features)
        )
        self.to(device)

    def forward(self, x):
        return self.net(x)

    def act(self, obs):
        obs_t = torch.as_tensor(obs, dtype=torch.float32).to(device)
        q_values = self.forward(obs_t.unsqueeze(0))
        max_q_index = torch.argmax(q_values, dim=1)[0]
        action = max_q_index.detach().item()
        return action