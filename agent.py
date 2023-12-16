import torch
from torch import nn
import numpy as np
import random
from network import Network


class Agent():
    def __init__(self, env, epsilon_start, epsilon_end, epsilon_decay, gamma, lr=5e-4):
        input_size = env.maze.shape
        out_features = 4
        kernel_size = 3
        num_filters = 6
        self.num_actions = out_features
        self.gamma = gamma
        self.targets = None
        self.action_qvalues = None
        self.online_net = Network(input_size, kernel_size, num_filters, out_features)
        self.target_net = Network(input_size, kernel_size, num_filters, out_features)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=lr)
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.counter = 1

    def choose_action(self, time, obs):
        epsilon = np.interp(time, [0, self.epsilon_decay], [self.epsilon_start, self.epsilon_end])
        rnd_sample = random.random()
        if rnd_sample <= epsilon:
            action = random.randrange(self.num_actions)
        else:
            action = self.online_net.act(obs)

        return action

    def calculate_target(self, new_obses_t, rews_t, dones_t):
        target_qvalues = self.target_net(new_obses_t)
        max_target_qvalues = target_qvalues.max(dim=1, keepdim=True)[0]

        self.targets = rews_t + self.gamma * (1 - dones_t) * max_target_qvalues

    def calculate_action_qvalues(self, obses_t, actions_t):
        q_values = self.online_net(obses_t)
        self.action_qvalues = torch.gather(input=q_values, dim=1, index=actions_t)

    def optimize_network(self):
        loss = nn.functional.smooth_l1_loss(self.action_qvalues, self.targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_net.load_state_dict(self.online_net.state_dict())