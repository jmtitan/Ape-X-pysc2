'''Neural network model'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import padding
from torch.nn.modules.activation import ReLU


class Network(nn.Module):
    def __init__(self, num_actions):
        super(Network, self).__init__()
        self.dense = nn.Sequential(
            nn.Linear(48, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU()
        )
        self.action_scores = nn.Linear(512, num_actions)

    def forward(self, x):
        x = self.dense(x)
        qvalue = self.action_scores(x)
        return qvalue

    @torch.no_grad()
    def step(self, x):
        x = self.dense(x)
        v = self.action_scores(x)
        a = torch.argmax(v, 0).item()
        return a, v



