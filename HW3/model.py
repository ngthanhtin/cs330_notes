import torch
from torch import nn

class Model(nn.Module):
    '''Define Q-model'''
    def __init__(self, num_bits):
        super(Model, self).__init__()
        hidden_dim = 256
        self.relu = nn.ReLU()
        self.linear = nn.Linear(2 * num_bits, hidden_dim)
        self.out = nn.Linear(hidden_dim, num_bits)

    def forward(self, inputs):
        linear = self.relu(self.linear(inputs))
        out = self.out(linear)

        return out