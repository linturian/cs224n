#!/usr/bin/env python3
# -*- coding: utf-8 -*-

### YOUR CODE HERE for part 1h
import torch
import torch.nn as nn


class Highway(torch.nn.Module):
    def __init__(self, filter_size):
        super(Highway, self).__init__()
        self.projection = torch.nn.Linear(filter_size, filter_size)
        self.gate = torch.nn.Linear(filter_size, filter_size)

    def forward(self, conv_out: torch.Tensor) -> torch.Tensor:
        proj = self.projection(conv_out)
        proj = nn.functional.relu(proj)
        gate = self.gate(conv_out)
        gate = torch.sigmoid(gate)
        highway = gate * proj + (1 - gate) * conv_out
        return highway

### END YOUR CODE
