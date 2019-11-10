#!/usr/bin/env python3
# -*- coding: utf-8 -*-

### YOUR CODE HERE for part 1h
import torch
import torch.nn as nn


class CNN(torch.nn.Module):
    def __init__(self, char_embedded_size, filter_size, kernel_size=5, max_pooling=1):
        super(CNN, self).__init__()
        self.conv = nn.Conv1d(in_channels=char_embedded_size,
                              out_channels=filter_size,
                              kernel_size=kernel_size)
        self.max_pooling = max_pooling

    def forward(self, emb: torch.Tensor):
        # Reshape
        emb = emb.permute(0, 2, 1)

        conv = self.conv(emb)
        conv_relu = nn.functional.relu(conv)
        conv_relu = conv_relu.permute(0, 2, 1)
        conv_out = torch.max(conv_relu, dim=self.max_pooling)
        return conv_out
### END YOUR CODE
