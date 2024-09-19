# from torch import nn
from einops import rearrange
# import torch
# import numpy

from ldm.modules.attention import FeedForward

import numpy as np
import torch

import torch.nn as nn

import torch
import math

class FixedPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(FixedPositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

 
class GlobalTextAdapter(nn.Module):
    def __init__(self, in_dim, max_len=768):
        super().__init__()
        self.in_dim = in_dim
        dim_out1 = in_dim*2
        dim_out2 = in_dim
        self.ff1 = FeedForward(in_dim, dim_out=dim_out1, mult=2, glu=True, dropout=0.1)
        self.ff2 = FeedForward(dim_out1, dim_out=dim_out2, mult=4, glu=True, dropout=0.3)
        self.norm1 = nn.LayerNorm(in_dim)
        self.norm2 = nn.LayerNorm(dim_out1)
        # self.positional_encoding = FixedPositionalEncoding(d_model=in_dim, max_len=max_len)

    def forward(self, x):
        x = self.ff1(self.norm1(x))
        x = self.ff2(self.norm2(x))
        # x = self.positional_encoding(x)
        return x

class LookUpTable(nn.Module):
    def __init__(self, channel, hidden_channel=128, num_candidate=32, w1=0.1, w2=0.9):
        super(LookUpTable, self).__init__()
        self.w1, self.w2 = w1, w2 # input lookup
        self.linear_q = nn.Linear(channel, hidden_channel)
        self.linear_k = nn.Linear(hidden_channel, hidden_channel)
        self.linear_v = nn.Linear(hidden_channel, channel)

        self.candidates = nn.Parameter(torch.randn((num_candidate, hidden_channel)))

    def forward(self, x):
        q = self.linear_q(x)
        k = self.linear_k(self.candidates)
        v = self.linear_v(self.candidates)
        attn = (q @ k.T) * (q.shape[1]**(-0.5))
        attn = nn.functional.softmax(attn, dim=-1)
        x_out = self.w1 * x + self.w2 * (attn @ v)

        return x_out

class GlobalContentAdapter(nn.Module):
    def __init__(self, in_dim, channel_mult=[2, 4]):
        super().__init__()
        dim_out1, mult1 = in_dim*channel_mult[0], channel_mult[0]*2
        dim_out2, mult2 = in_dim*channel_mult[1], channel_mult[1]*2//channel_mult[0]
        self.in_dim = in_dim
        self.channel_mult = channel_mult
        
        self.ff1 = FeedForward(in_dim, dim_out=dim_out1, mult=mult1, glu=True, dropout=0.0)
        self.ff2 = FeedForward(dim_out1, dim_out=dim_out2, mult=mult2, glu=True, dropout=0.0)
        self.norm1 = nn.LayerNorm(in_dim)
        self.norm2 = nn.LayerNorm(dim_out1)

        # self.lookup = LookUpTable(in_dim)

    def forward(self, x):
        # x = self.lookup(x)
        x = self.ff1(self.norm1(x))
        x = self.ff2(self.norm2(x))
        x = rearrange(x, 'b (n d) -> b n d', n=self.channel_mult[-1], d=self.in_dim).contiguous()
        return x