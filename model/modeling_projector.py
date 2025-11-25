import torch
from torch import nn


class NaiveLinear(nn.Module):
    def __init__(self, mm_size, text_size):
        super().__init__()
        self.weight = nn.Linear(mm_size, text_size)
        
    def forward(self, x):
        return self.weight(x)
    

class TwoLayerMLP(nn.Module):
    def __init__(self, mm_size, text_size):
        super().__init__()
        self.w1 = nn.Linear(mm_size, text_size)
        self.act = nn.SiLU()
        self.w2 = nn.Linear(text_size, text_size)
        
    def forward(self, x):
        return self.w2(self.act(self.w1(x)))
    
    
NAME2PROJ = {
    "naive_linear": NaiveLinear,
    "mlp": TwoLayerMLP
}
