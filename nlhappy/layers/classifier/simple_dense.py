import torch.nn as nn

class SimpleDense(nn.Module):
    def __init__(self,
                input_size: int,
                hidden_size: int,
                output_size: int):
        super(SimpleDense, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lin1 = nn.Linear(self.input_size, self.hidden_size)
        self.lin2 = nn.Linear(self.hidden_size, self.output_size)
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        self.activation = nn.ReLU()

        
    def forward(self, x):
        x = self.lin1(x)
        x = self.layer_norm(x)
        x = self.activation(x)
        x = self.lin2(x)
        return x