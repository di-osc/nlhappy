import torch.nn as nn

class SimpleDense(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.net = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.LayerNorm(normalized_shape=self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.output_size)
            )

        
    def forward(self, inputs):
        
        logits = self.net(inputs)

        return logits