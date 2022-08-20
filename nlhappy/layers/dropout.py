import torch.nn as nn
import torch


class MultiDropout(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dropout_0 = nn.Dropout(p=0)
        self.dropout_1 = nn.Dropout(p=0.1)
        self.dropout_2 = nn.Dropout(p=0.2)
        self.dropout_3 = nn.Dropout(p=0.3)
        self.dropout_4 = nn.Dropout(p=0.4)

        
    def forward(self, x):
        output_0 = self.dropout_0(x)
        output_1 = self.dropout_1(x)
        output_2 = self.dropout_2(x)
        output_3 = self.dropout_3(x)
        output_4 = self.dropout_4(x)
        return torch.mean(torch.stack([output_0,output_1,output_2,output_3,output_4], dim=0), dim=0)