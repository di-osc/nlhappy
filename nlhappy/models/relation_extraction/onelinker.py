import torch.nn as nn
import torch
from ...layers.embedding import SinusoidalPositionEmbedding


class TokenPairRepresentation(nn.Module):
    """通过attention机制得到token pair的向量表示
    """
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 output_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lin1 = nn.Linear(input_size*2, hidden_size*output_size*3)
        
    def forward(self, x, mask):
        batch_size, seq_len, hidden_size = x.shape
        head = x.unsqueeze(2).expand(batch_size, seq_len, seq_len, hidden_size).reshape(batch_size, seq_len*seq_len, hidden_size)
        tail = x.repeat(1, seq_len, 1)
        pairs = torch.cat([head, tail], dim=-1)
        pairs = self.lin1(pairs)  # batch_size, seq*seq, output*hidden*3
        pairs = torch.split(pairs, self.hidden_size * 3, dim=-1)
        pairs = torch.stack(pairs, dim=-2) # batch_size, seq*seq, output ,hidden*3 
        q, k, v = pairs[..., :self.hidden_size], pairs[..., self.hidden_size:self.hidden_size*2], pairs[..., self.hidden_size*2:]
        logits = torch.einsum('bmhd , bnhd -> bhmn', q, k) # batch_size, output, seq*seq, seq*seq
        logits = logits / self.hidden_size ** 0.5
        pad_mask = mask.unsqueeze(-1) * mask.unsqueeze(1)
        pad_mask = pad_mask.reshape(2, 100).unsqueeze(1).unsqueeze(1)
        logits = logits*pad_mask - (1-pad_mask)*1e12
        score = torch.softmax(logits, dim=-1)
        context_layer = torch.matmul(score, v.permute(0, 2, 1, 3))
        return context_layer, score
    
    
# class TokenPairRepresentation(nn.Module):
#     """通过attention机制得到token pair的向量表示
#     """
#     def __init__(self,
#                  input_size: int,
#                  hidden_size: int,
#                  tag_size: int,
#                  output_size: int,
#                  dropout: float,
#                  RoPE: bool) -> None:
#         super().__init__()
#         self.hidden_size = hidden_size
#         self.output_size = output_size
#         self.tag_size = tag_size
#         self.head_lin = nn.Linear(input_size, hidden_size * output_size)
#         self.tail_lin = nn.Linear(input_size, hidden_size * output_size)
#         self.pair_lin = nn.Linear(hidden_size*2, hidden_size)
#         self.out_lin = nn.Linear(hidden_size, tag_size)
#         self.RoPE = RoPE
#         self.layer_norm = nn.LayerNorm(hidden_size)
#         self.activation = nn.ReLU()  
#         self.dropout = nn.Dropout(dropout) 
        
#     def forward(self, x):
#         batch_size, seq_len, hidden_size = x.shape
#         head = self.head_lin(x).reshape(batch_size, seq_len, self.output_size, self.hidden_size)  # batch_size seq_len, self.hidden_size * self.output_size
#         tail = self.tail_lin(x).reshape(batch_size, seq_len, self.output_size, self.hidden_size) # batch_size seq_len, self.hidden_size * self.output_size
#         if self.RoPE:
#             pos = SinusoidalPositionEmbedding(self.hidden_size, 'zero')(x)
#             cos_pos = pos[..., None, 1::2].repeat(1, 1, 1, 2)
#             sin_pos = pos[..., None, ::2].repeat(1, 1, 1, 2)
#             head2 = torch.stack([-head[..., 1::2], head[..., ::2]], 4)
#             head2 = torch.reshape(head2, head.shape)
#             head = head * cos_pos + head2 * sin_pos
#             tail2 = torch.stack([-tail[..., 1::2], tail[..., ::2]], 4)
#             tail2 = torch.reshape(tail2, tail.shape)
#             tail = tail * cos_pos + tail2 * sin_pos
#         head= head.reshape(batch_size, self.output_size, seq_len, self.hidden_size)
#         tail = tail.reshape(batch_size, self.output_size, seq_len, self.hidden_size)
#         head = head.unsqueeze(3).expand(batch_size, self.output_size, seq_len, seq_len, self.hidden_size).reshape(batch_size, self.output_size, seq_len*seq_len, self.hidden_size)
#         tail = tail.repeat(1, 1, seq_len, 1)
#         pairs = torch.cat([head, tail], dim=-1) # batch, self.output_size, seq*seq, hidden_size*2
#         pairs = self.dropout(pairs)
#         pairs = self.pair_lin(pairs)
#         pairs = self.layer_norm(pairs)
#         pairs = self.activation(pairs)
#         context_layer = self.out_lin(pairs).reshape(batch_size, seq_len, seq_len, self.output_size, self.tag_size) # batch_size, output, seq*seq, tag_size

#         return context_layer

