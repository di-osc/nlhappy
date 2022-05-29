import torch.nn as nn
import torch
import torch.nn.functional as F 


class SkipGram(nn.Module):
    """word2vector模型skipgram模式的复现
    参考:
    - https://github.com/jasoncao11/nlp-notebook/blob/master/1-1.Word2Vec/model.py
    """
    def __init__(self,
                vocab_size,
                hidden_size) -> None:
        '''
        参数:
        - vocab_size: 词表大小
        - hidden_size: 词向量维度
        '''
        super().__init__()
        self.in_embedding = nn.Embedding(vocab_size, hidden_size)
        self.out_embedding = nn.Embedding(vocab_size, hidden_size)


    def forward(self, in_ids, pos_ids, neg_ids):
        """
        参数:
        - input_ids: 中心词的id, 也是词表的下标
        - pos_ids: 对应背景词的id
        - neg_ids: 负采样的词id
        """
        # batch_size * 1 -> batch_size * hidden_size *1
        in_embed = self.in_embedding(in_ids)
        in_embed = in_embed.unsqueeze(-1)
        # batch_size * (window*2) -> batch_size * (window*2) * hidden_size
        pos_embed = self.out_embedding(pos_ids)
        # batch_size * K -> batch_size * K * hidden_size
        neg_embed = self.out_embedding(neg_ids)
        pos_dot = torch.bmm(pos_embed, in_embed)
        pos_dot = pos_dot.squeeze(-1)

        # 负采样的词应该跟中心词距离越远
        neg_dot = torch.bmm(neg_embed, in_embed) * (-1)
        neg_dot = neg_dot.squeeze(-1)

        log_pos = F.logsigmoid(pos_dot)
        log_neg = F.logsigmoid(neg_dot)

        return -(log_pos + log_neg).mean()


if __name__ == "__main__":
    model = SkipGram(10, 20)
    model.eval()
    in_ids = torch.tensor([2])
    pos_ids = torch.tensor([[1,3]])
    neg_ids = torch.tensor([[4,5]])
    distance = model(in_ids, pos_ids, neg_ids)
    print(distance)





