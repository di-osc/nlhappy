from transformers import AutoModel
import torch

class BERTSequenceClassificationNet(torch.nn.Module):
    def __init__(
        self,
        encoder_name:str ,
        mid_size:int,
        output_size:int
    ):
        super().__init__()
        self.bert = AutoModel.from_pretrained(encoder_name)
        self.pooler = torch.nn.Linear(self.bert.config.hidden_size, mid_size)
        self.classifier = torch.nn.Linear(mid_size, output_size)

    def forward(self, inputs):
        x = self.bert(**inputs).last_hidden_state
        x = x.mean(dim=1)
        x = self.pooler(x)
        x = self.classifier(x)
        return x



