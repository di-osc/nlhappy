from .attention import MultiHeadAttentionLayer
from .normalization import LayerNorm as BertLayerNorm
import torch.nn as nn
import torch
from .normalization import LayerNorm
import os
from typing import Optional,  Tuple
from .activation import activations


class BertEmbeddings(nn.Module):
    """bert的Embedding"""
    def __init__(
        self,
        vocab_size,
        type_vocab_size,
        hidden_size,
        max_position_embeddings,
        hidden_dropout_prob,
        layer_norm_eps
        ) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.positon_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.layer_norm = LayerNorm(hidden_size=hidden_size, eps=layer_norm_eps)


    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        # 如果不传token_type 则默认为不需要做序列种类判断
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        word_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.positon_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = word_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertIntermediate(nn.Module):
    def __init__(self, hidden_size, intermediate_size, hidden_act) -> None:
        super().__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)
        self.intermediate_act_fn = activations[hidden_act]

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertAddNorm(nn.Module):
    def __init__(self, intermediate_size, hidden_size, hidden_dropout_prob, layer_norm_eps) -> None:
        super().__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.layer_norm = BertLayerNorm(hidden_size=hidden_size, eps=layer_norm_eps)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states

class BertAttention(nn.Module):
    def __init__(
        self, 
        hidden_size, 
        num_attention_heads, 
        attention_probs_dropout_prob, 
        return_attention_scores,
        hidden_dropout_prob,
        layer_norm_eps) -> None:
        super().__init__()
        self.self = MultiHeadAttentionLayer(hidden_size, num_attention_heads, attention_probs_dropout_prob, return_attention_scores)
        self.output = BertAddNorm(
            intermediate_size=hidden_size,
            hidden_size=hidden_size,
            hidden_dropout_prob=hidden_dropout_prob,
            layer_norm_eps=layer_norm_eps
            )
    def forward(self, input_tensor, attention_mask=None, head_mask=None):
        self_outputs = self.self(input_tensor, input_tensor, input_tensor, attention_mask, head_mask)
        attention_output = self.output(self_outputs[0], input_tensor)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


class BertLayer(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        hidden_act,
        intermediate_size,
        hidden_dropout_prob,
        layer_norm_eps
        ) -> None:
        super().__init__()
        self.attention = BertAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_probs_dropout_prob=hidden_dropout_prob,
            hidden_dropout_prob=hidden_dropout_prob,
            layer_norm_eps=layer_norm_eps
        )
        self.intermediate = BertIntermediate(hidden_size=hidden_size,intermediate_size=intermediate_size, hidden_act=hidden_act)
        self.output = BertAddNorm(intermediate_size, hidden_size, hidden_dropout_prob, layer_norm_eps)

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        attention_outputs = self.attention(hidden_states, attention_mask, head_mask)
        attention_output = attention_outputs[0]

        # 这里是左上的 Add & Norm，从而得到完整的 FFN
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)

        # attention_outputs[0] 是 embedding, [1] 是 attention_probs
        outputs = (layer_output,) + attention_outputs[1:]
        return outputs


class BertEncoder(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        hidden_act,
        intermediate_size,
        hidden_dropout_prob,
        layer_norm_eps,
        output_attentions,
        output_hidden_states
        ) -> None:
        super().__init__()
        self.layer = nn.ModuleList([BertLayer(hidden_size, num_attention_heads, hidden_act, intermediate_size, hidden_dropout_prob, layer_norm_eps)])
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states


    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states += (hidden_states, )
            layer_outputs = layer_module(hidden_states, attention_mask, head_mask[i])
            hidden_states = layer_outputs[0]
            if self.output_attentions:
                all_attentions += (layer_outputs[1], )
        
        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            # 把中间层的结果取出来，一些研究认为中间层的 embedding 也有价值
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)

        # last-layer hidden state, (all hidden states), (all attentions)
        return outputs


class BertPooler(nn.Module):
    def __init__(
        self,
        hidden_size) -> None:
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        cls = hidden_states[:, 0]
        pooled_output = self.dense(cls)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertOutput:
    last_hidden_state: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

    def __init__(self, last_hidden_state, pooler_output, attentions):
        self.last_hidden_state = last_hidden_state
        self.pooler_output = pooler_output
        self.attentions = attentions



class Bert(nn.Module):
    def __init__(self,
        vocab_size,
        type_vocab_size,
        hidden_size,
        max_position_embeddings,
        hidden_dropout_prob,
        num_attention_heads,
        hidden_act,
        intermediate_size,
        output_attentions,
        output_hidden_states,
        layer_norm_eps:float = 3e-12
        ) -> None:
        super().__init__()
        self.embeddings =BertEmbeddings(
            vocab_size=vocab_size,
            type_vocab_size=type_vocab_size,
            hidden_size=hidden_size,
            max_position_embeddings=max_position_embeddings,
            hidden_dropout_prob=hidden_dropout_prob,
            layer_norm_eps=layer_norm_eps
        )
        self.encoder = BertEncoder(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            hidden_act=hidden_act,
            intermediate_size=intermediate_size,
            hidden_dropout_prob=hidden_dropout_prob,
            layer_norm_eps=layer_norm_eps,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )
        self.pooler = BertPooler(hidden_size=hidden_size)


    def init_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def from_pretrained(self, pretrained_model_path):
        if not os.path.exists(pretrained_model_path):
            print(f"missing pretrained_model_path: {pretrained_model_path}")
            pass

        state_dict = torch.load(pretrained_model_path, map_location='cpu')

        # 名称可能存在不一致，进行替换
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = key
            if 'gamma' in key:
                new_key = new_key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = new_key.replace('beta', 'bias')
            if 'bert.' in key:
                new_key = new_key.replace('bert.', '')
            # 兼容部分不优雅的变量命名
            if 'LayerNorm' in key:
                new_key = new_key.replace('LayerNorm', 'layer_norm')


            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)

        for old_key, new_key in zip(old_keys, new_keys):

            if new_key in self.state_dict().keys():
                state_dict[new_key] = state_dict.pop(old_key)
            else:
                # 避免预训练模型里有多余的结构，影响 strict load_state_dict
                state_dict.pop(old_key)

        # 确保完全一致
        self.load_state_dict(state_dict, strict=True)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(
                    -1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype)  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(input_ids, position_ids=position_ids, token_type_ids=token_type_ids)
        encoder_outputs = self.encoder(embedding_output,
                                       extended_attention_mask,
                                       head_mask=head_mask)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = BertOutput(last_hidden_state=sequence_output, pooler_output=pooled_output,
                             attentions=encoder_outputs[1:])

        return outputs

