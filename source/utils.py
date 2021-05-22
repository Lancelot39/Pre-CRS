import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from collections import defaultdict
import numpy as np
import json
import os
import copy
import math
def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different
        (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) *
        (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class Attention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.hidden_dim = args.hidden_size
        self.projection = nn.Sequential(
            nn.Linear(args.hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, input_tensor):
        batch_size = input_tensor.size(0)
        # [B L 5 H] -> [B L 5 1]
        energy = self.projection(input_tensor)
        # [B L 5]
        weights = F.softmax(energy.squeeze(-1), dim=-2) # 可以返回weight 看attention的情况
        # [B L 5 H] * [B L 5 1] -> [B L 5 H]
        # 5个向量都已经乘过了权重 直接sum
        outputs = (input_tensor * weights.unsqueeze(-1)).sum(dim=-2)
        return outputs

def kmax_pooling(x, dim, k):
    index = x.topk(k, dim=dim)[1].sort(dim=dim)[0]
    return x.gather(dim, index).squeeze(dim)

def avg_pooling(x, dim):
    return x.sum(dim=dim)/x.size(1)

def get_item_audience():
    import pickle
    with open('item_audience_distribute.pkl') as fin:
        item_audience = pickle.load(fin)
    return item_audience

class Embeddings(nn.Module):
    """Construct the embeddings from item, position, attribute.
    """
    def __init__(self, args):
        super(Embeddings, self).__init__()

        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size) # 不要乱用padding_idx
        self.attr_embeddings = nn.Embedding(args.attr_size, args.hidden_size) # 只 性别 attr_size 为2 年龄 attr_size 为7
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)

        # self.item_attr_attention = Attention(args)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)

        attr_weight = torch.zeros(args.item_size, args.attr_size)
        import pickle
        with open('item_audience_distribute.pkl', 'rb') as fin: # 用户属性的分布
            self.item_audience = pickle.load(fin)
        for (item, v) in self.item_audience.items():
            attr_weight[item, :] = torch.tensor(v[1], dtype=torch.float) # v[0] 是性别属性分布 v[1] 是年龄属性分布
        self.item2attr_audi = nn.Embedding.from_pretrained(attr_weight, freeze=True)
        self.args = args

    def forward(self, input_ids, attr_ids, add_attr=True, cuda_condition=False):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        items_embeddings = self.item_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        # TODO 每一个item都加上  attr_embedding 按比例加进去 tensor([0, 1])
        # all_attr = torch.arange(0, self.args.attr_size)
        # if cuda_condition:
        #     all_attr = all_attr.cuda()
        # all_attr_embedding = self.attr_embeddings(all_attr) # 把所有属性向量都取出来 [attr_size hidden]
        # attr_weight = self.item2attr_audi(input_ids) # [batch len 2]
        # attr_embeddings = (attr_weight.unsqueeze(-1)*all_attr_embedding).sum(-2) # [batch len hidden]
        #

        # items_embeddings [batch seq_len hidden]
        # attr_embedding [batch hidden]

        # if add_attr:
        #     attr_embeddings = self.attr_embeddings(attr_ids)
        #     attr_embeddings = attr_embeddings.unsqueeze(1).expand_as(items_embeddings)
        #     embeddings = items_embeddings + position_embeddings + attr_embeddings
        # else:
        #     embeddings = items_embeddings + position_embeddings

        embeddings = items_embeddings + position_embeddings

        # 修改属性
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings

class SelfAttention(nn.Module):
    def __init__(self, opt):
        super(SelfAttention, self).__init__()
        self.num_attention_heads = opt['num_attention_heads']
        self.attention_head_size = int(opt['dim'] / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(opt['dim'], self.all_head_size)
        self.key = nn.Linear(opt['dim'], self.all_head_size)
        self.value = nn.Linear(opt['dim'], self.all_head_size)

        self.attn_dropout = nn.Dropout(opt['dropout'])

        # 做完self-attention 做一个前馈全连接 LayerNorm 输出
        self.dense = nn.Linear(opt['dim'], opt['dim'])
        self.LayerNorm = LayerNorm(opt['dim'], eps=1e-12)
        self.out_dropout = nn.Dropout(opt['dropout'])

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor, attention_mask):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)  #batch*head*p_l*hidden
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)

        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states

class Intermediate(nn.Module):
    def __init__(self, opt):
        super(Intermediate, self).__init__()
        self.dense_1 = nn.Linear(opt['dim'], opt['dim'] * 4)

        self.intermediate_act_fn = gelu

        self.dense_2 = nn.Linear(opt['dim'] * 4, opt['dim'])
        self.LayerNorm = LayerNorm(opt['dim'], eps=1e-12)
        self.dropout = nn.Dropout(opt['dropout'])

    def forward(self, input_tensor):
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class Layer(nn.Module):
    def __init__(self, opt):
        super(Layer, self).__init__()
        self.attention = SelfAttention(opt)
        self.intermediate = Intermediate(opt)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        return intermediate_output

class Encoder(nn.Module):
    def __init__(self, opt):
        super(Encoder, self).__init__()
        layer = Layer(opt)
        self.layer = nn.ModuleList([copy.deepcopy(layer)
                                    for _ in range(opt['num_layers'])])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers

class SASRecModel(nn.Module):
    def __init__(self, args):
        super(SASRecModel, self).__init__()
        self.embeddings = Embeddings(args)
        self.encoder =Encoder(args)
        self.args = args

        # fixme 引入多任务
        self.attr_dense = nn.Linear(args.hidden_size, args.hidden_size)
        self.act = nn.Tanh()
        self.dropout = nn.Dropout(p=args.hidden_dropout_prob)
        self.attr_class = nn.Linear(args.hidden_size, args.attr_size)

        # conact 之后做 转换
        self.convert_layer = nn.Linear(args.hidden_size*2, args.hidden_size)
        self.apply(self.init_sas_weights)


    def forward(self, task_type, input_ids, attr_ids, attention_mask=None, add_attr=False, add_gru=False,
                cuda_condition=False, output_all_encoded_layers=True):

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) # torch.int64
        # 添加mask 只关注前几个物品进行推荐
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1) # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long()

        if cuda_condition:
            subsequent_mask = subsequent_mask.cuda()
        extended_attention_mask = extended_attention_mask * subsequent_mask

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding = self.embeddings(input_ids, attr_ids,  add_attr, cuda_condition)

        encoded_layers = self.encoder(embedding,
                                      extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers)
        # [B L H]
        sequence_output = encoded_layers[-1]

        class_output = self.attr_dense(sequence_output[:, -1]) # [B H]
        class_output = self.dropout(class_output)
        class_output = self.act(class_output)
        class_output = self.attr_class(class_output)
        return class_output

    def init_sas_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()