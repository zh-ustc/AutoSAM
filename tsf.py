import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_, uniform_, constant_

class PositionalEmbedding(nn.Module):

    def __init__(self, max_len, d_model):
        super(PositionalEmbedding, self).__init__()

        # Compute the positional encodings once in log space.
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x):
        batch_size = x.size(0)
        return self.pe.weight.unsqueeze(0).repeat(batch_size, 1, 1)

    def get(self,x):
        return self.pe(x)


class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, mask=None, dropout=None,p=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))
        #print(scores[0,0,-1])
        if p is not None:
#            scores = scores * p

            scores = scores + (p+1e-12).log()

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        #scores = torch.exp(scores)
        # with torch.no_grad():
        #     if (scores!=scores).sum()>0:
        #         print('exp')
        #         exit()


        # with torch.no_grad():
        #     if (scores!=scores).sum()>0:
        #         if (p!=p).sum()>0:
        #             print('p')
        #         else: print('pp')
        #         exit()
        #scores = scores + 1e-13
        #s = scores.sum(dim=-1,keepdim=True) + 1e-12
        #p_attn = F.normalize(scores,1,-1)


 
        p_attn = F.softmax(scores, dim=-1)
        #p_attn = F.gumbel_softmax(scores,dim=-1)
        #p_attn[p_attn<1e-2] = 0
        #p_attn = p_attn.masked_fill(p_attn<1e-1,0)
        #p_attn = F.normalize(p_attn,1,-1)
        # with torch.no_grad():
        #     x = (p_attn[0,0,-1,:]*1000).long()
        #     print(x.tolist())
        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn


class MultiHeadAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None,p=None):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout,p=p)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    """

    def __init__(self, size, enable_res_parameter, dropout=0.1):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        self.enable = enable_res_parameter
        if enable_res_parameter:
            self.a = nn.Parameter(torch.tensor(1e-8))

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        if not self.enable:
            return self.norm(x + self.dropout(sublayer(x)))
        else:
            return self.norm(x + self.dropout(self.a * sublayer(x)))


class PointWiseFeedForward(nn.Module):
    """
    FFN implement
    """

    def __init__(self, d_model, d_ffn, dropout=0.1):
        super(PointWiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.linear2(self.activation(self.linear1(x))))


class TransformerBlock(nn.Module):
    """
    TRM layer
    """

    def __init__(self, d_model, attn_heads, d_ffn, enable_res_parameter, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attn = MultiHeadAttention(attn_heads, d_model, dropout)
        self.ffn = PointWiseFeedForward(d_model, d_ffn, dropout)
        self.skipconnect1 = SublayerConnection(d_model, enable_res_parameter, dropout)
        self.skipconnect2 = SublayerConnection(d_model, enable_res_parameter, dropout)

        #self.norm = nn.LayerNorm(d_model)
        #self.dropout = nn.Dropout(dropout)

    def forward(self,x,mask,p=None):
        x = self.skipconnect1(x, lambda _x: self.attn.forward(_x, _x, _x, mask=mask,p=p))
        #x = self.norm(self.dropout(self.attn.forward(x, x, x, mask=mask,p=p)))
        x = self.skipconnect2(x, self.ffn)
        return x
