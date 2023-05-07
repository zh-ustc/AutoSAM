import torch
from torch import nn
import numpy as np
from tsf import PositionalEmbedding,TransformerBlock
from torch.nn.init import xavier_normal_, uniform_, constant_
import torch.nn.functional as F

class sampler_trm(nn.Module):
    def __init__(self, args):
        super(sampler_trm, self).__init__()

        self.device = args.device
        d_model = args.d_model
        attn_heads = args.attn_heads
        d_ffn = args.d_ffn
        dropout = args.dropout
        self.max_len = args.max_len
        enable_res_parameter = args.enable_res_parameter
        self.attention_mask = torch.tril(torch.ones((self.max_len, self.max_len), dtype=torch.bool)).to(self.device)
        self.attention_mask.requires_grad = False
        self.TRM = TransformerBlock(d_model, attn_heads, d_ffn, enable_res_parameter, dropout)
        self.mlp = nn.Sequential(
            nn.Linear(2*d_model,d_model),
            nn.ReLU(),
            nn.Linear(d_model,2)#,
        )
        self.trans = nn.Linear(d_model,d_model)
        self.apply(self._init_weights)

        

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            stdv = np.sqrt(1. / self.num_item)
            uniform_(module.weight.data, -stdv, stdv)
        if isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0.1)


    def forward(self,id_embd,x,tau):
        item = self.trans(id_embd)
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
        mask *= self.attention_mask
        id_embd = self.TRM(id_embd, mask)
        p = self.mlp(torch.cat([id_embd,item],dim=-1))/tau
        p = p.softmax(dim=-1)
        return p[:,:,0]