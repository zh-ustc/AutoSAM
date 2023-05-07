import torch
from torch import nn
import numpy as np
from tsf import PositionalEmbedding,TransformerBlock
from torch.nn.init import xavier_normal_, uniform_, constant_,xavier_uniform_
import torch.nn.functional as F

                    
class SASRec(nn.Module):
    def __init__(self, args):
        super(SASRec, self).__init__()
        self.num_item = args.num_item + 1
        self.device = args.device
        d_model = args.d_model
        attn_heads = args.attn_heads
        d_ffn = args.d_ffn
        layers = args.bert_layers
        dropout = args.dropout
        self.max_len = args.max_len
        enable_res_parameter = args.enable_res_parameter
        self.token = nn.Embedding(self.num_item, d_model)
        self.attention_mask = torch.tril(torch.ones((self.max_len, self.max_len), dtype=torch.bool)).to(self.device)
        self.attention_mask.requires_grad = False
        self.TRMs_id = nn.ModuleList(
            [TransformerBlock(d_model, attn_heads, d_ffn, enable_res_parameter, dropout) for i in range(layers)])
        self.position = PositionalEmbedding(self.max_len, d_model)
        self.pred = nn.Linear(d_model,self.num_item)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            stdv = np.sqrt(1. / self.num_item)
            uniform_(module.weight.data, -stdv, stdv)
        if isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0.1)
    
    def forward(self,x,label,negs,position=None):
        '''
            x : input seq , [B,L]
            label : next item at each step, [B,L]
            negs : [N]
            position : position in original sequence [B,L] 
        '''

        if position is None:
            id_embd = self.token(x) + self.position(x)
        else:
            id_embd = self.token(x) + self.position.get(position)

        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
        mask *= self.attention_mask[:len(x[0]),:len(x[0])]
        
        for TRM in self.TRMs_id:
            id_embd = TRM(id_embd,mask)

        label = label.reshape(-1)        
        pos_item = label[label>0] # M
        pos_embd = self.token(pos_item) # M,d
        neg_embd = self.token(negs) # N,d
        id_embd = id_embd.view(x.size(0)*x.size(1),-1)[label>0] # M,d
        pos_score = (pos_embd * id_embd).sum(dim=-1,keepdim=True) # M,1
        neg_score = torch.matmul(id_embd,neg_embd.t()) # M,N
        mask = pos_item.unsqueeze(1) == negs.unsqueeze(0)
        neg_score[mask] = -1e9
        pre = torch.cat([pos_score,neg_score],dim=1) # M,N+1
        return pre


    def inf(self,x,position=None):
        '''
            x : input seq , [B,L]
            position : position in original sequence [B,L] 
        '''
        if position is None:
            id_embd = self.token(x) + self.position(x)
        else:
            id_embd = self.token(x) + self.position.get(position)
     
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
        mask *= self.attention_mask[:len(x[0]),:len(x[0])]
        for TRM in self.TRMs_id:
            id_embd = TRM(id_embd, mask)

        id_embd = id_embd[:,-1,:] # B,d
        pre = torch.matmul(id_embd,self.token.weight.t())
        return pre

