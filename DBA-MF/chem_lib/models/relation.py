

import torch
import torch.nn as nn
import torch.nn.functional as F

import os


class ProtoAttention(torch.nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.linear_q = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear_k = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear_v = torch.nn.Linear(hidden_dim, hidden_dim)
        self.scale = hidden_dim ** 0.5
        self.alpha = nn.Parameter(torch.tensor(0.3))  ####node:0.2043   coord:0.1775

    def forward(self, q_emb, proto):
        Q = self.linear_q(q_emb)  # [n_q, d]
        K = self.linear_k(proto)  # [n_p, d]
        V = self.linear_v(proto)  # [n_p, d]

        attn = torch.softmax(Q @ K.T / self.scale, dim=-1)  # [n_q, n_p]
        context = attn @ V  # [n_q, d]


        return q_emb + self.alpha * context
class Messnode(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Messnode, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(0.3))

    def topk_mask(self,A, k):
        _, topk_indices = torch.topk(A, k, dim=-1)
        return torch.zeros_like(A).scatter(-1, topk_indices, 1.0)

    def forward(self, A, X,k):
        sim = A.clone()
        A = self.topk_mask(sim, k)
        X_agg =  A @ X
        X_out = X + self.alpha * X_agg
        return X_out

class enhanced(nn.Module):
    def __init__(self, inp_dim, hidden_dim,top_k,dataset,n_shot):
        super(enhanced, self).__init__()

        self.top_k = top_k
        self.ProtoAttention = ProtoAttention(inp_dim)
        self.message_node=Messnode(in_dim=inp_dim, out_dim=hidden_dim)

        self.dataset=dataset
        self.n_shot_train=n_shot
    def forward(self, s_emb, q_emb):
        n_support = s_emb.size(0)
        n_query = q_emb.size(0)
        pos_emb = torch.chunk(s_emb, 2)[1]
        neg_emb = torch.chunk(s_emb, 2)[0]
        pos_proto = torch.mean(pos_emb, dim=0, keepdim=True)
        neg_proto = torch.mean(neg_emb, dim=0, keepdim=True)
        proto=torch.cat([pos_proto, neg_proto], dim=0)
        x_s = s_emb / torch.norm(s_emb, dim=-1, keepdim=True)  # 归一化
        sim_node = torch.mm(x_s, x_s.T)  # 余弦相似度
        enhanced_s_emb = self.message_node(sim_node, s_emb,k=self.top_k)
        enhanced_q=self.ProtoAttention(q_emb,proto)
        return enhanced_s_emb,enhanced_q
