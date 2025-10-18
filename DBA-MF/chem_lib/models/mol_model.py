import torch
import torch.nn as nn
from sympy.physics.vector.tests.test_printing import alpha

from .encoder import GNN_Encoder
from .relation import enhanced
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import radius_graph, global_mean_pool
from torch_scatter import scatter_add
import torch.nn.functional as F


class TinyEGNNLayer(MessagePassing):
    def __init__(self, hidden_dim,out_dim,edge_dim,coord_scale=0.1):
        super().__init__(aggr='add')
        self.coord_scale = torch.nn.Parameter(torch.tensor(0.3))
        # self.coord_scale = 0.3
        in_dim = 2* hidden_dim + edge_dim+1
        self.edge_mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.SiLU(),
            # nn.Linear(hidden_dim, hidden_dim)

        )  # φ_e

        self.coord_mlp = nn.Sequential(
            nn.Linear(out_dim, 1),
            nn.SiLU()
        )  # φ_x

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim+out_dim, out_dim),
            nn.SiLU(),
            # nn.Linear(hidden_dim, hidden_dim)
        )  # φ_h
    def rbf(self, dist2):
        dist = torch.sqrt(dist2 + 1e-8)  # [E,1]
        diff = dist - self.rbf_centers.view(1, -1)  # [E, K]
        rbf_feat = torch.exp(- (diff ** 2) / self.rbf_gamma)  # [E,K]
        return rbf_feat
    def forward(self, h, x, edge_index, edge_attr):
        self.num_nodes = h.size(0)
        self.edge_index=edge_index
        max_idx = edge_index.max().item()
        return self.propagate(edge_index=edge_index, h=h, x=x, edge_attr=edge_attr)

    def message(self, h_i, h_j, x_i, x_j, edge_attr):
        rel_pos = x_i - x_j
        dist2 = (rel_pos ** 2).sum(dim=-1, keepdim=True)
        edge_input = torch.cat([h_i, h_j, dist2, edge_attr], dim=-1)
        m_ij = self.edge_mlp(edge_input)
        coord_coeff = self.coord_mlp(m_ij)
        coord_update = rel_pos * coord_coeff
        return m_ij, coord_update

    def aggregate(self, inputs, index, dim_size=None):
        m_ij, coord_update = inputs
        agg_msg = scatter_add(m_ij, index, dim=0, dim_size=self.num_nodes)
        agg_coord = scatter_add(coord_update, index, dim=0, dim_size=self.num_nodes)
        return agg_msg, agg_coord

    def update(self, aggr_out, h, x):
        m_i, delta_x = aggr_out
        x_new = x + self.coord_scale * delta_x
        h_new = self.node_mlp(torch.cat([h, m_i], dim=-1))
        return h_new, x_new

class EGNN(nn.Module):
    def __init__(self, hidden_dim, normalize=True):
        super().__init__()
        self.egnn1 = TinyEGNNLayer(hidden_dim,out_dim=hidden_dim,edge_dim=300,)
        self.egnn2 = TinyEGNNLayer(hidden_dim,out_dim=128,edge_dim=300)
        self.egnn3 = TinyEGNNLayer(128,out_dim=128,edge_dim=300)
        self.egnn4 = TinyEGNNLayer(128,out_dim=128,edge_dim=300)
        self.norm1 = nn.BatchNorm1d(hidden_dim)
        self.norm2 = nn.BatchNorm1d(128)
        self.norm3 = nn.BatchNorm1d(128)
        self.norm4 = nn.BatchNorm1d(128)
        self.drop = nn.Dropout(0.2)
    def forward(self, h, pos, edge_index, edge_attr):
        h1, x1 = self.egnn1(h, pos, edge_index, edge_attr)
        h1 = self.norm1(h1)
        h1 = F.silu(h1)
        h1 = self.drop(h1)
        h2, x2 = self.egnn2(h1, x1, edge_index, edge_attr)
        h2 = self.norm2(h2)
        h2 = F.silu(h2)
        h2 = self.drop(h2)
        h3, x3 = self.egnn3(h2, x2, edge_index, edge_attr)
        h3= self.norm3(h3)
        h3= F.silu(h3)
        h3 = self.drop(h3)
        h4, x4 = self.egnn4(h3, x3, edge_index, edge_attr)
        h4= self.norm4(h4)
        h4 = self.drop(h4)
        return h4

class Classifer(nn.Module):
    def __init__(self, inp_dim,num_class=2, pre_dropout=0.0):
        super(Classifer, self).__init__()
        self.inp_dim = inp_dim
        self.pre_dropout = pre_dropout


        if self.pre_dropout > 0:
            self.predrop1 = nn.Dropout(p=self.pre_dropout)

        self.fc1 = nn.Sequential(nn.Linear(self.inp_dim, 128), nn.LeakyReLU())
        if self.pre_dropout > 0:
            self.predrop2 = nn.Dropout(p=self.pre_dropout)
        self.fc2 = nn.Linear(128, num_class)


    def forward(self, all_emb,s_size):
        node_feat = all_emb
        if self.pre_dropout > 0:
            node_feat = self.predrop1(node_feat)
        if self.pre_dropout > 0:
            node_feat = self.predrop2(node_feat)

        node_feat = self.fc1(node_feat)
        support_emb = node_feat[:s_size]
        query_emb = node_feat[s_size:]


        s_logits = self.fc2(support_emb)
        q_logits = self.fc2(query_emb)

        return s_logits, q_logits

class ContextAwareRelationNet(nn.Module):
    def __init__(self, args):
        super(ContextAwareRelationNet, self).__init__()
        self.n_shot_train = args.n_shot_train
        self.gpu_id = args.gpu_id
        self.top_k=args.top_k
        self.mol_encoder = GNN_Encoder(num_layer=args.enc_layer, emb_dim=args.emb_dim, JK=args.JK,
                                       drop_ratio=args.dropout, graph_pooling=args.enc_pooling, gnn_type=args.enc_gnn,
                                       batch_norm = args.enc_batch_norm)
        '''复现'''
        # if args.pretrained:
        #     model_file = args.pretrained_weight_path
        #     if args.enc_gnn != 'gin':
        #         temp = model_file.split('/')
        #         model_file = '/'.join(temp[:-1]) +'/'+args.enc_gnn +'_'+ temp[-1]
        #     print('load pretrained model from', model_file)
        #     self.mol_encoder.from_pretrained(model_file, self.gpu_id)
        self.dataset=args.dataset
        self.node_enhance = enhanced(inp_dim=128, hidden_dim=128,top_k=self.top_k, dataset=self.dataset,n_shot=self.n_shot_train)
        self.coord_enhance = enhanced(inp_dim=128, hidden_dim=128,top_k=self.top_k, dataset=self.dataset,n_shot=self.n_shot_train)
        self.egnn=EGNN(hidden_dim=args.emb_dim, normalize=True)
        self.pool = global_mean_pool
        self.pre_head=Classifer(inp_dim=256,num_class=2,pre_dropout=args.rel_dropout2)
        self.proj = nn.Sequential(
            nn.Linear(300, 128),
            nn.BatchNorm1d(128),
        )


    def forward(self, s_data, q_data, s_label=None):
        s_node_emb,s_emb,s_index,s_edge = self.mol_encoder(s_data.x, s_data.edge_index, s_data.edge_attr, s_data.batch)
        q_node_emb,q_emb,q_index,q_edge= self.mol_encoder(q_data.x, q_data.edge_index, q_data.edge_attr, q_data.batch)
        s_coord=self.egnn(s_node_emb,s_data.pos,s_index,s_edge)
        q_coord=self.egnn(q_node_emb,q_data.pos,q_index,q_edge)
        s_coord=self.pool(s_coord, s_data.batch)
        q_coord=self.pool(q_coord,q_data.batch)
        s_node_emb=self.proj(s_node_emb)
        q_node_emb=self.proj(q_node_emb)
        s_node=self.pool(s_node_emb,s_data.batch)
        q_node=self.pool(q_node_emb,q_data.batch)
        s_repre, q_repre= self.node_enhance(s_node,q_node)
        s_coord_repre, q_coord_repre= self.coord_enhance(s_coord,q_coord)
        support=torch.cat([s_repre,s_coord_repre],dim=1)
        query=torch.cat([q_repre,q_coord_repre],dim=1)
        all_emb=torch.cat([support, query], dim=0)
        s_logits, q_logits= self.pre_head(all_emb,s_node.size(0))
        return s_logits, q_logits
