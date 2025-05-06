import dgl.function as fn
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import edge_softmax
from torch.autograd import Variable


def laplace_transform(graph):
    rowsum_sqrt = sp.diags(1 / (np.sqrt(graph.sum(axis=1).A.ravel()) + 1e-8))
    colsum_sqrt = sp.diags(1 / (np.sqrt(graph.sum(axis=0).A.ravel()) + 1e-8))
    graph = rowsum_sqrt @ graph @ colsum_sqrt
    return graph


def to_tensor(graph):
    graph = graph.tocoo()
    indices = np.vstack((graph.row, graph.col))
    graph = torch.sparse.FloatTensor(torch.LongTensor(indices), torch.FloatTensor(graph.data), torch.Size(graph.shape))
    return graph


class graph_layer(nn.Module):
    def __init__(self,
                 out_dim,
                 num_heads,
                 attn_drop=0.2,
                 alpha=0.01,
                 attn_switch=False):
        super(graph_layer, self).__init__()
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.attn_switch = attn_switch

        self.rnn = nn.GRU(out_dim, num_heads * out_dim)

        if self.attn_switch:
            self.attn1 = nn.Linear(out_dim, num_heads, bias=False)
            self.attn2 = nn.Parameter(torch.empty(size=(1, num_heads, out_dim)))
        else:
            self.attn = nn.Parameter(torch.empty(size=(1, num_heads, out_dim)))
        self.leaky_relu = nn.LeakyReLU(alpha)
        self.softmax = edge_softmax
        if attn_drop:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = lambda x: x

        if self.attn_switch:
            nn.init.xavier_normal_(self.attn1.weight, gain=1.414)
            nn.init.xavier_normal_(self.attn2.data, gain=1.414)
        else:
            nn.init.xavier_normal_(self.attn.data, gain=1.414)

    def edge_softmax(self, g):
        attention = self.softmax(g, g.edata.pop('a'))
        g.edata['a_drop'] = self.attn_drop(attention)

    def message_passing(self, edges):
        ft = edges.data['eft'] * edges.data['a_drop']
        return {'ft': ft}

    def forward(self, g, features, edge_metapath_indices):
        edata = F.embedding(edge_metapath_indices, features)
        if edata.shape[0] == 0:
            return torch.Tensor(np.zeros(shape=(g.num_nodes(), self.num_heads, self.out_dim))).to(
                edge_metapath_indices.device), g
        _, hidden = self.rnn(edata.permute(1, 0, 2))
        eft = hidden.permute(1, 0, 2).view(-1, self.num_heads, self.out_dim)  # E x num_heads x out_dim
        if self.attn_switch:
            center_node_feat = F.embedding(edge_metapath_indices[:, -1], features)  # E x out_dim
            a1 = self.attn1(center_node_feat)  # E x num_heads
            a2 = (eft * self.attn2).sum(dim=-1)  # E x num_heads
            a = (a1 + a2).unsqueeze(dim=-1)  # E x num_heads x 1
        else:
            a = (eft * self.attn).sum(dim=-1).unsqueeze(dim=-1)  # E x num_heads x 1
        a = self.leaky_relu(a)
        g.edata.update({'eft': eft, 'a': a})
        self.edge_softmax(g)
        g.update_all(self.message_passing, fn.sum('ft', 'ft'))
        ret = g.ndata['ft']  # E x num_heads x out_dim
        return ret, g


class Trace_agg_layer(nn.Module):
    def __init__(self,
                 num_metapaths,
                 out_dim,
                 num_heads,
                 attn_vec_dim,
                 attn_drop=0.2):
        super(Trace_agg_layer, self).__init__()
        self.out_dim = out_dim
        self.num_heads = num_heads

        self.metapath_layers = nn.ModuleList()
        for i in range(num_metapaths):
            self.metapath_layers.append(graph_layer(out_dim,
                                                    num_heads,
                                                    attn_drop=attn_drop))

        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(out_dim * num_heads, out_dim * 2, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(out_dim * 2, out_dim, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(out_dim, attn_vec_dim, bias=True),
            torch.nn.ReLU(),
        )
        self.fc2 = nn.Linear(attn_vec_dim, 1, bias=False)
        nn.init.xavier_normal_(self.fc2.weight, gain=1.414)

    def forward(self, g_list, features, edge_metapath_indices_list):
        g_list = [g.to(features.device) for g in g_list]
        metapath_outs = []
        gs = []
        for g, edge_metapath_indices, metapath_layer in zip(g_list, edge_metapath_indices_list, self.metapath_layers):
            metapath_out, learnt_g = metapath_layer(g, features, edge_metapath_indices)
            metapath_out = metapath_out.view(-1, self.num_heads * self.out_dim)
            metapath_out = F.elu(metapath_out)
            metapath_outs.append(metapath_out)
            gs.append(learnt_g)
        beta = []
        for metapath_out in metapath_outs:
            fc1 = torch.tanh(self.fc1(metapath_out))
            fc1 = torch.mean(fc1, dim=0)
            beta.append(self.fc2(fc1))
        beta = torch.cat(beta, dim=0)
        beta = F.softmax(beta, dim=0)
        beta = torch.unsqueeze(beta, dim=-1)
        beta = torch.unsqueeze(beta, dim=-1)
        metapath_outs = [torch.unsqueeze(metapath_out.to(features.device), dim=0) for metapath_out in metapath_outs]
        metapath_outs = torch.cat(metapath_outs, dim=0)
        h = torch.sum(beta * metapath_outs, dim=0)
        return h, gs


class hetenode_layer(nn.Module):
    def __init__(self,
                 num_metapaths_list,
                 in_dim,
                 out_dim,
                 num_heads,
                 attn_vec_dim,
                 attn_drop=0.2,
                 ):
        super(hetenode_layer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_metapaths_list = num_metapaths_list
        self.ctr_ntype_layers = nn.ModuleList()
        for i in range(len(num_metapaths_list)):
            self.ctr_ntype_layers.append(Trace_agg_layer(num_metapaths_list[i],
                                                         in_dim,
                                                         num_heads,
                                                         attn_vec_dim,
                                                         attn_drop))
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_dim * num_heads, in_dim * 2, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(in_dim * 2, in_dim, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(in_dim, out_dim, bias=True),
            torch.nn.ReLU(),
        )

    def forward(self, g_lists, feature, type_mask, edge_metapath_indices_lists):
        learnt_g = []
        h = torch.zeros(type_mask.shape[0], self.in_dim * self.num_heads, device=feature.device)
        for i, (g_list, edge_metapath_indices_list, ctr_ntype_layer) in enumerate(
                zip(g_lists, edge_metapath_indices_lists, self.ctr_ntype_layers)):
            h[np.where(type_mask == i)[0]], gs = ctr_ntype_layer(g_list, feature, edge_metapath_indices_list)
            learnt_g.append(gs)
        h_ = self.fc(h)
        return h_, h, learnt_g


class ViewsFusion(nn.Module):
    def __init__(self, num_heads, hidden_dim):
        super(ViewsFusion, self).__init__()
        self.mlp1 = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * num_heads, hidden_dim * 2, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim * 2, hidden_dim, bias=True),
            torch.nn.ReLU(),
        )
        self.mlp2 = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * num_heads * 2, hidden_dim * num_heads, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim * num_heads, hidden_dim * 2, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim * 2, hidden_dim, bias=True),
            torch.nn.ReLU(),
        )

    def forward(self, trace, event=None):
        if event is None:
            return self.mlp1(trace)
        out = torch.concat([trace, event], dim=2)
        return self.mlp2(out)


class EventBlock(nn.Module):
    def __init__(self, num_metapaths_list, num_layers, feats_dim_list, dropout_rate, hidden_dim, out_dim, num_nodes,
                 num_heads, attn_vec_dim, seq_len):
        super(EventBlock, self).__init__()
        self.fc_list = nn.ModuleList(
            [nn.Linear(feats_dim * seq_len, hidden_dim, bias=True) for feats_dim in feats_dim_list])
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.num_nodes = num_nodes
        for l in range(num_layers - 1):
            self.layers.append(
                hetenode_layer(num_metapaths_list, hidden_dim, hidden_dim,
                               num_heads, attn_vec_dim, attn_drop=dropout_rate))
        self.layers.append(
            hetenode_layer(num_metapaths_list, hidden_dim, out_dim,
                           num_heads, attn_vec_dim, attn_drop=dropout_rate))

    def forward(self, features_list, g_lists, type_mask, edge_metapath_indices_lists, target_node_indices, device):
        transformed_features = torch.zeros(len(g_lists), type_mask.shape[0], self.hidden_dim, device=device)
        for i, fc in enumerate(self.fc_list):
            node_indices = np.where(type_mask == i)[0]
            features_list[i] = np.transpose(features_list[i], (0, 2, 1, 3))
            feature = torch.flatten(torch.Tensor(features_list[i]), start_dim=2, end_dim=3).to(device)
            feature = fc(feature)
            transformed_features[:, node_indices] = feature
        event_embeds = torch.Tensor(
            np.zeros(shape=[len(g_lists), len(target_node_indices), self.hidden_dim * self.num_heads])).to(
            device)

        for single_batch in range(len(g_lists)):
            h = transformed_features[single_batch]
            for l in range(self.num_layers - 1):
                h, _, learnt_g = self.layers[l](g_lists[single_batch], h, type_mask,
                                                edge_metapath_indices_lists[single_batch])
                h = F.elu(h)
            h, embeds, learnt_g = self.layers[-1](
                g_lists[single_batch], h, type_mask, edge_metapath_indices_lists[single_batch])

            event_embeds[single_batch] = embeds[target_node_indices]
        return event_embeds


class MicBlock(nn.Module):
    def __init__(self, batch_size, num_layers, num_metapaths_list, num_nodes,
                 feats_dim_list, hidden_dim, out_dim,
                 num_heads, attn_vec_dim, dropout_rate, seq_len, distill_step=1):
        super(MicBlock, self).__init__()
        self.meta = MetaBlock(num_layers=num_layers, num_metapaths_list=num_metapaths_list,
                              hidden_dim=hidden_dim, out_dim=out_dim, graph_num=batch_size,
                              num_heads=num_heads, attn_vec_dim=attn_vec_dim, num_nodes=num_nodes)
        self.event = EventBlock(num_metapaths_list, num_layers, feats_dim_list, dropout_rate, hidden_dim, out_dim,
                                num_nodes, num_heads, attn_vec_dim, seq_len)
        self.fusion = ViewsFusion(num_heads, hidden_dim)
        self.distill_step = distill_step

    def forward(self, g_lists, features_list, type_mask, edge_metapath_indices_lists,
                target_node_indices, cur_epochs, device):
        e1, g1 = self.meta(g_lists, type_mask, edge_metapath_indices_lists, target_node_indices, device)
        if cur_epochs % self.distill_step == 0:
            e2 = self.event(features_list, g1, type_mask, edge_metapath_indices_lists, target_node_indices, device)
            out = self.fusion(e1, e2)
            return out
        return self.fusion(e1)


class MetaBlock(nn.Module):
    def __init__(self, num_nodes,
                 num_layers,
                 num_metapaths_list,
                 hidden_dim,
                 out_dim,
                 num_heads,
                 attn_vec_dim, graph_num,
                 dropout_rate=0.5):
        super(MetaBlock, self).__init__()
        self.trace_feature = None
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.layers = nn.ModuleList()
        self.graph_num = graph_num
        self.num_nodes = num_nodes
        for l in range(num_layers - 1):
            self.layers.append(
                hetenode_layer(num_metapaths_list, hidden_dim, hidden_dim,
                               num_heads, attn_vec_dim, attn_drop=dropout_rate))
        self.layers.append(
            hetenode_layer(num_metapaths_list, hidden_dim, out_dim,
                           num_heads, attn_vec_dim, attn_drop=dropout_rate))
        self.init_emb()

    def init_emb(self):
        self.trace_feature = nn.Parameter(torch.FloatTensor(self.graph_num, self.num_nodes, self.hidden_dim))
        nn.init.xavier_normal_(self.trace_feature)

    def forward(self, g_lists, type_mask, edge_metapath_indices_lists, target_node_indices, device):
        trace_embeds = None
        learnt_gs = []
        for single_batch in range(len(g_lists)):
            h = self.trace_feature[single_batch]
            for l in range(self.num_layers - 1):
                h, embeds, learnt_g = self.layers[l](g_lists[single_batch], h, type_mask,
                                                     edge_metapath_indices_lists[single_batch])
                h = F.elu(h)
            h, embeds, learnt_g = self.layers[-1](
                g_lists[single_batch], h, type_mask, edge_metapath_indices_lists[single_batch])
            if trace_embeds is None:
                trace_embeds = torch.Tensor(
                    np.zeros(shape=[len(g_lists), len(target_node_indices), embeds.shape[1]])).to(
                    device)
            learnt_gs.append(learnt_g)
            self.trace_feature[single_batch].data = h
            trace_embeds[single_batch] = embeds[target_node_indices, :]
        return trace_embeds, learnt_gs


class TimeBlock2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride_size=1, padding_size=0,
                 dilation_size=1):
        super(TimeBlock2, self).__init__()
        self.k = kernel_size
        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size), (1, stride_size), (0, padding_size),
                               (1, dilation_size))
        self.conv2 = nn.Conv2d(in_channels, out_channels, (1, kernel_size), (1, stride_size), (0, padding_size),
                               (1, dilation_size))
        self.conv3 = nn.Conv2d(in_channels, out_channels, (1, kernel_size), (1, stride_size), (0, padding_size),
                               (1, dilation_size))

    def forward(self, X):
        X = X.permute(0, 3, 1, 2)
        temp = torch.tanh(self.conv1(X)) * torch.sigmoid(self.conv2(X))
        out = F.relu(temp * self.conv3(X))
        out = out.permute(0, 2, 3, 1)
        return out


class LSTMCell(nn.Module):
    def __init__(self, num_nodes, hidden_size, input_size, out_dim, num_layers=1):
        super(LSTMCell, self).__init__()
        self.num_classes = num_nodes
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, out_dim)
        self.hidden_size = hidden_size
        self.out_dim = out_dim
        nn.init.xavier_normal_(self.fc.weight, gain=1.414)
        self.num_layers = num_layers

    def forward(self, x):
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size)).to(x.device)

        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size)).to(x.device)
        out = torch.zeros(x.size(0), x.size(2), self.out_dim).to(x.device)
        for node in range(self.num_classes):
            ula, (h_out, _) = self.lstm(x[:, :, node], (h_0, c_0))
            h_out = h_out.view(-1, self.hidden_size)
            o = self.fc(h_out)
            out[:, node, :] = o
        return out


class MacGCNBlock(nn.Module):
    def __init__(self, device, num_agg_sites, num_warehouses, num_agg_warehouses, nhid, out_dim, raw_graph,
                 type_mask_mac, graph_nums, num_layers=2):
        super(MacGCNBlock, self).__init__()
        self.site_features = None
        self.warehouse_features = None
        self.device = device
        self.graph_nums = graph_nums
        self.num_agg_sites = num_agg_sites
        self.num_agg_warehouses = num_agg_warehouses
        self.embedding_size = nhid
        self.init_emb()
        self.num_warehouses = num_warehouses
        self.ws_graph_all = raw_graph
        self.out_dim = out_dim
        self.mac_level_graphs = []
        self.num_layers = num_layers
        self.type_mask_mac = type_mask_mac

    def init_emb(self):
        self.warehouse_features = nn.Parameter(
            torch.FloatTensor(self.graph_nums, self.num_agg_warehouses, self.embedding_size))
        nn.init.xavier_normal_(self.warehouse_features)
        self.site_features = nn.Parameter(torch.FloatTensor(self.graph_nums, self.num_agg_sites, self.embedding_size))
        nn.init.xavier_normal_(self.site_features)

    def get_graph(self):
        ws_graphs = self.ws_graph_all
        device = self.device
        mac_level_graphs = []
        for ws_graph in ws_graphs:
            mac_level_graph = sp.bmat([[sp.csr_matrix((ws_graph.shape[0], ws_graph.shape[0])), ws_graph],
                                       [ws_graph.T, sp.csr_matrix((ws_graph.shape[1], ws_graph.shape[1]))]])

            mac_level_graph = to_tensor(laplace_transform(mac_level_graph)).to(device)
            mac_level_graphs.append(mac_level_graph)
        self.mac_level_graphs = mac_level_graphs

    def one_propagate(self, graph, A_feature, B_feature):
        features = torch.cat((A_feature, B_feature), 0)  # 随机初始化的
        all_features = [features]

        for i in range(self.num_layers):
            features = torch.spmm(graph, features)
            features = features / (i + 2)
            all_features.append(F.normalize(features, p=2, dim=1))

        all_features = torch.stack(all_features, 1)
        all_features = torch.sum(all_features, dim=1).squeeze(1)

        A_feature, B_feature = torch.split(all_features, (A_feature.shape[0], B_feature.shape[0]), 0)

        return A_feature, B_feature

    # lightGCN的实现
    def propagate(self):
        warehouse_features = torch.Tensor(
            np.zeros(shape=[len(self.mac_level_graphs), self.num_agg_warehouses, self.embedding_size])).to(
            self.device)
        site_features = torch.Tensor(
            np.zeros(shape=[len(self.mac_level_graphs), self.num_agg_sites, self.embedding_size])).to(
            self.device)
        for one_batch in range(len(self.mac_level_graphs)):
            A_feat, B_feat = self.one_propagate(self.mac_level_graphs[one_batch],
                                                self.warehouse_features[one_batch],
                                                self.site_features[one_batch])
            warehouse_features[one_batch] = A_feat
            site_features[one_batch] = B_feat
        return warehouse_features, site_features

    def forward(self, mac_graph):
        self.ws_graph_all = mac_graph
        self.get_graph()
        return self.propagate()


class STBlock(nn.Module):
    def __init__(self, num_nodes, seq_len, cin, cout, num_layers=2, kernel_size=4):
        super(STBlock, self).__init__()

        self.start_conv = nn.Conv2d(in_channels=1,
                                    out_channels=cin,
                                    kernel_size=(1, 1))
        self.num_layers = num_layers
        self.lstm1 = LSTMCell(num_nodes, seq_len, 1, cout, num_layers=1)
        self.lstm2 = LSTMCell(num_nodes, 8, cin, cout, num_layers=1)

        self.temporal1 = nn.ModuleList()
        self.temporal2 = nn.ModuleList()
        self.temporal3 = nn.ModuleList()

        for i in range(self.num_layers):
            self.temporal1.append(
                TimeBlock2(in_channels=cin, out_channels=cin, kernel_size=kernel_size))
            self.temporal2.append(
                TimeBlock2(in_channels=cin, out_channels=cin, kernel_size=kernel_size))
            self.temporal3.append(
                TimeBlock2(in_channels=cin, out_channels=cout, kernel_size=kernel_size * 2))

    def forward(self, seq_data, spatio):
        out = self.start_conv(seq_data).permute(0, 2, 3, 1)
        skip = self.lstm1(seq_data.permute(0, 3, 2, 1))
        skip = skip.unsqueeze(2)
        for i in range(self.num_layers):
            residual = out
            out_ = torch.einsum("ijm,ijlm->ijlm", [spatio, out])
            o1 = self.temporal1[i](out_)

            o2 = self.temporal2[i](out)
            out = o1 * o2
            s = out
            s = self.temporal3[i](s)
            skip = s[:, :, -1:] + skip
            out = out + residual[:, :, -out.size(2):]

        out = self.lstm2(out.permute(0, 2, 1, 3)).unsqueeze(2) + skip
        return out.squeeze(1)


class SpatioFusion(nn.Module):
    def __init__(self, hid, mapping=None, device='cpu'):
        super(SpatioFusion, self).__init__()
        self.device = device
        self.fc = nn.Linear(hid * 2, hid, bias=True)
        nn.init.xavier_normal_(self.fc.weight, gain=1.414)
        self.mapping = mapping is not None
        if mapping is not None:
            self.mapping_matrix = torch.Tensor(mapping).to(self.device)

    def forward(self, embedding1, embedding2):
        if self.mapping:
            embedding1 = torch.matmul(embedding1.permute(0, 2, 1), self.mapping_matrix.permute(1, 0))
            embedding1 = embedding1.permute(0, 2, 1)
        combined_embedding = torch.cat([embedding1, embedding2], dim=2)  # 8, 509, 256
        gating = torch.sigmoid(self.fc(combined_embedding))
        fused_embedding = gating * embedding1 + embedding2
        return fused_embedding


class HSTGL(nn.Module):

    def __init__(self, device, num_metapaths_list, feature_dim_list, num_layers, num_nodes, num_site_nodes,
                 num_warehouse_nodes, num_mac_sites, batch_size,
                 num_mac_warehouses, mac_graphs, type_mask_mac, mapping_matrix, hid_dim, out_dim, num_heads, atten_dim,
                 dropout, seq_len=14):
        super(HSTGL, self).__init__()
        self.num_sites = num_site_nodes
        self.device = device
        self.num_layers = 2
        self.num_nodes = num_warehouse_nodes + num_site_nodes
        self.wh = num_warehouse_nodes
        self.st = num_site_nodes
        spatial_channel = int(hid_dim / 2)

        self.micblock = MicBlock(num_layers=num_layers, num_metapaths_list=num_metapaths_list,
                                 feats_dim_list=feature_dim_list, hidden_dim=hid_dim, out_dim=out_dim,
                                 batch_size=batch_size,
                                 num_heads=num_heads, attn_vec_dim=atten_dim, dropout_rate=dropout, num_nodes=num_nodes,
                                 seq_len=seq_len)

        self.macblock = MacGCNBlock(device=device, num_agg_sites=num_mac_sites, num_warehouses=num_warehouse_nodes,
                                    num_agg_warehouses=num_mac_warehouses,
                                    raw_graph=mac_graphs, nhid=hid_dim, out_dim=out_dim,
                                    type_mask_mac=type_mask_mac, graph_nums=batch_size)
        self.temporal_wh = STBlock(num_warehouse_nodes, seq_len, hid_dim, spatial_channel)
        self.temporal_st = STBlock(num_site_nodes, seq_len, hid_dim, spatial_channel)

        self.lstm = LSTMCell(self.num_nodes, seq_len, 1, out_dim, num_layers=1)
        self.mlp1 = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Conv2d(spatial_channel, int(spatial_channel / 2), 1, bias=True),
            torch.nn.ReLU(),
            torch.nn.Conv2d(int(spatial_channel / 2), out_dim, 1, bias=True)
        )
        self.mlp2 = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Conv2d(spatial_channel, int(spatial_channel / 2), 1, bias=True),
            torch.nn.ReLU(),
            torch.nn.Conv2d(int(spatial_channel / 2), out_dim, 1, bias=True)
        )
        self.sf1 = SpatioFusion(hid_dim, device=device, mapping=mapping_matrix)
        self.sf2 = SpatioFusion(hid_dim, device=device, mapping=None)

    def forward(self, g_lists, seq_data, type_mask, edge_metapath_indices_lists, features_list, mac_graph,
                target_node_indices, cur_epoch):
        macro_c, macro_s = self.macblock(mac_graph)
        s2 = self.micblock(g_lists, features_list, type_mask, edge_metapath_indices_lists,
                           target_node_indices, cur_epoch, seq_data.device)
        seq_data = seq_data[:, :, target_node_indices]
        residual = self.lstm(seq_data.permute(0, 3, 2, 1))

        fused_w = self.sf1(macro_c, s2[:, :self.wh])
        fused_s = self.sf2(macro_s, s2[:, -self.st:])

        out1 = self.temporal_wh(seq_data[:, :, :self.wh], fused_w)
        out2 = self.temporal_st(seq_data[:, :, -self.st:], fused_s)

        out = torch.concat((out1, out2), dim=1)
        fuse_z = out * residual.unsqueeze(2)
        out1 = self.mlp1(fuse_z[:, :self.wh].permute(0, 3, 1, 2)).squeeze(1)
        out2 = self.mlp2(fuse_z[:, -self.st:].permute(0, 3, 1, 2)).squeeze(1)
        out = torch.concat((out1, out2), dim=1)
        return out
