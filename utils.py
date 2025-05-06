import os

import dgl
import networkx as nx
import numpy
import numpy as np
import torch

NORMAL = False
normal = '_normal' if NORMAL else ''  # for predicting normal conditions
threshold = 80  # control the number of the metapath samples
expected_metapaths_bj = [
    [(0, 1, 2, 1, 0), (0, 1, 0), (0, 1, 1, 2, 1, 1, 0)],
    [(1, 1, 1)],
    [(2, 0, 2)]
]

expected_metapaths_sh = [
    [(0, 1, 0), (0, 1, 2, 1, 0), (0, 1, 1, 2, 1, 1, 0), (0, 1, 1, 1, 2, 1, 1, 1, 0)],
    [(1, 1, 1)],
    [(2, 0, 2)]
]


def metric(pred, real):
    mae = masked_mae(pred, real, 0.0).item()
    mape = masked_smape(pred, real, 0.0).item()
    rmse = masked_rmse(pred, real, 0.0).item()
    return mae, mape, rmse


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


class StandardScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds - labels) ** 2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_smape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = 2 * torch.abs(preds - labels) / (torch.abs(labels) + torch.abs(preds))
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return 100 * torch.mean(loss)


class DataSet:
    def __init__(self, et, ds, in_dim, dataset_dir, out_dim, batch_size, valid_batch_size=None, test_batch_size=None):
        data = {}
        self.type_mask = np.load(dataset_dir + f'{ds}/node_types.npy', allow_pickle=True, fix_imports=True,
                                 encoding='latin1')
        self.type_mask_mac = np.load(dataset_dir + f'{ds}/node_types_mac.npy', allow_pickle=True, fix_imports=True,
                                     encoding='latin1')
        self.mapping_matrix = np.load(dataset_dir + f'{ds}/node_types_mac_matrix.npy', allow_pickle=True,
                                      fix_imports=True,
                                      encoding='latin1')
        for category in ['train', 'val', 'test']:
            cat_data = np.load(os.path.join(dataset_dir, ds + '/' + category + f'_{et}{normal}.npz'),
                               allow_pickle=True)
            data['x_' + category] = cat_data['x']
            data['fl1_' + category] = cat_data['fl1']
            data['fl2_' + category] = cat_data['fl2']
            data['fl3_' + category] = cat_data['fl3']
            data['g_' + category] = cat_data['g']
            data['y_' + category] = cat_data['y']
            data['b_' + category] = cat_data['b']
        for category in ['train', 'val', 'test']:
            data['fl_' + category] = [data['fl1_' + category], data['fl2_' + category], data['fl3_' + category]]

        scaler_w = StandardScaler(mean=data['x_train'][..., np.where(self.type_mask == 0)[0], 0].mean(),
                                  std=data['x_train'][..., np.where(self.type_mask == 0)[0], 0].std())
        scaler_c = StandardScaler(mean=data['x_train'][..., np.where(self.type_mask == 1)[0], 0].mean(),
                                  std=data['x_train'][..., np.where(self.type_mask == 1)[0], 0].std())
        scaler_s = StandardScaler(mean=data['x_train'][..., np.where(self.type_mask == 2)[0], 0].mean(),
                                  std=data['x_train'][..., np.where(self.type_mask == 2)[0], 0].std())
        for category in ['train', 'val', 'test']:
            data['x_' + category][..., np.where(self.type_mask == 0)[0], 0] = scaler_w.transform(
                data['x_' + category][..., np.where(self.type_mask == 0)[0], 0])
            data['y_' + category][..., np.where(self.type_mask == 0)[0], 0] = scaler_w.transform(
                data['y_' + category][..., np.where(self.type_mask == 0)[0], 0])
            data['x_' + category][..., np.where(self.type_mask == 1)[0], 0] = scaler_c.transform(
                data['x_' + category][..., np.where(self.type_mask == 1)[0], 0])
            data['y_' + category][..., np.where(self.type_mask == 1)[0], 0] = scaler_c.transform(
                data['y_' + category][..., np.where(self.type_mask == 1)[0], 0])
            data['x_' + category][..., np.where(self.type_mask == 2)[0], 0] = scaler_s.transform(
                data['x_' + category][..., np.where(self.type_mask == 2)[0], 0])
            data['y_' + category][..., np.where(self.type_mask == 2)[0], 0] = scaler_s.transform(
                data['y_' + category][..., np.where(self.type_mask == 2)[0], 0])

        data['train_loader'] = DataLoader(data['x_train'], data['g_train'], data['fl_train'], data['y_train'],
                                          data['b_train'],
                                          batch_size)
        data['val_loader'] = DataLoader(data['x_val'], data['g_val'], data['fl_val'], data['y_val'], data['b_val'],
                                        valid_batch_size)
        data['test_loader'] = DataLoader(data['x_test'], data['g_test'], data['fl_test'], data['y_test'],
                                         data['b_test'],
                                         test_batch_size)
        data['scaler'] = [scaler_w, scaler_c, scaler_s]
        self.data = data


class DataLoader(object):
    def __init__(self, xs, gs, fs, ys, batches, batch_size, pad_with_last_sample=True):
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            fs_padding = [np.repeat(i[-1:], num_padding, axis=0) for i in fs]
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            g_padding = np.repeat(gs[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            gs = np.concatenate([gs, g_padding], axis=0)
            fs = [np.concatenate([fs[ind], i], axis=0) for ind, i in enumerate(fs_padding)]
            ys = np.concatenate([ys, y_padding], axis=0)
            batches = np.concatenate((batches, numpy.array(num_padding * [batches[-1]])))
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys
        self.gs = gs
        self.fs = fs
        self.bs = batches

    def shuffle(self):
        permutation = np.random.permutation(self.size)

        xs, gs, fs, ys, bs = self.xs[permutation], self.gs[permutation], [i[permutation] for i in self.fs], self.ys[
            permutation], self.bs[permutation]
        self.xs = xs
        self.ys = ys
        self.fs = fs
        self.gs = gs
        self.bs = bs

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                f_i = [f[start_ind: end_ind, ...] for f in self.fs]
                g_i = self.gs[start_ind: end_ind, ...]
                b_i = self.bs[start_ind: end_ind, ...]
                yield x_i, g_i, f_i, y_i, b_i
                self.current_ind += 1

        return _wrapper()


def get_metapath_neighbor_pairs_v2(M, type_mask, expected_metapaths):
    outs = []
    for metapath in expected_metapaths:
        mask = np.zeros(M.shape, dtype=bool)
        for i in range((len(metapath) - 1) // 2):
            temp = np.zeros(M.shape, dtype=bool)
            temp[np.ix_(type_mask == metapath[i], type_mask == metapath[i + 1])] = True  # 下标的笛卡尔积形成多个坐标对
            temp[np.ix_(type_mask == metapath[i + 1], type_mask == metapath[i])] = True
            mask = np.logical_or(mask, temp)
        partial_g_nx = nx.from_numpy_array((M.A * mask).astype(int))

        metapath_to_target = {}
        sources = ((type_mask == metapath[0]).nonzero()[0])
        if len(metapath) >= 5 or metapath[0] == 2 or metapath[0] == 1:
            if sources.shape[0] >= threshold:
                indx = np.random.choice(sources.shape[0], threshold, replace=False)
            else:
                indx = np.random.choice(sources.shape[0], sources.shape[0], replace=False)
            sources = np.sort(sources[indx])
        for source in sources:
            single_source_paths = nx.single_source_shortest_path(
                partial_g_nx, source, cutoff=(len(metapath) + 1) // 2 - 1)  # 计算当前源与所有可达节点的最短路径
            targets = ((type_mask == metapath[(len(metapath) - 1) // 2]).nonzero()[0])
            if len(metapath) > 5:
                indx = np.random.choice(targets.shape[0], threshold, replace=False)
                targets = np.sort(targets[indx])
            for target in targets:
                if target in single_source_paths:
                    shortests = [p for p in nx.all_shortest_paths(partial_g_nx, source, target) if
                                 len(p) == (len(metapath) + 1) // 2]
                    if len(shortests) > 0:
                        metapath_to_target[target] = metapath_to_target.get(target, []) + shortests
        metapath_neighbor_paris = {}
        for key, value in metapath_to_target.items():
            for p1 in value:
                for p2 in value:
                    metapath_neighbor_paris[(p1[0], p2[0])] = metapath_neighbor_paris.get((p1[0], p2[0]), []) + [
                        p1 + p2[-2::-1]]
        outs.append(metapath_neighbor_paris)
    return outs


def get_metapath_graph_pair(ds, graphs, type_mask, device):
    g_lists_all = []
    edge_metapath_indices_lists_all = []
    mac_raw_graph = []
    for i in range(graphs.shape[0]):
        nx_G_lists = list()
        edge_metapath_indices_lists = list()
        expected_metapaths = expected_metapaths_sh if ds == 'sh' else expected_metapaths_bj
        for j in range(len(expected_metapaths)):
            adjM = graphs[i, 0]
            neighbor_pairs = get_metapath_neighbor_pairs_v2(adjM, type_mask, expected_metapaths[j])
            G_list = get_networkx_graph(neighbor_pairs, type_mask, j)
            all_edge_metapath_idx_array = get_edge_metapath_idx_array(neighbor_pairs)
            nx_G_lists.append(G_list)
            edge_metapath_indices_lists.append(all_edge_metapath_idx_array)
        edge_metapath_indices_lists = [[torch.LongTensor(indices).to(device) for indices in indices_list] for
                                       indices_list in edge_metapath_indices_lists]

        g_lists = []
        for nx_G_list in nx_G_lists:
            g_lists.append([])
            for nx_G in nx_G_list:  # 以节点i为初始节点的元路径邻接图
                g = dgl.DGLGraph(multigraph=True)
                g.add_nodes(nx_G.number_of_nodes())
                if len(nx_G.edges()) > 0:
                    g.add_edges(*list(zip(*sorted(map(lambda tup: (int(tup[0]), int(tup[1])), nx_G.edges())))))
                g_lists[-1].append(g)
        g_lists_all.append(g_lists)
        edge_metapath_indices_lists_all.append(edge_metapath_indices_lists)
        mac_raw_graph.append(graphs[i, 1])
    return g_lists_all, edge_metapath_indices_lists_all, mac_raw_graph


def get_networkx_graph(neighbor_pairs, type_mask, ctr_ntype):
    indices = np.where(type_mask == ctr_ntype)[0]
    idx_mapping = {}
    for i, idx in enumerate(indices):
        idx_mapping[idx] = i
    G_list = []
    for metapaths in neighbor_pairs:
        sorted_metapaths = sorted(metapaths.items())
        G = nx.MultiGraph()
        G.add_nodes_from(range(len(indices)))
        for (src, dst), paths in sorted_metapaths:
            for _ in range(len(paths)):
                G.add_edge(idx_mapping[src], idx_mapping[dst])
        G_list.append(G)
    return G_list


def get_edge_metapath_idx_array(neighbor_pairs):
    all_edge_metapath_idx_array = []
    for metapath_neighbor_pairs in neighbor_pairs:
        sorted_metapath_neighbor_pairs = sorted(metapath_neighbor_pairs.items())
        edge_metapath_idx_array = []
        for _, paths in sorted_metapath_neighbor_pairs:
            paths = [p[:(len(p) + 1) // 2] for p in paths]
            edge_metapath_idx_array.extend(paths)
        edge_metapath_idx_array = np.array(edge_metapath_idx_array, dtype=int)
        all_edge_metapath_idx_array.append(edge_metapath_idx_array)
    return all_edge_metapath_idx_array
