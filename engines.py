import numpy as np
import torch
import torch.optim as optim

import utils
from model import HSTGL

load_initial = False
initial_checkpoint = ''


class Trainer:
    def __init__(self, scaler, num_metapaths_list, feature_dim_list, device, num_layers, num_nodes, num_site_nodes,
                 num_warehouse_nodes, batch_size,
                 num_mac_sites, num_mac_warehouses, lrate, wdecay, mac_graph, type_mask_mac, mapping_matrix, clip, nhid,
                 out_dim, nheads, natten, dropout):
        self.model = HSTGL(device, num_metapaths_list, feature_dim_list, num_layers, num_nodes, num_site_nodes,
                           num_warehouse_nodes, num_mac_sites, batch_size, num_mac_warehouses, mac_graph, type_mask_mac,
                           mapping_matrix, nhid, out_dim,
                           nheads, natten, dropout)

        if load_initial:
            print('load_state_dict...')
            res = self.model.load_state_dict(
                torch.load(initial_checkpoint, map_location=lambda storage, loc: storage), strict=False)
            print(res)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.loss = utils.masked_mae
        self.scaler = scaler
        self.clip = clip

    def train(self, input, real, g_lists, type_mask,
              edge_metapath_indices_lists, mac_graph, f, cur_epoch):
        # with autograd.detect_anomaly():
        self.model.train()

        self.optimizer.zero_grad()
        wh = np.where(type_mask == 0)[0]
        target_node_indices = np.where(type_mask != 1)[0]
        real = real[:, target_node_indices]

        # from thop import profile
        # macs, params = profile(self.model, inputs=(g_lists, input, type_mask, edge_metapath_indices_lists, f, mac_graph, target_node_indices,
        #                     cur_epoch,))
        # print(' FLOPs: ', macs * 2)  # 一般来讲，FLOPs是macs的两倍
        # print('params: ', params)
        # exit(0)
        output = self.model(g_lists, input, type_mask, edge_metapath_indices_lists, f, mac_graph, target_node_indices,
                            cur_epoch)
        loss_wh = self.loss(output[:, wh], real[:, wh], 0.0)
        loss_st = self.loss(output[:, len(wh):], real[:, len(wh):], 0.0)
        loss = loss_wh + loss_st
        loss.backward()

        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()

        predict, predict_w, predict_s = self.predict(output[:, wh], output[:, len(wh):])
        real, real_w, real_s = self.predict(real[:, wh], real[:, len(wh):])

        wh_mae = utils.masked_mae(predict_w, real_w, 0.0).item()
        st_mae = utils.masked_mae(predict_s, real_s, 0.0).item()

        return wh_mae, st_mae

    def eval(self, input, real, g_lists, type_mask, edge_metapath_indices_lists, mac_graph, f, cur_epoch
             ):
        self.model.eval()
        wh = np.where(type_mask == 0)[0]
        target_node_indices = np.where(type_mask != 1)[0]
        real = real[:, target_node_indices]

        output = self.model(g_lists, input, type_mask, edge_metapath_indices_lists, f, mac_graph, target_node_indices,
                            cur_epoch)

        predict, predict_w, predict_s = self.predict(output[:, wh], output[:, len(wh):])
        real, real_w, real_s = self.predict(real[:, wh], real[:, len(wh):])
        wh_mae = utils.masked_mae(predict_w, real_w, 0.0).item()
        st_mae = utils.masked_mae(predict_s, real_s, 0.0).item()
        wh_mape = utils.masked_smape(predict_w, real_w, 0.0).item()
        st_mape = utils.masked_smape(predict_s, real_s, 0.0).item()
        wh_rmse = utils.masked_rmse(predict_w, real_w, 0.0).item()
        st_rmse = utils.masked_rmse(predict_s, real_s, 0.0).item()

        return real, predict, wh_mae, st_mae, wh_mape, st_mape, wh_rmse, st_rmse

    def predict(self, a, b):
        predict_w = self.scaler[0].inverse_transform(a)
        predict_s = self.scaler[2].inverse_transform(b)
        predict_w = torch.exp(predict_w)
        predict = torch.concat((predict_w, predict_s), dim=1)
        return predict, predict_w, predict_s
