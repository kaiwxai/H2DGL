import argparse
import pickle
import time
import warnings

import numpy as np
import torch
from dgl.base import DGLWarning

import utils
from engines import Trainer

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0', help='cuda id')
parser.add_argument('--data_dir', type=str, default='data/', help='data path')
parser.add_argument('--evtype', type=str, default='weather', help='event type: weather or covid')
parser.add_argument('--dataset', type=str, default='bj', help='dataset: bj or sh')

parser.add_argument('--expid', type=int, default=1, help='experiment id')
parser.add_argument('--num_layers', type=int, default=2, help='layers number in graph learning module')
parser.add_argument('--in_dim', type=int, default=14, help='sequence inputs dimension')
parser.add_argument('--out_dim', type=int, default=1, help='sequence output dimension')

parser.add_argument('--nhid', type=int, default=128, help='hidden dim')
parser.add_argument('--num_heads', type=int, default=1, help='number of heads')
parser.add_argument('--atten_dim', type=int, default=128, help='attention dim')

parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')  #
parser.add_argument('--epochs', type=int, default=100, help='')

parser.add_argument('--patience', type=int, default=30, help='')
parser.add_argument('--clip', type=int, default=5, help='')
parser.add_argument('--save', type=str, default='data/res/', help='save path')
args = parser.parse_args()
num_metapaths_list = [3, 1, 1]
feature_dim_list = [5, 1, 2] if args.evtype == 'weather' else [5, 1, 3]
nodes_list = [509, 397, 24, 1145] if args.dataset == 'bj' else [85, 271, 12, 598]

if __name__ == '__main__':
    print(args)
    warnings.simplefilter(action='ignore', category=DGLWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    device = torch.device(args.device)
    dataloader = utils.DataSet(args.evtype, args.dataset, args.in_dim, args.data_dir, args.out_dim,
                               args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader.data['scaler']
    engine = Trainer(scaler, feature_dim_list=feature_dim_list, num_metapaths_list=num_metapaths_list, device=device,
                     num_layers=args.num_layers, batch_size=args.batch_size,
                     lrate=args.learning_rate, wdecay=args.weight_decay,
                     num_mac_sites=nodes_list[1], num_mac_warehouses=nodes_list[2],
                     num_warehouse_nodes=nodes_list[0], num_site_nodes=nodes_list[1], mac_graph=None,
                     type_mask_mac=dataloader.type_mask_mac, clip=args.clip, nhid=args.nhid,
                     out_dim=args.out_dim, natten=args.atten_dim, dropout=args.dropout,
                     nheads=args.num_heads, num_nodes=nodes_list[-1], mapping_matrix=dataloader.mapping_matrix)
    his_loss = []
    val_time = []
    train_time = []
    vloss_0 = []
    vloss_1 = []
    mape_0 = []
    mape_1 = []
    rmse_0 = []
    rmse_1 = []
    mae = []
    mape = []
    rmse = []
    min_val_loss = float('inf')
    wait = 0
    import gc

    gc.collect()
    with torch.no_grad():
        torch.cuda.empty_cache()
    for i in range(1, args.epochs + 1):
        train_loss_0 = []
        train_loss_1 = []
        t1 = time.time()
        dataloader.data['train_loader'].shuffle()
        for iter, (x, g, f, y, b) in enumerate(dataloader.data['train_loader'].get_iterator()):
            trainx = torch.Tensor(x).to(device)
            trainx = trainx.transpose(1, 3)
            trainy = torch.Tensor(y).to(device)
            trainy = trainy.transpose(1, 3)
            g_lists, edge_metapath_indices_lists, mac_graph = utils.get_metapath_graph_pair(args.dataset, g,
                                                                                            dataloader.type_mask,
                                                                                            device)

            metrics = engine.train(trainx, trainy[:, 0, :, :], g_lists, dataloader.type_mask,
                                   edge_metapath_indices_lists, mac_graph, f, cur_epoch=i)
            train_loss_0.append(metrics[0])
            train_loss_1.append(metrics[1])
        t2 = time.time()
        train_time.append(t2 - t1)
        # validation
        valid_loss_0 = []
        valid_loss_1 = []
        valid_rmse_0 = []
        valid_rmse_1 = []
        valid_mape_0 = []
        valid_mape_1 = []
        s1 = time.time()
        for iter, (x, g, f, y, b) in enumerate(dataloader.data['val_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1, 3)
            testy = torch.Tensor(y).to(device)
            testy = testy.transpose(1, 3)
            g_lists, edge_metapath_indices_lists, mac_graph = utils.get_metapath_graph_pair(args.dataset, g,
                                                                                            dataloader.type_mask,
                                                                                            device)
            metrics = engine.eval(testx, testy[:, 0, :, :], g_lists, dataloader.type_mask,
                                  edge_metapath_indices_lists, mac_graph, f, cur_epoch=i)
            valid_loss_0.append(metrics[2])
            valid_loss_1.append(metrics[3])
            valid_mape_0.append(metrics[4])
            valid_mape_1.append(metrics[5])
            valid_rmse_0.append(metrics[6])
            valid_rmse_1.append(metrics[7])

        s2 = time.time()
        log = 'Epoch: {:03d}, Inference Time: {:.2f} secs'
        print(log.format(i, (s2 - s1)))
        val_time.append(s2 - s1)
        mtrain_loss_0 = np.mean(train_loss_0)
        mtrain_loss_1 = np.mean(train_loss_1)

        mvalid_loss_0 = np.mean(valid_loss_0)
        mvalid_loss_1 = np.mean(valid_loss_1)
        mvalid_rmse_0 = np.mean(valid_rmse_0)
        mvalid_rmse_1 = np.mean(valid_rmse_1)
        mvalid_mape_0 = np.mean(valid_mape_0)
        mvalid_mape_1 = np.mean(valid_mape_1)
        his_loss.append(mvalid_loss_0 + mvalid_loss_1)
        log = 'Epoch: {:03d}, Train Loss(MAE): {:.2f}, {:.2f}, Valid Loss(MAE): {:.2f}, {:.2f}, Valid SMAPE: {:.2f},' \
              '{:.2f}, Valid RMSE: {:.2f},{:.2f} Training Time: {:.2f}/epoch'
        vloss_0.append(mvalid_loss_0)
        vloss_1.append(mvalid_loss_1)
        mape_0.append(mvalid_mape_0)
        mape_1.append(mvalid_mape_1)
        rmse_0.append(mvalid_rmse_0)
        rmse_1.append(mvalid_rmse_1)
        print(
            log.format(i, mtrain_loss_0, mtrain_loss_1, mvalid_loss_0, mvalid_loss_1, mvalid_mape_0, mvalid_mape_1,
                       mvalid_rmse_0, mvalid_rmse_1, (t2 - t1)), flush=True)
        torch.save(engine.model.state_dict(), args.save + "_epoch_" + str(args.expid) + "_" + str(i) + "_" + ".pth")
        loss = mtrain_loss_0 + mtrain_loss_1
        if loss < min_val_loss:
            wait = 0
            min_val_loss = loss
        else:
            wait += 1
            if wait == args.patience:
                print('Early stopping at epoch: %d' % i)
                break
    print("Average Training Time: {:.2f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.2f} secs".format(np.mean(val_time)))

    # testing
    bestid = np.argmin(his_loss)
    engine.model.load_state_dict(
        torch.load(args.save + "_epoch_" + str(args.expid) + "_" + str(bestid + 1) + "_" + ".pth"))
    outputs = []
    realy = torch.Tensor(dataloader.data['y_test']).to(device)
    realy = realy.transpose(1, 3)[:, 0, :, :]
    test_y = []
    for iter, (x, g, f, y, b) in enumerate(dataloader.data['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1, 3)
        g_lists, edge_metapath_indices_lists, mac_graph = utils.get_metapath_graph_pair(args.dataset, g,
                                                                                        dataloader.type_mask,
                                                                                        device)
        target_node_indices = np.where(dataloader.type_mask != 1)[0]
        with torch.no_grad():
            preds = engine.model(g_lists, testx, dataloader.type_mask,
                                 edge_metapath_indices_lists, f, mac_graph, target_node_indices, bestid + 1)
        outputs.append(preds)
        test_y.append([(scaler[0].std, scaler[0].mean, scaler[1].std, scaler[1].mean), b, preds.detach().cpu().numpy,
                       y])

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:realy.size(0), ...]

    print("Training finished")

    amae_0 = []
    amae_1 = []
    amape_0 = []
    amape_1 = []
    armse_0 = []
    armse_1 = []

    wh = np.where(dataloader.type_mask == 0)[0]
    st = np.where(dataloader.type_mask == 2)[0]
    for i in range(args.out_dim):
        predict_w = scaler[0].inverse_transform(yhat[:, wh, i])
        predict_w = torch.exp(predict_w)
        predict_s = scaler[2].inverse_transform(yhat[:, len(wh):, i])
        real = realy[:, :, i]
        real_w = scaler[0].inverse_transform(real[:, wh])
        real_w = torch.exp(real_w)
        real_s = scaler[2].inverse_transform(real[:, st])

        metrics0 = utils.metric(predict_w, real_w)
        metrics1 = utils.metric(predict_s, real_s)
        amae_0.append(metrics0[0])
        amape_0.append(metrics0[1])
        armse_0.append(metrics0[2])
        amae_1.append(metrics1[0])
        amape_1.append(metrics1[1])
        armse_1.append(metrics1[2])

    log = 'On average over {:.2f} horizons, Test MAE: {:.2f},  {:.2f}, Test SMAPE: {:.2f},   {:.2f}, Test RMSE: {:.2f},' \
          '  {:.2f}'
    print(
        log.format(args.out_dim, np.mean(amae_0), np.mean(amae_1), np.mean(amape_0), np.mean(amape_1), np.mean(armse_0),
                   np.mean(armse_1)))
    torch.save(engine.model.state_dict(),
               args.save + "_exp" + str(args.expid) + "_best_" + str(round(his_loss[bestid], 2)) + ".pth")
    pickle.dump(test_y, open(args.save + 'ys.pkl', 'wb'))
