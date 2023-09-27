import argparse
import os.path as osp
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import pandas as pd
import copy
import os
import wandb
import pickle

from torch_geometric.datasets import Amazon, Coauthor, CitationFull
from torch_geometric.logging import log
from torch_geometric.data import Data
from scipy.stats import pearsonr

from conformalized_gnn.model import GNN, ConfGNN, ConfMLP
from conformalized_gnn.calibrator import TS, VS, ETS, CaGCN, GATS
from conformalized_gnn.conformal import run_conformal_classification, run_conformal_regression

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='county_education_2012', choices = ['Cora_ML_CF', 'CiteSeer_CF', 'DBLP_CF', 'PubMed_CF', 'Amazon-Computers', 'Amazon-Photo', 'Coauthor-CS', 'Coauthor-Physics', 'Anaheim', 'ChicagoSketch', 'county_education_2012', 'county_election_2016', 'county_income_2012', 'county_unemployment_2012', 'twitch_PTBR'])
parser.add_argument('--hidden_channels', type=int, default=16)
parser.add_argument('--model', type=str, default='GCN', choices = ['GAT', 'GCN', 'GraphSAGE', 'SGC'])
parser.add_argument('--heads', type=int, default=1)
parser.add_argument('--aggr', type=str, default='sum')
parser.add_argument('--alpha', type=float, default=0.1)

parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--epochs', type=int, default=5000)
parser.add_argument('--wandb', action='store_true', help='Track experiment')
parser.add_argument('--device', type=str, default='cuda:4')
parser.add_argument('--conformal_score', type=str, default='cqr', choices = ['aps', 'cqr'])

parser.add_argument('--conftr', action='store_true', default = False)
parser.add_argument('--conftr_holdout', action='store_true', default = False)
parser.add_argument('--conftr_calib_holdout', action='store_true', default = False)
parser.add_argument('--conftr_valid_holdout', action='store_true', default = False)

parser.add_argument('--conftr_sep_test', action='store_true', default = False)
parser.add_argument('--conf_correct_model', type=str, default='gnn', choices = ['gnn', 'mlp', 'Calibrate', 'mcdropout', 'mcdropout_std', 'QR'])
parser.add_argument('--calibrator', type=str, default='NULL', choices = ['TS', 'VS', 'ETS', 'CaGCN', 'GATS'])

parser.add_argument('--quantile', action='store_true', default = False)
parser.add_argument('--bnn', action='store_true', default = False)

parser.add_argument('--target_size', type=int, default=0)
parser.add_argument('--confnn_hidden_dim', type=int, default=64)
parser.add_argument('--confgnn_num_layers', type=int, default=1)
parser.add_argument('--confgnn_base_model', type=str, default='GCN', choices = ['GAT', 'GCN', 'GraphSAGE', 'SGC'])
parser.add_argument('--confgnn_lr', type=float, default=1e-3)
parser.add_argument('--tau', type=float, default=0.1)
parser.add_argument('--size_loss_weight', type=float, default=1)
parser.add_argument('--reg_loss_weight', type=float, default=1)

parser.add_argument('--not_save_res', action='store_true', default = False)
parser.add_argument('--num_runs', type=int, default=10)
parser.add_argument('--retrain', action='store_true', default = False)
parser.add_argument('--verbose', action='store_true', default = False)
parser.add_argument('--data_seed', type=int, default=0)
parser.add_argument('--hyperopt', action='store_true', default = False)
parser.add_argument('--optimal', action='store_true', default = False)
parser.add_argument('--optimal_examine', action='store_true', default = False)
parser.add_argument('--cond_cov_loss', action='store_true', default = False)
parser.add_argument('--conformal_training', action='store_true', default = False)

parser.add_argument('--ablation', type=str, default='NULL', choices = ['NULL','mlp_conf_loss', 
                                                                      'gnn_no_conf_loss', 
                                                                      'Calibrate'
                                                                      ])

parser.add_argument('--calib_fraction', type=float, default=0.5)
parser.add_argument('--optimize_conformal_score', type=str, default='aps', choices = ['aps', 'raps'])


args = parser.parse_args()

global task
if args.dataset in ['Anaheim', 
                    'ChicagoSketch',  
                    'county_education_2012', 
                    'county_election_2016',
                    'county_income_2012',
                    'county_unemployment_2012',
                    'twitch_PTBR']:
    task = 'regression'
    metric = 'eff_valid_cqr'
    if args.conformal_score != 'cqr':
        raise ValueError('For regression task, the training conformal score should be cqr!')
else:    
    task = 'classification'
    if args.optimize_conformal_score == 'raps':
        metric = 'eff_valid_raps'
    else:
        metric = 'eff_valid_aps'
    
if args.optimal:
    print('Loading optimal set of parameters...')
    args.not_save_res = False
    if args.optimal_examine:
        args.num_runs = 1
    
    args.verbose = False
    args.conf_correct_model = 'gnn'
    args.conftr = True
    args.conftr_calib_holdout = True
    if task == 'classification':
        args.conformal_score = 'aps'
        if args.optimize_conformal_score == 'raps':
            metric = 'eff_valid_raps'
        else:
            metric = 'eff_valid_aps'
    else:
        args.conformal_score = 'cqr'
        args.quantile = True
        metric = 'eff_valid_cqr'
    
    if args.optimize_conformal_score == 'raps':
        with open('./params/optimal_param_set_raps.pkl', 'rb') as f:
            optimal_set = pickle.load(f)
        
    else:
        with open('./params/optimal_param_set.pkl', 'rb') as f:
            optimal_set = pickle.load(f)

    optimal_parameter = optimal_set[args.model][args.dataset]
    
    #print(args)
    
    d = vars(args)   
    for i, j in optimal_parameter.items():
        d[i] = j
        print(str(i) + ' set to ' + str(j))   

    #print(args)
    
if args.bnn or (task == 'classification'):
    args.quantile = False
else:    
    args.quantile = True 
    
if args.ablation == 'mlp_conf_loss':
    args.conf_correct_model = 'mlp'
elif args.ablation == 'gnn_no_conf_loss':
    args.conftr = False
    args.conftr_calib_holdout = False
elif args.ablation == 'TS':
    args.conf_correct_model = 'TS'
    args.conftr_calib_holdout = False

    
if args.hyperopt:
    args.not_save_res = True
    args.num_runs = 3
    args.verbose = False
    args.retrain = False    
    args.conf_correct_model = 'gnn'
    args.conftr = True
    args.conftr_calib_holdout = True
    if task == 'classification':
        args.conformal_score = 'aps'
    else:
        args.conformal_score = 'cqr'
        args.quantile = True

        
if args.conformal_training:
    args.conftr_calib_holdout = False
    args.conftr_holdout = False
    args.conftr_valid_holdout = False
    
device = torch.device(args.device)
    
if args.optimal:
    name = 'optimal_' + args.dataset + '_' + args.model
    if args.ablation != 'NULL':
        name += '_ablation_' + args.ablation   
    if args.calib_fraction != 0.5:
        name += '_calib_fraction_' + str(args.calib_fraction)
else:
    name = args.dataset + '_' + args.model
if args.conftr:
    name+= '_conftr'
if args.conftr_calib_holdout:
    name+='_calib_holdout'
if args.conf_correct_model == 'gnn':    
    name += '_confgnn'
if args.cond_cov_loss:
    name += '_cond_cov_loss'
    
if args.conf_correct_model == 'Calibrate':
    name += '_' + args.calibrator
elif args.conf_correct_model in ['mcdropout', 'QR', 'mcdropout_std']:
    name += '_' + args.conf_correct_model
    
if args.alpha != 0.1:
    name += '_alpha_' + str(args.alpha)    
        
if args.bnn:
    name += '_bnn'
        
if args.optimize_conformal_score == 'raps':
    name += '_raps'
        
if args.wandb:
    wandb.init(project='ConformalGNN_' + args.dataset + '_' + args.model, name=name, config = args)
    
    
    
def gaussian_nll_loss(mean, log_var, y_true):
    # Compute the negative log likelihood for a Gaussian distribution
    precision = torch.exp(-log_var)
    mse_loss = F.mse_loss(mean, y_true, reduction='none')
    nll_loss = 0.5 * (mse_loss * precision + log_var + torch.log(torch.tensor(2 * np.pi)))
    return torch.mean(nll_loss)


def train(epoch, model, data, optimizer, alpha):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index, data.edge_attr)
    if task == 'classification':
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    else:
        if args.quantile:
            mid = out[:, 0][data.train_mask].reshape(-1,1)
            label = data.y[data.train_mask].reshape(-1,1)
            mse_loss = F.mse_loss(mid, label)
            low_bound = alpha/2
            upp_bound = 1 - alpha/2
            lower = out[:, 1][data.train_mask].reshape(-1,1)
            upper = out[:, 2][data.train_mask].reshape(-1,1)
            low_loss = torch.mean(torch.max((low_bound - 1) * (label - lower), low_bound * (label - lower)))
            upp_loss = torch.mean(torch.max((upp_bound - 1) * (label - upper), upp_bound * (label - upper)))
            loss = mse_loss + low_loss + upp_loss
        elif args.bnn:
            mu = out[:, 0][data.train_mask].reshape(-1,1)
            logvar = out[:, 1][data.train_mask].reshape(-1,1)
            loss = gaussian_nll_loss(mu, logvar, data.y[data.train_mask])
        else:
            loss = F.mse_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    if args.quantile:
        return (float(loss), mse_loss, low_loss, upp_loss)
    elif args.bnn:
        return float(loss)
    else:
        return float(loss)


@torch.no_grad()
def test(model, data, alpha, tau, target_size, size_loss = False):
    model.eval()
    if size_loss:
        pred_raw, ori_pred_raw = model(data.x, data.edge_index)
    else:
        pred_raw = model(data.x, data.edge_index)
        
    if task == 'classification':
        pred = pred_raw.argmax(dim=-1)
    else:
        if args.quantile:
            pred = pred_raw[:, 0]
        elif args.bnn:
            pred = pred_raw[:, 0]
        else:
            pred = pred_raw
    accs = []
    for mask in [data.train_mask, data.valid_mask, data.calib_test_mask]:
        if task == 'classification':
            accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
        else:
            accs.append(pearsonr(pred[mask].detach().cpu().numpy().reshape(-1), 
                                 data.y[mask].detach().cpu().numpy().reshape(-1))[0])
            
    if size_loss:
        if task == 'regression':
            if args.quantile:
                query_idx = np.where(data.valid_mask)[0]
                np.random.seed(0)
                np.random.shuffle(query_idx)

                train_train_idx = query_idx[:int(len(query_idx)/2)]
                train_calib_idx = query_idx[int(len(query_idx)/2):]
                
                n_temp = len(train_calib_idx)
                ### use only train_train nodes
                mid = pred_raw[:, 0][train_calib_idx].reshape(-1,1)
                label = data.y[train_calib_idx].reshape(-1,1)
                mse_loss = F.mse_loss(mid, label)
                low_bound = alpha/2
                upp_bound = 1 - alpha/2
                lower = pred_raw[:, 1][train_calib_idx].reshape(-1,1)
                upper = pred_raw[:, 2][train_calib_idx].reshape(-1,1)
                low_loss = torch.mean(torch.max((low_bound - 1) * (label - lower), low_bound * (label - lower)))
                upp_loss = torch.mean(torch.max((upp_bound - 1) * (label - upper), upp_bound * (label - upper)))
                
                ## CQR loss
                size_loss = 0
                lower_calib = pred_raw[:, 1][train_train_idx].reshape(-1,1)
                upper_calib = pred_raw[:, 2][train_train_idx].reshape(-1,1)
                label_calib = data.y[train_train_idx].reshape(-1,1)

                cal_scores = torch.maximum(label_calib-upper_calib, lower_calib-label_calib)
                # Get the score quantile
                qhat = torch.quantile(cal_scores, np.ceil((n_temp+1)*(1-alpha))/n_temp, interpolation='higher')
                size_loss = torch.mean(upper_calib + qhat - (lower_calib - qhat))
                pred_loss = mse_loss + low_loss + upp_loss
        elif args.bnn:
            raise ValueError('Not implemented....')
        else:
            out_softmax = F.softmax(pred_raw, dim = 1)
            query_idx = np.where(data.valid_mask)[0]
            np.random.seed(0)
            np.random.shuffle(query_idx)

            train_train_idx = query_idx[:int(len(query_idx)/2)]
            train_calib_idx = query_idx[int(len(query_idx)/2):]

            n_temp = len(train_calib_idx)
            q_level = np.ceil((n_temp+1)*(1-alpha))/n_temp

            tps_conformal_score = out_softmax[train_calib_idx][torch.arange(len(train_calib_idx)), data.y[train_calib_idx]]
            qhat = torch.quantile(tps_conformal_score, 1 - q_level, interpolation='higher')
            c = torch.sigmoid((out_softmax[train_train_idx] - qhat)/tau)
            size_loss = torch.mean(torch.relu(torch.sum(c, axis = 1) - target_size))
            
        return accs, pred_raw, size_loss.item()
    else:
        return accs, pred_raw    
    
    
def main(args):
    #print(args)
    import torch_geometric.transforms as T

    if args.dataset in ['Cora_CF', 'Cora_ML_CF', 'CiteSeer_CF', 'DBLP_CF', 'PubMed_CF']:
        path = osp.join('data', 'CitationFull')
        dataset = CitationFull(path, args.dataset[:-3], transform=T.NormalizeFeatures())
        data = dataset[0]
    elif args.dataset in ['Amazon-Computers', 'Amazon-Photo']:
        path = osp.join('data', 'Amazon')
        dataset = Amazon(path, args.dataset.split('-')[1], transform=T.NormalizeFeatures())
        data = dataset[0]
    elif args.dataset in ['Coauthor-CS', 'Coauthor-Physics']:
        path = osp.join('data', 'coauthor')
        dataset = Coauthor(path, args.dataset.split('-')[1], transform=T.NormalizeFeatures())
        data = dataset[0]
    else:
        edges = pd.read_csv('./dataset_regression/' + args.dataset + '_edge_list.txt', sep = '\t', header = None) -1
        feats = pd.read_csv('./dataset_regression/' + args.dataset + '_features.txt', sep = '\t', header = None)
        labels = pd.read_csv('./dataset_regression/' + args.dataset + '_labels.txt', sep = '\t', header = None)
        edge_index = torch.tensor(edges[[0, 1]].values.T, dtype=torch.long)
        x = torch.tensor(feats.values, dtype=torch.float)
        y = torch.tensor(labels.values, dtype=torch.float)
        data = Data(x=x, edge_index=edge_index,y=y)

    if task == 'classification':
        y = data.y.detach().cpu().numpy()
        idx = np.array(range(len(y)))
        np.random.seed(args.data_seed)
        np.random.shuffle(idx)
        split_res = np.split(idx, [int(0.2 * len(idx)), int(0.3 * len(idx)), len(idx)])
        train_idx, valid, calib_test = split_res[0], split_res[1], split_res[2]

    elif task == 'regression':
        idx = np.array(range(len(y)))
        np.random.seed(args.data_seed)
        np.random.shuffle(idx)
        split_res = np.split(idx, [int(0.5 * len(idx)), int(0.6 * len(idx)), len(idx)])
        train_idx, valid, calib_test = split_res[0], split_res[1], split_res[2]

    data.train_mask = np.array([False] * len(y))
    data.train_mask[train_idx] = True

    data.valid_mask = np.array([False] * len(y))
    data.valid_mask[valid] = True

    data.calib_test_mask = np.array([False] * len(y))
    data.calib_test_mask[calib_test] = True

    n_trials = 100
    n = min(1000, int(calib_test.shape[0]/2))
    alpha = args.alpha
    tau = args.tau
    target_size = args.target_size
    num_conf_layers = args.confgnn_num_layers
    base_model = args.confgnn_base_model
    optimal_examine_res = {}
    tau2res = {}    
    for run in tqdm(range(args.num_runs)):
        if args.optimal_examine:
            run = 4242
        result_this_run = {}

        if args.quantile:
            if args.alpha == 0.1:
                model_checkpoint = './model/' + args.model + '_' + args.dataset + '_' + str(run+1) + '_quantile_0410.pt'
            else:
                model_checkpoint = './model/' + args.model + '_' + args.dataset + '_' + str(run+1) + '_quantile_' + str(args.alpha) + '_0410.pt'
        elif args.bnn:
            model_checkpoint = './model/' + args.model + '_' + args.dataset + '_' + str(run+1) + '_bnn_' + str(args.alpha) + '_0410.pt'
        else:
            model_checkpoint = './model/' + args.model + '_' + args.dataset + '_' + str(run+1) + '_0410.pt'
        if task == 'regression':
            if args.quantile:
                output_dim = 3
            elif args.bnn:
                output_dim = 2
            else:
                output_dim = 1
            num_features = x.shape[1]
        else:
            output_dim = dataset.num_classes
            num_features = dataset.num_features
        if (os.path.exists(model_checkpoint)) and (not args.retrain):
            print('loading saved base model...')
            model = torch.load(model_checkpoint, map_location = device)
            model, data = model.to(device), data.to(device)
            model.eval()
            pred = model(data.x, data.edge_index)
            best_model = model
            best_pred = pred
        else:
            print('training base model from scratch...')
            model = GNN(num_features, args.hidden_channels, output_dim, args.model, args.heads, args.aggr)    

            model, data = model.to(device), data.to(device)
            optimizer = torch.optim.Adam([
                dict(params=model.conv1.parameters(), weight_decay=5e-4),
                dict(params=model.conv2.parameters(), weight_decay=0)
            ], lr=args.lr)  # Only perform weight-decay on first convolution.

            best_val_acc = final_test_acc = 0
            for epoch in range(1, args.epochs + 1):
                loss = train(epoch, model, data, optimizer, alpha)
                if args.quantile:
                    mse = loss[1]
                    lower = loss[2]
                    upper = loss[3]
                    loss = loss[0]
                
                (train_acc, val_acc, tmp_test_calib_acc), pred = test(model, data, alpha, tau, target_size)
                if val_acc > best_val_acc:
                    #torch.save(best_model, model_checkpoint)
                    best_model = copy.deepcopy(model)
                    best_val_acc = val_acc
                    test_acc = tmp_test_calib_acc
                    best_pred = pred
                if args.quantile:
                    if args.verbose:
                        log(Epoch=epoch, Loss=loss, Train=train_acc, Val=val_acc, Calib_Test=tmp_test_calib_acc, upper=upper, lower=lower, mse=mse)

                else:
                    if args.verbose:
                        log(Epoch=epoch, Loss=loss, Train=train_acc, Val=val_acc, Calib_Test=tmp_test_calib_acc)

            torch.save(best_model, model_checkpoint)
            pred = best_pred
        
        (train_acc, val_acc, test_acc), _ = test(best_model, data, alpha, tau, target_size, size_loss = False)
        
        result_this_run['gnn'] = {}
        
        if args.bnn:
            pred = pred.detach().cpu().numpy()
            pred_all = pred[:, 0].reshape(-1)
            pred_logvar = pred[:, 1].reshape(-1)
            pred_std = np.sqrt(np.exp(pred_logvar))
            mu = pred_all
            pred = np.vstack([mu, mu-1.96 * pred_std,mu+1.96 * pred_std]).T        
        
        if task == 'regression':
            result_this_run['gnn']['CQR'] = run_conformal_regression(pred, data, n, alpha, calib_eval = False)
        else:
            result_this_run['gnn']['APS'] = run_conformal_classification(pred, data, n, alpha, score = 'aps', calib_eval = False)
            result_this_run['gnn']['RAPS'] = run_conformal_classification(pred, data, n, alpha, score = 'raps', calib_eval = False)
        
        condcov_epochs = []
        if args.optimal_examine:
            optimal_examine_res['gnn_pred'] = pred
            optimal_examine_res['data'] = data
        result_this_run['conf_gnn'] = {}
        if args.bnn:
            result_this_run['conf_gnn']['Raw'] = run_conformal_regression(pred, data, n, alpha, score = 'qr', calib_eval = False)        
        
        elif args.conf_correct_model == 'mcdropout':
            pred_all = []
            model.train()
            for i in tqdm(range(1000)):
                pred_all.append(model(data.x, data.edge_index).detach().cpu().numpy())
            model.eval()
            
            pred_all = [i[:,0] for i in pred_all]
            pred_mcdropout = np.vstack([np.mean(pred_all, axis = 0), np.quantile(pred_all, q = alpha/2, axis = 0), np.quantile(pred_all, q = 1-alpha/2, axis = 0)]).T
            result_this_run['conf_gnn']['Raw'] = run_conformal_regression(pred_mcdropout, data, n, alpha, score = 'qr', calib_eval = False)
        elif args.conf_correct_model == 'mcdropout_std':
            pred_all = []
            model.train()
            for i in tqdm(range(1000)):
                pred_all.append(model(data.x, data.edge_index).detach().cpu().numpy())
            model.eval()
            
            pred_all = [i[:,0] for i in pred_all]
            std = np.std(pred_all, axis = 0)
            mu = np.mean(pred_all, axis = 0)
            pred_mcdropout = np.vstack([mu, 
                                        mu - 1.96 * std, 
                                        mu + 1.96 * std]).T
            result_this_run['conf_gnn']['Raw'] = run_conformal_regression(pred_mcdropout, data, n, alpha, score = 'qr', calib_eval = False)

            
        elif args.conf_correct_model == 'QR':
            result_this_run['conf_gnn']['Raw'] = run_conformal_regression(pred, data, n, alpha, score = 'qr', calib_eval = False)
        
        elif args.conf_correct_model == 'Calibrate':
            #print('Use calibration model...')
            if task == 'regression':
                raise ValueError('Unavailable for regression task...')
            model_to_correct = copy.deepcopy(model)
            if args.calibrator == 'TS':
                temp_model = TS(model_to_correct, device)
            elif args.calibrator == 'VS':
                temp_model = VS(model_to_correct, output_dim, device)
            elif args.calibrator == 'ETS':
                temp_model = ETS(model_to_correct, output_dim, device)
            elif args.calibrator == 'CaGCN':
                temp_model = CaGCN(model_to_correct, data.x.shape[0], output_dim, 0.5, device)
            elif args.calibrator == 'GATS':
                temp_model = GATS(model_to_correct, data.edge_index, 
                                  data.x.shape[0], torch.tensor(data.train_mask), 
                                  output_dim, None, 2, 1, device)
            else:
                raise ValueError
            cal_wdecay = 0
            temp_model.fit(data, data['valid_mask'], data['train_mask'], cal_wdecay)
            with torch.no_grad():
                temp_model.eval()
                best_pred = temp_model(data.x, data.edge_index)
            result_this_run['conf_gnn']['Raw'] = run_conformal_classification(best_pred, data, n, args.alpha, score = 'threshold', calib_eval = False)
            
            
        else:    
            model_to_correct = copy.deepcopy(model)
            if args.conf_correct_model == 'gnn':
                confmodel = ConfGNN(model_to_correct, data, args, num_conf_layers, base_model, output_dim, task).to(args.device)
            elif args.conf_correct_model == 'mlp':
                confmodel = ConfMLP(model_to_correct, data, output_dim, task).to(args.device)
            optimizer = torch.optim.Adam(confmodel.parameters(), weight_decay=5e-4, lr=args.confgnn_lr)  # Only perform weight-decay on first convolution.
            pred_loss_hist, size_loss_hist, cons_loss_hist, val_size_loss_hist = [], [], [], []
            best_size_loss = 10000
            best_val_acc = 0

            if args.conftr_calib_holdout:
                calib_test_idx = np.where(data.calib_test_mask)[0]
                np.random.seed(run)
                np.random.shuffle(calib_test_idx)
                calib_eval_idx = calib_test_idx[:int(n * args.calib_fraction)]
                calib_test_real_idx = calib_test_idx[int(n * args.calib_fraction):]

                data.calib_eval_mask = np.array([False] * len(y))
                data.calib_eval_mask[calib_eval_idx] = True
                data.calib_test_real_mask = np.array([False] * len(y))
                data.calib_test_real_mask[calib_test_real_idx] = True
                if args.verbose:
                    print('Using a separate calibration holdout...')
                calib_eval_idx = np.where(data.calib_eval_mask)[0]
                np.random.seed(run)
                np.random.shuffle(calib_eval_idx)
                train_calib_idx = calib_eval_idx[int(len(calib_eval_idx)/2):]
                train_test_idx = calib_eval_idx[:int(len(calib_eval_idx)/2)]
                train_train_idx = np.where(data.train_mask)[0]

            if args.conftr_valid_holdout:
                if args.verbose:
                    print('Using the validation set as holdout...')
                calib_eval_idx = np.where(data.valid_mask)[0]
                np.random.seed(run)
                np.random.shuffle(calib_eval_idx)
                train_calib_idx = calib_eval_idx[int(len(calib_eval_idx)/2):]
                train_test_idx = calib_eval_idx[:int(len(calib_eval_idx)/2)]
                train_train_idx = np.where(data.train_mask)[0]

            if args.conftr_holdout:
                train_idx = np.where(data.train_mask)[0]
                np.random.seed(run)
                np.random.shuffle(train_idx)

                train_train_idx = train_idx[:int(len(train_idx)/2)]

                if args.conftr_sep_test:
                    train_calib_test_idx = train_idx[int(len(train_idx)/2):]
                    np.random.seed(run)
                    np.random.shuffle(train_calib_test_idx)
                    train_calib_idx = train_calib_test_idx[int(len(train_calib_test_idx)/2):]
                    train_test_idx = train_calib_test_idx[:int(len(train_calib_test_idx)/2)]
                else:
                    train_calib_idx = train_idx[int(len(train_idx)/2):]
                    train_test_idx = train_train_idx
            
            print('Starting topology-aware conformal correction...')
            for epoch in range(1, args.epochs + 1):  
                if (not args.conftr_holdout) and (not args.conftr_calib_holdout) and (not args.conftr_valid_holdout):
                    train_idx = np.where(data.train_mask)[0]
                    np.random.seed(epoch)
                    np.random.shuffle(train_idx)
                    train_train_idx = train_idx[:int(len(train_idx)/2)]
                    train_calib_idx = train_idx[int(len(train_idx)/2):]
                    train_test_idx = train_train_idx

                confmodel.train()
                optimizer.zero_grad()
                out, ori_out = confmodel(data.x, data.edge_index)
                if task == 'regression':
                    if args.quantile:
                        ### use only train_train nodes
                        mid = out[:, 0][train_train_idx].reshape(-1,1)
                        label = data.y[train_train_idx].reshape(-1,1)
                        mse_loss = F.mse_loss(mid, label)
                        low_bound = alpha/2
                        upp_bound = 1 - alpha/2
                        lower = out[:, 1][train_train_idx].reshape(-1,1)
                        upper = out[:, 2][train_train_idx].reshape(-1,1)
                        low_loss = torch.mean(torch.max((low_bound - 1) * (label - lower), low_bound * (label - lower)))
                        upp_loss = torch.mean(torch.max((upp_bound - 1) * (label - upper), upp_bound * (label - upper)))
                        pred_loss = mse_loss + low_loss + upp_loss

                        n_temp = len(train_calib_idx)
                        ## CQR loss
                        lower_calib = out[:, 1][train_calib_idx].reshape(-1,1)
                        upper_calib = out[:, 2][train_calib_idx].reshape(-1,1)
                        label_calib = data.y[train_calib_idx].reshape(-1,1)

                        cal_scores = torch.maximum(label_calib-upper_calib, lower_calib-label_calib)
                        # Get the score quantile
                        qhat = torch.quantile(cal_scores, np.ceil((n_temp+1)*(1-alpha))/n_temp, interpolation='higher')

                        lower_test = out[:, 1][train_test_idx].reshape(-1,1)
                        upper_test = out[:, 2][train_test_idx].reshape(-1,1)
                        
                        lower_deviate_loss = F.mse_loss(out[:, 1].reshape(-1,1), ori_out[:, 1].reshape(-1,1))
                        upper_deviate_loss = F.mse_loss(out[:, 2].reshape(-1,1), ori_out[:, 2].reshape(-1,1))
                                                
                        size_loss = torch.mean(upper_test + qhat - (lower_test - qhat))

                        if args.wandb:
                            wandb.log({'run_' + str(run) + '_train_size_loss': size_loss.item(),
                                      'run_' + str(run) + '_train_low_loss': low_loss.item(),
                                      'run_' + str(run) + '_train_up_loss': upp_loss.item(),
                                      'run_' + str(run) + '_train_mse_loss': mse_loss.item(),
                                      'run_' + str(run) + '_pred_loss': pred_loss.item(),
                                      'run_' + str(run) + '_train_qhat': qhat.item(),
                                      'run_' + str(run) + '_train_lower_test': torch.mean(lower_test).item(),
                                      'run_' + str(run) + '_train_upper_test': torch.mean(upper_test).item(),
                                      'run_' + str(run) + '_train_lower_deviate_loss': lower_deviate_loss.item(),
                                      'run_' + str(run) + '_train_upper_deviate_loss': upper_deviate_loss.item(),
                                       
                                      })
                    if args.conftr:
                        if epoch <= 1000:
                            loss = pred_loss
                        else:
                            loss = pred_loss + args.size_loss_weight * size_loss
                            loss += args.reg_loss_weight + lower_deviate_loss
                            loss += args.reg_loss_weight + upper_deviate_loss
                    else:
                        loss = pred_loss

                else:
                    out_softmax = F.softmax(out, dim = 1)
                    ori_out_softmax = F.softmax(ori_out, dim = 1)

                    n_temp = len(train_calib_idx)
                    q_level = np.ceil((n_temp+1)*(1-alpha))/n_temp

                    tps_conformal_score = out_softmax[train_calib_idx][torch.arange(len(train_calib_idx)), data.y[train_calib_idx]]
                    qhat = torch.quantile(tps_conformal_score, 1 - q_level, interpolation='higher')

                    c = torch.sigmoid((out_softmax[train_test_idx] - qhat)/tau)
                    size_loss = torch.mean(torch.relu(torch.sum(c, axis = 1) - target_size))
                    if args.cond_cov_loss:
                        ## coverage loss
                        unique_classes = torch.unique(data.y)
                        y = data.y[train_test_idx]
                        loss_cov = torch.zeros(1).to(device)
                        for i in unique_classes:
                            class_mask = y == i
                            loss_cov += -torch.mean(c[torch.arange(c.shape[0]), y][class_mask])
                        loss_cov = (1/len(unique_classes)) * loss_cov

                        loss_cov = loss_cov.squeeze()
                        #print(loss_cov.item())
                        #print(run_conformal_classification(out, data, n, alpha, score = 'aps', validation_set = True))
                    pred_loss = F.cross_entropy(out[train_train_idx], data.y[train_train_idx])

                    if args.conftr:
                        if epoch <= 1000:
                            loss = pred_loss
                        elif args.cond_cov_loss:
                            if epoch <=3000:
                                loss = pred_loss + args.size_loss_weight * size_loss
                            else:
                                loss = pred_loss + args.size_loss_weight * size_loss + loss_cov
                        else:
                            loss = pred_loss + args.size_loss_weight * size_loss
                    else:
                        loss = pred_loss
                    '''
                    cov_all, eff_all, pred_set_all, val_labels_all, idx_all = run_conformal_classification(out, 
                                                                                       data, n, args.alpha, 
                                                                                       score = 'aps', 
                                                                                       calib_eval = True, 
                                                                                       validation_set = False, 
                                                                                       use_additional_calib = False, 
                                                                                       return_prediction_sets = True)
                    condcov_all = {}
                    for run in range(100):
                        pred_set = pred_set_all[run]
                        val_labels = val_labels_all[run]
                        cov_per_data = pred_set[np.arange(pred_set.shape[0]),val_labels]
                        for l in np.unique(val_labels):
                            if l in condcov_all:
                                condcov_all[l].append(cov_per_data[np.where(val_labels == l)[0]].mean())
                            else:
                                condcov_all[l] = [cov_per_data[np.where(val_labels == l)[0]].mean()]
                    condcov_epochs.append({i: np.mean(j) for i,j in condcov_all.items()})
                    '''
                    
                loss.backward()
                
                #torch.nn.utils.clip_grad_norm_(confmodel.parameters(),0.1)

                optimizer.step()
                if args.verbose:
                    log(Epoch = epoch, Prediction_loss = pred_loss.item(), size_loss = size_loss.item())
                loss = float(loss)
                pred_loss_hist.append(pred_loss.item())
                size_loss_hist.append(size_loss.item())
                
                (train_acc, val_acc, tmp_test_calib_acc), pred, size_loss = test(confmodel, data, alpha, tau, target_size, size_loss = True)
                
                if task == 'regression':
                    eff_valid = run_conformal_regression(pred, data, n, alpha, validation_set = True)[1]
                else:
                    eff_valid = run_conformal_classification(pred, data, n, alpha, score = 'aps', validation_set = True)[1]
                    
                if args.wandb:
                    wandb.log({'run_' + str(run) + '_eff_valid': eff_valid})
                    
                val_size_loss_hist.append(size_loss)
                if args.conftr:
                    if eff_valid < best_size_loss: 
                        best_size_loss = eff_valid
                        test_acc = tmp_test_calib_acc
                        best_pred = pred
                        best_epoch = epoch
                else:
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        test_acc = tmp_test_calib_acc
                        best_pred = pred    

            result_this_run['conf_gnn'] = {}
            
            if task == 'regression':
                result_this_run['conf_gnn']['CQR'] = run_conformal_regression(best_pred, data, n, alpha, calib_eval = args.conftr_calib_holdout, calib_fraction = args.calib_fraction)
                result_this_run['conf_gnn']['eff_valid'] = run_conformal_regression(best_pred, data, n, alpha, validation_set = True)[1]
            
            else:
                result_this_run['conf_gnn']['APS'] = run_conformal_classification(best_pred, data, n, alpha, score = 'aps', calib_eval = args.conftr_calib_holdout, calib_fraction = args.calib_fraction)
                result_this_run['conf_gnn']['RAPS'] = run_conformal_classification(best_pred, data, n, alpha, score = 'raps', calib_eval = args.conftr_calib_holdout, calib_fraction = args.calib_fraction)
                result_this_run['conf_gnn']['eff_valid'] = run_conformal_classification(best_pred, data, n, alpha, score = 'aps', validation_set = True)[1]
                result_this_run['conf_gnn']['eff_valid_raps'] = run_conformal_classification(best_pred, data, n, alpha, score = 'raps', validation_set = True)[1]
            
        if args.optimal_examine:
            optimal_examine_res['confgnn_pred'] = best_pred
            optimal_examine_res['condcov_epochs'] = condcov_epochs
            return optimal_examine_res
        tau2res[run] = result_this_run
        print('Finished training this run!')
      
    if not os.path.exists('./pred'):
        os.mkdir('./pred')
    if not args.not_save_res:
        print('Saving results to', './pred/' + name +'.pkl')
        with open('./pred/' + name +'.pkl', 'wb') as f:
            pickle.dump(tau2res, f)
        
    if args.hyperopt:        
        if task == 'classification':
            wandb.log({'gnn_aps_eff': np.mean([result_this_run['gnn']['APS'][1] for i, result_this_run in tau2res.items()])})
            wandb.log({'gnn_raps_eff': np.mean([result_this_run['gnn']['RAPS'][1] for i, result_this_run in tau2res.items()])})
            wandb.log({'confgnn_aps_eff': np.mean([result_this_run['conf_gnn']['APS'][1] for i, result_this_run in tau2res.items()])})
            wandb.log({'confgnn_raps_eff': np.mean([result_this_run['conf_gnn']['RAPS'][1] for i, result_this_run in tau2res.items()])})
            wandb.log({'eff_valid_aps': np.mean([result_this_run['conf_gnn']['eff_valid'] for i, result_this_run in tau2res.items()])})
            wandb.log({'eff_valid_raps': np.mean([result_this_run['conf_gnn']['eff_valid_raps'] for i, result_this_run in tau2res.items()])})
            
        else:
            wandb.log({'confgnn_cqr_eff': np.mean([result_this_run['conf_gnn']['CQR'][1] for i, result_this_run in tau2res.items()])})
            wandb.log({'gnn_cqr_eff': np.mean([result_this_run['gnn']['CQR'][1] for i, result_this_run in tau2res.items()])})
            wandb.log({'eff_valid_cqr': np.mean([result_this_run['conf_gnn']['eff_valid'] for i, result_this_run in tau2res.items()])})

def hyperopt_loop():
    run = wandb.init()
    args_hyperopt = copy.deepcopy(args)
    if task == 'regression':
        args_hyperopt.reg_loss_weight = wandb.config.reg_loss_weight
    else:
        args_hyperopt.target_size = wandb.config.target_size
        args_hyperopt.tau = wandb.config.tau
    args_hyperopt.confnn_hidden_dim = wandb.config.confnn_hidden_dim
    args_hyperopt.confgnn_num_layers = wandb.config.confgnn_num_layers
    args_hyperopt.confgnn_base_model = wandb.config.confgnn_base_model
    args_hyperopt.confgnn_lr = wandb.config.confgnn_lr
    args_hyperopt.size_loss_weight = wandb.config.size_loss_weight
    
    main(args_hyperopt)
    
            
if args.hyperopt:
    if task == 'regression':
        parameter_set = {
            'confnn_hidden_dim': {'values': [16, 32, 64, 128, 256]},
            'confgnn_lr': {'values': [1e-1,1e-2,1e-3,1e-4]},
            'confgnn_num_layers': {'values': [1,2,3,4]},
            'confgnn_base_model': {'values': ['GAT', 'GCN', 'GraphSAGE', 'SGC']},
            'size_loss_weight': {'values': [1,1e-1,1e-2,1e-3]},
            'reg_loss_weight': {'values': [1,1e-1]}
         }
    else:
        parameter_set = {
            'target_size': {'values': [0, 1]},
            'confnn_hidden_dim': {'values': [16, 32, 64, 128, 256]},
            'confgnn_lr': {'values': [1e-1,1e-2,1e-3,1e-4]},
            'confgnn_num_layers': {'values': [1,2,3,4]},
            'confgnn_base_model': {'values': ['GAT', 'GCN', 'GraphSAGE', 'SGC']},
            'tau': {'values': [10, 1, 1e-1,1e-2,1e-3]},
            'size_loss_weight': {'values': [1,1e-1,1e-2,1e-3]}
         }
    
    sweep_configuration = {
        'method': 'bayes',
        'name': 'sweep',
        'metric': {
            'goal': 'minimize', 
            'name': metric
        },
        'parameters': parameter_set
    }
        
    if args.optimize_conformal_score == 'raps':
        sweep_id = wandb.sweep(sweep=sweep_configuration, project='Conformalized_' + args.model + '_' + args.dataset + '_raps')
        wandb.agent(sweep_id, function=hyperopt_loop, count=100)
    else:
        sweep_id = wandb.sweep(sweep=sweep_configuration, project='Conformalized_' + args.model + '_' + args.dataset + '_' + str(args.alpha))
        wandb.agent(sweep_id, function=hyperopt_loop, count=100)
else:
    
    if args.optimal_examine:
        res = main(args)
        
    else:
        main(args)
