import argparse
from pathlib import Path
import os
import pandas as pd
import networkx as nx
from itertools import product
import pickle
import sys
from scipy.stats import spearmanr
from random import sample
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

import torch
from torch_geometric.utils.convert import from_networkx
from torch.nn import Linear, MSELoss
from torch_geometric.nn import GCNConv, Sequential
from torch.optim import Adam, SGD
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.transforms import to_device


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--met_path", type=str,
                        help="path to the input metabolome in .tsv format",
                        required=True, default=None)
    parser.add_argument("-p", "--prior_path", type=str,
                        help="path to the prior knolwedge based edges in .tsv format",
                        required=True, default=None)
    parser.add_argument("-f", "--flag_prior", action=argparse.BooleanOptionalAction,
                       help="Whethe to use prior edges. If set to No, then prior_path argument is not used.")
    parser.add_argument("-o", "--out_dir", type=str,
                        help="path to the output directory",
                        required=True, default=None)
    args = parser.parse_args()
    return args

def construct_spearman_network(met_df, threshold, out_path):
    corr_pval = pd.DataFrame(columns=met_df.columns)
    corr_edges = []
    for row in met_df.columns:
        for col in met_df.columns:
            common = met_df[met_df[row].notnull() & met_df[col].notnull()]
            spearman_res = spearmanr(common[row], common[col])
            corr_pval.at[row, col] = spearman_res.pvalue
            if(spearman_res.pvalue <= threshold):
                corr_edges.append((row, col))
    G = nx.Graph()
    G.add_nodes_from(met_df.columns)
    G.add_edges_from(corr_edges)
    G.remove_edges_from(list(nx.selfloop_edges(G)))
    pickle.dump(G, open(out_path, 'wb'))
    return G
            
    
def construct_prior_network(mets, prior_path, out_path):
    G = nx.Graph()
    G.add_nodes_from(mets)
    prior_df = pd.read_csv(prior_path, sep='\t')
    prior_edges = list(zip(prior_df.compound_1, prior_df.compound_2))
    G.add_edges_from(prior_edges)
    G.remove_edges_from(list(nx.selfloop_edges(G)))
    print('nodes', G.number_of_nodes(), 'isolates', nx.number_of_isolates(G))
    
    pickle.dump(G, open(out_path, 'wb'))
    return G

def isnan(t):
    return torch.isnan(t).any()
    
def construct_network(met_df, prior_path, flag_prior):
    mets = list(met_df.columns)
    G = construct_spearman_network(met_df, 0.05, 'spearman-network.pkl')
    if ((flag_prior is not None) and (flag_prior)):
        print('Including prior edges.')
        prior_ntw = construct_prior_network(mets, prior_path, 'prior-network.pkl')
        G = nx.compose(G, prior_ntw)
    pickle.dump(G, open('network', 'wb'))
    print('Network constructed')
    return G

class Encoder(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(Encoder, self).__init__()
        self.conv = GCNConv(in_dim, hidden_dim)
        self.lin = Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        print('Encoding')
        print('x', x.shape, isnan(x))
        h = self.conv(x, edge_index)
        print('h', h.shape, isnan(h))
        h = h.relu()
        print('h', h.shape, isnan(h))
        out = self.lin(h)
        print('out', out.shape, isnan(out))
        return out
    
class Decoder(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(Decoder, self).__init__()
        self.lin1 = Linear(in_dim, hidden_dim)
        self.lin2 = Linear(hidden_dim, out_dim)

    def forward(self, x):
        print('Decoding')
        print('x', x.shape, isnan(x))
        h = self.lin1(x)
        print('h', h.shape, isnan(h))
        h = h.relu()
        print('h', h.shape, isnan(h))
        out = self.lin2(h)
        print('out', out.shape, isnan(out))
        return out
    
class Imputer(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(Imputer, self).__init__()
        self.encoder = Encoder(in_dim, hidden_dim[0], hidden_dim[1])
        self.decoder = Decoder(hidden_dim[1], hidden_dim[0], in_dim)

    def forward(self, x, edge_index):
        h = self.encoder(x, edge_index)
        out = self.decoder(h)
        print('out', out.shape, isnan(out))
        return out
    
def plot(x, y, legends, xlabel, ylabel, title, png_path):
    for i in range(len(y)):
        plt.plot(x, y[i], label=legends[i])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.savefig(png_path)
    plt.close()
    
def run_with_hparams(graph, hidden_layer_count, hidden_dim_factor, optimizer, learning_rate, train_index, val_index, train_true, val_true, sample_names, met_names, tsv_path, png_path):
    in_dim = graph.x.shape[1]
    hidden_dim = []
    for i in range(1, hidden_layer_count+1):
        hidden_dim.append(in_dim//(hidden_dim_factor**i))
        
    imputer = Imputer(in_dim, hidden_dim)
    criterion = MSELoss()
    if(optimizer == 'Adam'):
        optimizer = Adam(imputer.parameters(), learning_rate)
    elif(optimizer == 'SGD'):
        optimizer = SGD(imputer.parameters(), learning_rate)
    else:
        raise Exception('Unrecognized optimizer: ', optimizer)
    train_losses = []
    val_losses = []
    
    prev_x = None
    epoch = 0
    while(True):
        optimizer.zero_grad()
        res = imputer(graph.x, graph.edge_index)
        val_pred = res[val_index[:, 0], val_index[:, 1]]
        print('val_pred', val_pred.shape, val_pred)
        train_pred = res[train_index[:, 0], train_index[:, 1]]
        print('train_pred', train_pred.shape, train_pred)
        train_loss = criterion(train_pred, train_true)
        val_loss = criterion(val_pred, val_true)
        train_loss.backward()
        optimizer.step()
        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())
        if(epoch > 1):
            if(val_losses[-1] > val_losses[-2]):
                df = pd.DataFrame(prev_x.transpose(), index=sample_names, columns=met_names)
                df.index.name = 'Sample'
                df.to_csv(tsv_path, sep="\t", index=True)
                break
        else:
            prev_x = graph.x.detach().clone().cpu().numpy()
        epoch += 1
            
            
    plot(list(range(1, len(train_losses)+1)), [train_losses, val_losses], ['Train', 'Validation'], 'Epoch', 'MSE Loss', png_path.replace('.png', ''), png_path)
    return train_losses, val_losses
       
        
def find_best_hparams(graph, train_index, val_index, train_true, val_true, sample_names, met_names):
    learning_rate = [0.001, 0.01, 0.1]
    hidden_layer_count = [2, 3, 4]
    hidden_dim_factor = [2, 3]
    optimizer = ['Adam', 'SGD']
    
    hparam_combo = list(product(hidden_layer_count, hidden_dim_factor, optimizer, learning_rate))
    hparam_labels = ['hparams-' +str(i) for i in range(1, len(hparam_combo)+1)]
    hparams = dict(zip(hparam_labels, hparam_combo))
    best_hparam_label = None
    best_hparam = None
    best_val_loss = float('inf')
    
    plot_x = []
    plot_y = []
    
    for label, hparam in hparams.items():
        train_losses, val_losses = run_with_hparams(graph, hparam[0], hparam[1], hparam[2], hparam[3], train_index, val_index, train_true, val_true, sample_names, met_names, label + ' Imputation.tsv', label + ' Loss.png')
        plot_x.append(label)
        plot_y.append(val_losses[-2])
        
        if(val_losses[-2] < best_val_loss):
            best_val_loss = val_losses[-2]
            best_hparam_label = label
            best_hparam = hparam
    plot(plot_x, [plot_y], 'Hyperparameters', 'Early stopping validation MSE loss', 'Hyperparameter tuning', 'Tune.png')
    
    with open('best_hparam.txt', 'w') as file:
        file.write(best_hparam_label + ' ' + best_hparam) 
        
def impute(met_path, prior_path, flag_prior):
    met_df = pd.read_csv(met_path, sep='\t', index_col='Sample') # row->sample, col->met
    met_arr = met_df.to_numpy().transpose()
    # row->met, col->sample, so that later can get dict of format {met1: [sample values], met2: [sample values]...}
    print('met_df', met_df.shape, 'met_arr', met_arr.shape)
    
    observed_index = np.argwhere(~np.isnan(met_arr)) # np array    
    observed_count = len(observed_index)
    
    val_ratio = 0.25 # to tune hyperparams
    val_count = round(observed_count * val_ratio)
    val_index_pos = np.random.randint(low=0, high=len(observed_index), size=val_count)
    val_index = observed_index[val_index_pos, :]
    train_index = np.delete(observed_index, axis=0, obj=val_index_pos)
    print('observed_count', observed_count, 'train_count', train_index.shape[0], 'val_count', val_count, val_index.shape[0])
    
    val_index = torch.tensor(val_index)
    train_index = torch.tensor(train_index)
    
    G = construct_network(met_df, prior_path, flag_prior)
    node_feat = dict(zip(met_df.columns, met_arr.tolist()))
    nx.set_node_attributes(G, node_feat, "x")
    
    graph = from_networkx(G)
    val_true = graph.x[val_index[:, 0], val_index[:, 1]]
    train_true = graph.x[train_index[:, 0], train_index[:, 1]]
    
    print('nan count before:', torch.isnan(graph.x).sum())
    graph.x[val_index[:, 0], val_index[:, 1]] = float('nan')
    print('nan count after:', torch.isnan(graph.x).sum())
    pd.DataFrame(graph.x.numpy().transpose(), columns=met_df.columns, index=met_df.index)
    print('graph.x', graph.x.shape, 'after replace', torch.nan_to_num(graph.x, nan=0.0).shape)
    graph.x = torch.nan_to_num(graph.x, nan=0.0)
    
    find_best_hparams(graph, train_index, val_index, train_true, val_true, met_df.index.values.tolist(), met_df.columns.tolist())
    
def main(args):
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    os.chdir(args.out_dir)
    
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    log_file = open('prior-based-imputation.log', 'w')
    sys.stdout = log_file
    sys.stderr = log_file
    
    print(args)
    
    impute(args.met_path, args.prior_path, args.flag_prior)

    sys.stdout = orig_stdout 
    sys.stderr = orig_stderr 
    log_file.close()
    
if __name__ == "__main__":
    main(parse_args())

