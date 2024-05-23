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

import torch
from torch_geometric.utils.convert import from_networkx
from torch.nn import Linear, MSELoss
from torch_geometric.nn import GCNConv, Sequential
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--met_path", type=str,
                        help="path to the input metabolome in .tsv format",
                        required=True, default=None)
    parser.add_argument("-p", "--prior_path", type=str,
                        help="path to the prior knolwedge based edges in .tsv format",
                        required=True, default=None)
    parser.add_argument("-o", "--out_dir", type=str,
                        help="path to the output directory",
                        required=True, default=None)
    args = parser.parse_args()
    print(args)
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
    
def construct_network(met_df, prior_path):
    mets = list(met_df.columns)
    prior_ntw = construct_prior_network(mets, prior_path, 'prior-network.pkl')
    corr_ntw = construct_spearman_network(met_df, 0.05, 'spearman-network.pkl')
    G = nx.compose(prior_ntw, corr_ntw)
    pickle.dump(G, open('network', 'wb'))
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
    
def plot(x, y, legends, xlabel, ylabel, title):
    for i in range(len(y)):
        plt.plot(x, y[i], label=legends[i])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.savefig('Loss.png')
    plt.close()
        
    
#hyperparameters -> hidden_dims, learning rate, optimizer, epoch count, loss change threshold
    
def impute(met_path, prior_path):
    met_df = pd.read_csv(met_path, sep='\t', index_col='Sample')
    met_arr = met_df.to_numpy().transpose()
    print('met_df', met_df.shape, 'met_arr', met_arr.shape)
    
    observed_index = np.argwhere(~np.isnan(met_arr))
    #print('observed_index', observed_index)
    print(len(observed_index))
    observed_count = len(observed_index)
    
    val_ratio = 0.25 # to tune hyperparams
    val_count = round(observed_count * val_ratio)
    val_index_pos = np.random.randint(low=0, high=len(observed_index), size=val_count)
    val_index = observed_index[val_index_pos, :]
    #print('val_index', val_index.shape, val_index)
    
    train_index = np.delete(observed_index, axis=0, obj=val_index_pos)
    #print('train_index', train_index.shape, train_index)
    
    met_arr[val_index[:0], val_index[:1]] = np.nan
    print('met_arr', met_arr.shape)
    met_arr = np.nan_to_num(met_arr, nan=0)
    print('met_arr', met_arr.shape)
    
    G = construct_network(met_df, prior_path)
    print('Network constructed')
    print('met_arr', len(met_arr.tolist()))
    print('met_cols', len(met_df.columns))
    node_feat = dict(zip(met_df.columns, met_arr.tolist()))
    
    nx.set_node_attributes(G, node_feat, "x")
    #print(G.nodes(data=True))
    graph = from_networkx(G)
    val_index = torch.tensor(val_index)
    train_index = torch.tensor(train_index)
    print('val_index', val_index)
    print(val_index[:0])
    print(val_index[:1])
    print('graph.x', graph.x.shape, graph.x)
    print('train_index', train_index)
    val_true = graph.x[val_index[:, 0], val_index[:, 1]]
    print('val_true', val_true.shape, val_true)
    train_true = graph.x[train_index[:, 0], train_index[:, 1]]
    print('train_true', train_true.shape, train_true)
    in_dim = met_arr.shape[1]
    hidden_dim = [in_dim//2, in_dim//4]
    imputer = Imputer(in_dim, hidden_dim)
    
    lr = 0.01
    criterion = MSELoss()  #Initialize the CrossEntropyLoss function.
    optimizer = Adam(imputer.parameters(), lr=lr)  # Initialize the Adam optimizer.
    
    train_losses = []
    val_losses = []
    max_epoch = 500
    writer = SummaryWriter('./tensorboard')
    for epoch in range(max_epoch):
        print('epoch', epoch)
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
        writer.add_scalar("Train Loss", train_loss.item(), epoch)
        writer.add_scalar("Val Loss", val_loss.item(), epoch)
        if(epoch > 1):
            if(val_losses[-1] > val_losses[-2]):
                break
    writer.flush()
    writer.close()
    print('train_losses', train_losses)
    print('val_losses', val_losses)
    
    plot(list(range(1, len(train_losses)+1)), [train_losses, val_losses], ['Train', 'Validation'], 'Epoch', 'MSE Loss', 'Train and Validation Loss')
    
    
def main(args):
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    os.chdir(args.out_dir)
    
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    log_file = open('prior-based-imputation.log', 'w')
    sys.stdout = log_file
    sys.stderr = log_file
    
    impute(args.met_path, args.prior_path)

    sys.stdout = orig_stdout 
    sys.stderr = orig_stderr 
    log_file.close()
    
if __name__ == "__main__":
    main(parse_args())
