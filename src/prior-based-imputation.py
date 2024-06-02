import argparse, os, pickle, sys, json, shutil
from pathlib import Path
import pandas as pd
import networkx as nx
from itertools import product
from scipy.stats import spearmanr
from random import sample
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from functools import reduce

import torch
from torch_geometric.utils.convert import from_networkx
from torch.nn import Linear, MSELoss
from torch_geometric.nn import GCNConv
from torch.optim import Adam, SGD
from torch_geometric.loader import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--met_path", type=str,
                        help="path to the input metabolome in .tsv format",
                        required=True, default=None)
    parser.add_argument("-p", "--prior_path", type=str,
                        help="path to the prior knolwedge based edges in .tsv format",
                        required=False, default=None)
    parser.add_argument("-u", "--use_prior", action=argparse.BooleanOptionalAction,
                       help="whether to use prior edges - if not set, then prior_path argument is unused.")
    parser.add_argument("-t", "--threshold", type=float,
                       help="p-value threshold for correlation edges between metabolites")
    parser.add_argument("-o", "--out_dir", type=str,
                        help="path to the output directory",
                        required=True, default=None)
    args = parser.parse_args()
    return args

def construct_spearman_graph(met_df, threshold, out_path):
    #print('Constructing spearman correlation graph.')
    corr_pval = pd.DataFrame(columns=met_df.columns, index=met_df.columns)
    corr_edges = []
    for row in met_df.columns:
        for col in met_df.columns:
            common = met_df[met_df[row].notnull() & met_df[col].notnull()]
            spearman_res = spearmanr(common[row], common[col])
            corr_pval.at[row, col] = spearman_res.pvalue
            if(spearman_res.pvalue <= threshold):
                corr_edges.append((row, col))
                
    corr_pval.to_csv('spearman_pvalue.tsv', sep='\t', index=True)
    G = nx.Graph()
    G.add_nodes_from(met_df.columns)
    G.add_edges_from(corr_edges)
    G.remove_edges_from(list(nx.selfloop_edges(G)))
    pickle.dump(G, open(out_path, 'wb'))

    #print('nodes', G.number_of_nodes(), 'isolates', nx.number_of_isolates(G), 'edges', G.number_of_edges())
    cc = sorted(nx.connected_components(G), key=len, reverse=True)
    #print('Connected component count:', len(cc))
    #print('Largest connected component size:', len(cc[0]))
    return G
            
    
def construct_prior_graph(mets, prior_path, out_path):
    #print('Constructing prior graph.')
    G = nx.Graph()
    G.add_nodes_from(mets)
    prior_df = pd.read_csv(prior_path, sep='\t')
    prior_edges = list(zip(prior_df.compound_1, prior_df.compound_2))
    G.add_edges_from(prior_edges)
    G.remove_edges_from(list(nx.selfloop_edges(G)))
    pickle.dump(G, open(out_path, 'wb'))
    
    #print('nodes', G.number_of_nodes(), 'isolates', nx.number_of_isolates(G), 'edges', G.number_of_edges())
    cc = sorted(nx.connected_components(G), key=len, reverse=True)
    #print('Connected component count:', len(cc))
    #print('Largest connected component size:', len(cc[0]))
    return G
    
def construct_graph(met_df, prior_path, use_prior, threshold):
    mets = list(met_df.columns)
    G = construct_spearman_graph(met_df, threshold, 'spearman-graph.pkl')
    
    degree_df = []
    
    degree = list(G.degree())
    #print('Spearman node degrees', degree)
    degree_sp = pd.DataFrame(data=degree, columns=['Node', 'Spearman_Degree'])
    degree_df.append(degree_sp)
    #print()
    
    if ((use_prior is not None) and (use_prior)):
        #print('use_prior is turned on.')
        prior_G = construct_prior_graph(mets, prior_path, 'prior-graph.pkl')
        degree = list(prior_G.degree())
        #print('Prior node degrees', degree)
        degree_pr = pd.DataFrame(data=degree, columns=['Node', 'Prior_Degree'])
        degree_df.append(degree_pr)
        #print()
        
        #print('Constructing merged graph.')
        G = nx.compose(G, prior_G)
        
        #print('nodes', G.number_of_nodes(), 'isolates', nx.number_of_isolates(G), 'edges', G.number_of_edges())
        cc = sorted(nx.connected_components(G), key=len, reverse=True)
        #print('Connected component count:', len(cc))
        #print('Largest connected component size:', len(cc[0]))
        
        degree = list(G.degree())
        #print('Merged node degrees', degree)
        degree_mg = pd.DataFrame(data=degree, columns=['Node', 'Merged_Degree'])
        degree_df.append(degree_mg)
        #print()
    
    degree_df = reduce(lambda x, y: pd.merge(x, y, on='Node', how='inner'), degree_df)
    degree_df.to_csv('node_degree.tsv', sep='\t', index=False)
    
    pickle.dump(G, open('graph.pkl', 'wb'))
    return G

class Encoder(torch.nn.Module):
    def __init__(self, dim): # dim -> variable length list
        #print('Encoder init start')
        super(Encoder, self).__init__()
        self.conv = []
        for i in range(len(dim)-2):
            #print('conv', i)
            self.conv.append(GCNConv(in_channels=dim[i], out_channels=dim[i+1]))
            #print(self.conv[i].lin.weight.shape)
            #print(self.conv[i].lin.bias.shape)
        self.conv = torch.nn.ModuleList(self.conv)
        self.lin = Linear(dim[-2], dim[-1])
        #print('Encoder init end')

    def forward(self, x, edge_index):
        '''print('Encoding forward')
        print('x', x.is_cuda)
        for i in range(len(self.conv)):
            print('conv', i, 'bias', self.conv[i].state_dict()['bias'].is_cuda)
            print('conv', i, 'lin.weight', self.conv[i].state_dict()['lin.weight'].is_cuda)
        print('lin', 'bias', self.lin.state_dict()['bias'].is_cuda)
        print('lin', 'weight', self.lin.state_dict()['weight'].is_cuda)'''
        
        for i in range(len(self.conv)):
            h = self.conv[i](x, edge_index).relu()
            x = h
        out = self.lin(x)
        return out
    
class Decoder(torch.nn.Module):
    def __init__(self, dim): # dim -> variable length list
        super(Decoder, self).__init__()
        self.lin = []
        for i in range(len(dim)-1):
            self.lin.append(Linear(dim[i], dim[i+1]))
        self.lin = torch.nn.ModuleList(self.lin)
        #print('Decoder')
    
    def forward(self, x):
        '''print('Decoding')
        print('x', x.is_cuda)
        for i in range(len(self.lin)):
            print('lin', 'weight', i, self.lin[i].state_dict()['weight'].is_cuda)
            print('lin', 'bias', i, self.lin[i].state_dict()['bias'].is_cuda)'''
        for i in range(len(self.lin)-1):
            h = self.lin[i](x).relu()
            x = h
        out = self.lin[len(self.lin)-1](x)
        return out
    
    
class Imputer(pl.LightningModule):
    def __init__(self, dim, train_index, val_index, train_true, val_true, optimizer, learning_rate):
        #print('Constructing Imputer')
        print('dim', dim, 'train_index', train_index, 'val_index', val_index, 'train_true', train_true, 'val_true', val_true, 'optimizer', optimizer, 'learning_rate', learning_rate)
        super(Imputer, self).__init__()
        self.save_hyperparameters()
        self.encoder = Encoder(dim)
        #print('Here 1')
        rev_dim = list(reversed(dim))
        self.decoder = Decoder(rev_dim)
        #print('Here 2')
        
        self.train_index = train_index
        self.val_index = val_index
        
        '''print('self.device', self.device)
        print('Before device conversion')
        print('train_true', train_true.is_cuda)
        print('val_true', val_true.is_cuda)'''
        
        self.train_true = train_true.to(self.device)
        self.val_true = val_true.to(self.device)
        
        #print('After device conversion')
        #print('train_true', self.train_true.is_cuda)
        #print('val_true', self.val_true.is_cuda)
        
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        
        self.loss = MSELoss()
        #print('Imputer init done.')

    def forward(self, x, edge_index):
        h = self.encoder(x, edge_index)
        out = self.decoder(h)
        return out
    
    def training_step(self, batch, batch_index):
        out = self.forward(batch.x, batch.edge_index)
        
        train_pred = out[self.train_index[:, 0], self.train_index[:, 1]]
        train_true = self.train_true.type_as(train_pred)
        train_mse = self.loss(train_pred, train_true)
        
        val_pred = out[self.val_index[:, 0], self.val_index[:, 1]]
        val_true = self.val_true.type_as(val_pred)
        val_mse = self.loss(val_pred, val_true)
        self.log_dict({"train_mse": train_mse, "val_mse": val_mse}, on_step=False, on_epoch=True)
        
        return train_mse
    
    '''def validation_step(self, batch, batch_index):
        out = self.forward(batch.x, batch.edge_index)
        val_pred = out[self.val_index[:, 0], self.val_index[:, 1]]
        val_true = self.val_true.type_as(val_pred)
        #print('val_pred', val_pred.shape, 'val_true', val_true.shape)
        #print('val_pred', val_pred.is_cuda, 'val_true', val_true.is_cuda)
        val_mse = self.loss(val_pred, val_true)
        self.log("val_mse", val_mse)
        return val_mse'''
    
    def predict_step(self, batch, batch_index):
        out = self.forward(batch.x, batch.edge_index)
        val_pred = out[self.val_index[:, 0], self.val_index[:, 1]]
        val_true = self.val_true.type_as(val_pred)
        val_mse = self.loss(val_pred, val_true)
        return {'out': out, 'val_mse': val_mse}
    
    def configure_optimizers(self):
        if(self.optimizer == 'Adam'):
            return Adam(self.parameters(), lr=self.learning_rate)
        elif(self.optimizer == 'SGD'):
            return SGD(self.parameters(), lr=self.learning_rate)
        else:
            raise Exception('Unrecognized optimizer:', self.optimizer)
    
def plot(x, y, legends, xlabel, ylabel, title, png_path):
    print('Plotting...')
    if(legends is None):
        for i in range(len(y)):
            plt.plot(x, y[i])
    else:
        for i in range(len(y)):
            plt.plot(x, y[i], label=legends[i])
            plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(png_path)
    print('Plot saved to', png_path)
    plt.close()
    
def run_with_hparams(data_loader, dim, optimizer, learning_rate, train_index, val_index, train_true, val_true, hparam_label):
    max_epochs = 1000
    #print('data_loader', data_loader)
    imputer = Imputer(dim, train_index, val_index, train_true, val_true, optimizer, learning_rate)
    #print(imputer)
    
    early_stopping = EarlyStopping(monitor='val_mse', mode='min', patience=1)
    checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="val_mse", mode="min")
    
    if(os.path.exists("logs/" + hparam_label)):
        shutil.rmtree("logs/" + hparam_label)
    
    logger = CSVLogger("logs", name=hparam_label)

    trainer = pl.Trainer(max_epochs=max_epochs, callbacks=[early_stopping, checkpoint_callback], log_every_n_steps=1, accelerator="gpu", devices=1, enable_checkpointing=True, logger=logger)
    trainer.fit(imputer, train_dataloaders=data_loader, val_dataloaders=data_loader)
    #print('Fitting done')
    best_imputer = Imputer.load_from_checkpoint(checkpoint_callback.best_model_path)
    #print('best_imputer', best_imputer)
    pred = trainer.predict(model=best_imputer, dataloaders=data_loader)
    
    mse_path = "logs/" + hparam_label + '/version_' + str(logger.version) + '/metrics.csv'
    print('mse_path', mse_path)
    mse_df = pd.read_csv(mse_path, sep=',')
    x = mse_df['epoch'].tolist()
    train_y = mse_df['train_mse'].tolist()
    val_y = mse_df['val_mse'].tolist()
    y = [train_y, val_y]
    png_path = "logs/" + hparam_label + '/version_' + str(logger.version) + '/MSE_Loss.png'
    plot(x, y, ['Train', 'Validation'], 'Epoch', 'MSE Loss', 'MSE Loss across epochs', png_path)
    return pred
     
def tune_hyperparameters(graph, train_index, val_index, train_true, val_true, sample_names, met_names):
    #hidden_layer_count = [2, 3, 4]
    #hidden_dim_factor = [2, 3]
    #optimizer = ['Adam', 'SGD']
    #learning_rate = [0.001, 0.01, 0.1]
    
    hidden_layer_count = [3]
    hidden_dim_factor = [2]
    optimizer = ['Adam']
    learning_rate = [0.01, 0.1]
    
    hparam_combo = list(product(hidden_layer_count, hidden_dim_factor, optimizer, learning_rate))
    in_dim = graph.x.shape[1]
    data_loader = DataLoader([graph])
    hparam_val_loss = {}
    
    hparam_file = open('hyperparameters.tsv', 'w')
    hparam_file.write('hparam_no' + '\t' + 'hidden_layer_count' + '\t' + 'hidden_dim_factor' + '\t' + 'optimizer' + '\t' + 'learning_rate' + '\n')
 
    for hparam_no in range(len(hparam_combo)):
        hparam_label = 'hparams-'+str(hparam_no)
        Path(hparam_label).mkdir(parents=True, exist_ok=True)
        hparam = hparam_combo[hparam_no]
        hparam_file.write(hparam_label + '\t' + str(hparam[0]) + '\t' + str(hparam[1]) + '\t' + str(hparam[2]) + '\t' + str(hparam[3]) + '\n')
        dim = [in_dim]
        for i in range(1, hparam[0]+1):
            dim.append(in_dim//(hparam[1]**i))
        #print(hparam_label, ': hidden_layer_count', hparam[0], 'hidden_dim_factor', hparam[1], 'dim', dim, 'optimizer', hparam[2], 'learning_rate', hparam[3], flush=True)
        res = run_with_hparams(data_loader, dim, hparam[2], hparam[3], train_index, val_index, train_true, val_true, hparam_label)
        print('res', res)
        met_arr = res[0]['out'].detach().cpu().numpy().transpose()
        met_df = pd.DataFrame(data=met_arr, index=sample_names, columns=met_names)
        met_df.index.name = 'Sample'
        met_df.to_csv(hparam_label + '/' + hparam_label + '.tsv', sep='\t', index=True)
        hparam_val_loss[hparam_no] = res[0]['val_mse'].item()
        
    hparam_file.flush()
    hparam_file.close()
        
    best_hparam_no = min(hparam_val_loss, key=hparam_val_loss.get)
    best_hparam = dict(zip(['hidden_layer_count', 'hidden_dim_factor', 'optimizer', 'learning_rate'], hparam_combo[best_hparam_no]))
    with open('best_hparam.json', 'w') as file:
        json.dump(best_hparam, file)
    plot(hparam_val_loss.keys(), [hparam_val_loss.values()], None, 'Hyperparameters', 'Early stopping validation MSE loss', 'Hyperparameter tuning', 'Tune.png')
    
    
def impute(met_path, prior_path, use_prior, threshold):
    met_df = pd.read_csv(met_path, sep='\t', index_col='Sample') # row->sample, col->met
    met_arr = met_df.to_numpy().transpose()    # row->met, col->sample, so that later can get dict of format {met1: [sample values], met2: [sample values]...}
    print(met_df.shape[0], 'samples', met_df.shape[1], 'metabolites')
    
    observed_index = np.argwhere(~np.isnan(met_arr)) # np array    
    observed_count = len(observed_index)
    
    val_ratio = 0.25 # to tune hyperparams
    val_count = round(observed_count * val_ratio)
    val_index_pos = np.random.choice(len(observed_index), size=val_count, replace=False)
    val_index = observed_index[val_index_pos, :]
    train_index = np.delete(observed_index, axis=0, obj=val_index_pos)
    print('observed_count', observed_count, 'train_count', train_index.shape[0], 'val_count', val_count, val_index.shape[0]) # check train + val = observed?
    
    val_index = torch.tensor(val_index)
    train_index = torch.tensor(train_index)
    
    G = construct_graph(met_df, prior_path, use_prior, threshold)
    node_feat = dict(zip(met_df.columns, met_arr.tolist()))
    nx.set_node_attributes(G, node_feat, "x")
    
    graph = from_networkx(G)
    val_true = graph.x[val_index[:, 0], val_index[:, 1]]
    train_true = graph.x[train_index[:, 0], train_index[:, 1]]
    
    #print('nan count before:', torch.isnan(graph.x).sum())
    graph.x[val_index[:, 0], val_index[:, 1]] = float('nan')
    #print('nan count after:', torch.isnan(graph.x).sum())
    mask_df = pd.DataFrame(graph.x.numpy().transpose(), columns=met_df.columns, index=met_df.index)
    mask_df.to_csv("masked.tsv", sep="\t", index=True)
    graph.x = torch.nan_to_num(graph.x, nan=0.0)
 
    tune_hyperparameters(graph, train_index, val_index, train_true, val_true, list(met_df.index), list(met_df.columns))
    
def main(args):
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    os.chdir(args.out_dir)
    
    '''orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    log_file = open('prior-based-imputation.log', 'w')
    sys.stdout = log_file
    sys.stderr = log_file'''
    
    print(args)
    
    impute(args.met_path, args.prior_path, args.use_prior, args.threshold)

    '''sys.stdout = orig_stdout 
    sys.stderr = orig_stderr 
    log_file.close()'''
    
if __name__ == "__main__":
    main(parse_args())
