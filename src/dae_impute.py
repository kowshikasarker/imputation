import argparse, os, sys, json, pickle, warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from pathlib import Path
from math import ceil
from itertools import product
from shutil import rmtree
from subprocess import check_output
from io import StringIO

import torch
import torch.nn as nn
import torch_geometric.nn as geom_nn
import torch_geometric.data as geom_data

from torch.optim import Adam
from torch_geometric.transforms import ToUndirected
from torch_geometric.utils import to_networkx

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--met_path", type=str,
                        help="Metabolome in .tsv format after preprocessing, the first column name should be 'Sample' containing sample identifiers and the rest of the columns should contain metabolomic concentrations",
                        required=True, default=None)
    parser.add_argument("--edge_path", type=str,
                        help="Edges in .tsv format, the first column 'Node1' and seconds column 'Node2' mention metabolites that match with column names of in_path argument",
                        required=True, default=None)
    parser.add_argument("--out_dir", type=str,
                        help="path to output dir",
                        required=True, default=None)
    args = parser.parse_args()
    return args

def plot_metric(x, y, xlabel, ylabel, title, png_path):
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(png_path)
    plt.close()

def get_free_gpu():
    gpu_stats = check_output(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"])
    gpu_stats = gpu_stats.decode("utf-8")
    gpu_df = pd.read_csv(StringIO(gpu_stats),
                         names=['memory.used',
                                'memory.free'],
                         skiprows=1)
    gpu_df.to_csv('gpu_memory.tsv', sep='\t')
    gpu_df['memory.used'] = gpu_df['memory.used'].str.replace(" MiB", "").astype(int)
    gpu_df['memory.free'] = gpu_df['memory.free'].str.replace(" MiB", "").astype(int)
    print('GPU usage:\n{}'.format(gpu_df))
    gpu_id = gpu_df['memory.free'].idxmax()
    print('Returning GPU{} with {} free MiB'.format(gpu_id, gpu_df.iloc[gpu_id]['memory.free']))
    return gpu_id  

class Imputer(pl.LightningModule):
    def __init__(self, feature_dim, hidden_dim, lr):
        print('Imputer', Imputer)
        print('feature_dim', feature_dim, 'hidden_dim', hidden_dim, 'lr', lr)
        super(Imputer, self).__init__()
        self.save_hyperparameters()
        self.model = DenoisingAutoencoder(feature_dim, hidden_dim)
        print('self.model', self.model)
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.loss_module = torch.nn.MSELoss()
        
    def forward(self, x, edge_index):
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        return self.model(x, edge_index)
    
    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)  
    
    def training_step(self, batch):
        out = self.forward(batch.x, batch.edge_index)
        loss = self.loss_module(out, batch.x)
        self.log_dict({'loss': loss.item()})
        return loss
    
    def predict_step(self, batch):
        out = self.forward(batch.x, batch.edge_index)
        return out
    
def get_free_gpu():
    gpu_stats = check_output(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"])
    gpu_stats = gpu_stats.decode("utf-8")
    gpu_df = pd.read_csv(StringIO(gpu_stats),
                         names=['memory.used',
                                'memory.free'],
                         skiprows=1)
    gpu_df.to_csv('gpu_memory.tsv', sep='\t')
    gpu_df['memory.used'] = gpu_df['memory.used'].str.replace(" MiB", "").astype(int)
    gpu_df['memory.free'] = gpu_df['memory.free'].str.replace(" MiB", "").astype(int)
    print('GPU usage:\n{}'.format(gpu_df))
    gpu_id = gpu_df['memory.free'].idxmax()
    print('Returning GPU{} with {} free MiB'.format(gpu_id, gpu_df.iloc[gpu_id]['memory.free']))
    return gpu_id
    
class DenoisingAutoencoder(nn.Module):
    def __init__(self, feature_dim, hidden_dim):
        print('DenoisingAutoencoder')
        super().__init__()
        layers = []
        in_dim = feature_dim
        for h in range(len(hidden_dim)):
            out_dim = hidden_dim[h]
            layers += [
                geom_nn.GATConv(in_channels=in_dim, out_channels=out_dim),
                nn.ReLU(inplace=True),
            ]
            in_dim = out_dim
        layers += [geom_nn.GATConv(in_channels=hidden_dim[-1], out_channels=feature_dim)]
        self.layers = nn.ModuleList(layers)
        print('self.layers', self.layers)

    def forward(self, x, edge_index):
        for layer in self.layers:
            if isinstance(layer, geom_nn.MessagePassing):
                x = layer(x, edge_index)
            else:
                x = layer(x)
        return x
    
def dae_impute(met_path, edge_path):
    met_df = pd.read_csv(met_path, sep='\t', index_col='Sample')
    print('met_df', met_df.shape)
    met_df = met_df.fillna(met_df.mean(axis=0))
    corruption_pct = 0.50
    corruption_count = int(ceil(met_df.size * corruption_pct))
    print('corruption_count', corruption_count)
    
    mask_pos = np.random.choice(list(range(met_df.size)), size=corruption_count, replace=False)
    print('mask_pos', mask_pos)
    mask = np.array([False] * met_df.size)
    mask[mask_pos] = True
    mask = mask.reshape(met_df.shape)
    print('mask', 'sum', mask.sum())
    assert corruption_count == mask.sum()
    met_df[mask] = 0
    
    met_df = met_df.transpose() # row -> metabolite, col -> sample
    met_node_id = dict(zip(list(met_df.index), list(range(met_df.shape[0]))))
    json.dump(met_node_id, open("met_node_id.json", "w"), indent = 4)
    
    x = met_df.to_numpy()
    print('x', x.shape)

    edge_df = pd.read_csv(edge_path, sep='\t')
    
    edge_df = pd.DataFrame(np.sort(edge_df.values, axis=1), columns=edge_df.columns, index=edge_df.index)
    edge_df.to_csv('sorted_edge.tsv', sep='\t', index=False)
    edge_df = edge_df.drop_duplicates()
    edge_df['Node1'] = edge_df['Node1'].map(met_node_id)
    edge_df['Node2'] = edge_df['Node2'].map(met_node_id)
    
    edge_index = edge_df[['Node1', 'Node2']].to_numpy().transpose()
    print('edge_index', edge_index.shape)
    
    graph = geom_data.Data()
    graph.x = torch.from_numpy(x).float()
    graph.edge_index = torch.from_numpy(edge_index).long()
    graph.mask = torch.from_numpy(mask)
    graph.sample_names = list(met_df.columns)
    graph.met_names = list(met_df.index)
    graph = ToUndirected()(graph)
    torch.save(graph, 'graph.pt')
    
    nx_graph = to_networkx(graph)
    pickle.dump(nx_graph, open("nx_graph.pkl", "wb"))
    
    feature_dim = x.shape[1]
    hidden_layer_count = [1, 2]
    lr = [1e-1, 1e-3, 1e-5, 1e-7]
    batch_size = [16, 32, 64]
    
    hparams = list(product(hidden_layer_count, lr, batch_size))
    hparam_label = ['hparam-'+str(i) for i in range(len(hparams))]
    
    log_dir = os.getcwd() + '/logs'
    if(os.path.exists(log_dir)):
        rmtree(log_dir)
    Path(log_dir).mkdir(parents=True)
    
    hparam_df = pd.DataFrame(hparams,
                             columns=['hidden_layer_count', 'learning_rate', 'batch_size'],
                             index=hparam_label)
    hparam_df.to_csv(log_dir + '/hyperparameters.tsv', sep='\t', index=True)
    
    for i in range(len(hparam_label)):
        hparam_no = hparam_label[i]
        hparam = hparams[i]
        print(hparam_no, hparam, end='\n')
        
        if(os.path.exists(hparam_no)):
            rmtree(hparam_no)
        Path(hparam_no).mkdir(parents=True)
        os.chdir(hparam_no)
        
        data_loader = geom_data.DataLoader([graph], batch_size=hparam[2], shuffle=False)
        early_stopping = EarlyStopping(monitor='loss',
                                       mode='min',
                                       patience=3,
                                       min_delta=0,
                                       check_finite=True,
                                       stopping_threshold=0)
        checkpoint_callback = ModelCheckpoint(save_top_k=1,
                                              monitor='loss',
                                              mode='min',
                                              auto_insert_metric_name=True)
        hparam_log_dir = log_dir + '/' + hparam_no
        csv_logger = CSVLogger(hparam_log_dir, name=None)
        
        hidden_dim = []
        for h in range(1, hparam[0]+1):
            hidden_dim.append(feature_dim//(2**h))
        imputer = Imputer(feature_dim,
                          hidden_dim,
                          hparam[1])
        hparam_df.loc[hparam_no, 'param_count'] = sum(p.numel() for p in imputer.model.parameters())
        trainer = pl.Trainer(max_epochs=100, # try different values
                             callbacks=[early_stopping,
                                        checkpoint_callback],
                             log_every_n_steps=1,
                             accelerator="cuda",
                             devices=[get_free_gpu()], # pick the gpu with the max free space -> processes will get distributed across gpus
                             enable_checkpointing=True,
                             limit_val_batches=0,
                             logger=[csv_logger],
                             enable_progress_bar=False,
                             detect_anomaly=True,
                             num_sanity_val_steps=0)
        trainer.fit(imputer, train_dataloaders=data_loader)
        metric_path = hparam_log_dir + '/version_0/metrics.csv'
        metric_df = pd.read_csv(metric_path, sep=',')
        hparam_df.loc[hparam_no, 'last_epoch'] = metric_df['epoch'].max()
        metrics = ['loss']
        plot_dir = hparam_log_dir + '/plot'
        Path(plot_dir).mkdir(parents=True, exist_ok=True)
        for metric in metrics:
            print(metric)
            mdf = metric_df.dropna(subset=metric).sort_values(by='epoch', ascending=True)
            x = list(mdf['epoch'])
            y = list(mdf[metric])
            plot_metric(x, y, 'Epoch', metric, metric + ' across epochs', plot_dir + '/' + metric + '.png')
    
        hparam_df.loc[hparam_no, 'loss'] = metric_df['loss'].min(skipna=True)
        
        best_model = Imputer.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
        best_model.eval()

        out = trainer.predict(best_model, data_loader)
        out = torch.concat(out, dim=0)
        out = out.cpu().detach().numpy()
        out = out.transpose()
        out_df = pd.DataFrame(out, index=graph.sample_names, columns=graph.met_names)
        out_df.index.name = 'Sample'
        out_df.to_csv('imputed.tsv', sep='\t', index=True)
        os.chdir('..')
        print('\n')
    hparam_df.to_csv(log_dir + '/hyperparameters.tsv', sep='\t', index=True)
    best_hparam = hparam_df['loss'].idxmin()
    with open(log_dir + '/best_hparam.txt', 'w') as fp:
        fp.write(best_hparam)
        
def main(args):
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    os.chdir(args.out_dir)
    
    log_file = open(args.out_dir + '/dae_impute.log', 'w')
    
    original_stdout = sys.stdout
    sys.stdout = log_file
    
    original_stderr = sys.stderr
    sys.stderr = log_file
    
    print(args)
    
    kwargs = vars(args)
    del kwargs['out_dir']
    dae_impute(**kwargs)

    sys.stdout = original_stdout
    sys.stderr = original_stderr
    
    log_file.flush()
    log_file.close()

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main(parse_args())
