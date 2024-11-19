import argparse, os, sys, dcor

import pandas as pd
import numpy as np

from pathlib import Path
from random import shuffle
from math import ceil
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr, pearsonr

def parse_args():
    parser = argparse.ArgumentParser(description='Preprocesses metabolomic data before imputation: replaces zero with NaNs, standardizes metabolites across samples, detects correlation edges.')
    parser.add_argument("--met_path", type=str,
                        help="metabolomic data in .tsv format, each row is a sample and each column is a metabolite",
                        required=True, default=None)
    parser.add_argument('--corr_type', type=str,
                        nargs='+', choices=['pr', 'sp', 'dcov', 'dcol'],
                        help='Correlation measures to construct the correlation graph',
                        required=True, default=None)
    parser.add_argument("--corr_pct", type=float,
                        help="Correlation percentile cutoff to include an edge between a metabolite pair in the correlation graph",
                        required=True, default=None)
    parser.add_argument('--prior_path', type=str,
                        help='Path to prior edges in .tsv format',
                        required=False, default=None)
    parser.add_argument("--prior_pct", type=float,
                        help="Weight percentile cutoff to include a prior edge between a metabolite pair",
                        required=False, default=None)
    parser.add_argument("--out_dir", type=str,
                        help="path to output dir",
                        required=True, default=None)
    args = parser.parse_args()
    return args

def get_corr_edges(met_df, corr_type, corr_pct):
    # smaller value means greater dependence/correlation
    # takes smallest connections
    
    assert corr_type in ('sp', 'pr', 'dcol', 'dcov')
    
    print('get_corr_edges')
    print('corr_method', corr_type, flush=True)
    
    def dcov_corr(a, b):
        print('dcov_corr', flush=True)
        return dcor.distance_covariance(a, b)
    
    def dcol_corr(a, b):
        print('dcol_corr', flush=True)
        df = pd.DataFrame(zip(a, b), columns=['a', 'b'])
        
        df = df.sort_values(by='a')
        b1 = df.iloc[:-1, :]['b'].to_numpy()
        b2 = df.iloc[1:, :]['b'].to_numpy()
        print('b1', b1.shape, 'b2', b2.shape)
        b_dist = np.subtract(b2, b1)
        b_dist = np.abs(b_dist)
        b_dist = np.mean(b_dist)
        return b_dist
    
    def sp_corr(a, b):
        print('sp_corr', flush=True)
        return -abs(spearmanr(a, b)[0])
    
    def pr_corr(a, b):
        print('pr_corr', flush=True)
        return -abs(pearsonr(a, b)[0])
    
    corr_func = {
        'sp': sp_corr,
        'pr': pr_corr,
        'dcov': dcov_corr,
        'dcol': dcol_corr
    }
    
    pairwise_corr = []
    
    cols = list(met_df.columns)    
    func = corr_func[corr_type]
    for i in range(len(cols)):
        row = []
        for j in range(len(cols)):
            print('i', i, 'j', j, flush=True)
            if(i == j):
                row.append(np.inf)
            else:
                row.append(func(met_df[cols[i]].to_numpy(), met_df[cols[j]].to_numpy()))
        pairwise_corr.append(row)
    corr_df = pd.DataFrame(pairwise_corr, index=cols, columns=cols)
    corr_df.to_csv(corr_type + '_correlation.tsv', sep='\t', index=True)
    
    corr_edge_cnt = ceil((1 - corr_pct) * met_df.shape[1])
    print('corr_top_cnt', corr_edge_cnt, flush=True)
    
    top_cols = ['Top-' + str(i) for i in range(1, corr_edge_cnt+1)]
    top_corr = pd.DataFrame(corr_df.apply(lambda x: x.nsmallest(corr_edge_cnt).index.astype(str).tolist(), axis=1).tolist(), 
                               columns=top_cols, index=corr_df.index)
    top_corr = top_corr.stack()
    top_corr = top_corr.droplevel(axis=0, level=1).reset_index()
    top_corr.columns = ['Node1', 'Node2']
    top_corr = pd.DataFrame(np.sort(top_corr.to_numpy(), axis=1),
                            columns=top_corr.columns,
                            index=top_corr.index)
    # the sorting is done so that duplicate edges like (Node1, Node2) and (Node2, Node1) become identical and later gets removed by drop_duplicates
    top_corr = top_corr.drop_duplicates()
    loop = top_corr[top_corr['Node1'] == top_corr['Node2']]
    assert loop.empty
    return top_corr

def generate_mcar(met_df, nan_pct):
    print('generate_mcar', 'nan_pct', nan_pct)
    mask = np.random.choice([True, False], size=met_df.size, p=[nan_pct, 1 - nan_pct])
    mask = mask.reshape(met_df.shape)
    mcar_df = met_df.copy()
    mcar_df[mask] = pd.NA
    
    out_dir = os.getcwd() + '/MCAR-' + str(int(nan_pct * 100))
    Path(out_dir).mkdir(exist_ok=True, parents=True)
    mcar_df.to_csv(out_dir + '/MCAR-' + str(int(nan_pct * 100)) + '.tsv', sep='\t', index=True)
    
def generate_mnar(met_df, col_pct):
    print('generate_mnar', 'col_pct', col_pct)
    mnar_df = met_df.copy()
    cols = list(met_df.columns)
    print('cols', cols)
    shuffle(cols)
    print('cols', cols)
    mnar_col_cnt = int(ceil(len(cols) * col_pct))
    for i in range(mnar_col_cnt):
        col = cols[i]
        print('col', col)
        quantile = np.random.uniform(0.3, 0.6)
        print('quantile', quantile)
        
        cutoff = np.quantile(mnar_df[col], quantile)
        print('cutoff', cutoff)
        mnar_df[col][mnar_df[col] < cutoff] = pd.NA

    out_dir = os.getcwd() + '/MNAR-' + str(int(col_pct * 100))
    Path(out_dir).mkdir(exist_ok=True, parents=True)
    mnar_df.to_csv(out_dir + '/MNAR-' + str(int(col_pct * 100)) + '.tsv', sep='\t', index=True)

def preprocess(met_path, corr_type, corr_pct, prior_path, prior_pct):
    met_df = pd.read_csv(met_path, sep='\t', index_col='Sample')
    print('met_df', met_df.shape)
    met_df = met_df.replace(0, pd.NA)
    met_df = met_df.dropna(axis=1, how='any')
    print('met_df', met_df.shape)
    standard_scaler = StandardScaler()
    met_df = pd.DataFrame(standard_scaler.fit_transform(met_df), columns=met_df.columns, index=met_df.index)
    met_df.to_csv('metabolome.tsv', sep='\t', index=True)
    corr_dfs = []
    for c in range(len(corr_type)):
        corr_df = get_corr_edges(met_df, corr_type[c], corr_pct)
        corr_df['Type'] = corr_type[c]
        corr_dfs.append(corr_df)
    corr_df = pd.concat(corr_dfs, axis=0)
    corr_df.to_csv('corr_edges.tsv', sep='\t', index=False)

    for pct in range(10, 51, 10):
        generate_mcar(met_df, pct/100)
        generate_mnar(met_df, pct/100)
    
    
def main(args):
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    os.chdir(args.out_dir)
    
    log_file = open(args.out_dir + '/preprocess.log', 'w')
    
    original_stdout = sys.stdout
    sys.stdout = log_file
    
    original_stderr = sys.stderr
    sys.stderr = log_file
    
    print(args)
    
    kwargs = vars(args)
    del kwargs['out_dir']
    preprocess(**kwargs)

    sys.stdout = original_stdout
    sys.stderr = original_stderr
    
    log_file.flush()
    log_file.close()

if __name__ == "__main__":
    main(parse_args())
