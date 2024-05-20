import argparse
from pathlib import Path
import os
import pandas as pd
from torch_geometric.utils.convert import from_networkx
import networkx as nx
from itertools import product
import pickle
import sys

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--in_path", type=str,
                        help="path to the input metabolome in .tsv format",
                        required=True, default=None)
    parser.add_argument("-g", "--gem_path", type=str,
                        help="path to the Human-GEM in .xlsx format",
                        required=True, default=None)
    parser.add_argument("-m", "--met_path", type=str,
                        help="path to the metabolite identifiers of Human-GEM in .tsv format",required=True, default=None)
    parser.add_argument("-o", "--out_dir", type=str,
                        help="path to the output directory",
                        required=True, default=None)
    args = parser.parse_args()
    print(args)
    return args

def extract_metabolites(eqn):
    eqn = eqn.replace("=>", "+").replace("<=>", "+")
    old = eqn.split(' + ')
    met = set()
    flag = False
    for i_old in old:
        i_old_split = i_old.split(' ')
        if(len(i_old_split)>1):
            if(i_old_split[0].replace(".", "").isnumeric()): # If a metabolite has a coefficient in reaction equation
                i_new = i_old[len(i_old_split[0])+1:]
                met.add(i_new)
                flag = True
            else:
                met.add(i_old)
        else:
            met.add(i_old)
    return met

def get_hmdb_set(met_set, met_to_hmdb):
    hmdb_set = set()
    for met in met_set:
        if(met in met_to_hmdb):
            hmdb_set.add(met_to_hmdb[met])
    return hmdb_set

def construct_network(in_path, gem_path, met_path):
    # nodes -> hmdb # undirected edges -> two hmdb in same reaction
    
    G = nx.Graph()
    
    data_mets = pd.read_csv(in_path, sep='\t', index_col='Sample')
    data_hmdb = set(data_mets.columns)
    print('data_hmdb', len(data_hmdb))
    G.add_nodes_from(data_hmdb)
    #print(data_hmdb)
    
    prior_mets_1 = pd.read_csv(met_path, sep='\t', usecols=['mets', 'metHMDBID'])
    prior_mets_1 = prior_mets_1[~prior_mets_1.metHMDBID.isna()]
    
    prior_mets_2 = pd.read_excel(gem_path, sheet_name='METS', usecols=['ID', 'REPLACEMENT ID'])
    prior_mets = pd.merge(prior_mets_1, prior_mets_2, how='inner', left_on='mets', right_on='REPLACEMENT ID')
    #print(prior_mets.head())
    met_to_hmdb = dict(zip(prior_mets.ID, prior_mets.metHMDBID))
    #print(met_to_hmdb)
    common_hmdb = set(data_mets.columns).intersection(set(prior_mets['metHMDBID']))
    print('common_hmdb', len(common_hmdb))
    #print(common_hmdb)
    
    gem_df = pd.read_excel(gem_path, sheet_name='RXNS', usecols=['EQUATION', 'SUBSYSTEM'])
    #pattern = '|'.join(['\[e\]', '\[x\]', '\[m\]', '\[c\]', '\[l\]', '\[r\]', '\[g\]', '\[n\]', '\[i\]'])
    #gem_df['EQUATION'] = gem_df['EQUATION'].str.replace(pattern, '')
    gem_df['Met_Set'] = gem_df['EQUATION'].apply(lambda eqn: extract_metabolites(eqn))
    gem_df['HMDB_Set'] = gem_df['Met_Set'].apply(lambda met_set: get_hmdb_set(met_set, met_to_hmdb))
    gem_df['Common_HMDB'] = gem_df['HMDB_Set'].apply(lambda hmdb_set: data_hmdb.intersection(hmdb_set))
    gem_df.to_csv('Processed-Human-GEM.tsv', sep='\t', index=False)
    
    gem_gb = gem_df.groupby('SUBSYSTEM').agg({'Common_HMDB': lambda x: set.union(*x)})
    gem_gb = gem_gb.reset_index()
    gem_gb.to_csv('Subsystem-HMDB.tsv', sep='\t', index=False)
    for _, row in gem_df.iterrows():
        if(len(row['Common_HMDB']) > 2):
            edges = list(product(list(row['Common_HMDB']), repeat=2))
            print(row['EQUATION'], row['Common_HMDB'], edges)
            G.add_edges_from(edges)
    G.remove_edges_from(list(nx.selfloop_edges(G)))
    print('nodes', G.number_of_nodes(), 'isolates', nx.number_of_isolates(G))
    pickle.dump(G, open('reaction-coparticipation-undirected-network.pkl', 'wb'))
    
    # nodes -> hmdb # undirected edges -> two hmdb in same subsystem
        
    G = nx.Graph()
    G.add_nodes_from(data_hmdb)
    for _, row in gem_gb.iterrows():
        if(len(row['Common_HMDB']) > 2):
            edges = list(product(list(row['Common_HMDB']), repeat=2))
            print(row['Common_HMDB'], edges)
            G.add_edges_from(edges)
    G.remove_edges_from(list(nx.selfloop_edges(G)))
    print('nodes', G.number_of_nodes(), 'isolates', nx.number_of_isolates(G))
    pickle.dump(G, open('subsystem-coparticipation-undirected-network.pkl', 'wb'))
            
def main(args):
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    os.chdir(args.out_dir)
    
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    log_file = open('reaction-coparticipation-log.txt', 'w')
    sys.stdout = log_file
    sys.stderr = log_file
    
    construct_network(args.in_path, args.gem_path, args.met_path)

    sys.stdout = orig_stdout 
    sys.stderr = orig_stderr 
    log_file.close()
    
if __name__ == "__main__":
    main(parse_args())

