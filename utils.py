import numpy as np
import scipy.sparse as sp
import pickle as pkl
import torch
from sklearn.preprocessing import StandardScaler
from networkx.readwrite import json_graph
import json
from collections import defaultdict
import networkx as nx
import sys
import os
from torch_geometric.datasets import Planetoid, Yelp, Flickr, Reddit2
from torch_geometric.utils import to_undirected
from ogb.nodeproppred import PygNodePropPredDataset, NodePropPredDataset
from typing import Tuple
from linkx.dataset import load_snap_patents_mat
import scipy.io
from torch.utils.data import random_split
from torch_geometric.data import Batch, Data
from torch import Tensor 

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def load_torch_geom_data(dataset_name, root_dir):
    """load datasets from pyg """
    if dataset_name.lower() == 'cora':
        data = Planetoid(root=f'{root_dir}/Planetoid', name='Cora', split='full')[0]
    elif dataset_name.lower() == 'citeseer':
        data = Planetoid(root=f'{root_dir}/Planetoid', name='CiteSeer', split='full')[0]
    elif dataset_name.lower() == 'pubmed':
        data = Planetoid(root=f'{root_dir}/Planetoid', name='PubMed', split='full')[0]
    elif dataset_name.lower() == 'reddit':
        data = Reddit2(root=f'{root_dir}/Reddit2')[0]
    elif dataset_name.lower() == 'yelp':
        data = Yelp(root=f'{root_dir}/YELP')[0]
    elif dataset_name.lower() == 'flickr':
        data = Flickr(root=f'{root_dir}/Flickr')[0]
    else:
	    raise ValueError("Unknown dataset name. Supported datasets: reddit, yelp, flickr")

    # Convert to undirected graph
    edges = to_undirected(data.edge_index)
    edges = edges.numpy().T  # convert to numpy in format (edge_start, edge_end)

    # Features and Labels
    features = data.x.numpy()  # Convert to numpy
    labels = data.y.numpy()

    idx_train = data.train_mask.nonzero(as_tuple=False).squeeze().numpy()
    idx_val = data.val_mask.nonzero(as_tuple=False).squeeze().numpy()
    idx_test = data.test_mask.nonzero(as_tuple=False).squeeze().numpy()

    if dataset_name.lower() in ['reddit', 'yelp']:
        features = (features - features.mean(axis=0)) / features.std(axis=0)

    # Num classes
    if labels.ndim == 1:
        num_classes = int(np.max(labels) + 1)
    else:
        num_classes = labels.shape[1]

    return np.array(edges), labels, features, num_classes, idx_train, idx_val, idx_test


def load_data(dataset_str, data_dir=None):

    if dataset_str.lower() in ['cora', 'citeseer', 'pubmed', 'reddit', 'yelp', 'flickr']:
        assert data_dir is not None
        edges, labels, features, num_classes, idx_train, idx_val, idx_test = load_torch_geom_data(dataset_str,
                                                                                                  root_dir=f'{data_dir}')

        return edges, labels, features, num_classes, idx_train, idx_val, idx_test
    elif dataset_str.lower() in ['ogbn-arxiv', 'arxiv']:
        # Load the arxiv dataset
        dataset = NodePropPredDataset(name="ogbn-arxiv", root=f'{data_dir}/OGB')

        split_idx = dataset.get_idx_split()
        idx_train, idx_val, idx_test = split_idx["train"], split_idx["valid"], split_idx["test"]

        # Edge info
        edge_index = dataset.graph['edge_index'].T
        edges = edge_index

        # Features and labels
        features = dataset.graph['node_feat']
        labels = dataset.labels.flatten()

        # Standardize the features
        # train_feats = features[idx_train]
        # scaler = StandardScaler()
        # scaler.fit(train_feats)
        # features = scaler.transform(features)

        # Num classes
        num_classes = dataset.num_classes

        return edges, labels, features, num_classes, idx_train, idx_val, idx_test

    elif dataset_str.lower() in ['ogbn-products', 'products']:
        # Load the products dataset
        dataset = NodePropPredDataset(name="ogbn-products", root=f'{data_dir}/OGB')
        split_idx = dataset.get_idx_split()
        idx_train, idx_val, idx_test = split_idx["train"], split_idx["valid"], split_idx["test"]

        # Edge info
        edge_index = dataset.graph['edge_index'].T
        edges = edge_index

        # Features and labels
        features = dataset.graph['node_feat']
        labels = dataset.labels.flatten()

        # Standardize the features
        # train_feats = features[idx_train]
        # scaler = StandardScaler()
        # scaler.fit(train_feats)
        # features = scaler.transform(features)

        # Num classes
        num_classes = dataset.num_classes

        return edges, labels, features, num_classes, idx_train, idx_val, idx_test
    elif dataset_str.lower() == 'ppi':
        prefix = './ppi/ppi'
        G_data = json.load(open(prefix + "-G.json"))
        G = json_graph.node_link_graph(G_data)
        if isinstance(G.nodes()[0], int):
            conversion = lambda n : int(n)
        else:
            conversion = lambda n : n

        if os.path.exists(prefix + "-feats.npy"):
            feats = np.load(prefix + "-feats.npy")
        else:
            print("No features present.. Only identity features will be used.")
            feats = None
        id_map = json.load(open(prefix + "-id_map.json"))
        id_map = {conversion(k):int(v) for k,v in id_map.items()}
        walks = []
        class_map = json.load(open(prefix + "-class_map.json"))
        if isinstance(list(class_map.values())[0], list):
            lab_conversion = lambda n : n
        else:
            lab_conversion = lambda n : int(n)

        class_map = {conversion(k):lab_conversion(v) for k,v in class_map.items()}

        ## Remove all nodes that do not have val/test annotations
        ## (necessary because of networkx weirdness with the Reddit data)
        broken_count = 0
        for node in G.nodes():
            if not 'val' in G.node[node] or not 'test' in G.node[node]:
                G.remove_node(node)
                broken_count += 1
        print("Removed {:d} nodes that lacked proper annotations due to networkx versioning issues".format(broken_count))

        ## Make sure the graph has edge train_removed annotations
        ## (some datasets might already have this..)
        print("Loaded data.. now preprocessing..")
        for edge in G.edges():
            if (G.node[edge[0]]['val'] or G.node[edge[1]]['val'] or
                G.node[edge[0]]['test'] or G.node[edge[1]]['test']):
                G[edge[0]][edge[1]]['train_removed'] = True
            else:
                G[edge[0]][edge[1]]['train_removed'] = False

        train_ids = np.array([id_map[str(n)] for n in G.nodes() if not G.node[n]['val'] and not G.node[n]['test']])
        train_feats = feats[train_ids]
        scaler = StandardScaler()
        scaler.fit(train_feats)
        features = scaler.transform(feats)
        
        degrees = np.zeros(len(G), dtype=np.int64)
        edges = []
        labels = []
        idx_train = []
        idx_val   = []
        idx_test  = []
        for s in G:
            if G.nodes[s]['test']:
                idx_test += [s]
            elif G.nodes[s]['val']:
                idx_val += [s]
            else:
                idx_train += [s]
            for t in G[s]:
                edges += [[s, t]]
            degrees[s] = len(G[s])
            labels += [class_map[str(s)]]
        
        return np.array(edges), np.array(degrees), np.array(labels), np.array(features),\
                np.array(idx_train), np.array(idx_val), np.array(idx_test)
    elif dataset_str.lower() == 'ogbn-proteins':
        return get_proteins(data_dir)
    elif dataset_str.lower() == 'snap-patents':
        return get_linkx_dataset(data_dir, 'snap-patents', 0)
    elif dataset_str.lower() in ['blogcat']:
        return get_blogcat(data_dir, dataset_str, 0)
    elif dataset_str.lower() == 'dblp':
        return get_dblp(data_dir, dataset_str, 42)


    # elif dataset_str.lower() in ['cora', 'pubmed', 'citeseer']:
    #     names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    #     objects = []
    #     for i in range(len(names)):
    #         with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
    #             if sys.version_info > (3, 0):
    #                 objects.append(pkl.load(f, encoding='latin1'))
    #             else:
    #                 objects.append(pkl.load(f))
    #
    #     x, y, tx, ty, allx, ally, graph = tuple(objects)
    #     test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    #     test_idx_range = np.sort(test_idx_reorder)
    #
    #     if dataset_str == 'citeseer':
    #         # Fix citeseer dataset (there are some isolated nodes in the graph)
    #         # Find isolated nodes, add them as zero-vecs into the right position
    #         test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
    #         tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
    #         tx_extended[test_idx_range-min(test_idx_range), :] = tx
    #         tx = tx_extended
    #         ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
    #         ty_extended[test_idx_range-min(test_idx_range), :] = ty
    #         ty = ty_extended
    #
    #     features = sp.vstack((allx, tx)).tolil()
    #     features[test_idx_reorder, :] = features[test_idx_range, :]
    #
    #     labels = np.vstack((ally, ty))
    #     labels[test_idx_reorder, :] = labels[test_idx_range, :]
    #
    #     idx_test = np.array(test_idx_range.tolist())
    #     idx_train = np.array(range(len(y)))
    #     idx_val = np.array(range(len(y), len(y)+500))
    #
    #     degrees = np.zeros(len(labels), dtype=np.int64)
    #     edges = []
    #     for s in graph:
    #         for t in graph[s]:
    #             edges += [[s, t]]
    #         degrees[s] = len(graph[s])
    #     labels = np.argmax(labels, axis=1)
    #     return np.array(edges), labels, features, np.max(labels)+1,  idx_train, idx_val, idx_test
    else:
        raise NotImplementedError()


def get_proteins(root: str, node_emb_dim: int = 128):
    dataset = PygNodePropPredDataset('ogbn-proteins', root)
    data = dataset[0]

    split_idx = dataset.get_idx_split()
    
    idx_train, idx_val, idx_test = split_idx["train"], split_idx["valid"], split_idx["test"]

    # Edge info
    edge_index = dataset.edge_index
    edges = edge_index

    # Features and labels
    features = torch.empty(data.num_nodes, node_emb_dim)
    #nn.init.normal_(features)
    #features = nn.Parameter(features, requires_grad=True)
    labels = dataset.y.numpy()

        # Standardize the features
        # train_feats = features[idx_train]
        # scaler = StandardScaler()
        # scaler.fit(train_feats)
        # features = scaler.transform(features)

        # Num classes
    num_classes = labels.shape[1]
    #import pdb;pdb.set_trace()
    return edges, labels, features, num_classes, idx_train, idx_val, idx_test
    # This is a multi-label binary classification dataset, so we need
    # float targets for BCEWithLogitsLoss
    #data.y = data.y.float()

    #return data, dataset.num_features, data.y.shape[1]


def get_linkx_dataset(root: str, name: str, seed: int = None):
    if name.lower() == 'snap-patents':
        dataset = load_snap_patents_mat(root)
        split_idx = dataset.get_idx_split(seed=seed)
        num_nodes = dataset.graph['num_nodes']
        train_mask = index2mask(split_idx['train'], num_nodes)
        valid_mask = index2mask(split_idx['valid'], num_nodes)
        test_mask = index2mask(split_idx['test'], num_nodes)

        edge_index = dataset.graph['edge_index']
        edge_index = to_undirected(edge_index, num_nodes=num_nodes)

        data = Data(x=dataset.graph['node_feat'],
                    edge_index=edge_index,
                    y=dataset.label,
                    train_mask=train_mask,
                    val_mask=valid_mask,
                    test_mask=test_mask)
        num_classes = len(data.y.unique())
    else:
        raise ValueError(f'Unknown dataset name: {name}')
    return edge_index, dataset.label, dataset.graph['node_feat'], len(data.y.unique()), split_idx['train'], split_idx['valid'], split_idx['test']


def get_blogcat(root: str, name: str, split_id: int=0) -> Tuple[Data, int, int]:
    dataset = torch.load(f'{root}/blogcat/blogcatalog_0.6/split_{split_id}.pt')
    graph = scipy.io.loadmat(f'{root}/blogcat/blogcatalog_0.6/blogcatalog.mat')
    edges = graph['network'].nonzero()
    edge_index = torch.tensor(np.vstack((edges[0], edges[1])), dtype=torch.long)
    train_mask = torch.zeros(10312, dtype=torch.bool)
    test_mask = torch.zeros(10312, dtype=torch.bool)
    val_mask = torch.zeros(10312, dtype=torch.bool)
    train_mask[dataset['train_mask']] = True
    test_mask[dataset['test_mask']] = True
    val_mask[dataset['val_mask']] = True
 
    data = Data(y=torch.tensor(graph['group'].todense()), edge_index=edge_index,
                train_mask=train_mask, test_mask=test_mask, val_mask=val_mask,
                num_classes=39, num_nodes=10312)
    data.node_stores[0].x = torch.empty(data.num_nodes, 64)
    return edge_index, graph['group'].todense(), data.x, data.num_classes, dataset['train_mask'], dataset['val_mask'], dataset['test_mask']


def get_dblp(root: str, name: str, seed: int = None) -> Tuple[Data, int, int]:
    features = []
    with open(f'{root}/dblp/features.txt', 'r') as file:
        content = file.read()
        values = content.strip().split('\n')
        for value in values:
            feature = value.split(',')
            number = [float(i) for i in feature]
            features.append(number)
    edges = []
    with open(f'{root}/dblp/dblp.edgelist', 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split()
            source_node = int(parts[0])
            target_node = int(parts[1])
            edges.append((source_node, target_node))
            edges.append((source_node, target_node))

    edge_index = torch.tensor(list(zip(*edges)), dtype=torch.long)
    labels = []
    with open(f'{root}/dblp/labels.txt', 'r') as file:
        content = file.read()
        values = content.strip().split('\n')
        for value in values:
            label = value.split(',')
            number = [1. if float(x)==1 else 0. for x in label]
            labels.append(number)
    torch.manual_seed(seed)
    num_nodes = 28702
    train_ratio = 0.8
    test_ratio = 0.1
    num_train = int(train_ratio * num_nodes)
    num_test = int(test_ratio * num_nodes)
    num_val = num_nodes - num_train - num_test

    train_idx, test_idx, val_idx = random_split(list(range(num_nodes)), [num_train, num_test, num_val])
    train_mask = torch.zeros(28702, dtype=torch.bool)
    test_mask = torch.zeros(28702, dtype=torch.bool)
    val_mask = torch.zeros(28702, dtype=torch.bool)
    train_mask[train_idx.indices] = True
    test_mask[test_idx.indices] = True
    val_mask[val_idx.indices] = True

    data = Data(edge_index=edge_index, x=torch.tensor(features),y=torch.tensor(labels), train_mask=train_mask,
                test_mask=test_mask, val_mask=val_mask, num_features=300)
    data.node_stores[0].num_classes = 4
    return edge_index, np.array(labels), np.array(features), 4, np.array(train_idx.indices), np.array(val_idx.indices), np.array(test_idx.indices)


def load_cora():
    num_nodes = 2708
    num_feats = 1433
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes,1), dtype=np.int64)
    node_map = {}
    label_map = {}
    with open("cora/cora.content") as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            feat_data[i,:] = info[1:-1]
            node_map[info[0]] = i
            if not info[-1] in label_map:
                label_map[info[-1]] = len(label_map)
            labels[i] = label_map[info[-1]]
    degrees = np.zeros(num_nodes, dtype=np.int64)
    adj_lists = []
    with open("cora/cora.cites") as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            paper1 = node_map[info[0]]
            paper2 = node_map[info[1]]
            adj_lists += [[paper1, paper2]]
            adj_lists += [[paper2, paper1]]
            degrees[paper1] += 1
            degrees[paper2] += 1
    adj_lists = np.array(adj_lists)
    return feat_data, labels, adj_lists, np.array(degrees)

def sym_normalize(mx):
    """Sym-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1/2).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    
    colsum = np.array(mx.sum(0))
    c_inv = np.power(colsum, -1/2).flatten()
    c_inv[np.isinf(c_inv)] = 0.
    c_mat_inv = sp.diags(c_inv)
    
    mx = r_mat_inv.dot(mx).dot(c_mat_inv)
    return mx

def row_normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)

    mx = r_mat_inv.dot(mx)
    return mx


def generate_random_graph(n, e, prob = 0.1):
    idx = np.random.randint(2)
    g = nx.powerlaw_cluster_graph(n, e, prob) 
    adj_lists = defaultdict(set)
    num_feats = 8
    degrees = np.zeros(len(g), dtype=np.int64)
    edges = []
    for s in g:
        for t in g[s]:
            edges += [[s, t]]
            degrees[s] += 1
            degrees[t] += 1
    edges = np.array(edges)
    return degrees, edges, g, None 

def get_sparse(edges, num_nodes):
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                    shape=(num_nodes, num_nodes), dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    return sparse_mx_to_torch_sparse_tensor(adj) 

def norm(l):
    return (l - np.average(l)) / np.std(l)

def stat(l):
    return np.average(l), np.sqrt(np.var(l))

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    if len(sparse_mx.row) == 0 and len(sparse_mx.col) == 0:
        indices = torch.LongTensor([[], []])
    else:
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return indices, values, shape


def get_adj(edges, num_nodes):
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                    shape=(num_nodes, num_nodes), dtype=np.float32)
    return adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
def get_laplacian(adj):
    adj = row_normalize(adj + sp.eye(adj.shape[0]))
    return sparse_mx_to_torch_sparse_tensor(adj) 

def index2mask(idx: Tensor, size: int) -> Tensor:
    mask = torch.zeros(size, dtype=torch.bool, device=idx.device)
    mask[idx] = True
    return mask


def gen_masks(y: Tensor, train_per_class: int = 20, val_per_class: int = 30,
              num_splits: int = 20) -> Tuple[Tensor, Tensor, Tensor]:
    num_classes = int(y.max()) + 1

    train_mask = torch.zeros(y.size(0), num_splits, dtype=torch.bool)
    val_mask = torch.zeros(y.size(0), num_splits, dtype=torch.bool)

    for c in range(num_classes):
        idx = (y == c).nonzero(as_tuple=False).view(-1)
        perm = torch.stack(
            [torch.randperm(idx.size(0)) for _ in range(num_splits)], dim=1)
        idx = idx[perm]

        train_idx = idx[:train_per_class]
        train_mask.scatter_(0, train_idx, True)
        val_idx = idx[train_per_class:train_per_class + val_per_class]
        val_mask.scatter_(0, val_idx, True)

    test_mask = ~(train_mask | val_mask)

    return train_mask, val_mask, test_mask
