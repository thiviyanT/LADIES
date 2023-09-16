from ogb.nodeproppred import NodePropPredDataset
import dgl
import numpy as np


def load_ogb(name):
    # Load the dataset and split
    dataset = NodePropPredDataset(name=name)
    split_idx = dataset.get_idx_split()
    graph, label = dataset[0]

    # Convert DGL graph to NetworkX graph for processing
    G = dgl.to_networkx(graph)
    features = graph.ndata['feat'].numpy()
    labels = label.numpy().flatten()

    # Extract edges, degrees, and train/val/test split
    edges = np.array(graph.edges())
    degrees = np.array(G.degree())[:, 1]
    idx_train, idx_val, idx_test = split_idx["train"].numpy(), split_idx["valid"].numpy(), split_idx["test"].numpy()

    return edges, labels, features, np.max(labels) + 1, idx_train, idx_val, idx_test


edges, labels, features, num_labels, idx_train, idx_val, idx_test = load_ogb('ogbn-arxiv')
