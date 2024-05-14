#!/usr/bin/env python
# coding: utf-8
import pdb
from utils import *
import argparse
import scipy
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score
import time
import math

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)


parser = argparse.ArgumentParser(description='Training GCN on Cora/CiteSeer/PubMed/Reddit Datasets')

'''
    Dataset arguments
'''
parser.add_argument('--data_dir', type=str, default='.',
                    help='Dataset directory')
parser.add_argument('--dataset', type=str, default='reddit',
                    help='Dataset name: Cora/CiteSeer/PubMed/Reddit/Yelp/Flickr/Arxiv/Products')
parser.add_argument('--nhid', type=int, default=256,
                    help='Hidden state dimension')
parser.add_argument('--epoch_num', type=int, default= 100,
                    help='Number of Epoch')
parser.add_argument('--batch_size', type=int, default=256,
                    help='size of output node in a batch')
parser.add_argument('--n_layers', type=int, default=2,
                    help='Number of GCN layers')
parser.add_argument('--n_iters', type=int, default=1,
                    help='Number of iteration to run on a batch')
parser.add_argument('--n_stops', type=int, default=200,
                    help='Stop after number of batches that f1 dont increase')
parser.add_argument('--samp_num', type=int, default=256,
                    help='Number of sampled nodes per layer')
parser.add_argument('--sample_method', type=str, default='ladies',
                    help='Sampled Algorithms: ladies/fastgcn/full')
parser.add_argument('--cuda', type=int, default=0,
                    help='Avaiable GPU ID')
parser.add_argument('--runs', type=int, default=1,
                    help='Number of runs')
parser.add_argument('--dropout', type=float, default=0.0,
                    help='Dropout rate')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Learning rate')
parser.add_argument('--embed_nodes', type=bool, default=False,
                    help='training the node embeddings')
parser.add_argument('--node_emb_dim', type=float, default=64,
                    help='node embeddings dimension')

args = parser.parse_args()


class GraphConvolution(nn.Module):
    def __init__(self, n_in, n_out, bias=True):
        super(GraphConvolution, self).__init__()
        self.n_in  = n_in
        self.n_out = n_out
        self.linear = nn.Linear(n_in,  n_out)
    def forward(self, x, adj):
        out = self.linear(x)
        return F.relu(torch.spmm(adj, out))


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclasses, layers, dropout):
        super(GCN, self).__init__()
        self.layers = layers
        self.nhid = nhid
        self.gcs = nn.ModuleList()
        self.gcs.append(GraphConvolution(nfeat,  nhid))
        self.dropout = nn.Dropout(dropout)
        for i in range(layers-1):
            self.gcs.append(GraphConvolution(nhid,  nclasses))

    def forward(self, x, adjs):
        '''
            The difference here with the original GCN implementation is that
            we will receive different adjacency matrix for different layer.
        '''
        for idx in range(len(self.gcs)):
            x = self.dropout(self.gcs[idx](x, adjs[idx]))
        return x


class SuGCN(nn.Module):
    def __init__(self, encoder, num_classes, dropout, inp):
        super(SuGCN, self).__init__()
        self.encoder = encoder
        # self.dropout = nn.Dropout(dropout)
        # self.linear = nn.Linear(self.encoder.nhid, num_classes)

    def forward(self, feat, adjs):
        x = self.encoder(feat, adjs)
        # x = self.dropout(x)
        # x = self.linear(x)
        return x


def fastgcn_sampler(seed, batch_nodes, samp_num_list, num_nodes, lap_matrix, depth):
    '''
        FastGCN_Sampler: Sample a fixed number of nodes per layer. The sampling probability (importance)
                         is pre-computed based on the global degree (lap_matrix)
    '''
    np.random.seed(seed)
    previous_nodes = batch_nodes
    adjs  = []
    #     pre-compute the sampling probability (importance) based on the global degree (lap_matrix)
    pi = np.array(np.sum(lap_matrix.multiply(lap_matrix), axis=0))[0]
    p = pi / np.sum(pi)
    '''
        Sample nodes from top to bottom, based on the pre-computed probability. Then reconstruct the adjacency matrix.
    '''
    for d in range(depth):
        #     row-select the lap_matrix (U) by previously sampled nodes
        U = lap_matrix[previous_nodes , :]
        #     sample the next layer's nodes based on the pre-computed probability (p).
        s_num = np.min([np.sum(p > 0), samp_num_list[d]])
        after_nodes = np.random.choice(num_nodes, s_num, p = p, replace = False)
        #     col-select the lap_matrix (U), and then devided by the sampled probability for 
        #     unbiased-sampling. Finally, conduct row-normalization to avoid value explosion.         
        adj = row_normalize(U[: , after_nodes].multiply(1/p[after_nodes]))
        #     Turn the sampled adjacency matrix into a sparse matrix. If implemented by PyG
        #     This sparse matrix can also provide index and value.
        adjs += [sparse_mx_to_torch_sparse_tensor(row_normalize(adj))]
        #     Turn the sampled nodes as previous_nodes, recursively conduct sampling.
        previous_nodes = after_nodes
    #     Reverse the sampled probability from bottom to top. Only require input how the lastly sampled nodes.
    adjs.reverse()
    return adjs, previous_nodes, batch_nodes

def ladies_sampler(seed, batch_nodes, samp_num_list, num_nodes, lap_matrix, depth):
    '''
        LADIES_Sampler: Sample a fixed number of nodes per layer. The sampling probability (importance)
                         is computed adaptively according to the nodes sampled in the upper layer.
    '''
    np.random.seed(seed)
    previous_nodes = batch_nodes
    adjs  = []
    #pdb.set_trace()
    '''
        Sample nodes from top to bottom, based on the probability computed adaptively (layer-dependent).
    '''
    for d in range(depth):
        #     row-select the lap_matrix (U) by previously sampled nodes
        U = lap_matrix[previous_nodes , :]
        #     Only use the upper layer's neighborhood to calculate the probability.
        pi = np.array(np.sum(U.multiply(U), axis=0))[0]
        p = pi / np.sum(pi)
        s_num = np.min([np.sum(p > 0), samp_num_list[d]])
        #     sample the next layer's nodes based on the adaptively probability (p).
        after_nodes = np.random.choice(num_nodes, s_num, p = p, replace = False)
        #     Add output nodes for self-loop
        after_nodes = np.unique(np.concatenate((after_nodes, batch_nodes)))
        #     col-select the lap_matrix (U), and then devided by the sampled probability for 
        #     unbiased-sampling. Finally, conduct row-normalization to avoid value explosion.      
        adj = U[: , after_nodes].multiply(1/p[after_nodes])
        adjs += [sparse_mx_to_torch_sparse_tensor(row_normalize(adj))]
        #     Turn the sampled nodes as previous_nodes, recursively conduct sampling.
        previous_nodes = after_nodes
    #     Reverse the sampled probability from bottom to top. Only require input how the lastly sampled nodes.
    adjs.reverse()
    return adjs, previous_nodes, batch_nodes


def default_sampler(seed, batch_nodes, samp_num_list, num_nodes, lap_matrix, depth):
    mx = sparse_mx_to_torch_sparse_tensor(lap_matrix)
    return [mx for i in range(depth)], np.arange(num_nodes), batch_nodes


def prepare_data_unparallel(sampler, train_nodes, samp_num_list, num_nodes, lap_matrix, depth):
    adjs, previous_nodes, batch_nodes = sampler(np.random.randint(2**32 - 1), train_nodes, samp_num_list, num_nodes, lap_matrix, depth)
    return adjs, previous_nodes, batch_nodes


def package_mxl(mxl, device):
    return [torch.sparse.FloatTensor(mx[0], mx[1], mx[2]).to(device) for mx in mxl]


if args.cuda != -1 and torch.cuda.is_available():
    device = torch.device("cuda:" + str(args.cuda))
else:
    device = torch.device("cpu")
    
    
print(args.dataset, args.sample_method)
edges, labels, feat_data, num_classes, train_nodes, valid_nodes, test_nodes = load_data(args.dataset, data_dir=args.data_dir)

adj_matrix = get_adj(edges, feat_data.shape[0])

lap_matrix = row_normalize(adj_matrix + sp.eye(adj_matrix.shape[0]))

embedding_params = []
if args.embed_nodes:
    print(f'Using learned node embeddings for features')
    embeddings = torch.FloatTensor(data.num_nodes, args.node_emb_dim)
    nn.init.normal_(embeddings)
    embeddings = nn.Parameter(embeddings, requires_grad=True)
    feat_data = embeddings
    embedding_params.append(embeddings)

if type(feat_data) == scipy.sparse.lil.lil_matrix:
    feat_data = torch.FloatTensor(feat_data.todense()).to(device) 
else:
    feat_data = torch.FloatTensor(feat_data).to(device)
labels = torch.LongTensor(labels).to(device)

if labels.dim() == 1:
    loss_fn = nn.CrossEntropyLoss()
else:
    loss_fn = nn.BCEWithLogitsLoss()


if args.sample_method == 'ladies':
    sampler = ladies_sampler
elif args.sample_method == 'fastgcn':
    sampler = fastgcn_sampler
elif args.sample_method == 'full':
    sampler = default_sampler

samp_num_list = np.array([args.samp_num, args.samp_num, args.samp_num, args.samp_num, args.samp_num])


results = torch.empty(args.runs)
for oiter in range(args.runs):
    encoder = GCN(nfeat = feat_data.shape[1], nhid=args.nhid, nclasses=num_classes,layers=args.n_layers, dropout = args.dropout).to(device)
    susage  = SuGCN(encoder = encoder, num_classes=num_classes, dropout= args.dropout, inp = feat_data.shape[1])
    susage.to(device)
    
    optimizer = optim.Adam(filter(lambda p : p.requires_grad, list(susage.parameters()) + embedding_params), lr=args.lr)
    best_val = 0
    best_tst = -1
    cnt = 0
    times = []
    res   = []
    print('-' * 10)

    num_batches = len(train_nodes) // args.batch_size if len(train_nodes) // args.batch_size != 0 else 1

    for epoch in np.arange(args.epoch_num):
        susage.train()
        train_losses = []

        for i in range(num_batches):
            # Get the current mini-batch
            start_idx = i * args.batch_size
            end_idx = start_idx + args.batch_size
            train_batch_nodes = train_nodes[start_idx:end_idx]
            #print('batch_number:', i) #pdb.set_trace()
            adjs, input_nodes, output_nodes = prepare_data_unparallel(sampler,train_batch_nodes,samp_num_list,len(feat_data),lap_matrix,args.n_layers)

            # Training
            for _iter in range(args.n_iters):
                adjs = package_mxl(adjs, device)
                optimizer.zero_grad()
                t1 = time.time()
                susage.train()
                output = susage.forward(feat_data[input_nodes], adjs)
                if args.sample_method == 'full':
                    output = output[output_nodes]
                if labels.dim() != 1:
                    labels = labels.float()
                loss_train = loss_fn(output, labels[output_nodes])
                loss_train.backward()
                torch.nn.utils.clip_grad_norm_(susage.parameters(), 0.2)
                optimizer.step()
                times += [time.time() - t1]
                train_losses += [loss_train.detach().tolist()]
                del loss_train

        idx = torch.randperm(len(valid_nodes))[:args.batch_size]
        valid_batch_nodes = valid_nodes[idx]
        valid_data = default_sampler(np.random.randint(2**32 - 1),
                                     valid_batch_nodes,
                                     samp_num_list*20,
                                     len(feat_data),
                                     lap_matrix,
                                     args.n_layers)

        # Evaluation
        susage.eval()
        adjs, input_nodes, output_nodes = valid_data
        adjs = package_mxl(adjs, device)
        output = susage.forward(feat_data[input_nodes], adjs)
        output = output[output_nodes]
        if labels.dim() != 1:
            labels = labels.float()
        loss_valid = loss_fn(output, labels[output_nodes]).detach().tolist()

        if labels[output_nodes].dim() == 1:
            predictions = output.argmax(dim=1).cpu()
            targets = labels[output_nodes].cpu()
            valid_f1 = f1_score(targets, predictions, average='micro')
        else:
            pred = output.cpu()
            labl = labels.cpu()
            y_pred = pred > 0
            y_true = labl[output_nodes] > 0.5

            tp = int((y_true & y_pred).sum())
            fp = int((~y_true & y_pred).sum())
            fn = int((y_true & ~y_pred).sum())

            try:
                precision = tp / (tp + fp)
                recall = tp / (tp + fn)
                valid_f1 = accuracy = 2 * (precision * recall) / (precision + recall)
            except ZeroDivisionError:
                valid_f1 = 0.
        print(("Epoch: %d (%.1fs) Train Loss: %.2f    Valid Loss: %.2f Valid F1: %.3f") %                   (epoch, np.sum(times), np.average(train_losses), loss_valid, valid_f1))
    print(f'Final Val Acc: {100*valid_f1:.2f}')
    #     if valid_f1 > best_val + 1e-2:
    #         best_val = valid_f1
    #         torch.save(susage, './save/best_model.pt')
    #         cnt = 0
    #     else:
    #         cnt += 1
    #     if cnt == args.n_stops // args.batch_num:
    #         break
    # best_model = torch.load('./save/best_model.pt')
    # best_model.eval()

    
    '''
    If using batch sampling for inference:
    '''
    #     for b in np.arange(len(test_nodes) // args.batch_size):
    #         batch_nodes = test_nodes[b * args.batch_size : (b+1) * args.batch_size]
    #         adjs, input_nodes, output_nodes = sampler(np.random.randint(2**32 - 1), batch_nodes,
    #                                     samp_num_list * 20, len(feat_data), lap_matrix, args.n_layers)
    #         adjs = package_mxl(adjs, device)
    #         output = best_model.forward(feat_data[input_nodes], adjs)[output_nodes]
    #         test_f1 = f1_score(output.argmax(dim=1).cpu(), labels[output_nodes].cpu(), average='micro')
    #         test_f1s += [test_f1]

    '''
    If using full-batch inference:
    '''
    susage.eval()
    batch_nodes = test_nodes
    adjs, input_nodes, output_nodes = default_sampler(np.random.randint(2**32 - 1), batch_nodes,
                                    samp_num_list * 20, len(feat_data), lap_matrix, args.n_layers)
    adjs = package_mxl(adjs, device)
    output = susage.forward(feat_data[input_nodes], adjs)
    if labels[output_nodes].dim() == 1:
        predictions = output.argmax(dim=1)[output_nodes].cpu()
        targets = labels[output_nodes].cpu()
        test_f1s = f1_score(targets, predictions, average='micro')
    else:
        pred = output[output_nodes].cpu()
        labl = labels.cpu()
        y_pred = pred > 0
        y_true = labl[output_nodes] > 0.5

        tp = int((y_true & y_pred).sum())
        fp = int((~y_true & y_pred).sum())
        fn = int((y_true & ~y_pred).sum())

        try:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            test_f1s = accuracy = 2 * (precision * recall) / (precision + recall)
        except ZeroDivisionError:
            test_f1s = 0.
    
    print('Iteration: %d, Test F1: %.3f' % (oiter+1, np.average(test_f1s)))
    results[oiter] = np.average([test_f1s])

print('-' * 10)
print(f'Number of runs: {args.runs}')
print(f'Mini Acc: {100 * results.mean():.2f} ± {100 * results.std():.2f}')
