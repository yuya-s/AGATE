import numpy as np
import networkx as nx
import torch
import random
import glob
import re
from scipy.io import mmread
import os
import sys
current_dir = os.path.dirname(os.path.abspath("__file__"))
sys.path.append( str(current_dir) + '/../../../' )
from setting_param import ratio_test
from setting_param import ratio_valid
from setting_param import adj_shape
from setting_param import max_nnz_am

from setting_param import all_node_num
from setting_param import n_expanded
from setting_param import L
from setting_param import Model_link_prediction_appeared_InputDir as InputDir


def load_paths_from_dir(dir_path):
    # dir
    path_list = glob.glob(dir_path + "/*")
    # ー (ー)
    path_list = np.array(sorted(path_list, key=lambda s: int(re.findall(r'\d+', s)[-1])))
    return path_list

def dev_test_split(all_idx, n_samples, ratio_test):
    n_test = int(n_samples * ratio_test)
    return all_idx[:-n_test], all_idx[-n_test:]

def train_valid_split(dev_idx, n_samples, ratio_valid):
    n_valid = int(n_samples * ratio_valid)
    return dev_idx[:-n_valid], dev_idx[-n_valid:]

def coo_scipy2coo_numpy(coo_scipy, max_nnz):
    # scipy_coovalue, indicesnumpy
    # max_nnz
    coo_numpy = np.zeros((3, max_nnz))
    coo_numpy[:, :len(coo_scipy.data)] = np.vstack((coo_scipy.data, coo_scipy.row, coo_scipy.col))
    return coo_numpy

def in_out_generate(coo_numpy, n_node):
    coo_numpy_in = coo_numpy.copy()
    coo_numpy_out = np.zeros_like(coo_numpy)
    coo_numpy_out[0] = coo_numpy[0]
    coo_numpy_out[1] = coo_numpy[2] % n_node
    coo_numpy_out[2] = (coo_numpy[2] // n_node) * n_node + coo_numpy[1]
    return np.stack((coo_numpy_in, coo_numpy_out))

def get_A_in_per_batch(A, batch):
    A_in = A[batch][0]  # (3, max_nnz)
    nnz = int(A_in[0].sum().item())
    A_in = A_in[:, :nnz]  # (3, nnz)
    return A_in

def get_adj_per_t(A_in, t):
    col = A_in[2]
    idx = (t * all_node_num) <= col
    idx = idx * (col < ((t + 1) * all_node_num))
    A_t = A_in[:, idx]  # (3, nnz_per_t)
    A_t[2] = A_t[2] % all_node_num  # adjacency matrix per t
    return A_t

def get_cur_adj(A_t):
    cur_adj = {}
    cur_adj['vals'] = A_t[0]  # (nnz_per_t, )
    cur_adj['idx'] = A_t[1:].t().long()  # (nnz_per_t, 2)
    return cur_adj

def make_sparse_tensor(adj,tensor_type,torch_size):
    if len(torch_size) == 2:
        tensor_size = torch.Size(torch_size)
    elif len(torch_size) == 1:
        tensor_size = torch.Size(torch_size*2)

    if tensor_type == 'float':
        test = torch.sparse.FloatTensor(adj['idx'].t(),
                                      adj['vals'].type(torch.float),
                                      tensor_size)
        return torch.sparse.FloatTensor(adj['idx'].t(),
                                      adj['vals'].type(torch.float),
                                      tensor_size)
    elif tensor_type == 'long':
        return torch.sparse.LongTensor(adj['idx'].t(),
                                      adj['vals'].type(torch.long),
                                      tensor_size)
    else:
        raise NotImplementedError('only make floats or long sparse tensors')

def sparse_prepare_tensor(tensor,torch_size):
    tensor = make_sparse_tensor(tensor,
                                tensor_type = 'float',
                                torch_size = torch_size)
    return tensor

def get_A_last(A):
    """

    """
    batch_adj_list = []
    for batch in range(A.shape[0]):
        A_in = get_A_in_per_batch(A, batch) # (3, nnz)
        A_t = get_adj_per_t(A_in, L-1) # (3, nnz_per_t)
        cur_adj = get_cur_adj(A_t)
        cur_adj = sparse_prepare_tensor(cur_adj, torch_size=[all_node_num])
        batch_adj_list.append(cur_adj)
    A_last = torch.stack(batch_adj_list, 0)
    return A_last

class BADataset():
    def __init__(self, path, L, is_train, is_valid, is_test):
        # PATH
        attribute_paths = load_paths_from_dir(path + '/input')
        adjacency_paths = load_paths_from_dir(InputDir + '/input/adjacency')
        label_paths = load_paths_from_dir(path + '/label')
        mask_paths = load_paths_from_dir(path + '/mask')

        # split data
        n_samples = len(label_paths)
        all_idx = list(range(n_samples))
        dev_idx, test_idx = dev_test_split(all_idx, n_samples, ratio_test)
        train_idx, valid_idx = dev_test_split(dev_idx, n_samples, ratio_valid)

        if is_train:
            #target_idx = train_idx
            target_idx = all_idx[-18:-4]
        elif is_valid:
            #target_idx = valid_idx
            target_idx = all_idx[-4:-2]
        elif is_test:
            #target_idx = test_idx
            target_idx = all_idx[-2:]
        else:
            target_idx = all_idx

        self.idx_list = target_idx
        self.attribute = [np.load(attribute_paths[idx]) for idx in target_idx]
        self.label = [mmread(label_paths[idx]).toarray() for idx in target_idx]
        self.mask = [mmread(mask_paths[idx]).toarray() for idx in target_idx]

        """
        # adjacency: [n_sample * (2, 3, max_nnz_am)] → (n_sample, all_node_num, all_node_num) 
        self.adjacency = [in_out_generate(coo_scipy2coo_numpy(mmread(adjacency_paths[idx]), max_nnz_am), adj_shape[0])
                          for idx in target_idx]
        self.adjacency = get_A_last(torch.Tensor(self.adjacency))
        """
        self.L = L

        self.indic_test_ones = []
        self.indic = []
        self.indic_mask = []
        for n in range(len(self.label)):
            """
            indic = self.adjacency[n]._indices().numpy().transpose(1, 0) # (nnz, 2) (：[0, 1][1, 0])
            indic_frozenset = set([frozenset({i, j}) for [i, j] in indic])
            graph = nx.Graph()
            graph.add_edges_from(indic.tolist())
            indic = np.array(sorted(graph.edges)) # (nnz/2, 2) (：[0, 1][1, 0])

            # test_ones
            indic_test_ones = torch.Tensor(self.label[n]).to_sparse()._indices().numpy().transpose(1, 0)  # (nnz, 2) (：[0, 1][1, 0])
            graph = nx.Graph()
            graph.add_edges_from(indic_test_ones.tolist())
            indic_test_ones = np.array(sorted(graph.edges))  # (nnz/2, 2) (：[0, 1][1, 0])
            indic_test_ones = np.array([sorted(edge) for edge in indic_test_ones])  # (existing_node, new_node)
            indic_test_ones_frozenset = set([frozenset({i, j}) for [i, j] in indic_test_ones])

            # test_zeros
            indic_test_zeros = set()
            while len(indic_test_zeros) < len(indic_test_ones):
                i = int(indic_test_ones[random.sample(list(range(indic_test_ones.shape[0])), 1), 1])
                j = int(indic[random.sample(list(range(indic.shape[0])), 1), 0])
                while i == j:
                    j = int(indic[random.sample(list(range(indic.shape[0])), 1), 0])
                if not frozenset([i, j]) in indic_test_ones_frozenset:
                    indic_test_zeros.add(frozenset([i, j]))
            indic_test_zeros = np.array(sorted([sorted(edge) for edge in indic_test_zeros]))
            """
            # test_ones
            indic_test_ones = torch.Tensor(self.label[n]).to_sparse()._indices().numpy().transpose(1, 0)  # (nnz, 2) (：[0, 1][1, 0])
            graph = nx.Graph()
            graph.add_edges_from(indic_test_ones.tolist())
            indic_test_ones = np.array(sorted(graph.edges))  # (nnz/2, 2) (：[0, 1][1, 0])
            indic_test_ones = np.array([sorted(edge) for edge in indic_test_ones])  # (existing_node, new_node)
            self.indic_test_ones.append(indic_test_ones)

            # mask ー
            indic_mask = torch.Tensor(self.mask[n]).to_sparse()._indices().numpy().transpose(1, 0)  # (nnz, 2) (：[0, 1][1, 0])
            graph = nx.Graph()
            graph.add_edges_from(indic_mask.tolist())
            indic_mask = np.array(sorted(graph.edges))  # (nnz/2, 2) (：[0, 1][1, 0])
            indic_mask = np.array([sorted(edge, reverse=True) for edge in indic_mask])  # (new_node, existing_node)
            self.indic_mask.append(indic_mask)

            #
            tmp = []
            for i in range(all_node_num, all_node_num + n_expanded):
                for j in range(all_node_num + n_expanded):
                    tmp.append([i, j])
            self.indic.append(np.array(tmp))

    def __getitem__(self, index):
        sample_idx = self.idx_list[index] + self.L
        annotation = self.attribute[index]
        label = self.indic_test_ones[index]
        mask = self.indic_mask[index]
        indic = self.indic[index]
        return sample_idx, annotation, label, mask, indic

    def __len__(self):
        return len(self.idx_list)