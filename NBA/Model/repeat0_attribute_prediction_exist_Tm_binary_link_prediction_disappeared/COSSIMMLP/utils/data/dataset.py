import numpy as np
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
from setting_param import attribute_dim


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

class BADataset():
    def __init__(self, path, L, is_train, is_valid, is_test):
        # PATH
        attribute_paths = load_paths_from_dir(path + '/input/node_attribute')
        adjacency_paths = load_paths_from_dir(path + '/input/adjacency')
        label_paths = load_paths_from_dir(path + '/label')
        mask_paths = load_paths_from_dir(path + '/mask')

        # split data
        n_samples = len(label_paths)
        all_idx = list(range(n_samples))
        dev_idx, test_idx = dev_test_split(all_idx, n_samples, ratio_test)
        train_idx, valid_idx = dev_test_split(dev_idx, n_samples, ratio_valid)

        if is_train:
            target_idx = train_idx
        elif is_valid:
            target_idx = valid_idx
        elif is_test:
            target_idx = test_idx
        else:
            target_idx = all_idx

        self.idx_list = target_idx
        self.attribute = [mmread(attribute_paths[idx]).toarray().reshape(all_node_num, 1) for idx in target_idx]
        self.label = [mmread(label_paths[idx]).toarray() for idx in target_idx]
        self.mask = [mmread(mask_paths[idx]).toarray() for idx in target_idx]

        # 
        self.L = L
        self.n_node = adj_shape[0]
        self.n_edge_types = adj_shape[1] // adj_shape[0]

    def __getitem__(self, index):
        sample_idx = self.idx_list[index] + self.L
        annotation = self.attribute[index]
        label = self.label[index]
        mask = self.mask[index]
        return sample_idx, annotation, label, mask

    def __len__(self):
        return len(self.idx_list)