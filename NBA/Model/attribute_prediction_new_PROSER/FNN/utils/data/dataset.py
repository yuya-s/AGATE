import numpy as np
import glob
import re
import os
import sys
current_dir = os.path.dirname(os.path.abspath("__file__"))
sys.path.append( str(current_dir) + '/../../../' )
from setting_param import ratio_test
from setting_param import ratio_valid


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


class BADataset():
    def __init__(self, path, L, is_train, is_valid, is_test):
        # PATH
        input_paths = load_paths_from_dir(path + '/input')
        label_paths = load_paths_from_dir(path + '/label')
        input_num_paths = load_paths_from_dir(path + '/input_num')

        # split data
        n_samples = len(input_paths)
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
        self.input = [np.load(input_paths[idx]) for idx in target_idx]
        self.label = [np.load(label_paths[idx]) for idx in target_idx]
        self.input_num = [np.load(input_num_paths[idx]) for idx in target_idx]

        #
        self.L = L

    def __getitem__(self, index):
        sample_idx = self.idx_list[index] + self.L
        input = self.input[index]
        label = self.label[index]
        input_num = self.input_num[index]
        return sample_idx, input, label, input_num

    def __len__(self):
        return len(self.idx_list)