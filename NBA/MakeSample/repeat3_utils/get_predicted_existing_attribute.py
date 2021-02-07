import numpy as np
from scipy.io import mmread
import seaborn as sns
import glob
import re
import os
import sys
sns.set_style(style='whitegrid')

# moduleー
current_dir = os.path.dirname(os.path.abspath("__file__"))
sys.path.append( str(current_dir) + '/../' )
from setting_param import ratio_test
from setting_param import ratio_valid
from setting_param import MakeSample_attribute_prediction_exist_InputDir
EXIST_TABLE = np.load(MakeSample_attribute_prediction_exist_InputDir + '/exist_table.npy')

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

def true_pred_mask_split(input_dir):
    paths = load_paths_from_dir(input_dir + '/output')
    true_ls = []
    pred_ls = []
    mask_ls = []
    for path in paths:
        if 'true' in path.split('/')[-1]:
            true_ls.append(path)
        elif 'pred' in path.split('/')[-1]:
            pred_ls.append(path)
        elif 'mask' in path.split('/')[-1]:
            mask_ls.append(path)
    return np.array(true_ls), np.array(pred_ls), np.array(mask_ls)

def get_performance(InputDir, is_train, is_valid, is_test):
    true_paths, pred_paths, mask_paths = true_pred_mask_split(InputDir)
    n_samples = len(true_paths)
    all_idx = list(range(n_samples))
    dev_idx, test_idx = dev_test_split(all_idx, n_samples, ratio_test)
    train_idx, valid_idx = dev_test_split(dev_idx, n_samples, ratio_valid)

    if is_train:
        target_idx = train_idx
    elif is_valid:
        target_idx = valid_idx
    elif is_test:
        target_idx = test_idx

    y_pred = np.array([mmread(pred_paths[idx]).toarray() for idx in target_idx])
    return y_pred

def get_predicted_existing_attribute(InputDir, is_train, is_valid, is_test):
    y_pred = get_performance(InputDir, is_train, is_valid, is_test)
    return y_pred