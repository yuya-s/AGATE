import numpy as np
from scipy.io import mmread
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import glob
import re
import os
import sys
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
sns.set(style='darkgrid')
sns.set_style(style='whitegrid')

# moduleー
current_dir = os.path.dirname(os.path.abspath("__file__"))
sys.path.append( str(current_dir) + '/../' )
from setting_param import ratio_test
from setting_param import ratio_valid
from setting_param import MakeSample_attribute_prediction_exist_InputDir
EXIST_TABLE = np.load(MakeSample_attribute_prediction_exist_InputDir + '/exist_table.npy')

from setting_param import attribute_prediction_exist_PTS_mean as a_mean
from setting_param import attribute_prediction_exist_PTS_std as a_std

from setting_param import Evaluation_repeat1_attribute_prediction_exist_PTS_utilize_disappeared_Baseline_InputDir as Baseline_InputDir
from setting_param import Evaluation_repeat1_attribute_prediction_exist_PTS_utilize_disappeared_LSTM_InputDir as LSTM_InputDir
from setting_param import Evaluation_repeat1_attribute_prediction_exist_PTS_utilize_disappeared_STGGNN_InputDir as STGGNN_InputDir
from setting_param import Evaluation_repeat1_attribute_prediction_exist_PTS_utilize_disappeared_EGCNh_InputDir as EGCNh_InputDir
from setting_param import Evaluation_repeat1_attribute_prediction_exist_PTS_utilize_disappeared_STGCN_InputDir as STGCN_InputDir
from setting_param import Evaluation_repeat1_attribute_prediction_exist_PTS_utilize_disappeared_EGCNo_InputDir as EGCNo_InputDir
from setting_param import Evaluation_repeat1_attribute_prediction_exist_PTS_utilize_disappeared_GCN_InputDir as GCN_InputDir

from setting_param import Evaluation_repeat1_attribute_prediction_exist_PTS_utilize_disappeared_OutputDir as OutputDir

InputDirs = [Baseline_InputDir, LSTM_InputDir, GCN_InputDir, STGCN_InputDir, EGCNh_InputDir, EGCNo_InputDir, STGGNN_InputDir]
methods = ['Baseline', 'LSTM', 'GCN', 'STGCN', 'EvolveGCN-H', 'EvolveGCN-O', 'TGGNN']
# InputDirs = [STGGNN_InputDir]
# methods = ['TGGNN']
os.makedirs(OutputDir, exist_ok=True)

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


def calc_mae(true_paths, pred_paths, mask_paths, target_idx):
    y_true = sum([((mmread(true_paths[idx]).toarray()[0 < mmread(mask_paths[idx]).toarray()] * a_std) + a_mean).tolist() for idx in target_idx], [])
    y_pred = sum([((mmread(pred_paths[idx]).toarray()[0 < mmread(mask_paths[idx]).toarray()] * a_std) + a_mean).tolist() for idx in target_idx], [])
    return mean_absolute_error(y_true, y_pred)


def calc_mse(true_paths, pred_paths, mask_paths, target_idx):
    y_true = sum([((mmread(true_paths[idx]).toarray()[0 < mmread(mask_paths[idx]).toarray()] * a_std) + a_mean).tolist() for idx in target_idx], [])
    y_pred = sum([((mmread(pred_paths[idx]).toarray()[0 < mmread(mask_paths[idx]).toarray()] * a_std) + a_mean).tolist() for idx in target_idx], [])
    return mean_squared_error(y_true, y_pred)


def calc_rmse(true_paths, pred_paths, mask_paths, target_idx):
    y_true = sum([((mmread(true_paths[idx]).toarray()[0 < mmread(mask_paths[idx]).toarray()] * a_std) + a_mean).tolist() for idx in target_idx], [])
    y_pred = sum([((mmread(pred_paths[idx]).toarray()[0 < mmread(mask_paths[idx]).toarray()] * a_std) + a_mean).tolist() for idx in target_idx], [])
    return np.sqrt(mean_squared_error(y_true, y_pred))


def calc_r2(true_paths, pred_paths, mask_paths, target_idx):
    y_true = sum([((mmread(true_paths[idx]).toarray()[0 < mmread(mask_paths[idx]).toarray()] * a_std) + a_mean).tolist() for idx in target_idx], [])
    y_pred = sum([((mmread(pred_paths[idx]).toarray()[0 < mmread(mask_paths[idx]).toarray()] * a_std) + a_mean).tolist() for idx in target_idx], [])
    return r2_score(y_true, y_pred)

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

    mae = calc_mae(true_paths, pred_paths, mask_paths, target_idx)
    mse = calc_mse(true_paths, pred_paths, mask_paths, target_idx)
    rmse = calc_rmse(true_paths, pred_paths, mask_paths, target_idx)
    r2 = calc_r2(true_paths, pred_paths, mask_paths, target_idx)

    return mae, mse, rmse, r2

# Loss
for idx, method in enumerate(methods):
    if method == 'Baseline':
        continue
    InputDir = InputDirs[idx]
    loss = pd.read_csv(InputDir + '/loss.csv')
    epoch = loss['epoch'].values
    train_loss = loss['train_loss'].values
    valid_loss = loss['valid_loss'].values
    test_loss = loss['test_loss'].values

    plt.figure()
    plt.plot(epoch, train_loss, marker=".", label='train')
    plt.plot(epoch, valid_loss, marker=".", label='valid')
    plt.plot(epoch, test_loss, marker=".", label='test')
    plt.title(method)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(OutputDir + '/' + method + '_loss.pdf')

# train
result_dic = {'Method':[], 'MAE':[], 'MSE':[], 'RMSE':[], 'R2':[]}
for idx, method in enumerate(methods):
    InputDir = InputDirs[idx]
    mae, mse, rmse, r2 = get_performance(InputDir, True, False, False)
    result_dic['Method'].append(method)
    result_dic['MAE'].append(mae)
    result_dic['MSE'].append(mse)
    result_dic['RMSE'].append(rmse)
    result_dic['R2'].append(r2)
pd.DataFrame(result_dic).to_csv(OutputDir + '/performance_train.csv')

# valid
result_dic = {'Method':[], 'MAE':[], 'MSE':[], 'RMSE':[], 'R2':[]}
for idx, method in enumerate(methods):
    InputDir = InputDirs[idx]
    mae, mse, rmse, r2 = get_performance(InputDir, False, True, False)
    result_dic['Method'].append(method)
    result_dic['MAE'].append(mae)
    result_dic['MSE'].append(mse)
    result_dic['RMSE'].append(rmse)
    result_dic['R2'].append(r2)
pd.DataFrame(result_dic).to_csv(OutputDir + '/performance_valid.csv')

# test
result_dic = {'Method':[], 'MAE':[], 'MSE':[], 'RMSE':[], 'R2':[]}
for idx, method in enumerate(methods):
    InputDir = InputDirs[idx]
    mae, mse, rmse, r2 = get_performance(InputDir, False, False, True)
    result_dic['Method'].append(method)
    result_dic['MAE'].append(mae)
    result_dic['MSE'].append(mse)
    result_dic['RMSE'].append(rmse)
    result_dic['R2'].append(r2)
pd.DataFrame(result_dic).to_csv(OutputDir + '/performance_test.csv')