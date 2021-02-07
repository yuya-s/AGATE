import numpy as np
from scipy.io import mmread
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import glob
import re
import os
import sys
import itertools
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
sns.set_style(style='whitegrid')

# moduleー
current_dir = os.path.dirname(os.path.abspath("__file__"))
sys.path.append( str(current_dir) + '/../' )
from setting_param import ratio_test
from setting_param import ratio_valid
from setting_param import MakeSample_attribute_prediction_exist_InputDir
EXIST_TABLE = np.load(MakeSample_attribute_prediction_exist_InputDir + '/exist_table.npy')

from setting_param import attribute_prediction_exist_Pos_ID as Pos_ID
from setting_param import attribute_prediction_exist_Pos_n_class as n_class
df = pd.read_csv(MakeSample_attribute_prediction_exist_InputDir + "/attribute_idx.csv")
Pos_list = [df['class' + str(i)][Pos_ID] for i in range(n_class)]

from setting_param import Evaluation_attribute_prediction_exist_Pos_Baseline_InputDir as Baseline_InputDir
from setting_param import Evaluation_attribute_prediction_exist_Pos_LSTM_InputDir as LSTM_InputDir
from setting_param import Evaluation_attribute_prediction_exist_Pos_STGGNN_InputDir as STGGNN_InputDir
from setting_param import Evaluation_attribute_prediction_exist_Pos_EGCNh_InputDir as EGCNh_InputDir
from setting_param import Evaluation_attribute_prediction_exist_Pos_STGCN_InputDir as STGCN_InputDir
from setting_param import Evaluation_attribute_prediction_exist_Pos_EGCNo_InputDir as EGCNo_InputDir
from setting_param import Evaluation_attribute_prediction_exist_Pos_GCN_InputDir as GCN_InputDir

from setting_param import Evaluation_attribute_prediction_exist_Pos_OutputDir as OutputDir

InputDirs = [Baseline_InputDir, LSTM_InputDir, GCN_InputDir, STGCN_InputDir, EGCNh_InputDir, EGCNo_InputDir, STGGNN_InputDir]
methods = ['Baseline', 'LSTM', 'GCN', 'STGCN', 'EvolveGCN-H', 'EvolveGCN-O', 'TGGNN']
os.makedirs(OutputDir, exist_ok=True)
os.makedirs(OutputDir + '/train', exist_ok=True)
os.makedirs(OutputDir + '/valid', exist_ok=True)
os.makedirs(OutputDir + '/test', exist_ok=True)

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

def calc_classification_score(true_paths, pred_paths, mask_paths, target_idx):
    y_true = sum([mmread(true_paths[idx]).toarray()[0 < mmread(mask_paths[idx]).toarray()].tolist() for idx in target_idx], [])
    y_pred = sum([mmread(pred_paths[idx]).toarray()[0 < mmread(mask_paths[idx]).toarray()].tolist() for idx in target_idx], [])
    return accuracy_score(y_true, y_pred), recall_score(y_true, y_pred, average='macro'), precision_score(y_true, y_pred, average='macro'), f1_score(y_true, y_pred, average='macro'), confusion_matrix(y_true, y_pred, labels=list(range(n_class)))

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

    acc, recall, precision, f1, cm = calc_classification_score(true_paths, pred_paths, mask_paths, target_idx)

    return acc, recall, precision, f1, cm

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
result_dic = {'Method':[], 'ACC':[], 'Recall (macro)':[], 'Precision (macro)':[], 'F1 (macro)':[]}
for idx, method in enumerate(methods):
    InputDir = InputDirs[idx]
    acc, recall, precision, f1, cm = get_performance(InputDir, True, False, False)
    result_dic['Method'].append(method)
    result_dic['ACC'].append(acc)
    result_dic['Recall (macro)'].append(recall)
    result_dic['Precision (macro)'].append(precision)
    result_dic['F1 (macro)'].append(f1)

    # plot cm
    plt.figure()
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title(method)
    plt.colorbar()
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    tick_marks = np.arange(n_class)
    plt.xticks(tick_marks, Pos_list, rotation=90, fontsize=4)
    plt.yticks(tick_marks, Pos_list, rotation=0, fontsize=4)
    plt.grid(False)
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 verticalalignment="center",
                 color="white" if cm[i, j] > cm.max() / 2 else "black",
                 fontsize=4)
    plt.savefig(OutputDir + '/train/confusion_matrix_' + method + '.pdf')

pd.DataFrame(result_dic).to_csv(OutputDir + '/performance_train.csv')

# valid
result_dic = {'Method':[], 'ACC':[], 'Recall (macro)':[], 'Precision (macro)':[], 'F1 (macro)':[]}
for idx, method in enumerate(methods):
    InputDir = InputDirs[idx]
    acc, recall, precision, f1, cm = get_performance(InputDir, False, True, False)
    result_dic['Method'].append(method)
    result_dic['ACC'].append(acc)
    result_dic['Recall (macro)'].append(recall)
    result_dic['Precision (macro)'].append(precision)
    result_dic['F1 (macro)'].append(f1)

    # plot cm
    plt.figure()
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title(method)
    plt.colorbar()
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    tick_marks = np.arange(n_class)
    plt.xticks(tick_marks, Pos_list, rotation=90, fontsize=4)
    plt.yticks(tick_marks, Pos_list, rotation=0, fontsize=4)
    plt.grid(False)
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 verticalalignment="center",
                 color="white" if cm[i, j] > cm.max() / 2 else "black",
                 fontsize=4)
    plt.savefig(OutputDir + '/valid/confusion_matrix_' + method + '.pdf')

pd.DataFrame(result_dic).to_csv(OutputDir + '/performance_valid.csv')

# test
result_dic = {'Method':[], 'ACC':[], 'Recall (macro)':[], 'Precision (macro)':[], 'F1 (macro)':[]}
for idx, method in enumerate(methods):
    InputDir = InputDirs[idx]
    acc, recall, precision, f1, cm = get_performance(InputDir, False, False, True)
    result_dic['Method'].append(method)
    result_dic['ACC'].append(acc)
    result_dic['Recall (macro)'].append(recall)
    result_dic['Precision (macro)'].append(precision)
    result_dic['F1 (macro)'].append(f1)

    # plot cm
    plt.figure()
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title(method)
    plt.colorbar()
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    tick_marks = np.arange(n_class)
    plt.xticks(tick_marks, Pos_list, rotation=90, fontsize=4)
    plt.yticks(tick_marks, Pos_list, rotation=0, fontsize=4)
    plt.grid(False)
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 verticalalignment="center",
                 color="white" if cm[i, j] > cm.max() / 2 else "black",
                 fontsize=4)
    plt.savefig(OutputDir + '/test/confusion_matrix_' + method + '.pdf')

pd.DataFrame(result_dic).to_csv(OutputDir + '/performance_test.csv')