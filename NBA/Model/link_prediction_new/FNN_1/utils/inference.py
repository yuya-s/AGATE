from torch.autograd import Variable
import numpy as np
import os
import torch
from scipy.sparse import lil_matrix
from scipy.io import mmwrite

import sys
import os
current_dir = os.path.dirname(os.path.abspath("__file__"))
sys.path.append( str(current_dir) + '/../../../' )
from setting_param import all_node_num
from setting_param import n_expanded

os.environ['KMP_DUPLICATE_LIB_OK']='True' # mmwriteー

def inference(dataloader, net, opt, OutputDir):
    net.eval()

    for i, (sample_idx, data_arrays_link, dists, ind_train_A, ind_train_X, nodes_keep, A, X, label, mask) in enumerate(dataloader, 0):
        sample_idx, data_arrays_link, dists, ind_train_A, ind_train_X, nodes_keep, A, X, label, mask = sample_idx[0], data_arrays_link, dists[0], ind_train_A[0], ind_train_X[0], nodes_keep[0], A[0], X[0], label[0], mask[0]
        indic_train_ones = data_arrays_link[0][0]
        indic_train_zeros = data_arrays_link[1][0]
        indic_valid_ones = data_arrays_link[2][0]
        indic_valid_zeros = data_arrays_link[3][0]
        indic_test = data_arrays_link[4][0]

        if opt.cuda:
            label = label.cuda()
            mask = mask.cuda()
        target = Variable(label)
        mask = Variable(mask)

        pred_ = []
        for j in range(indic_test.shape[0]):
            X_0 = X[indic_test[j, 0]]
            X_1 = X[indic_test[j, 1]]
            X_batch = torch.stack([X_0, X_1])[None, :]
            output = net(X_batch)
            pred_.append(output.item())
        pred_ = np.array(pred_)

        pred = np.zeros((all_node_num+n_expanded, all_node_num+n_expanded))
        pred[all_node_num:] = pred_.reshape(n_expanded, all_node_num+n_expanded)
        pred_between_new_nodes = pred[all_node_num:, all_node_num:]
        pred = pred + pred.T
        pred[all_node_num:, all_node_num:] = pred_between_new_nodes

        #
        os.makedirs(OutputDir + "/output", exist_ok=True)
        p = pred
        t = target.numpy()
        m = mask.numpy()
        mmwrite(OutputDir + "/output/pred" + str(sample_idx.numpy()), lil_matrix(p * m)) #
        mmwrite(OutputDir + "/output/true" + str(sample_idx.numpy()), lil_matrix(t))
        mmwrite(OutputDir + "/output/mask" + str(sample_idx.numpy()), lil_matrix(m))