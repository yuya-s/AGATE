import torch
from torch.autograd import Variable
import numpy as np
import os
import sys
from scipy.sparse import lil_matrix
from scipy.io import mmwrite
os.environ['KMP_DUPLICATE_LIB_OK']='True' # mmwriteー
current_dir = os.path.dirname(os.path.abspath("__file__"))
sys.path.append( str(current_dir) + '/../../../' )

from setting_param import Model_attribute_prediction_exist_Tm_binary_STGGNN_OutputDir as binary_InputDir
from setting_param import Model_attribute_prediction_exist_Tm_with_transfer_GCN_OutputDir as transfer_InputDir
from setting_param import L
from utils.attribute_prediction_exist_Tm_binary import get_performance as get_binary
from utils.attribute_prediction_exist_Tm_with_transfer import get_performance as get_transfer
from utils.attribute_prediction_exist_Tm_binary import EXIST_TABLE
threshold = 0.3305

def concat_train_valid_test(train_result, valid_result, test_result):
    result = []
    for train_r in train_result:
        result.append(train_r)
    for valid_r in valid_result:
        result.append(valid_r)
    for test_r in test_result:
        result.append(test_r)
    ts_result_dic = {}
    for t_idx, ts in enumerate(range(L, EXIST_TABLE.shape[1]-L)):
        ts_result_dic[ts] = result[t_idx]
    return ts_result_dic

train_result = get_binary(binary_InputDir, True, False, False)
valid_result = get_binary(binary_InputDir, False, True, False)
test_result = get_binary(binary_InputDir, False, False, True)
pred_binary = concat_train_valid_test(train_result, valid_result, test_result)

train_result = get_transfer(transfer_InputDir, True, False, False)
valid_result = get_transfer(transfer_InputDir, False, True, False)
test_result = get_transfer(transfer_InputDir, False, False, True)
pred_transfer = concat_train_valid_test(train_result, valid_result, test_result)

def inference(dataloader, opt, OutputDir, Attribute_idx):
    for i, (sample_idx, annotation, adj_matrix, label, mask) in enumerate(dataloader, 0):
        target = Variable(label)
        mask = Variable(mask)
        output = annotation[:, :, -1][:, :, Attribute_idx]
        output = output.argmax(axis=2)[:, :, np.newaxis]

        for batch in range(opt.batchSize):
            ts = int(sample_idx[batch].numpy())
            output[batch][pred_binary[ts] > threshold] = torch.LongTensor(pred_transfer[ts][pred_binary[ts] > threshold])

        # 
        os.makedirs(OutputDir + "/output", exist_ok=True)
        for batch in range(opt.batchSize):
            p = output.detach().numpy()[batch]
            t = target[batch].numpy()
            m = mask[batch].numpy()
            mmwrite(OutputDir + "/output/pred" + str(sample_idx.numpy()[batch]), lil_matrix(p))
            mmwrite(OutputDir + "/output/true" + str(sample_idx.numpy()[batch]), lil_matrix(t))
            mmwrite(OutputDir + "/output/mask" + str(sample_idx.numpy()[batch]), lil_matrix(m))