import argparse

from utils.inference import inference
from utils.data.dataset import BADataset
from utils.data.dataloader import BADataloader

import sys
import os
current_dir = os.path.dirname(os.path.abspath("__file__"))
sys.path.append( str(current_dir) + '/../../../' )
from setting_param import Model_link_prediction_disappeared_InputDir as InputDir
from setting_param import Model_link_prediction_disappeared_Random_OutputDir as OutputDir
from setting_param import link_prediction_disappeared_worker
from setting_param import link_prediction_disappeared_batchSize
from setting_param import link_prediction_disappeared_init_L
from setting_param import link_prediction_disappeared_state_dim
from setting_param import link_prediction_disappeared_output_dim

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, help='number of data loading workers', default=link_prediction_disappeared_worker)
parser.add_argument('--batchSize', type=int, default=link_prediction_disappeared_batchSize, help='input batch size')
parser.add_argument('--state_dim', type=int, default=link_prediction_disappeared_state_dim, help='GGNN hidden state size')
parser.add_argument('--output_dim', type=int, default=link_prediction_disappeared_output_dim, help='Model output state size')
parser.add_argument('--init_L', type=int, default=link_prediction_disappeared_init_L, help='number of observation time step')

opt = parser.parse_args()
print(opt)

opt.dataroot = InputDir

opt.L = opt.init_L

def main(opt):
    all_dataset = BADataset(opt.dataroot, opt.L, False, False, False)
    all_dataloader = BADataloader(all_dataset, batch_size=opt.batchSize, \
                                     shuffle=False, num_workers=opt.workers, drop_last=False)

    opt.annotation_dim = 10
    opt.n_edge_types = all_dataset.n_edge_types
    opt.n_node = all_dataset.n_node

    inference(all_dataloader, opt, OutputDir)


if __name__ == "__main__":
    main(opt)
