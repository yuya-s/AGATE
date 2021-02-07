import os
import sys
current_dir = os.path.dirname(os.path.abspath("__file__"))
sys.path.append( str(current_dir) + '/../../' )
from setting_param import L
from setting_param import all_node_num
from setting_param import n_expanded
from setting_param import attribute_dim

from setting_param import MakeSample_repeat1_link_prediction_appeared_utilize_disappeared_DynGEM_InputDir as link_prediction_appeared_InputDir
from setting_param import MakeSample_repeat1_link_prediction_disappeared_utilize_disappeared_DynGEM_InputDir as link_prediction_disappeared_InputDir
from setting_param import MakeSample_repeat1_node_prediction_lost_utilize_disappeared_DynGEM_InputDir as node_prediction_lost_InputDir

from setting_param import MakeSample_repeat1_link_prediction_appeared_utilize_disappeared_DynGEM_OutputDir as link_prediction_appeared_OutputDir
from setting_param import MakeSample_repeat1_link_prediction_disappeared_utilize_disappeared_DynGEM_OutputDir as link_prediction_disappeared_OutputDir
from setting_param import MakeSample_repeat1_node_prediction_lost_utilize_disappeared_DynGEM_OutputDir as node_prediction_lost_OutputDir

import numpy as np
import glob
import re
from scipy.io import mmread, mmwrite
from scipy.sparse import lil_matrix
import networkx as nx
from dynamicgem.embedding.ae_static import AE

def load_paths_from_dir(dir_path):
    # dir
    path_list = glob.glob(dir_path + "/*")
    # ー (ー)
    path_list = np.array(sorted(path_list, key=lambda s: int(re.findall(r'\d+', s)[-1])))
    return path_list

intr = './intermediate'
if not os.path.exists(intr):
    os.mkdir(intr)

# PATH
adjacency_paths = load_paths_from_dir(link_prediction_appeared_InputDir + '/input/adjacency')
appeared_label_paths = load_paths_from_dir(link_prediction_appeared_InputDir + '/label')
appeared_mask_paths = load_paths_from_dir(link_prediction_appeared_InputDir + '/mask')
disappeared_label_paths = load_paths_from_dir(link_prediction_disappeared_InputDir + '/label')
disappeared_mask_paths = load_paths_from_dir(link_prediction_disappeared_InputDir + '/mask')
lost_label_paths = load_paths_from_dir(node_prediction_lost_InputDir + '/label')
lost_mask_paths = load_paths_from_dir(node_prediction_lost_InputDir + '/mask')

n_samples = len(appeared_label_paths)
all_idx = list(range(n_samples))
target_idx = all_idx

idx_list = target_idx
adjacency = [lil_matrix(mmread(adjacency_paths[idx])) for idx in target_idx]

appeared_label = [mmread(appeared_label_paths[idx]) for idx in target_idx]
appeared_mask = [mmread(appeared_mask_paths[idx]) for idx in target_idx]
disappeared_label = [mmread(disappeared_label_paths[idx]) for idx in target_idx]
disappeared_mask = [mmread(disappeared_mask_paths[idx]) for idx in target_idx]
lost_label = [mmread(lost_label_paths[idx]) for idx in target_idx]
lost_mask = [mmread(lost_mask_paths[idx]) for idx in target_idx]

os.mkdir(link_prediction_appeared_OutputDir)
os.mkdir(link_prediction_appeared_OutputDir + "/input/")
os.mkdir(link_prediction_appeared_OutputDir + "/input/node_attribute/")
os.mkdir(link_prediction_appeared_OutputDir + "/input/adjacency")
os.mkdir(link_prediction_appeared_OutputDir + "/label/")
os.mkdir(link_prediction_appeared_OutputDir + "/mask/")

os.mkdir(link_prediction_disappeared_OutputDir)
os.mkdir(link_prediction_disappeared_OutputDir + "/input/")
os.mkdir(link_prediction_disappeared_OutputDir + "/input/node_attribute/")
os.mkdir(link_prediction_disappeared_OutputDir + "/input/adjacency")
os.mkdir(link_prediction_disappeared_OutputDir + "/label/")
os.mkdir(link_prediction_disappeared_OutputDir + "/mask/")

os.mkdir(node_prediction_lost_OutputDir)
os.mkdir(node_prediction_lost_OutputDir + "/input/")
os.mkdir(node_prediction_lost_OutputDir + "/input/node_attribute/")
os.mkdir(node_prediction_lost_OutputDir + "/input/adjacency")
os.mkdir(node_prediction_lost_OutputDir + "/label/")
os.mkdir(node_prediction_lost_OutputDir + "/mask/")

for s in range(len(adjacency)):
    save_idx = L + s
    adj = adjacency[s]
    appeared_lab = appeared_label[s]
    appeared_msk = appeared_mask[s]
    disappeared_lab = disappeared_label[s]
    disappeared_msk = disappeared_mask[s]
    lost_lab = lost_label[s]
    lost_msk = lost_mask[s]

    graphs = []
    for t in range(L):
        A = adj[:, (all_node_num + n_expanded) * t : (all_node_num + n_expanded) * (t+1)]
        A = nx.from_scipy_sparse_matrix(A)
        graphs.append(A)

    # AE Static
    embedding = AE(d=attribute_dim,
               beta=5,
               nu1=1e-6,
               nu2=1e-6,
               K=3,
               n_units=[500, 300],
               n_iter=10,
               xeta=1e-4,
               n_batch=100,
               modelfile=['./intermediate/enc_modelsbm.json',
                          './intermediate/dec_modelsbm.json'],
               weightfile=['./intermediate/enc_weightssbm.hdf5',
                           './intermediate/dec_weightssbm.hdf5'])

    embs = np.zeros(((all_node_num + n_expanded), L*attribute_dim))
    # ae static
    for temp_var in range(L):
        emb, _ = embedding.learn_embeddings(graphs[temp_var])
        embs[:, temp_var*attribute_dim : (temp_var+1)*attribute_dim] = emb

    print(save_idx)
    print(adj.shape)
    print(lil_matrix(embs).shape)

    print(appeared_lab.shape)
    print(appeared_msk.shape)
    print(disappeared_lab.shape)
    print(disappeared_msk.shape)
    print(lost_lab.shape)
    print(lost_msk.shape)

    mmwrite(link_prediction_appeared_OutputDir + "/input/node_attribute/" + str(save_idx), lil_matrix(embs))
    mmwrite(link_prediction_appeared_OutputDir + "/input/adjacency/" + str(save_idx), adj)
    mmwrite(link_prediction_appeared_OutputDir + "/label/" + str(save_idx), appeared_lab)
    mmwrite(link_prediction_appeared_OutputDir + "/mask/" + str(save_idx), appeared_msk)

    mmwrite(link_prediction_disappeared_OutputDir + "/input/node_attribute/" + str(save_idx), lil_matrix(embs))
    mmwrite(link_prediction_disappeared_OutputDir + "/input/adjacency/" + str(save_idx), adj)
    mmwrite(link_prediction_disappeared_OutputDir + "/label/" + str(save_idx), disappeared_lab)
    mmwrite(link_prediction_disappeared_OutputDir + "/mask/" + str(save_idx), disappeared_msk)

    mmwrite(node_prediction_lost_OutputDir + "/input/node_attribute/" + str(save_idx), lil_matrix(embs))
    mmwrite(node_prediction_lost_OutputDir + "/input/adjacency/" + str(save_idx), adj)
    mmwrite(node_prediction_lost_OutputDir + "/label/" + str(save_idx), lost_lab)
    mmwrite(node_prediction_lost_OutputDir + "/mask/" + str(save_idx), lost_msk)