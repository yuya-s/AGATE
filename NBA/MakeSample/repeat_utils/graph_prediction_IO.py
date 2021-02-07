import os
import sys
# moduleー
current_dir = os.path.dirname(os.path.abspath("__file__"))
sys.path.append( str(current_dir) + '/../' )


def get_appeared_InputDirs(p_appeared, n_appeared):
    from setting_param import Evaluation_link_prediction_appeared_Baseline_InputDir as appeared_Baseline_InputDir
    from setting_param import Evaluation_link_prediction_appeared_Random_InputDir as appeared_Random_InputDir
    from setting_param import Evaluation_link_prediction_appeared_COSSIMMLP_InputDir as appeared_COSSIMMLP_InputDir
    from setting_param import Evaluation_link_prediction_appeared_LSTM_InputDir as appeared_LSTM_InputDir
    from setting_param import Evaluation_link_prediction_appeared_DynGEM_InputDir as appeared_DynGEM_InputDir
    from setting_param import Evaluation_link_prediction_appeared_STGGNN_InputDir as appeared_STGGNN_InputDir
    from setting_param import Evaluation_link_prediction_appeared_EGCNh_InputDir as appeared_EGCNh_InputDir
    from setting_param import Evaluation_link_prediction_appeared_STGCN_InputDir as appeared_STGCN_InputDir
    from setting_param import Evaluation_link_prediction_appeared_EGCNo_InputDir as appeared_EGCNo_InputDir
    from setting_param import Evaluation_link_prediction_appeared_GCN_InputDir as appeared_GCN_InputDir

    if p_appeared == 'Baseline':
        probability_appeared_InputDir = appeared_Baseline_InputDir
    elif p_appeared == 'Random':
        probability_appeared_InputDir = appeared_Random_InputDir
    elif p_appeared == 'COSSIMMLP':
        probability_appeared_InputDir = appeared_COSSIMMLP_InputDir
    elif p_appeared == 'LSTM':
        probability_appeared_InputDir = appeared_LSTM_InputDir
    elif p_appeared == 'DynGEM':
        probability_appeared_InputDir = appeared_DynGEM_InputDir
    elif p_appeared == 'STGGNN':
        probability_appeared_InputDir = appeared_STGGNN_InputDir
    elif p_appeared == 'EGCNh':
        probability_appeared_InputDir = appeared_EGCNh_InputDir
    elif p_appeared == 'STGCN':
        probability_appeared_InputDir = appeared_STGCN_InputDir
    elif p_appeared == 'EGCNo':
        probability_appeared_InputDir = appeared_EGCNo_InputDir
    elif p_appeared == 'GCN':
        probability_appeared_InputDir = appeared_GCN_InputDir

    from setting_param import Evaluation_prediction_num_of_edge_appeared_LSTM_InputDir as num_appeared_LSTM_InputDir
    from setting_param import \
        Evaluation_prediction_num_of_edge_appeared_Baseline_InputDir as num_appeared_Baseline_InputDir
    if n_appeared == 'Baseline':
        num_appeared_InputDir = num_appeared_Baseline_InputDir
    elif n_appeared == 'LSTM':
        num_appeared_InputDir = num_appeared_LSTM_InputDir

    return probability_appeared_InputDir, num_appeared_InputDir


def get_disappeared_InputDirs(p_disappeared, n_disappeared):
    from setting_param import Evaluation_link_prediction_disappeared_Baseline_InputDir as disappeared_Baseline_InputDir
    from setting_param import Evaluation_link_prediction_disappeared_Random_InputDir as disappeared_Random_InputDir
    from setting_param import Evaluation_link_prediction_disappeared_COSSIMMLP_InputDir as disappeared_COSSIMMLP_InputDir
    from setting_param import Evaluation_link_prediction_disappeared_LSTM_InputDir as disappeared_LSTM_InputDir
    from setting_param import Evaluation_link_prediction_disappeared_DynGEM_InputDir as disappeared_DynGEM_InputDir
    from setting_param import Evaluation_link_prediction_disappeared_STGGNN_InputDir as disappeared_STGGNN_InputDir
    from setting_param import Evaluation_link_prediction_disappeared_EGCNh_InputDir as disappeared_EGCNh_InputDir
    from setting_param import Evaluation_link_prediction_disappeared_STGCN_InputDir as disappeared_STGCN_InputDir
    from setting_param import Evaluation_link_prediction_disappeared_EGCNo_InputDir as disappeared_EGCNo_InputDir
    from setting_param import Evaluation_link_prediction_disappeared_GCN_InputDir as disappeared_GCN_InputDir
    from setting_param import Evaluation_repeat0_attribute_prediction_exist_Tm_binary_link_prediction_disappeared_COSSIMMLP_InputDir as disappeared_repeat0_InputDir

    if p_disappeared == 'Baseline':
        probability_disappeared_InputDir = disappeared_Baseline_InputDir
    elif p_disappeared == 'Random':
        probability_disappeared_InputDir = disappeared_Random_InputDir
    elif p_disappeared == 'COSSIMMLP':
        probability_disappeared_InputDir = disappeared_COSSIMMLP_InputDir
    elif p_disappeared == 'LSTM':
        probability_disappeared_InputDir = disappeared_LSTM_InputDir
    elif p_disappeared == 'DynGEM':
        probability_disappeared_InputDir = disappeared_DynGEM_InputDir
    elif p_disappeared == 'STGGNN':
        probability_disappeared_InputDir = disappeared_STGGNN_InputDir
    elif p_disappeared == 'EGCNh':
        probability_disappeared_InputDir = disappeared_EGCNh_InputDir
    elif p_disappeared == 'STGCN':
        probability_disappeared_InputDir = disappeared_STGCN_InputDir
    elif p_disappeared == 'EGCNo':
        probability_disappeared_InputDir = disappeared_EGCNo_InputDir
    elif p_disappeared == 'GCN':
        probability_disappeared_InputDir = disappeared_GCN_InputDir
    elif p_disappeared == 'repeat0':
        probability_disappeared_InputDir = disappeared_repeat0_InputDir

    from setting_param import \
        Evaluation_prediction_num_of_edge_disappeared_LSTM_InputDir as num_disappeared_LSTM_InputDir
    from setting_param import \
        Evaluation_prediction_num_of_edge_disappeared_Baseline_InputDir as num_disappeared_Baseline_InputDir
    if n_disappeared == 'Baseline':
        num_disappeared_InputDir = num_disappeared_Baseline_InputDir
    elif n_disappeared == 'LSTM':
        num_disappeared_InputDir = num_disappeared_LSTM_InputDir

    return probability_disappeared_InputDir, num_disappeared_InputDir


def get_new_InputDirs(p_new, n_new):
    from setting_param import \
        Evaluation_link_prediction_new_COSSIMMLP_Baseline_mix_InputDir as new_COSSIMMLP_Baseline_mix_InputDir
    from setting_param import \
        Evaluation_link_prediction_new_COSSIMMLP_Baseline_inference_InputDir as new_COSSIMMLP_Baseline_inference_InputDir
    from setting_param import \
        Evaluation_link_prediction_new_COSSIMMLP_FNN_mix_InputDir as new_COSSIMMLP_FNN_mix_InputDir
    from setting_param import \
        Evaluation_link_prediction_new_COSSIMMLP_FNN_inference_InputDir as new_COSSIMMLP_FNN_inference_InputDir
    from setting_param import \
        Evaluation_link_prediction_new_COSSIMMLP_DeepMatchMax_mix_InputDir as new_COSSIMMLP_DeepMatchMax_mix_InputDir
    from setting_param import \
        Evaluation_link_prediction_new_COSSIMMLP_DeepMatchMax_inference_InputDir as new_COSSIMMLP_DeepMatchMax_inference_InputDir
    from setting_param import \
        Evaluation_link_prediction_new_COSSIMMLP_PROSER_mix_InputDir as new_COSSIMMLP_PROSER_mix_InputDir
    from setting_param import \
        Evaluation_link_prediction_new_COSSIMMLP_PROSER_inference_InputDir as new_COSSIMMLP_PROSER_inference_InputDir
    from setting_param import \
        Evaluation_link_prediction_new_DEAL_PROSER_inference_InputDir as new_DEAL_PROSER_inference_InputDir
    if p_new == 'COSSIMMLP_Baseline_mix':
        probability_new_InputDir = new_COSSIMMLP_Baseline_mix_InputDir
    elif p_new == 'COSSIMMLP_Baseline_inference':
        probability_new_InputDir = new_COSSIMMLP_Baseline_inference_InputDir
    elif p_new == 'COSSIMMLP_FNN_mix':
        probability_new_InputDir = new_COSSIMMLP_FNN_mix_InputDir
    elif p_new == 'COSSIMMLP_FNN_inference':
        probability_new_InputDir = new_COSSIMMLP_FNN_inference_InputDir
    elif p_new == 'COSSIMMLP_DeepMatchMax_mix':
        probability_new_InputDir = new_COSSIMMLP_DeepMatchMax_mix_InputDir
    elif p_new == 'COSSIMMLP_DeepMatchMax_inference':
        probability_new_InputDir = new_COSSIMMLP_DeepMatchMax_inference_InputDir
    elif p_new == 'COSSIMMLP_PROSER_mix':
        probability_new_InputDir = new_COSSIMMLP_PROSER_mix_InputDir
    elif p_new == 'COSSIMMLP_PROSER_inference':
        probability_new_InputDir = new_COSSIMMLP_PROSER_inference_InputDir
    elif p_new == 'DEAL_PROSER_inference':
        probability_new_InputDir = new_DEAL_PROSER_inference_InputDir

    from setting_param import Evaluation_prediction_num_of_edge_new_LSTM_InputDir as num_new_LSTM_InputDir
    from setting_param import Evaluation_prediction_num_of_edge_new_Baseline_InputDir as num_new_Baseline_InputDir
    if n_new == 'Baseline':
        num_new_InputDir = num_new_Baseline_InputDir
    elif n_new == 'LSTM':
        num_new_InputDir = num_new_LSTM_InputDir

    return probability_new_InputDir, num_new_InputDir


def get_lost_InputDirs(p_lost, n_lost):
    from setting_param import Evaluation_node_prediction_lost_Baseline_InputDir as lost_Baseline_InputDir
    from setting_param import Evaluation_node_prediction_lost_Random_InputDir as lost_Random_InputDir
    from setting_param import Evaluation_node_prediction_lost_FNN_InputDir as lost_FNN_InputDir
    from setting_param import Evaluation_node_prediction_lost_LSTM_InputDir as lost_LSTM_InputDir
    from setting_param import Evaluation_node_prediction_lost_DynGEM_InputDir as lost_DynGEM_InputDir
    from setting_param import Evaluation_node_prediction_lost_STGGNN_InputDir as lost_STGGNN_InputDir
    from setting_param import Evaluation_node_prediction_lost_EGCNh_InputDir as lost_EGCNh_InputDir
    from setting_param import Evaluation_node_prediction_lost_STGCN_InputDir as lost_STGCN_InputDir
    from setting_param import Evaluation_node_prediction_lost_EGCNo_InputDir as lost_EGCNo_InputDir
    from setting_param import Evaluation_node_prediction_lost_GCN_InputDir as lost_GCN_InputDir

    if p_lost == 'Baseline':
        probability_lost_InputDir = lost_Baseline_InputDir
    elif p_lost == 'Random':
        probability_lost_InputDir = lost_Random_InputDir
    elif p_lost == 'FNN':
        probability_lost_InputDir = lost_FNN_InputDir
    elif p_lost == 'LSTM':
        probability_lost_InputDir = lost_LSTM_InputDir
    elif p_lost == 'DynGEM':
        probability_lost_InputDir = lost_DynGEM_InputDir
    elif p_lost == 'STGGNN':
        probability_lost_InputDir = lost_STGGNN_InputDir
    elif p_lost == 'EGCNh':
        probability_lost_InputDir = lost_EGCNh_InputDir
    elif p_lost == 'STGCN':
        probability_lost_InputDir = lost_STGCN_InputDir
    elif p_lost == 'EGCNo':
        probability_lost_InputDir = lost_EGCNo_InputDir
    elif p_lost == 'GCN':
        probability_lost_InputDir = lost_GCN_InputDir

    from setting_param import Evaluation_prediction_num_of_node_lost_LSTM_InputDir as num_lost_LSTM_InputDir
    from setting_param import Evaluation_prediction_num_of_node_lost_Baseline_InputDir as num_lost_Baseline_InputDir
    if n_lost == 'Baseline':
        num_lost_InputDir = num_lost_Baseline_InputDir
    elif n_lost == 'LSTM':
        num_lost_InputDir = num_lost_LSTM_InputDir

    return probability_lost_InputDir, num_lost_InputDir
