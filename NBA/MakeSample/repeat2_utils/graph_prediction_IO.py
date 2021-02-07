import os
import sys
# moduleー
current_dir = os.path.dirname(os.path.abspath("__file__"))
sys.path.append( str(current_dir) + '/../' )


def get_appeared_InputDirs(p_appeared, n_appeared):
    from setting_param import Evaluation_repeat1_link_prediction_appeared_utilize_existing_attribute_COSSIMMLP_InputDir as appeared_COSSIMMLP_InputDir

    if p_appeared == 'repeat1_utilize_existing_attribute_COSSIMMLP':
        probability_appeared_InputDir = appeared_COSSIMMLP_InputDir

    from setting_param import Evaluation_prediction_num_of_edge_appeared_LSTM_InputDir as num_appeared_LSTM_InputDir
    from setting_param import \
        Evaluation_prediction_num_of_edge_appeared_Baseline_InputDir as num_appeared_Baseline_InputDir
    if n_appeared == 'Baseline':
        num_appeared_InputDir = num_appeared_Baseline_InputDir
    elif n_appeared == 'LSTM':
        num_appeared_InputDir = num_appeared_LSTM_InputDir

    return probability_appeared_InputDir, num_appeared_InputDir


def get_disappeared_InputDirs(p_disappeared, n_disappeared):
    from setting_param import Evaluation_repeat1_link_prediction_disappeared_utilize_new_attribute_link_DynGEM_InputDir as disappeared_DynGEM_InputDir

    if p_disappeared == 'repeat1_utilize_new_attribute_link_DynGEM':
        probability_disappeared_InputDir = disappeared_DynGEM_InputDir

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
    from setting_param import Evaluation_link_prediction_new_DEAL_PROSER_inference_InputDir as new_DEAL_PROSER_inference_InputDir

    if p_new == 'DEAL_PROSER_inference':
        probability_new_InputDir = new_DEAL_PROSER_inference_InputDir

    from setting_param import Evaluation_prediction_num_of_edge_new_LSTM_InputDir as num_new_LSTM_InputDir
    from setting_param import Evaluation_prediction_num_of_edge_new_Baseline_InputDir as num_new_Baseline_InputDir
    if n_new == 'Baseline':
        num_new_InputDir = num_new_Baseline_InputDir
    elif n_new == 'LSTM':
        num_new_InputDir = num_new_LSTM_InputDir

    return probability_new_InputDir, num_new_InputDir


def get_lost_InputDirs(p_lost, n_lost):
    from setting_param import Evaluation_repeat1_node_prediction_lost_utilize_new_attribute_link_STGGNN_InputDir as lost_STGGNN_InputDir

    if p_lost == 'repeat1_utilize_new_attribute_link_STGGNN':
        probability_lost_InputDir = lost_STGGNN_InputDir

    from setting_param import Evaluation_prediction_num_of_node_lost_LSTM_InputDir as num_lost_LSTM_InputDir
    from setting_param import Evaluation_prediction_num_of_node_lost_Baseline_InputDir as num_lost_Baseline_InputDir
    if n_lost == 'Baseline':
        num_lost_InputDir = num_lost_Baseline_InputDir
    elif n_lost == 'LSTM':
        num_lost_InputDir = num_lost_LSTM_InputDir

    return probability_lost_InputDir, num_lost_InputDir
