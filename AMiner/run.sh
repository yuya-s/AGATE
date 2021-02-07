#!/usr/bin/env bash
WORK_PATH=$(pwd)

#============================================= 0 =============================================
cd $WORK_PATH/MakeSample
python prediction_num_of_edge.py
python prediction_num_of_node.py
#============================================= 1 =============================================
cd $WORK_PATH/Model/prediction_num_of_edge/Baseline
python main.py appeared
python main.py disappeared
python main.py new
cd $WORK_PATH/Model/prediction_num_of_edge/LSTM
python main.py appeared
python main.py disappeared
python main.py new
cd $WORK_PATH/Model/prediction_num_of_node/Baseline
python main.py new
python main.py lost
cd $WORK_PATH/Model/prediction_num_of_node/LSTM
python main.py new
python main.py lost
cd $WORK_PATH/Evaluation
python prediction_num_of_edge.py appeared
python prediction_num_of_edge.py disappeared
python prediction_num_of_edge.py new
python prediction_num_of_node.py new
python prediction_num_of_node.py lost
#============================================= 2 =============================================
cd $WORK_PATH/MakeSample
python confirm_n_expanded.py
#============================================= 3 =============================================
cd $WORK_PATH/MakeSample
python link_prediction_appeared.py
cd $WORK_PATH/Model/confirm_max_nnz_am/print_max_nnz_am/
python main.py
#============================================= 4 =============================================
cd $WORK_PATH/MakeSample
python attribute_prediction_new.py
cd $WORK_PATH/Model/attribute_prediction_new/Baseline
python main.py
cd $WORK_PATH/Model/attribute_prediction_new/FNN
python main.py
cd $WORK_PATH/Model/attribute_prediction_new/DeepMatchMax
python main.py
cd $WORK_PATH/Model/attribute_prediction_new_PROSER
python attribute_prediction_new_PROSER_oracle.py
cd $WORK_PATH/MakeSample
python attribute_prediction_new_PROSER.py
cd $WORK_PATH/Model/attribute_prediction_new_PROSER/FNN
python main.py
cd $WORK_PATH/Evaluation
python attribute_prediction_new_PROSER.py
cd $WORK_PATH/Model/attribute_prediction_new_PROSER/
python selecter.py
cd $WORK_PATH/Evaluation
python attribute_prediction_new.py
#============================================= 5 =============================================
cd $WORK_PATH/MakeSample
python link_prediction_new.py
#python link_prediction_disappeared.py
#python node_prediction_lost.py
#============================================= 6 =============================================
# node_prediction_lost
#cd $WORK_PATH/Model/node_prediction_lost/Baseline
#python main.py
#cd $WORK_PATH/Model/node_prediction_lost/EGCNh
#python main.py
#cd $WORK_PATH/Model/node_prediction_lost/EGCNo
#python main.py
#cd $WORK_PATH/Model/node_prediction_lost/GCN
#python main.py
#cd $WORK_PATH/Model/node_prediction_lost/LSTM
#python main.py
#cd $WORK_PATH/Model/node_prediction_lost/Random
#python main.py
#cd $WORK_PATH/Model/node_prediction_lost/STGCN
#python main.py
#cd $WORK_PATH/Model/node_prediction_lost/STGGNN
#python main.py
#cd $WORK_PATH/Evaluation
#python node_prediction_lost.py

#link_prediction_new&
cd $WORK_PATH/Model/link_prediction_new/COSSIMMLP
#python main.py Baseline mix
#python main.py Baseline learning
python main.py Baseline inference
#python main.py FNN mix
#python main.py FNN learning
python main.py FNN inference
#python main.py DeepMatchMax mix
#python main.py DeepMatchMax learning
python main.py DeepMatchMax inference
#python main.py PROSER mix
#python main.py PROSER learning
python main.py PROSER inference
cd $WORK_PATH/Model/link_prediction_new/DEAL
python main.py Baseline inference
python main.py FNN inference
python main.py DeepMatchMax inference
python main.py PROSER inference
cd $WORK_PATH/Model/link_prediction_new/FNN
python main.py Baseline inference
python main.py FNN inference
python main.py DeepMatchMax inference
python main.py PROSER inference
cd $WORK_PATH/Evaluation
python link_prediction_new.py

#link_prediction_appeared&
#cd $WORK_PATH/Model/link_prediction_appeared/Baseline
#python main.py
#cd $WORK_PATH/Model/link_prediction_appeared/COSSIMMLP
#python main.py
#cd $WORK_PATH/Model/link_prediction_appeared/EGCNh
#python main.py
#cd $WORK_PATH/Model/link_prediction_appeared/EGCNo
#python main.py
#cd $WORK_PATH/Model/link_prediction_appeared/GCN
#python main.py
#cd $WORK_PATH/Model/link_prediction_appeared/Random
#python main.py
#cd $WORK_PATH/Model/link_prediction_appeared/STGCN
#python main.py
#cd $WORK_PATH/Model/link_prediction_appeared/STGGNN
#python main.py
#cd $WORK_PATH/Evaluation
#python link_prediction_appeared.py

#link_prediction_disappeared&
#cd $WORK_PATH/Model/link_prediction_disappeared/Baseline
#python main.py
#cd $WORK_PATH/Model/link_prediction_disappeared/COSSIMMLP
#python main.py
#cd $WORK_PATH/Model/link_prediction_disappeared/EGCNh
#python main.py
#cd $WORK_PATH/Model/link_prediction_disappeared/EGCNo
#python main.py
#cd $WORK_PATH/Model/link_prediction_disappeared/GCN
#python main.py
#cd $WORK_PATH/Model/link_prediction_disappeared/Random
#python main.py
#cd $WORK_PATH/Model/link_prediction_disappeared/STGCN
#python main.py
#cd $WORK_PATH/Model/link_prediction_disappeared/STGGNN
#python main.py
#cd $WORK_PATH/Evaluation
#python link_prediction_disappeared.py

# repeat
cd $WORK_PATH/MakeSample
#python repeat1_link_prediction_new.py
python repeat1_attribute_prediction_new.py
#cd $WORK_PATH/MakeSample/DynGEM_repeat1
#python utilize_new_attribute_link.py

cd $WORK_PATH/Model/confirm_max_nnz_am/print_max_nnz_am/
python main.py

#repeat1_link_prediction_new_utilize_new_attribute_link&
#cd $WORK_PATH/Model/repeat1_link_prediction_new_utilize_new_attribute_link/Baseline
#python main.py
#cd $WORK_PATH/Model/repeat1_link_prediction_new_utilize_new_attribute_link/COSSIMMLP
#python main.py
#cd $WORK_PATH/Model/repeat1_link_prediction_new_utilize_new_attribute_link/EGCNh
#python main.py
#cd $WORK_PATH/Model/repeat1_link_prediction_new_utilize_new_attribute_link/EGCNo
#python main.py
#cd $WORK_PATH/Model/repeat1_link_prediction_new_utilize_new_attribute_link/GCN
#python main.py
#cd $WORK_PATH/Model/repeat1_link_prediction_new_utilize_new_attribute_link/Random
#python main.py
#cd $WORK_PATH/Model/repeat1_link_prediction_new_utilize_new_attribute_link/STGCN
#python main.py
#cd $WORK_PATH/Model/repeat1_link_prediction_new_utilize_new_attribute_link/STGGNN
#python main.py
#cd $WORK_PATH/Model/repeat1_link_prediction_new_utilize_new_attribute_link/DynGEM
#python main.py
#cd $WORK_PATH/Model/repeat1_link_prediction_new_utilize_new_attribute_link/LSTM
#python main.py
#cd $WORK_PATH/Evaluation
#python repeat1_link_prediction_new_utilize_new_attribute_link.py

# repeat1_attribute_prediction_new_utilize_new_attribute_link
cd $WORK_PATH/Model/repeat1_attribute_prediction_new_utilize_new_attribute_link/Baseline
python main.py
cd $WORK_PATH/Model/repeat1_attribute_prediction_new_utilize_new_attribute_link/EGCNh
python main.py
cd $WORK_PATH/Model/repeat1_attribute_prediction_new_utilize_new_attribute_link/EGCNo
python main.py
cd $WORK_PATH/Model/repeat1_attribute_prediction_new_utilize_new_attribute_link/GCN
python main.py
cd $WORK_PATH/Model/repeat1_attribute_prediction_new_utilize_new_attribute_link/LSTM
python main.py
cd $WORK_PATH/Model/repeat1_attribute_prediction_new_utilize_new_attribute_link/STGCN
python main.py
cd $WORK_PATH/Model/repeat1_attribute_prediction_new_utilize_new_attribute_link/STGGNN
python main.py
cd $WORK_PATH/Model/repeat1_attribute_prediction_new_utilize_new_attribute_link/DynGEM
python main.py
cd $WORK_PATH/Model/repeat1_attribute_prediction_new_utilize_new_attribute_link/FNN
python main.py
cd $WORK_PATH/Evaluation
python repeat1_attribute_prediction_new_utilize_new_attribute_link.py

# repeat1_link_prediction_new_AGATE
cd $WORK_PATH/MakeSample
python repeat1_link_prediction_new_AGATE.py
cd $WORK_PATH/Model/repeat1_link_prediction_new_AGATE/COSSIMMLP
python main.py
cd $WORK_PATH/Model/repeat1_link_prediction_new_AGATE/FNN
python main.py
cd $WORK_PATH/Model/repeat1_link_prediction_new_AGATE/DEAL
python main.py
cd $WORK_PATH/Evaluation
python repeat1_link_prediction_new_AGATE.py