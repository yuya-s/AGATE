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
cd $WORK_PATH/Evaluation
python attribute_prediction_new.py
#============================================= 5 =============================================
cd $WORK_PATH/MakeSample
python link_prediction_new.py
python link_prediction_disappeared.py
python node_prediction_lost.py
cd $WORK_PATH/MakeSample/DynGEM
python main.py
cd $WORK_PATH/MakeSample
python attribute_prediction_exist_Age.py
cd $WORK_PATH/Model/attribute_prediction_exist_Age/Baseline
python main.py
cd $WORK_PATH/Model/attribute_prediction_exist_Age/EGCNh
python main.py
cd $WORK_PATH/Model/attribute_prediction_exist_Age/EGCNo
python main.py
cd $WORK_PATH/Model/attribute_prediction_exist_Age/GCN
python main.py
cd $WORK_PATH/Model/attribute_prediction_exist_Age/LSTM
python main.py
cd $WORK_PATH/Model/attribute_prediction_exist_Age/STGCN
python main.py
cd $WORK_PATH/Model/attribute_prediction_exist_Age/STGGNN
python main.py
cd $WORK_PATH/Evaluation
python attribute_prediction_exist_Age.py

cd $WORK_PATH/MakeSample
python attribute_prediction_exist_Pos.py
cd $WORK_PATH/Model/attribute_prediction_exist_Pos/Baseline
python main.py
cd $WORK_PATH/Model/attribute_prediction_exist_Pos/EGCNh
python main.py
cd $WORK_PATH/Model/attribute_prediction_exist_Pos/EGCNo
python main.py
cd $WORK_PATH/Model/attribute_prediction_exist_Pos/GCN
python main.py
cd $WORK_PATH/Model/attribute_prediction_exist_Pos/LSTM
python main.py
cd $WORK_PATH/Model/attribute_prediction_exist_Pos/STGCN
python main.py
cd $WORK_PATH/Model/attribute_prediction_exist_Pos/STGGNN
python main.py
cd $WORK_PATH/Evaluation
python attribute_prediction_exist_Pos.py

cd $WORK_PATH/MakeSample
python attribute_prediction_exist_PTS.py
cd $WORK_PATH/Model/attribute_prediction_exist_PTS/Baseline
python main.py
cd $WORK_PATH/Model/attribute_prediction_exist_PTS/EGCNh
python main.py
cd $WORK_PATH/Model/attribute_prediction_exist_PTS/EGCNo
python main.py
cd $WORK_PATH/Model/attribute_prediction_exist_PTS/GCN
python main.py
cd $WORK_PATH/Model/attribute_prediction_exist_PTS/LSTM
python main.py
cd $WORK_PATH/Model/attribute_prediction_exist_PTS/STGCN
python main.py
cd $WORK_PATH/Model/attribute_prediction_exist_PTS/STGGNN
python main.py
cd $WORK_PATH/Evaluation
python attribute_prediction_exist_PTS.py

cd $WORK_PATH/MakeSample
python attribute_prediction_exist_Tm.py
cd $WORK_PATH/Model/attribute_prediction_exist_Tm/Baseline
python main.py
cd $WORK_PATH/Model/attribute_prediction_exist_Tm/binary_transfer
python main.py
cd $WORK_PATH/Model/attribute_prediction_exist_Tm/EGCNh
python main.py
cd $WORK_PATH/Model/attribute_prediction_exist_Tm/EGCNo
python main.py
cd $WORK_PATH/Model/attribute_prediction_exist_Tm/GCN
python main.py
cd $WORK_PATH/Model/attribute_prediction_exist_Tm/LSTM
python main.py
cd $WORK_PATH/Model/attribute_prediction_exist_Tm/STGCN
python main.py
cd $WORK_PATH/Model/attribute_prediction_exist_Tm/STGGNN
python main.py
cd $WORK_PATH/Evaluation
python attribute_prediction_exist_Tm.py

#============================================= 6 =============================================
# node_prediction_lost
cd $WORK_PATH/Model/node_prediction_lost/Baseline
python main.py
cd $WORK_PATH/Model/node_prediction_lost/EGCNh
python main.py
cd $WORK_PATH/Model/node_prediction_lost/EGCNo
python main.py
cd $WORK_PATH/Model/node_prediction_lost/GCN
python main.py
cd $WORK_PATH/Model/node_prediction_lost/LSTM
python main.py
cd $WORK_PATH/Model/node_prediction_lost/Random
python main.py
cd $WORK_PATH/Model/node_prediction_lost/STGCN
python main.py
cd $WORK_PATH/Model/node_prediction_lost/STGGNN
python main.py
cd $WORK_PATH/Model/node_prediction_lost/DynGEM
python main.py
cd $WORK_PATH/Model/node_prediction_lost/FNN
python main.py
cd $WORK_PATH/Evaluation
python node_prediction_lost.py

#link_prediction_new&
cd $WORK_PATH/Model/link_prediction_new/COSSIMMLP
python main.py Baseline mix
python main.py Baseline learning
python main.py Baseline inference
python main.py FNN mix
python main.py FNN learning
python main.py FNN inference
python main.py DeepMatchMax mix
python main.py DeepMatchMax learning
python main.py DeepMatchMax inference
python main.py PROSER mix
python main.py PROSER learning
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
cd $WORK_PATH/Model/link_prediction_appeared/Baseline
python main.py
cd $WORK_PATH/Model/link_prediction_appeared/COSSIMMLP
python main.py
cd $WORK_PATH/Model/link_prediction_appeared/EGCNh
python main.py
cd $WORK_PATH/Model/link_prediction_appeared/EGCNo
python main.py
cd $WORK_PATH/Model/link_prediction_appeared/GCN
python main.py
cd $WORK_PATH/Model/link_prediction_appeared/Random
python main.py
cd $WORK_PATH/Model/link_prediction_appeared/STGCN
python main.py
cd $WORK_PATH/Model/link_prediction_appeared/STGGNN
python main.py
cd $WORK_PATH/Model/link_prediction_appeared/DynGEM
python main.py
cd $WORK_PATH/Model/link_prediction_appeared/LSTM
python main.py
cd $WORK_PATH/Evaluation
python link_prediction_appeared.py

#link_prediction_disappeared&
cd $WORK_PATH/Model/link_prediction_disappeared/Baseline
python main.py
cd $WORK_PATH/Model/link_prediction_disappeared/COSSIMMLP
python main.py
cd $WORK_PATH/Model/link_prediction_disappeared/EGCNh
python main.py
cd $WORK_PATH/Model/link_prediction_disappeared/EGCNo
python main.py
cd $WORK_PATH/Model/link_prediction_disappeared/GCN
python main.py
cd $WORK_PATH/Model/link_prediction_disappeared/Random
python main.py
cd $WORK_PATH/Model/link_prediction_disappeared/STGCN
python main.py
cd $WORK_PATH/Model/link_prediction_disappeared/STGGNN
python main.py
cd $WORK_PATH/Model/link_prediction_disappeared/DynGEM
python main.py
cd $WORK_PATH/Model/link_prediction_disappeared/LSTM
python main.py
cd $WORK_PATH/Evaluation
python link_prediction_disappeared.py

#attribute_prediction_exist&
# Tm_transfer
cd $WORK_PATH/Model/attribute_prediction_exist_Tm_with_transfer/Baseline
python main.py
cd $WORK_PATH/Model/attribute_prediction_exist_Tm_with_transfer/EGCNh
python main.py
cd $WORK_PATH/Model/attribute_prediction_exist_Tm_with_transfer/EGCNo
python main.py
cd $WORK_PATH/Model/attribute_prediction_exist_Tm_with_transfer/GCN
python main.py
cd $WORK_PATH/Model/attribute_prediction_exist_Tm_with_transfer/LSTM
python main.py
cd $WORK_PATH/Model/attribute_prediction_exist_Tm_with_transfer/STGCN
python main.py
cd $WORK_PATH/Model/attribute_prediction_exist_Tm_with_transfer/STGGNN
python main.py
cd $WORK_PATH/Evaluation
python attribute_prediction_exist_Tm_with_transfer.py

# repeat
cd $WORK_PATH/MakeSample
python repeat1_link_prediction_appeared.py
python repeat1_link_prediction_disappeared.py
python repeat1_node_prediction_lost.py
python repeat1_attribute_prediction_exist_PTS.py
python repeat1_attribute_prediction_exist_Tm_binary.py
python repeat1_link_prediction_new.py
python repeat1_attribute_prediction_new.py
cd $WORK_PATH/MakeSample/DynGEM_repeat1
python utilize_existing_attribute.py
python utilize_lost.py
python utilize_new_attribute_link.py
python utilize_appeared.py
python utilize_disappeared.py
python utilize_all.py

cd $WORK_PATH/Model/confirm_max_nnz_am/print_max_nnz_am/
python main.py

# repeat1_node_prediction_lost_utilize_existing_attribute
cd $WORK_PATH/Model/repeat1_node_prediction_lost_utilize_existing_attribute/Baseline
python main.py
cd $WORK_PATH/Model/repeat1_node_prediction_lost_utilize_existing_attribute/EGCNh
python main.py
cd $WORK_PATH/Model/repeat1_node_prediction_lost_utilize_existing_attribute/EGCNo
python main.py
cd $WORK_PATH/Model/repeat1_node_prediction_lost_utilize_existing_attribute/GCN
python main.py
cd $WORK_PATH/Model/repeat1_node_prediction_lost_utilize_existing_attribute/LSTM
python main.py
cd $WORK_PATH/Model/repeat1_node_prediction_lost_utilize_existing_attribute/Random
python main.py
cd $WORK_PATH/Model/repeat1_node_prediction_lost_utilize_existing_attribute/STGCN
python main.py
cd $WORK_PATH/Model/repeat1_node_prediction_lost_utilize_existing_attribute/STGGNN
python main.py
cd $WORK_PATH/Model/repeat1_node_prediction_lost_utilize_existing_attribute/DynGEM
python main.py
cd $WORK_PATH/Model/repeat1_node_prediction_lost_utilize_existing_attribute/FNN
python main.py
cd $WORK_PATH/Evaluation
python repeat1_node_prediction_lost_utilize_existing_attribute.py

# repeat1_node_prediction_lost_utilize_new_attribute_link
cd $WORK_PATH/Model/repeat1_node_prediction_lost_utilize_new_attribute_link/Baseline
python main.py
cd $WORK_PATH/Model/repeat1_node_prediction_lost_utilize_new_attribute_link/EGCNh
python main.py
cd $WORK_PATH/Model/repeat1_node_prediction_lost_utilize_new_attribute_link/EGCNo
python main.py
cd $WORK_PATH/Model/repeat1_node_prediction_lost_utilize_new_attribute_link/GCN
python main.py
cd $WORK_PATH/Model/repeat1_node_prediction_lost_utilize_new_attribute_link/LSTM
python main.py
cd $WORK_PATH/Model/repeat1_node_prediction_lost_utilize_new_attribute_link/Random
python main.py
cd $WORK_PATH/Model/repeat1_node_prediction_lost_utilize_new_attribute_link/STGCN
python main.py
cd $WORK_PATH/Model/repeat1_node_prediction_lost_utilize_new_attribute_link/STGGNN
python main.py
cd $WORK_PATH/Model/repeat1_node_prediction_lost_utilize_new_attribute_link/DynGEM
python main.py
cd $WORK_PATH/Model/repeat1_node_prediction_lost_utilize_new_attribute_link/FNN
python main.py
cd $WORK_PATH/Evaluation
python repeat1_node_prediction_lost_utilize_new_attribute_link.py

# repeat1_node_prediction_lost_utilize_appeared
cd $WORK_PATH/Model/repeat1_node_prediction_lost_utilize_appeared/Baseline
python main.py
cd $WORK_PATH/Model/repeat1_node_prediction_lost_utilize_appeared/EGCNh
python main.py
cd $WORK_PATH/Model/repeat1_node_prediction_lost_utilize_appeared/EGCNo
python main.py
cd $WORK_PATH/Model/repeat1_node_prediction_lost_utilize_appeared/GCN
python main.py
cd $WORK_PATH/Model/repeat1_node_prediction_lost_utilize_appeared/LSTM
python main.py
cd $WORK_PATH/Model/repeat1_node_prediction_lost_utilize_appeared/Random
python main.py
cd $WORK_PATH/Model/repeat1_node_prediction_lost_utilize_appeared/STGCN
python main.py
cd $WORK_PATH/Model/repeat1_node_prediction_lost_utilize_appeared/STGGNN
python main.py
cd $WORK_PATH/Model/repeat1_node_prediction_lost_utilize_appeared/DynGEM
python main.py
cd $WORK_PATH/Model/repeat1_node_prediction_lost_utilize_appeared/FNN
python main.py
cd $WORK_PATH/Evaluation
python repeat1_node_prediction_lost_utilize_appeared.py

# repeat1_node_prediction_lost_utilize_disappeared
cd $WORK_PATH/Model/repeat1_node_prediction_lost_utilize_disappeared/Baseline
python main.py
cd $WORK_PATH/Model/repeat1_node_prediction_lost_utilize_disappeared/EGCNh
python main.py
cd $WORK_PATH/Model/repeat1_node_prediction_lost_utilize_disappeared/EGCNo
python main.py
cd $WORK_PATH/Model/repeat1_node_prediction_lost_utilize_disappeared/GCN
python main.py
cd $WORK_PATH/Model/repeat1_node_prediction_lost_utilize_disappeared/LSTM
python main.py
cd $WORK_PATH/Model/repeat1_node_prediction_lost_utilize_disappeared/Random
python main.py
cd $WORK_PATH/Model/repeat1_node_prediction_lost_utilize_disappeared/STGCN
python main.py
cd $WORK_PATH/Model/repeat1_node_prediction_lost_utilize_disappeared/STGGNN
python main.py
cd $WORK_PATH/Model/repeat1_node_prediction_lost_utilize_disappeared/DynGEM
python main.py
cd $WORK_PATH/Model/repeat1_node_prediction_lost_utilize_disappeared/FNN
python main.py
cd $WORK_PATH/Evaluation
python repeat1_node_prediction_lost_utilize_disappeared.py

#repeat1_link_prediction_appeared_utilize_disappeared&
cd $WORK_PATH/Model/repeat1_link_prediction_appeared_utilize_disappeared/Baseline
python main.py
cd $WORK_PATH/Model/repeat1_link_prediction_appeared_utilize_disappeared/COSSIMMLP
python main.py
cd $WORK_PATH/Model/repeat1_link_prediction_appeared_utilize_disappeared/EGCNh
python main.py
cd $WORK_PATH/Model/repeat1_link_prediction_appeared_utilize_disappeared/EGCNo
python main.py
cd $WORK_PATH/Model/repeat1_link_prediction_appeared_utilize_disappeared/GCN
python main.py
cd $WORK_PATH/Model/repeat1_link_prediction_appeared_utilize_disappeared/Random
python main.py
cd $WORK_PATH/Model/repeat1_link_prediction_appeared_utilize_disappeared/STGCN
python main.py
cd $WORK_PATH/Model/repeat1_link_prediction_appeared_utilize_disappeared/STGGNN
python main.py
cd $WORK_PATH/Model/repeat1_link_prediction_appeared_utilize_disappeared/DynGEM
python main.py
cd $WORK_PATH/Model/repeat1_link_prediction_appeared_utilize_disappeared/LSTM
python main.py
cd $WORK_PATH/Evaluation
python repeat1_link_prediction_appeared_utilize_disappeared.py

#repeat1_link_prediction_appeared_utilize_existing_attribute&
cd $WORK_PATH/Model/repeat1_link_prediction_appeared_utilize_existing_attribute/Baseline
python main.py
cd $WORK_PATH/Model/repeat1_link_prediction_appeared_utilize_existing_attribute/COSSIMMLP
python main.py
cd $WORK_PATH/Model/repeat1_link_prediction_appeared_utilize_existing_attribute/EGCNh
python main.py
cd $WORK_PATH/Model/repeat1_link_prediction_appeared_utilize_existing_attribute/EGCNo
python main.py
cd $WORK_PATH/Model/repeat1_link_prediction_appeared_utilize_existing_attribute/GCN
python main.py
cd $WORK_PATH/Model/repeat1_link_prediction_appeared_utilize_existing_attribute/Random
python main.py
cd $WORK_PATH/Model/repeat1_link_prediction_appeared_utilize_existing_attribute/STGCN
python main.py
cd $WORK_PATH/Model/repeat1_link_prediction_appeared_utilize_existing_attribute/STGGNN
python main.py
cd $WORK_PATH/Model/repeat1_link_prediction_appeared_utilize_existing_attribute/DynGEM
python main.py
cd $WORK_PATH/Model/repeat1_link_prediction_appeared_utilize_existing_attribute/LSTM
python main.py
cd $WORK_PATH/Evaluation
python repeat1_link_prediction_appeared_utilize_existing_attribute.py

#repeat1_link_prediction_appeared_utilize_lost&
cd $WORK_PATH/Model/repeat1_link_prediction_appeared_utilize_lost/Baseline
python main.py
cd $WORK_PATH/Model/repeat1_link_prediction_appeared_utilize_lost/COSSIMMLP
python main.py
cd $WORK_PATH/Model/repeat1_link_prediction_appeared_utilize_lost/EGCNh
python main.py
cd $WORK_PATH/Model/repeat1_link_prediction_appeared_utilize_lost/EGCNo
python main.py
cd $WORK_PATH/Model/repeat1_link_prediction_appeared_utilize_lost/GCN
python main.py
cd $WORK_PATH/Model/repeat1_link_prediction_appeared_utilize_lost/Random
python main.py
cd $WORK_PATH/Model/repeat1_link_prediction_appeared_utilize_lost/STGCN
python main.py
cd $WORK_PATH/Model/repeat1_link_prediction_appeared_utilize_lost/STGGNN
python main.py
cd $WORK_PATH/Model/repeat1_link_prediction_appeared_utilize_lost/DynGEM
python main.py
cd $WORK_PATH/Model/repeat1_link_prediction_appeared_utilize_lost/LSTM
python main.py
cd $WORK_PATH/Evaluation
python repeat1_link_prediction_appeared_utilize_lost.py

#repeat1_link_prediction_appeared_utilize_new_attribute_link&
cd $WORK_PATH/Model/repeat1_link_prediction_appeared_utilize_new_attribute_link/Baseline
python main.py
cd $WORK_PATH/Model/repeat1_link_prediction_appeared_utilize_new_attribute_link/COSSIMMLP
python main.py
cd $WORK_PATH/Model/repeat1_link_prediction_appeared_utilize_new_attribute_link/EGCNh
python main.py
cd $WORK_PATH/Model/repeat1_link_prediction_appeared_utilize_new_attribute_link/EGCNo
python main.py
cd $WORK_PATH/Model/repeat1_link_prediction_appeared_utilize_new_attribute_link/GCN
python main.py
cd $WORK_PATH/Model/repeat1_link_prediction_appeared_utilize_new_attribute_link/Random
python main.py
cd $WORK_PATH/Model/repeat1_link_prediction_appeared_utilize_new_attribute_link/STGCN
python main.py
cd $WORK_PATH/Model/repeat1_link_prediction_appeared_utilize_new_attribute_link/STGGNN
python main.py
cd $WORK_PATH/Model/repeat1_link_prediction_appeared_utilize_new_attribute_link/DynGEM
python main.py
cd $WORK_PATH/Model/repeat1_link_prediction_appeared_utilize_new_attribute_link/LSTM
python main.py
cd $WORK_PATH/Evaluation
python repeat1_link_prediction_appeared_utilize_new_attribute_link.py

#repeat1_link_prediction_disappeared_utilize_appeared&
cd $WORK_PATH/Model/repeat1_link_prediction_disappeared_utilize_appeared/Baseline
python main.py
cd $WORK_PATH/Model/repeat1_link_prediction_disappeared_utilize_appeared/COSSIMMLP
python main.py
cd $WORK_PATH/Model/repeat1_link_prediction_disappeared_utilize_appeared/EGCNh
python main.py
cd $WORK_PATH/Model/repeat1_link_prediction_disappeared_utilize_appeared/EGCNo
python main.py
cd $WORK_PATH/Model/repeat1_link_prediction_disappeared_utilize_appeared/GCN
python main.py
cd $WORK_PATH/Model/repeat1_link_prediction_disappeared_utilize_appeared/Random
python main.py
cd $WORK_PATH/Model/repeat1_link_prediction_disappeared_utilize_appeared/STGCN
python main.py
cd $WORK_PATH/Model/repeat1_link_prediction_disappeared_utilize_appeared/STGGNN
python main.py
cd $WORK_PATH/Model/repeat1_link_prediction_disappeared_utilize_appeared/DynGEM
python main.py
cd $WORK_PATH/Model/repeat1_link_prediction_disappeared_utilize_appeared/LSTM
python main.py
cd $WORK_PATH/Evaluation
python repeat1_link_prediction_disappeared_utilize_appeared.py

#repeat1_link_prediction_disappeared_utilize_existing_attribute&
cd $WORK_PATH/Model/repeat1_link_prediction_disappeared_utilize_existing_attribute/Baseline
python main.py
cd $WORK_PATH/Model/repeat1_link_prediction_disappeared_utilize_existing_attribute/COSSIMMLP
python main.py
cd $WORK_PATH/Model/repeat1_link_prediction_disappeared_utilize_existing_attribute/EGCNh
python main.py
cd $WORK_PATH/Model/repeat1_link_prediction_disappeared_utilize_existing_attribute/EGCNo
python main.py
cd $WORK_PATH/Model/repeat1_link_prediction_disappeared_utilize_existing_attribute/GCN
python main.py
cd $WORK_PATH/Model/repeat1_link_prediction_disappeared_utilize_existing_attribute/Random
python main.py
cd $WORK_PATH/Model/repeat1_link_prediction_disappeared_utilize_existing_attribute/STGCN
python main.py
cd $WORK_PATH/Model/repeat1_link_prediction_disappeared_utilize_existing_attribute/STGGNN
python main.py
cd $WORK_PATH/Model/repeat1_link_prediction_disappeared_utilize_existing_attribute/DynGEM
python main.py
cd $WORK_PATH/Model/repeat1_link_prediction_disappeared_utilize_existing_attribute/LSTM
python main.py
cd $WORK_PATH/Evaluation
python repeat1_link_prediction_disappeared_utilize_existing_attribute.py

#repeat1_link_prediction_disappeared_utilize_lost&
cd $WORK_PATH/Model/repeat1_link_prediction_disappeared_utilize_lost/Baseline
python main.py
cd $WORK_PATH/Model/repeat1_link_prediction_disappeared_utilize_lost/COSSIMMLP
python main.py
cd $WORK_PATH/Model/repeat1_link_prediction_disappeared_utilize_lost/EGCNh
python main.py
cd $WORK_PATH/Model/repeat1_link_prediction_disappeared_utilize_lost/EGCNo
python main.py
cd $WORK_PATH/Model/repeat1_link_prediction_disappeared_utilize_lost/GCN
python main.py
cd $WORK_PATH/Model/repeat1_link_prediction_disappeared_utilize_lost/Random
python main.py
cd $WORK_PATH/Model/repeat1_link_prediction_disappeared_utilize_lost/STGCN
python main.py
cd $WORK_PATH/Model/repeat1_link_prediction_disappeared_utilize_lost/STGGNN
python main.py
cd $WORK_PATH/Model/repeat1_link_prediction_disappeared_utilize_lost/DynGEM
python main.py
cd $WORK_PATH/Model/repeat1_link_prediction_disappeared_utilize_lost/LSTM
python main.py
cd $WORK_PATH/Evaluation
python repeat1_link_prediction_disappeared_utilize_lost.py

#repeat1_link_prediction_disappeared_utilize_new_attribute_link&
cd $WORK_PATH/Model/repeat1_link_prediction_disappeared_utilize_new_attribute_link/Baseline
python main.py
cd $WORK_PATH/Model/repeat1_link_prediction_disappeared_utilize_new_attribute_link/COSSIMMLP
python main.py
cd $WORK_PATH/Model/repeat1_link_prediction_disappeared_utilize_new_attribute_link/EGCNh
python main.py
cd $WORK_PATH/Model/repeat1_link_prediction_disappeared_utilize_new_attribute_link/EGCNo
python main.py
cd $WORK_PATH/Model/repeat1_link_prediction_disappeared_utilize_new_attribute_link/GCN
python main.py
cd $WORK_PATH/Model/repeat1_link_prediction_disappeared_utilize_new_attribute_link/Random
python main.py
cd $WORK_PATH/Model/repeat1_link_prediction_disappeared_utilize_new_attribute_link/STGCN
python main.py
cd $WORK_PATH/Model/repeat1_link_prediction_disappeared_utilize_new_attribute_link/STGGNN
python main.py
cd $WORK_PATH/Model/repeat1_link_prediction_disappeared_utilize_new_attribute_link/DynGEM
python main.py
cd $WORK_PATH/Model/repeat1_link_prediction_disappeared_utilize_new_attribute_link/LSTM
python main.py
cd $WORK_PATH/Evaluation
python repeat1_link_prediction_disappeared_utilize_new_attribute_link.py


#  all
#repeat1_link_prediction_appeared_utilize_all&
cd $WORK_PATH/Model/repeat1_link_prediction_appeared_utilize_all/COSSIMMLP
python main.py
cd $WORK_PATH/Evaluation
python repeat1_link_prediction_appeared_utilize_all.py

#repeat1_link_prediction_disappeared_utilize_all&
cd $WORK_PATH/Model/repeat1_link_prediction_disappeared_utilize_all/DynGEM
python main.py
cd $WORK_PATH/Evaluation
python repeat1_link_prediction_disappeared_utilize_all.py

# repeat1_node_prediction_lost_utilize_all
cd $WORK_PATH/Model/repeat1_node_prediction_lost_utilize_all/STGGNN
python main.py
cd $WORK_PATH/Evaluation
python repeat1_node_prediction_lost_utilize_all.py

# repeat1_attribute_prediction_exist_PTS_utilize_all
cd $WORK_PATH/Model/repeat1_attribute_prediction_exist_PTS_utilize_all/LSTM
python main.py
cd $WORK_PATH/Evaluation
python repeat1_attribute_prediction_exist_PTS_utilize_all.py

# repeat1_attribute_prediction_exist_Tm_binary_utilize_all
cd $WORK_PATH/Model/repeat1_attribute_prediction_exist_Tm_binary_utilize_all/STGGNN
python main.py
cd $WORK_PATH/Evaluation
python repeat1_attribute_prediction_exist_Tm_binary_utilize_all.py

#repeat1_link_prediction_new_utilize_new_attribute_link& ()
cd $WORK_PATH/Model/repeat1_link_prediction_new_utilize_new_attribute_link/Baseline
python main.py
cd $WORK_PATH/Model/repeat1_link_prediction_new_utilize_new_attribute_link/COSSIMMLP
python main.py
cd $WORK_PATH/Model/repeat1_link_prediction_new_utilize_new_attribute_link/EGCNh
python main.py
cd $WORK_PATH/Model/repeat1_link_prediction_new_utilize_new_attribute_link/EGCNo
python main.py
cd $WORK_PATH/Model/repeat1_link_prediction_new_utilize_new_attribute_link/GCN
python main.py
cd $WORK_PATH/Model/repeat1_link_prediction_new_utilize_new_attribute_link/Random
python main.py
cd $WORK_PATH/Model/repeat1_link_prediction_new_utilize_new_attribute_link/STGCN
python main.py
cd $WORK_PATH/Model/repeat1_link_prediction_new_utilize_new_attribute_link/STGGNN
python main.py
cd $WORK_PATH/Model/repeat1_link_prediction_new_utilize_new_attribute_link/DynGEM
python main.py
cd $WORK_PATH/Model/repeat1_link_prediction_new_utilize_new_attribute_link/LSTM
python main.py
cd $WORK_PATH/Evaluation
python repeat1_link_prediction_new_utilize_new_attribute_link.py

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

cd $WORK_PATH/MakeSample
python repeat1_link_prediction_new_AGATE.py

#repeat1_link_prediction_new_AGATE
cd $WORK_PATH/Model/repeat1_link_prediction_new_AGATE/COSSIMMLP
python main.py
cd $WORK_PATH/Model/repeat1_link_prediction_new_AGATE/DEAL
python main.py
cd $WORK_PATH/Model/repeat1_link_prediction_new_AGATE/FNN
python main.py
cd $WORK_PATH/Evaluation
python repeat1_link_prediction_new_AGATE.py



# repeat2
cd $WORK_PATH/MakeSample
python repeat2_link_prediction_appeared.py
python repeat2_link_prediction_disappeared.py
python repeat2_node_prediction_lost.py
python repeat2_attribute_prediction_exist_PTS.py
python repeat2_attribute_prediction_exist_Tm_binary.py
python repeat2_attribute_prediction_new.py
cd $WORK_PATH/MakeSample/DynGEM_repeat2
python utilize_all.py

#repeat2_link_prediction_appeared_utilize_all&
cd $WORK_PATH/Model/repeat2_link_prediction_appeared_utilize_all/COSSIMMLP
python main.py
cd $WORK_PATH/Evaluation
python repeat2_link_prediction_appeared_utilize_all.py

#repeat2_link_prediction_disappeared_utilize_all&
cd $WORK_PATH/Model/repeat2_link_prediction_disappeared_utilize_all/DynGEM
python main.py
cd $WORK_PATH/Evaluation
python repeat2_link_prediction_disappeared_utilize_all.py

# repeat2_node_prediction_lost_utilize_all
cd $WORK_PATH/Model/repeat2_node_prediction_lost_utilize_all/STGGNN
python main.py
cd $WORK_PATH/Evaluation
python repeat2_node_prediction_lost_utilize_all.py

# repeat2_attribute_prediction_exist_PTS_utilize_all
cd $WORK_PATH/Model/repeat2_attribute_prediction_exist_PTS_utilize_all/LSTM
python main.py
cd $WORK_PATH/Evaluation
python repeat2_attribute_prediction_exist_PTS_utilize_all.py

# repeat2_attribute_prediction_exist_Tm_binary_utilize_all
cd $WORK_PATH/Model/repeat2_attribute_prediction_exist_Tm_binary_utilize_all/STGGNN
python main.py
cd $WORK_PATH/Evaluation
python repeat2_attribute_prediction_exist_Tm_binary_utilize_all.py

# repeat2_attribute_prediction_new_utilize_new_attribute_link
cd $WORK_PATH/Model/repeat2_attribute_prediction_new_utilize_new_attribute_link/LSTM
python main.py
cd $WORK_PATH/Evaluation
python repeat2_attribute_prediction_new_utilize_new_attribute_link.py

cd $WORK_PATH/MakeSample
python repeat2_link_prediction_new_AGATE.py

repeat2_link_prediction_new_AGATE
cd $WORK_PATH/Model/repeat2_link_prediction_new_AGATE/DEAL
python main.py
cd $WORK_PATH/Evaluation
python repeat2_link_prediction_new_AGATE.py


# repeat3
cd $WORK_PATH/MakeSample
python repeat3_link_prediction_appeared.py
python repeat3_link_prediction_disappeared.py
python repeat3_node_prediction_lost.py
python repeat3_attribute_prediction_exist_PTS.py
python repeat3_attribute_prediction_exist_Tm_binary.py
python repeat3_attribute_prediction_new.py
cd $WORK_PATH/MakeSample/DynGEM_repeat3
python utilize_all.py

#repeat3_link_prediction_appeared_utilize_all&
cd $WORK_PATH/Model/repeat3_link_prediction_appeared_utilize_all/COSSIMMLP
python main.py
cd $WORK_PATH/Evaluation
python repeat3_link_prediction_appeared_utilize_all.py

#repeat3_link_prediction_disappeared_utilize_all&
cd $WORK_PATH/Model/repeat3_link_prediction_disappeared_utilize_all/DynGEM
python main.py
cd $WORK_PATH/Evaluation
python repeat3_link_prediction_disappeared_utilize_all.py

# repeat3_node_prediction_lost_utilize_all
cd $WORK_PATH/Model/repeat3_node_prediction_lost_utilize_all/STGGNN
python main.py
cd $WORK_PATH/Evaluation
python repeat3_node_prediction_lost_utilize_all.py

# repeat3_attribute_prediction_exist_PTS_utilize_all
cd $WORK_PATH/Model/repeat3_attribute_prediction_exist_PTS_utilize_all/LSTM
python main.py
cd $WORK_PATH/Evaluation
python repeat3_attribute_prediction_exist_PTS_utilize_all.py

# repeat3_attribute_prediction_exist_Tm_binary_utilize_all
cd $WORK_PATH/Model/repeat3_attribute_prediction_exist_Tm_binary_utilize_all/STGGNN
python main.py
cd $WORK_PATH/Evaluation
python repeat3_attribute_prediction_exist_Tm_binary_utilize_all.py

# repeat3_attribute_prediction_new_utilize_new_attribute_link
cd $WORK_PATH/Model/repeat3_attribute_prediction_new_utilize_new_attribute_link/LSTM
python main.py
cd $WORK_PATH/Evaluation
python repeat3_attribute_prediction_new_utilize_new_attribute_link.py

cd $WORK_PATH/MakeSample
python repeat3_link_prediction_new_AGATE.py

repeat3_link_prediction_new_AGATE
cd $WORK_PATH/Model/repeat3_link_prediction_new_AGATE/DEAL
python main.py
cd $WORK_PATH/Evaluation
python repeat3_link_prediction_new_AGATE.py



# repeat1_utilize_all
# repeat1_node_prediction_lost_utilize_all
cd $WORK_PATH/Model/repeat1_node_prediction_lost_utilize_all/Baseline
python main.py
cd $WORK_PATH/Model/repeat1_node_prediction_lost_utilize_all/EGCNh
python main.py
cd $WORK_PATH/Model/repeat1_node_prediction_lost_utilize_all/EGCNo
python main.py
cd $WORK_PATH/Model/repeat1_node_prediction_lost_utilize_all/GCN
python main.py
cd $WORK_PATH/Model/repeat1_node_prediction_lost_utilize_all/LSTM
python main.py
cd $WORK_PATH/Model/repeat1_node_prediction_lost_utilize_all/Random
python main.py
cd $WORK_PATH/Model/repeat1_node_prediction_lost_utilize_all/STGCN
python main.py
cd $WORK_PATH/Model/repeat1_node_prediction_lost_utilize_all/STGGNN
python main.py
cd $WORK_PATH/Model/repeat1_node_prediction_lost_utilize_all/DynGEM
python main.py
cd $WORK_PATH/Model/repeat1_node_prediction_lost_utilize_all/FNN
python main.py
cd $WORK_PATH/Evaluation
python repeat1_node_prediction_lost_utilize_all.py

#repeat1_link_prediction_appeared_utilize_all&
cd $WORK_PATH/Model/repeat1_link_prediction_appeared_utilize_all/Baseline
python main.py
cd $WORK_PATH/Model/repeat1_link_prediction_appeared_utilize_all/COSSIMMLP
python main.py
cd $WORK_PATH/Model/repeat1_link_prediction_appeared_utilize_all/EGCNh
python main.py
cd $WORK_PATH/Model/repeat1_link_prediction_appeared_utilize_all/EGCNo
python main.py
cd $WORK_PATH/Model/repeat1_link_prediction_appeared_utilize_all/GCN
python main.py
cd $WORK_PATH/Model/repeat1_link_prediction_appeared_utilize_all/Random
python main.py
cd $WORK_PATH/Model/repeat1_link_prediction_appeared_utilize_all/STGCN
python main.py
cd $WORK_PATH/Model/repeat1_link_prediction_appeared_utilize_all/STGGNN
python main.py
cd $WORK_PATH/Model/repeat1_link_prediction_appeared_utilize_all/DynGEM
python main.py
cd $WORK_PATH/Model/repeat1_link_prediction_appeared_utilize_all/LSTM
python main.py
cd $WORK_PATH/Evaluation
python repeat1_link_prediction_appeared_utilize_all.py

#repeat1_link_prediction_disappeared_utilize_all&
cd $WORK_PATH/Model/repeat1_link_prediction_disappeared_utilize_all/Baseline
python main.py
cd $WORK_PATH/Model/repeat1_link_prediction_disappeared_utilize_all/COSSIMMLP
python main.py
cd $WORK_PATH/Model/repeat1_link_prediction_disappeared_utilize_all/EGCNh
python main.py
cd $WORK_PATH/Model/repeat1_link_prediction_disappeared_utilize_all/EGCNo
python main.py
cd $WORK_PATH/Model/repeat1_link_prediction_disappeared_utilize_all/GCN
python main.py
cd $WORK_PATH/Model/repeat1_link_prediction_disappeared_utilize_all/Random
python main.py
cd $WORK_PATH/Model/repeat1_link_prediction_disappeared_utilize_all/STGCN
python main.py
cd $WORK_PATH/Model/repeat1_link_prediction_disappeared_utilize_all/STGGNN
python main.py
cd $WORK_PATH/Model/repeat1_link_prediction_disappeared_utilize_all/DynGEM
python main.py
cd $WORK_PATH/Model/repeat1_link_prediction_disappeared_utilize_all/LSTM
python main.py
cd $WORK_PATH/Evaluation
python repeat1_link_prediction_disappeared_utilize_all.py

#  existing node

# repeat1_attribute_prediction_exist_PTS_utilize_appeared
cd $WORK_PATH/Model/repeat1_attribute_prediction_exist_PTS_utilize_appeared/Baseline
python main.py
cd $WORK_PATH/Model/repeat1_attribute_prediction_exist_PTS_utilize_appeared/EGCNh
python main.py
cd $WORK_PATH/Model/repeat1_attribute_prediction_exist_PTS_utilize_appeared/EGCNo
python main.py
cd $WORK_PATH/Model/repeat1_attribute_prediction_exist_PTS_utilize_appeared/GCN
python main.py
cd $WORK_PATH/Model/repeat1_attribute_prediction_exist_PTS_utilize_appeared/LSTM
python main.py
cd $WORK_PATH/Model/repeat1_attribute_prediction_exist_PTS_utilize_appeared/Random
python main.py
cd $WORK_PATH/Model/repeat1_attribute_prediction_exist_PTS_utilize_appeared/STGCN
python main.py
cd $WORK_PATH/Model/repeat1_attribute_prediction_exist_PTS_utilize_appeared/STGGNN
python main.py
cd $WORK_PATH/Model/repeat1_attribute_prediction_exist_PTS_utilize_appeared/DynGEM
python main.py
cd $WORK_PATH/Model/repeat1_attribute_prediction_exist_PTS_utilize_appeared/FNN
python main.py
cd $WORK_PATH/Evaluation
python repeat1_attribute_prediction_exist_PTS_utilize_appeared.py

# repeat1_attribute_prediction_exist_PTS_utilize_disappeared
cd $WORK_PATH/Model/repeat1_attribute_prediction_exist_PTS_utilize_disappeared/Baseline
python main.py
cd $WORK_PATH/Model/repeat1_attribute_prediction_exist_PTS_utilize_disappeared/EGCNh
python main.py
cd $WORK_PATH/Model/repeat1_attribute_prediction_exist_PTS_utilize_disappeared/EGCNo
python main.py
cd $WORK_PATH/Model/repeat1_attribute_prediction_exist_PTS_utilize_disappeared/GCN
python main.py
cd $WORK_PATH/Model/repeat1_attribute_prediction_exist_PTS_utilize_disappeared/LSTM
python main.py
cd $WORK_PATH/Model/repeat1_attribute_prediction_exist_PTS_utilize_disappeared/Random
python main.py
cd $WORK_PATH/Model/repeat1_attribute_prediction_exist_PTS_utilize_disappeared/STGCN
python main.py
cd $WORK_PATH/Model/repeat1_attribute_prediction_exist_PTS_utilize_disappeared/STGGNN
python main.py
cd $WORK_PATH/Model/repeat1_attribute_prediction_exist_PTS_utilize_disappeared/DynGEM
python main.py
cd $WORK_PATH/Model/repeat1_attribute_prediction_exist_PTS_utilize_disappeared/FNN
python main.py
cd $WORK_PATH/Evaluation
python repeat1_attribute_prediction_exist_PTS_utilize_disappeared.py

# repeat1_attribute_prediction_exist_PTS_utilize_lost
cd $WORK_PATH/Model/repeat1_attribute_prediction_exist_PTS_utilize_lost/Baseline
python main.py
cd $WORK_PATH/Model/repeat1_attribute_prediction_exist_PTS_utilize_lost/EGCNh
python main.py
cd $WORK_PATH/Model/repeat1_attribute_prediction_exist_PTS_utilize_lost/EGCNo
python main.py
cd $WORK_PATH/Model/repeat1_attribute_prediction_exist_PTS_utilize_lost/GCN
python main.py
cd $WORK_PATH/Model/repeat1_attribute_prediction_exist_PTS_utilize_lost/LSTM
python main.py
cd $WORK_PATH/Model/repeat1_attribute_prediction_exist_PTS_utilize_lost/Random
python main.py
cd $WORK_PATH/Model/repeat1_attribute_prediction_exist_PTS_utilize_lost/STGCN
python main.py
cd $WORK_PATH/Model/repeat1_attribute_prediction_exist_PTS_utilize_lost/STGGNN
python main.py
cd $WORK_PATH/Model/repeat1_attribute_prediction_exist_PTS_utilize_lost/DynGEM
python main.py
cd $WORK_PATH/Model/repeat1_attribute_prediction_exist_PTS_utilize_lost/FNN
python main.py
cd $WORK_PATH/Evaluation
python repeat1_attribute_prediction_exist_PTS_utilize_lost.py

# repeat1_attribute_prediction_exist_PTS_utilize_new_attribute_link
cd $WORK_PATH/Model/repeat1_attribute_prediction_exist_PTS_utilize_new_attribute_link/Baseline
python main.py
cd $WORK_PATH/Model/repeat1_attribute_prediction_exist_PTS_utilize_new_attribute_link/EGCNh
python main.py
cd $WORK_PATH/Model/repeat1_attribute_prediction_exist_PTS_utilize_new_attribute_link/EGCNo
python main.py
cd $WORK_PATH/Model/repeat1_attribute_prediction_exist_PTS_utilize_new_attribute_link/GCN
python main.py
cd $WORK_PATH/Model/repeat1_attribute_prediction_exist_PTS_utilize_new_attribute_link/LSTM
python main.py
cd $WORK_PATH/Model/repeat1_attribute_prediction_exist_PTS_utilize_new_attribute_link/Random
python main.py
cd $WORK_PATH/Model/repeat1_attribute_prediction_exist_PTS_utilize_new_attribute_link/STGCN
python main.py
cd $WORK_PATH/Model/repeat1_attribute_prediction_exist_PTS_utilize_new_attribute_link/STGGNN
python main.py
cd $WORK_PATH/Model/repeat1_attribute_prediction_exist_PTS_utilize_new_attribute_link/DynGEM
python main.py
cd $WORK_PATH/Model/repeat1_attribute_prediction_exist_PTS_utilize_new_attribute_link/FNN
python main.py
cd $WORK_PATH/Evaluation
python repeat1_attribute_prediction_exist_PTS_utilize_new_attribute_link.py

# repeat1_attribute_prediction_exist_PTS_utilize_all
cd $WORK_PATH/Model/repeat1_attribute_prediction_exist_PTS_utilize_all/Baseline
python main.py
cd $WORK_PATH/Model/repeat1_attribute_prediction_exist_PTS_utilize_all/EGCNh
python main.py
cd $WORK_PATH/Model/repeat1_attribute_prediction_exist_PTS_utilize_all/EGCNo
python main.py
cd $WORK_PATH/Model/repeat1_attribute_prediction_exist_PTS_utilize_all/GCN
python main.py
cd $WORK_PATH/Model/repeat1_attribute_prediction_exist_PTS_utilize_all/LSTM
python main.py
cd $WORK_PATH/Model/repeat1_attribute_prediction_exist_PTS_utilize_all/Random
python main.py
cd $WORK_PATH/Model/repeat1_attribute_prediction_exist_PTS_utilize_all/STGCN
python main.py
cd $WORK_PATH/Model/repeat1_attribute_prediction_exist_PTS_utilize_all/STGGNN
python main.py
cd $WORK_PATH/Model/repeat1_attribute_prediction_exist_PTS_utilize_all/DynGEM
python main.py
cd $WORK_PATH/Model/repeat1_attribute_prediction_exist_PTS_utilize_all/FNN
python main.py
cd $WORK_PATH/Evaluation
python repeat1_attribute_prediction_exist_PTS_utilize_all.py

# repeat1_attribute_prediction_exist_Tm_binary_utilize_appeared
cd $WORK_PATH/Model/repeat1_attribute_prediction_exist_Tm_binary_utilize_appeared/Baseline
python main.py
cd $WORK_PATH/Model/repeat1_attribute_prediction_exist_Tm_binary_utilize_appeared/EGCNh
python main.py
cd $WORK_PATH/Model/repeat1_attribute_prediction_exist_Tm_binary_utilize_appeared/EGCNo
python main.py
cd $WORK_PATH/Model/repeat1_attribute_prediction_exist_Tm_binary_utilize_appeared/GCN
python main.py
cd $WORK_PATH/Model/repeat1_attribute_prediction_exist_Tm_binary_utilize_appeared/LSTM
python main.py
cd $WORK_PATH/Model/repeat1_attribute_prediction_exist_Tm_binary_utilize_appeared/Random
python main.py
cd $WORK_PATH/Model/repeat1_attribute_prediction_exist_Tm_binary_utilize_appeared/STGCN
python main.py
cd $WORK_PATH/Model/repeat1_attribute_prediction_exist_Tm_binary_utilize_appeared/STGGNN
python main.py
cd $WORK_PATH/Model/repeat1_attribute_prediction_exist_Tm_binary_utilize_appeared/DynGEM
python main.py
cd $WORK_PATH/Model/repeat1_attribute_prediction_exist_Tm_binary_utilize_appeared/FNN
python main.py
cd $WORK_PATH/Evaluation
python repeat1_attribute_prediction_exist_Tm_binary_utilize_appeared.py

# repeat1_attribute_prediction_exist_Tm_binary_utilize_disappeared
cd $WORK_PATH/Model/repeat1_attribute_prediction_exist_Tm_binary_utilize_disappeared/Baseline
python main.py
cd $WORK_PATH/Model/repeat1_attribute_prediction_exist_Tm_binary_utilize_disappeared/EGCNh
python main.py
cd $WORK_PATH/Model/repeat1_attribute_prediction_exist_Tm_binary_utilize_disappeared/EGCNo
python main.py
cd $WORK_PATH/Model/repeat1_attribute_prediction_exist_Tm_binary_utilize_disappeared/GCN
python main.py
cd $WORK_PATH/Model/repeat1_attribute_prediction_exist_Tm_binary_utilize_disappeared/LSTM
python main.py
cd $WORK_PATH/Model/repeat1_attribute_prediction_exist_Tm_binary_utilize_disappeared/Random
python main.py
cd $WORK_PATH/Model/repeat1_attribute_prediction_exist_Tm_binary_utilize_disappeared/STGCN
python main.py
cd $WORK_PATH/Model/repeat1_attribute_prediction_exist_Tm_binary_utilize_disappeared/STGGNN
python main.py
cd $WORK_PATH/Model/repeat1_attribute_prediction_exist_Tm_binary_utilize_disappeared/DynGEM
python main.py
cd $WORK_PATH/Model/repeat1_attribute_prediction_exist_Tm_binary_utilize_disappeared/FNN
python main.py
cd $WORK_PATH/Evaluation
python repeat1_attribute_prediction_exist_Tm_binary_utilize_disappeared.py

# repeat1_attribute_prediction_exist_Tm_binary_utilize_lost
cd $WORK_PATH/Model/repeat1_attribute_prediction_exist_Tm_binary_utilize_lost/Baseline
python main.py
cd $WORK_PATH/Model/repeat1_attribute_prediction_exist_Tm_binary_utilize_lost/EGCNh
python main.py
cd $WORK_PATH/Model/repeat1_attribute_prediction_exist_Tm_binary_utilize_lost/EGCNo
python main.py
cd $WORK_PATH/Model/repeat1_attribute_prediction_exist_Tm_binary_utilize_lost/GCN
python main.py
cd $WORK_PATH/Model/repeat1_attribute_prediction_exist_Tm_binary_utilize_lost/LSTM
python main.py
cd $WORK_PATH/Model/repeat1_attribute_prediction_exist_Tm_binary_utilize_lost/Random
python main.py
cd $WORK_PATH/Model/repeat1_attribute_prediction_exist_Tm_binary_utilize_lost/STGCN
python main.py
cd $WORK_PATH/Model/repeat1_attribute_prediction_exist_Tm_binary_utilize_lost/STGGNN
python main.py
cd $WORK_PATH/Model/repeat1_attribute_prediction_exist_Tm_binary_utilize_lost/DynGEM
python main.py
cd $WORK_PATH/Model/repeat1_attribute_prediction_exist_Tm_binary_utilize_lost/FNN
python main.py
cd $WORK_PATH/Evaluation
python repeat1_attribute_prediction_exist_Tm_binary_utilize_lost.py

# repeat1_attribute_prediction_exist_Tm_binary_utilize_new_attribute_link
cd $WORK_PATH/Model/repeat1_attribute_prediction_exist_Tm_binary_utilize_new_attribute_link/Baseline
python main.py
cd $WORK_PATH/Model/repeat1_attribute_prediction_exist_Tm_binary_utilize_new_attribute_link/EGCNh
python main.py
cd $WORK_PATH/Model/repeat1_attribute_prediction_exist_Tm_binary_utilize_new_attribute_link/EGCNo
python main.py
cd $WORK_PATH/Model/repeat1_attribute_prediction_exist_Tm_binary_utilize_new_attribute_link/GCN
python main.py
cd $WORK_PATH/Model/repeat1_attribute_prediction_exist_Tm_binary_utilize_new_attribute_link/LSTM
python main.py
cd $WORK_PATH/Model/repeat1_attribute_prediction_exist_Tm_binary_utilize_new_attribute_link/Random
python main.py
cd $WORK_PATH/Model/repeat1_attribute_prediction_exist_Tm_binary_utilize_new_attribute_link/STGCN
python main.py
cd $WORK_PATH/Model/repeat1_attribute_prediction_exist_Tm_binary_utilize_new_attribute_link/STGGNN
python main.py
cd $WORK_PATH/Model/repeat1_attribute_prediction_exist_Tm_binary_utilize_new_attribute_link/DynGEM
python main.py
cd $WORK_PATH/Model/repeat1_attribute_prediction_exist_Tm_binary_utilize_new_attribute_link/FNN
python main.py
cd $WORK_PATH/Evaluation
python repeat1_attribute_prediction_exist_Tm_binary_utilize_new_attribute_link.py

# repeat1_attribute_prediction_exist_Tm_binary_utilize_all
cd $WORK_PATH/Model/repeat1_attribute_prediction_exist_Tm_binary_utilize_all/Baseline
python main.py
cd $WORK_PATH/Model/repeat1_attribute_prediction_exist_Tm_binary_utilize_all/EGCNh
python main.py
cd $WORK_PATH/Model/repeat1_attribute_prediction_exist_Tm_binary_utilize_all/EGCNo
python main.py
cd $WORK_PATH/Model/repeat1_attribute_prediction_exist_Tm_binary_utilize_all/GCN
python main.py
cd $WORK_PATH/Model/repeat1_attribute_prediction_exist_Tm_binary_utilize_all/LSTM
python main.py
cd $WORK_PATH/Model/repeat1_attribute_prediction_exist_Tm_binary_utilize_all/Random
python main.py
cd $WORK_PATH/Model/repeat1_attribute_prediction_exist_Tm_binary_utilize_all/STGCN
python main.py
cd $WORK_PATH/Model/repeat1_attribute_prediction_exist_Tm_binary_utilize_all/STGGNN
python main.py
cd $WORK_PATH/Model/repeat1_attribute_prediction_exist_Tm_binary_utilize_all/DynGEM
python main.py
cd $WORK_PATH/Model/repeat1_attribute_prediction_exist_Tm_binary_utilize_all/FNN
python main.py
cd $WORK_PATH/Evaluation
python repeat1_attribute_prediction_exist_Tm_binary_utilize_all.py