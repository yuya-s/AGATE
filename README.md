# AGATE
This repository contains the code necessary to run the AGATE.  
See our report for details on the algorithm and the result.  
**Report link**: [Holistic Prediction on a Time-Evolving Attributed Graph]

## Dataset
We created graph data from three datasets; [NBA](https://www.basketball-reference.com), [Reddit](http://snap.stanford.edu/data/soc-RedditHyperlinks.html), and [AMiner](https://www.aminer.cn/citation).  
Time-evolving attributed graphs for AGATE can be downloaded from [here](https://drive.google.com/drive/folders/1D3HeC-2pbShwbQHzzj7HBbnxM-g3j27D?usp=sharing).
Unzip graph.zip and place it under [DATASET]/data/ to run it.

## Code
You can find source code that works with each dataset in the AMiner, NBA, and Reddit.  
You can run AGATE by executing the scripts in the order described in run.sh  
All hyperparameters are described in setting_param.py.  
MakeSample has codes that create a dataset for a sub-prediction task.  
Model has codes for training methods of each subprediction task.  
Evaluation has codes for evaluating subprediction accuracy.

## citing
If you find our code is useful for your research, please consider citing the following paper:

    @inproceedings{yamasaki2023holistic,
    title={Holistic Prediction on a Time-Evolving Attributed Graph},
    author={Yamasaki, Shohei and Sasaki, Yuya and Karras, Panagiotis and Onizuka, Makoto},
    booktitle={Proceedings of the the 61st annual meeting of the association for computational linguistics},
    year={2023}
    }

## contact
Please let me know if you have questions to sasaki@ist.osaka-u.ac.jp
