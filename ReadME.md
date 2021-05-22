This code is the first version of our paper, **Leveraging historical interaction data for improving conversational recommender system**

This paper presented a pre-training approach for conversational recommendation task, which focused on leveraging the item sequence from user history and attribute sequence from conversation data effectively. Based on a self-attentive architecture, our approach designed two pre-training tasks, namely Masked Item Prediction (MIP) and the Substituted Attributes Discrimination (SAD). We further improved our pre-training method by introducing a negative generator to produce high-quality negative samples. Experimental results on two datasets demonstrated the effectiveness of our approach for conversational recommendation task.

<img src="./table1.png" width=400 height=240 />


# Environment
pytorch==1.3.0

# Training
To use our code and data, we present a pipeline as following:

**1.Pre-training our model** via negative sampling from SASRec. For convenience, we give a pre-learned checkpoint file of SASRec for usage.
```
python run.py --load_dict_gen model_gen/net_parameter1.pkl 
```

**2.Fine-tuning our model** on Downstream tasks. And our code will record the performance on test set during training. (Due to the privacy-protection policy, one of our dataset Meituan can not be released.)
```
python run.py --is_finetune True --load_dict_gen model_gen/net_parameter1.pkl --load_dict_dis model/net_parameter1.pkl --save_dict model/ft_parameter1.pkl
```

# Thanks for your citation
If you use our code, please kindly cite our paper as following:
```
@inproceedings{zhou2020leveraging,
  title={Leveraging historical interaction data for improving conversational recommender system},
  author={Zhou, Kun and Zhao, Wayne Xin and Wang, Hui and Wang, Sirui and Zhang, Fuzheng and Wang, Zhongyuan and Wen, Ji-Rong},
  booktitle={Proceedings of the 29th ACM International Conference on Information \& Knowledge Management},
  pages={2349--2352},
  year={2020}
}
```
