Thanks for your understanding. This code is the first version of our paper, and we will update it in the Github page.

To run this code, we give a pre-learned checkpoint file of SASRec for usage.

Pre-training our model via negative sampling from SASRec
```
python run.py --load_dict_gen model_gen/net_parameter1.pkl 
```

Fine-tuning our model on Downstream tasks. And our code will record the performance on test set during training.
```
python run.py --is_finetune True --load_dict_gen model_gen/net_parameter1.pkl --load_dict_dis model/net_parameter1.pkl --save_dict model/ft_parameter1.pkl
```
