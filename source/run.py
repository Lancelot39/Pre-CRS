#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from tqdm import tqdm
import os
os.environ['CUDA_VISIBLE_DEVICES']='3'
import argparse
from dataset import dataset4train_bert,dataset4train,dataset4test,CRSdataset4train,CRSdataset4test
from model_dis import Model_dis
from model_gen import Model_gen
import torch.nn as nn
from torch import optim
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
#os.environ['CUDA_VISIBLE_DEVICES']='2'
try:
    import torch.version
    import torch.distributed as dist
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

def is_distributed():
    """
    Returns True if we are in distributed mode.
    """
    return TORCH_AVAILABLE and dist.is_available() and dist.is_initialized()

def setup_args():
    train = argparse.ArgumentParser()
    train.add_argument("-max_length","--max_length",type=int,default=50)
    train.add_argument("-batch_size","--batch_size",type=int,default=256)
    train.add_argument("-use_cuda","--use_cuda",type=bool,default=True)
    train.add_argument("-learningrate","--learningrate",type=float,default=1e-4)
    train.add_argument("-optimizer","--optimizer",type=str,default='adam')
    train.add_argument("-momentum","--momentum",type=float,default=0)
    train.add_argument("-is_finetune","--is_finetune",type=bool,default=False)
    train.add_argument("-embedding_type","--embedding_type",type=str,default='random')
    train.add_argument("-epoch","--epoch",type=int,default=50)
    train.add_argument("-gpu","--gpu",type=str,default='1,2')
    train.add_argument("-gradient_clip","--gradient_clip",type=float,default=0.1)
    train.add_argument("-embedding_size","--embedding_size",type=int,default=64)
    train.add_argument("-dim","--dim",type=int,default=64)
    train.add_argument("-num_layers","--num_layers",type=int,default=2)
    train.add_argument("-vocab_size","--vocab_size",type=int,default=1994)
    train.add_argument("-user_size","--user_size",type=int,default=2101)
    train.add_argument("-load_dict_dis", "--load_dict_dis", type=str, default=None)
    train.add_argument("-load_dict_gen", "--load_dict_gen", type=str, default=None)
    train.add_argument("-save_dict", "--save_dict", type=str, default='model/net_parameter1.pkl')

    train.add_argument("-dropout","--dropout",type=float,default=0.1)

    train.add_argument("-num_attention_heads","--num_attention_heads",type=int,default=2)

    train.add_argument("-l2_weight","--l2_weight",type=float,default=0)	#1e-5)

    return train

class TrainLoop():
    def __init__(self, opt, is_pre=True):
        self.opt=opt
        if is_pre:
            self.train_dataset=dataset4train_bert()
        else:
            self.train_dataset=dataset4train()
        self.test_dataset=dataset4test(test=True)
        self.dev_dataset=dataset4test(test=False)

        self.batch_size=self.opt['batch_size']
        self.epoch=self.opt['epoch']

        self.use_cuda=opt['use_cuda']

        # Note: we cannot change the type of metrics ahead of time, so you
        # should correctly initialize to floats or ints here
        self.build_model()

        if opt['load_dict_dis'] is not None:
            # load model parameters if available
            print('[ Loading existing model params from {} ]'
                  ''.format(opt['load_dict_dis']))
            self.model_dis.load_model(opt['load_dict_dis'])


        if opt['load_dict_gen'] is not None:
            # load model parameters if available
            print('[ Loading existing model params from {} ]'
                  ''.format(opt['load_dict_gen']))
            self.model_gen.load_model(opt['load_dict_gen'])

    def build_model(self):
        self.model_dis = Model_dis(self.opt)
        self.model_gen = Model_gen(self.opt)
        if self.use_cuda:
            self.model_dis.cuda()
            self.model_gen.cuda()

    def train(self):
        self.optimizer = optim.Adam(self.model_dis.parameters(), lr=self.opt['learningrate'],
                                    weight_decay=self.opt['l2_weight'])
        losses=[]
        best_val_item=0
        for i in range(self.epoch):
            train_set=CRSdataset4train(self.train_dataset)
            train_dataset_loader = torch.utils.data.DataLoader(dataset=train_set,
                                                            batch_size=self.batch_size,
                                                            shuffle=True)	#False)
            num=0
            print('start fine-tuning in epoch', i)
            for user,seq,seq_l,preferences,pre_l,pos_item, neg_item,rejected,rej_len in tqdm(train_dataset_loader):
                self.model_dis.train()
                self.zero_grad()

                v_loss = self.model_dis(user.cuda(), seq.cuda(), seq_l.cuda(), preferences.cuda(), rejected.cuda(),
                                         pos_item.cuda(), neg_item.cuda())
                joint_loss=v_loss

                losses.append(joint_loss)
                self.backward(joint_loss)
                self.update_params()
                if num%200==0:
                    print('loss is %f'%(sum(losses)/len(losses)))
                    losses=[]
                num+=1
            #_ = self.val_item()
            print('Dev performance: ')
            item_score = self.val_item_all(self.dev_dataset)
            print("the best mrr is %f"%best_val_item)
            if best_val_item < item_score:
                best_val_item = item_score
                self.model_dis.save_model(self.opt['save_dict'])
                print(" model saved once------------------------------------------------")
                print('Test performance: ')
                item_score = self.val_item_all(self.test_dataset)

    def pretrain(self):
        self.optimizer = optim.Adam(self.model_dis.parameters(), lr=1e-3,
                                    weight_decay=self.opt['l2_weight'])
        v_losses=[]
        p_losses=[]
        best_val_item=0

        for i in range(self.epoch):
            train_set = CRSdataset4train(self.train_dataset)
            train_dataset_loader = torch.utils.data.DataLoader(dataset=train_set,
                                                            batch_size=self.batch_size,
                                                            shuffle=True)	#False)
            num=0
            print('start pre-training in epoch', i)
            for user,seq,mask_seq,neg_seq,pre_seq,neg_pre_seq,pos_item, neg_item,rejected,rej_len in tqdm(train_dataset_loader):
                self.model_dis.train()
                self.zero_grad()

                v_loss,p_loss = self.model_dis.pretrain(seq.cuda(), mask_seq.cuda(), neg_seq.cuda(), pre_seq.cuda(), neg_pre_seq.cuda())
                joint_loss=v_loss+0.1*p_loss

                v_losses.append(v_loss)
                p_losses.append(p_loss)
                self.backward(joint_loss)
                self.update_params()
                if num%200==0:
                    print('item loss is %f'%(sum(v_losses)/len(v_losses)))
                    print('prefer loss is %f'%(sum(p_losses)/len(p_losses)))
                    v_losses=[]
                    p_losses=[]
                num+=1
            self.model_dis.save_model(self.opt['save_dict'])
            print(" model saved once------------------------------------------------")

    def pretrain_adv(self):
        self.optimizer = optim.Adam(self.model_dis.parameters(), lr=1e-4,
                                    weight_decay=self.opt['l2_weight'])
        def build_negative_items_sample(mask_sequence,sequence,scores):
            neg_sequences=[]
            for mask_items,items,score in zip(mask_sequence,sequence,scores):
                neg_sequence=[]
                for mask_v,v,prob in zip(mask_items,items,score):
                    if mask_v==v:
                        neg_sequence.append(v)
                    else:
                        quest = np.random.choice(range(len(prob)), p=prob)
                        count=0
                        while quest==v:
                            count+=1
                            quest = np.random.choice(range(len(prob)), p=prob)
                            if count>5:
                                break
                        neg_sequence.append(quest)
                neg_sequences.append(neg_sequence)
            return torch.LongTensor(neg_sequences)
        v_losses=[]
        p_losses=[]
        best_val_item=0

        for i in range(self.epoch):
            train_set = CRSdataset4train(self.train_dataset)
            train_dataset_loader = torch.utils.data.DataLoader(dataset=train_set,
                                                            batch_size=self.batch_size,
                                                            shuffle=True)	#False)
            num=0
            print('start NG pre-training in epoch', i)
            for user,seq,mask_seq,neg_seq,pre_seq,neg_pre_seq,pos_item, neg_item,rejected,rej_len in tqdm(train_dataset_loader):
                self.model_gen.train()
                self.model_dis.train()
                self.zero_grad()
                #build neg sequence
                v_scores=self.model_gen.pre_output(mask_seq.cuda())    #batch*p_l*v_l

                adv_neg_seq=build_negative_items_sample(mask_seq.numpy(),seq.numpy(),v_scores.detach().cpu().numpy())

                v_loss,p_loss = self.model_dis.pretrain(seq.cuda(), mask_seq.cuda(), adv_neg_seq.cuda(), pre_seq.cuda(), neg_pre_seq.cuda())
                joint_loss=v_loss+0.1*p_loss

                v_losses.append(v_loss)
                p_losses.append(p_loss)
                self.backward(joint_loss)
                self.update_params()
                if num%200==0:
                    print('item loss is %f'%(sum(v_losses)/len(v_losses)))
                    print('prefer loss is %f'%(sum(p_losses)/len(p_losses)))
                    v_losses=[]
                    p_losses=[]
                num+=1
            self.model_dis.save_model(self.opt['save_dict'])
            print(" model saved once------------------------------------------------")

    def val_item_all(self, dataset):
        import math
        self.model_dis.eval()
        val_set=CRSdataset4test(dataset)
        val_dataset_loader = torch.utils.data.DataLoader(dataset=val_set,
                                                           batch_size=100,	#self.batch_size,
                                                           shuffle=False)

        num=0
        ndcgs=[]
        mrrs=[]
        for user,seq, seq_l,preferences,pre_l,pos_item,rejected,rej_len in tqdm(val_dataset_loader):
            with torch.no_grad():
                items_scores = self.model_dis.output(user.cuda(), seq.cuda(), seq_l.cuda(),
                                            preferences.cuda(), rejected.cuda())

            items_scores=items_scores.cpu().numpy()
            #pos_scores=items_scores[pos_item].tolist()

            batch_size=len(pos_item)
            candidates=dataset.candidate_data[num:num+batch_size]
            #print(np.shape(items_scores))
            for can,score,pos_v in zip(candidates,items_scores,pos_item):
                #print(score)
                #print(can)
                ranks=[]

                pos_score=score[pos_v]

                rank=np.sum(score[list(can)[:100]]>pos_score)
                if rank==-1:
                    continue
                MRR=1 / (rank + 1)
                if rank < 10:
                    #if rank==-1:
                    #    continue
                    NDCG = 1 / math.log(rank + 2.0, 2)
                else:
                    NDCG = 0
                ndcgs.append(NDCG)
                mrrs.append(MRR)
            num+=batch_size
        print('the NDCG score of item is %f'%(sum(ndcgs)/len(ndcgs)))
        print('the MRR score of item is %f'%(sum(mrrs)/len(mrrs)))
        return sum(ndcgs)/len(ndcgs)

    def recall100(self,golden,inference):
        import math
        def DCG(labels):
            dcg=0
            for i,label in enumerate(labels):
                dcg+=(2**label - 1)/math.log(i+2, 2)
            return dcg
        def get_ndcg(ranks):
            labels=[pair[0] for pair in ranks]
            dcg=DCG(labels)
            sort_pairs=sorted(ranks,key=lambda x:x[1],reverse=True)
            label_news=[pair[0] for pair in sort_pairs]
            ideal_dcg=DCG(label_news)
            return ideal_dcg/dcg
        ranks=[]
        MRR=[]
        NDCG=[]
        for label,score in tqdm(zip(golden,inference)):
            if label==1:
                if len(ranks)!=0:
                    #NDCG.append(get_ndcg(ranks))
                    ranks.reverse()
                    sorted_rank=sorted(ranks,key=lambda x:x[1],reverse=True)
                    for i,(label_l,score_l) in enumerate(sorted_rank):
                        if label_l==1.0:
                            if i < 10:
                                ndcg = 1 / math.log(i + 2.0, 2)
                            else:
                                ndcg = 0
                            NDCG.append(ndcg)
                            MRR.append(1/(i+1))
                            break
                    ranks=[]
            ranks.append((label,score))
        #NDCG.append(get_ndcg(ranks))
        ranks.reverse()
        sorted_rank = sorted(ranks, key=lambda x: x[1], reverse=True)
        for i, (label, score) in enumerate(sorted_rank):
            if label == 1.0:
                if i < 10:
                    ndcg = 1 / math.log(i + 2.0, 2)
                else:
                    ndcg = 0
                NDCG.append(ndcg)
                MRR.append(1 / (i + 1))
                break
        print('MRR is %f'%(sum(MRR)/len(MRR)))
        print('NDCG is %f'%(sum(NDCG)/len(NDCG)))

    def backward(self, loss):
        """
        Perform a backward pass. It is recommended you use this instead of
        loss.backward(), for integration with distributed training and FP16
        training.
        """
        loss.backward()

    def update_params(self):
        """
        Perform step of optimization, clipping gradients and adjusting LR
        schedule if needed. Gradient accumulation is also performed if agent
        is called with --update-freq.

        It is recommended (but not forced) that you call this in train_step.
        """
        update_freq = 1
        if update_freq > 1:
            # we're doing gradient accumulation, so we don't only want to step
            # every N updates instead
            self._number_grad_accum = (self._number_grad_accum + 1) % update_freq
            if self._number_grad_accum != 0:
                return

        if self.opt['gradient_clip'] > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model_dis.parameters(), self.opt['gradient_clip']
            )

        self.optimizer.step()

    def zero_grad(self):
        """
        Zero out optimizer.

        It is recommended you call this in train_step. It automatically handles
        gradient accumulation if agent is called with --update-freq.
        """
        self.optimizer.zero_grad()

    def embedding_output(self):
        embedding=self.model.embeddings.weight.cpu().detach().numpy()
        np.save('embedding.npy',embedding)

if __name__ == '__main__':
    args=setup_args().parse_args()
    print(vars(args))
    #loop=TrainLoop_fusion_rec(vars(args),is_finetune=False)
    #loop.train()
    if args.is_finetune==True:
        loop = TrainLoop(vars(args),is_pre=False)
        loop.train()
    else:
        loop = TrainLoop(vars(args), is_pre=True)
        loop.pretrain()
        loop.pretrain_adv()
    #python run.py --load_dict_gen model_gen/net_parameter1.pkl
