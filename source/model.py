import os
os.environ['CUDA_VISIBLE_DEVICES']='3'
import torch
import torch.nn as nn
import numpy as np
import json
from utils import Encoder,LayerNorm

def concept_edge_list4GCN():
    graph=json.load(open('graph.json',encoding='utf-8'))
    edge_set=[[co[0] for co in graph],[co[1] for co in graph]]
    return torch.LongTensor(edge_set).cuda()

def _create_embeddings(vocab_length, embedding_size, padding_idx):
    """Create and initialize word embeddings."""
    #e=nn.Embedding.from_pretrained(data, freeze=False, padding_idx=0).double()
    e = nn.Embedding(vocab_length, embedding_size, padding_idx)
    nn.init.normal_(e.weight, mean=0, std=embedding_size ** -0.5)
    #e.weight=data
    #nn.init.constant_(e.weight[padding_idx], 0)
    return e

class Model(nn.Module):
    def __init__(self, opt, padding_idx4item=0, padding_idx4prefer=0):
        super().__init__()  # self.pad_idx, self.start_idx, self.end_idx)
        self.batch_size = opt['batch_size']
        self.max_length = opt['max_length']
        self.dropout = opt['dropout']
        self.num_layers = opt['num_layers']
        self.vocab_size=opt['vocab_size']
        self.user_size=opt['user_size']
        self.dim=opt['dim']
        self.embedding_size=opt['embedding_size']
        self.mask4item=1922
        self.prefer_length=20
        self.item_length=50

        self.pad_idx4item = padding_idx4item
        self.pad_idx4prefer = padding_idx4prefer

        self.embeddings = _create_embeddings(
            self.vocab_size, self.embedding_size, self.pad_idx4item
        )
        self.user_embeddings = _create_embeddings(
            self.user_size, self.embedding_size, self.pad_idx4item
        )
        self.position_embeddings = nn.Embedding(opt['max_length'], opt['dim'])
        self.LayerNorm = LayerNorm(opt['dim'], eps=1e-12)
        self.dropout = nn.Dropout(opt['dropout'])

        #self.edge_sets = concept_edge_list4GCN()
        #self.GCN = GCNConv(self.dim, self.dim)

        #self.gru = nn.GRU(self.embedding_size, self.dim, self.num_layers, dropout=self.dropout)

        self.SAS_encoder=Encoder(opt)
        self.prefer_SAS_encoder=Encoder(opt)
        self.neg_SAS_encoder=Encoder(opt)

        self.item_norm = nn.Linear(opt['dim'], opt['dim'])
        self.prefer_norm=nn.Linear(opt['dim'], opt['dim'])

        self.criterion=nn.BCELoss(reduce=False)
        self.cs_loss=nn.CrossEntropyLoss()

    def sequence_pretrain(self, u_emb, v_mat):
        '''
        batch*p_l*hidden
        batch*p_l*hidden
        '''
        #prob=torch.matmul(self.item_norm(u_emb),v_mat.transpose(0,1))   #batch*p_l*w_l
        vector=self.item_norm(u_emb.view([-1,self.dim])) #batchXp_l*2hidden
        probs=torch.mul(vector,v_mat.view([-1,self.dim]))
        prob=torch.sum(probs,-1)
        return torch.sigmoid(prob)

    def sequence_pretrain_prefer(self, u_emb, prefer_emb):
        '''
        input:
        batchXp_l*hidden
        batchXp_l*pre_l*hidden

        output: batch*p_l*pre_l
        '''
        #prob=torch.matmul(self.item_norm(u_emb),v_mat.transpose(0,1))   #batch*p_l*w_l
        pre_emb=self.prefer_norm(prefer_emb)
        user_emb=u_emb.view([-1,self.dim,1])
        probs=torch.matmul(pre_emb,user_emb) #batchXp_l*pre_l
        return torch.sigmoid(probs.squeeze(-1))

    def pretrain(self, sequence, mask_sequence, neg_sequence, preferences, neg_preferences):
        # graph network
        '''
        nodes_features=self.GCN(self.embeddings.weight,self.edge_sets)
        '''
        # gru4rec
        seq_emb = self.embeddings(mask_sequence)
        #seq_emb_pad=nn.utils.rnn.pack_padded_sequence(seq_emb, seq_length, batch_first=True, enforce_sorted=False)
        seq_length = mask_sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=mask_sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(mask_sequence)

        position_embeddings = self.position_embeddings(position_ids)
        seq_emb_pos = seq_emb + position_embeddings

        seq_emb_pos = self.LayerNorm(seq_emb_pos)
        seq_emb_pos = self.dropout(seq_emb_pos)

        seq_mask=(mask_sequence==self.pad_idx4item).float()*-1e8
        seq_mask=torch.unsqueeze(torch.unsqueeze(seq_mask,1),1)
        encoded_layers = self.SAS_encoder(seq_emb_pos,
                                      seq_mask,
                                      output_all_encoded_layers=True)
        # [B L H]
        sequence_output = encoded_layers[-1]
        user_emb=sequence_output    #batch*p_l*hidden

        #item_matrix=self.embeddings.weight[:30840,:]
        pos_embs=self.embeddings(sequence)
        neg_embs=self.embeddings(neg_sequence)

        #preference emb

        pos_prob=self.sequence_pretrain(user_emb,pos_embs)
        neg_prob=self.sequence_pretrain(user_emb,neg_embs)

        # item_scores = self.item_match_matrix(item_embs, user_emb, prefer_emb, pre_mask, reject_emb, rej_mask)
        #item bias loss
        pos_loss=self.criterion(pos_prob,torch.ones_like(pos_prob,dtype=torch.float32))
        neg_loss=self.criterion(neg_prob,torch.zeros_like(neg_prob,dtype=torch.float32))
        
        bert_mask=(mask_sequence==self.mask4item).float()
        '''
        v_distance = torch.sigmoid(pos_prob - neg_prob) #batchXp_l

        v_loss = self.criterion(v_distance, torch.ones_like(v_distance, dtype=torch.float32))
        '''
        item_loss=torch.sum(pos_loss*bert_mask.flatten()+neg_loss*bert_mask.flatten())

        # neg preferences
        neg_preference = neg_preferences.view([-1, self.prefer_length])
        neg_prefer_sequence = self.embeddings(neg_preference)
        neg_prefer_masks = (neg_preference == self.pad_idx4item).float() * -1e8
        neg_pre_mask = torch.unsqueeze(torch.unsqueeze(neg_prefer_masks, 1), 1)
        neg_prefer_encoded_layers = self.prefer_SAS_encoder(neg_prefer_sequence, neg_pre_mask,
                                                            output_all_encoded_layers=True)
        neg_prefer_emb4ele = neg_prefer_encoded_layers[-1]
        neg_prefer_emb = neg_prefer_emb4ele.view(-1, self.prefer_length, self.dim)

        prefer_prob = self.sequence_pretrain_prefer(user_emb, neg_prefer_emb)  # batchXp_l*pre_l

        # prefer bias loss
        neg_prefer_mask = (mask_sequence != neg_sequence).float() * (
                    mask_sequence != self.pad_idx4item).float()  # batch*p_l

        prefer_label = (preferences == neg_preferences).float()  # batch*p_l*pre_l

        prefer_loss = self.criterion(prefer_prob, prefer_label.view(-1, self.prefer_length))

        prefer_loss = torch.sum(prefer_loss * neg_prefer_mask.flatten().unsqueeze(-1))

        return item_loss,prefer_loss

    def item_match(self, v_emb, u_emb):
        u_emb=self.item_norm(u_emb)
        score = torch.sum(torch.mul(u_emb, v_emb), dim=-1)
        return torch.sigmoid(score)	# + pre_score+rej_score)

    def item_match_matrix(self, v_mat, u_emb):
        #print(v_mat.size())
        u_emb=self.item_norm(u_emb)
        pre_mat = torch.transpose(v_mat, 0, 1)  # emb*p_l
        score = torch.matmul(u_emb, pre_mat)  # batch*p_l
        return score	#+pre_score+rej_score

    def forward(self, user, sequence, seq_length, pos_item, neg_item):
        # graph network
        '''
        nodes_features=self.GCN(self.embeddings.weight,self.edge_sets)
        '''
        # gru4rec
        seq_emb = self.embeddings(sequence)
        #seq_emb_pad=nn.utils.rnn.pack_padded_sequence(seq_emb, seq_length, batch_first=True, enforce_sorted=False)
        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)

        position_embeddings = self.position_embeddings(position_ids)
        seq_emb_pos = seq_emb + position_embeddings

        seq_emb_pos = self.LayerNorm(seq_emb_pos)
        seq_emb_pos = self.dropout(seq_emb_pos)

        seq_mask=(sequence==self.pad_idx4item).float()*-1e8
        seq_mask=torch.unsqueeze(torch.unsqueeze(seq_mask,1),1)
        encoded_layers = self.SAS_encoder(seq_emb_pos,
                                      seq_mask,
                                      output_all_encoded_layers=True)
        # [B L H]
        sequence_output = encoded_layers[-1]
        user_emb=sequence_output[:, -1,:]

        #gru_encoding,hidden=self.gru(seq_emb_pad,None)

        #user_emb=hidden.squeeze(0)   #batch*p_l*hidden batch
        user_emb_origin=self.user_embeddings(user) #
        #print(user_emb.size())

        #item_embs = self.embeddings.weight[:30840, :]
        pos_v_emb=self.embeddings(pos_item)
        neg_v_emb=self.embeddings(neg_item)

        #pre_mask=(preferences!=self.pad_idx4prefer).float()
        #rej_mask=(rejected!=self.pad_idx4prefer).float()
        #item_mask=(sequence!=self.pad_idx4item).float()

        #negative emb

        pos_v_score=self.item_match(pos_v_emb,user_emb)
        neg_v_score=self.item_match(neg_v_emb,user_emb)

        pos_loss=self.criterion(pos_v_score,torch.ones_like(pos_v_score,dtype=torch.float32))
        neg_loss=self.criterion(neg_v_score,torch.zeros_like(neg_v_score,dtype=torch.float32))

        '''
        v_distance=torch.sigmoid(pos_v_score-neg_v_score)
        #item_scores = self.item_match_matrix(item_embs, user_emb, prefer_emb, pre_mask, reject_emb, rej_mask)

        v_loss=self.criterion(v_distance,torch.ones_like(v_distance,dtype=torch.float32))
        #v_loss=self.cs_loss(item_scores,pos_item)
        '''
        return torch.sum(pos_loss+neg_loss)

    def save_model(self, filename):
        torch.save(self.state_dict(), filename)

    def load_model(self, filename):
        self.load_state_dict(torch.load(filename))

    def output(self, user, sequence):
        # project back to vocabulary
        # graph network
        '''
        nodes_features = self.GCN(self.embeddings.weight, self.edge_sets)

        # gru4rec
        seq_emb = nodes_features[sequence]
        seq_emb_pad = nn.utils.rnn.pack_padded_sequence(seq_emb, seq_length, batch_first=True, enforce_sorted=False)
        gru_encoding, hidden = self.gru(seq_emb_pad, None)
        '''
        seq_emb = self.embeddings(sequence)
        # seq_emb_pad=nn.utils.rnn.pack_padded_sequence(seq_emb, seq_length, batch_first=True, enforce_sorted=False)
        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)

        position_embeddings = self.position_embeddings(position_ids)
        seq_emb_pos = seq_emb + position_embeddings

        seq_emb_pos = self.LayerNorm(seq_emb_pos)
        seq_emb_pos = self.dropout(seq_emb_pos)

        seq_mask = (sequence == self.pad_idx4item).float() * -1e8
        seq_mask = torch.unsqueeze(torch.unsqueeze(seq_mask, 1), 1)
        encoded_layers = self.SAS_encoder(seq_emb_pos,
                                      seq_mask,
                                      output_all_encoded_layers=True)
        # [B L H]
        sequence_output = encoded_layers[-1]

        user_emb=sequence_output[:, -1,:]
        user_emb_origin = self.user_embeddings(user)  #
        # print(user_emb.size())

        #pos_v_emb = self.embeddings(pos_item)
        item_embs=self.embeddings.weight[:1922,:]

        #pre_mask = (preferences != self.pad_idx4prefer).float()
        #rej_mask = (rejected!=self.pad_idx4prefer).float()
        # item_mask=(sequence!=self.pad_idx4item).float()
        # prefer emb

        item_scores = self.item_match_matrix(item_embs, user_emb)
        return item_scores
