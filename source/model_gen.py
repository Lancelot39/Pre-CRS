import os
os.environ['CUDA_VISIBLE_DEVICES']='2'
import torch
import torch.nn as nn
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

class Model_gen(nn.Module):
    def __init__(self, opt, padding_idx4item=0, padding_idx4prefer=0):
        super().__init__()  # self.pad_idx, self.start_idx, self.end_idx)
        self.batch_size = opt['batch_size']
        self.max_length = opt['max_length']
        self.dropout = opt['dropout']
        self.num_layers = 2	#opt['num_layers']
        self.vocab_size=opt['vocab_size']
        self.user_size=opt['user_size']
        self.dim=opt['dim']
        self.embedding_size=opt['embedding_size']

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

        opt['num_layers']=2

        self.SAS_encoder=Encoder(opt)
        self.prefer_SAS_encoder=Encoder(opt)
        self.neg_SAS_encoder=Encoder(opt)

        self.item_norm = nn.Linear(opt['dim'], opt['dim'])

        self.criterion=nn.BCELoss()
        self.cs_loss=nn.CrossEntropyLoss()


    def item_match(self, v_emb, u_emb, u_emb_orin, p_emb, rej_emb):
        '''
        total_sum=v_emb+u_emb+u_emb_ori+torch.sum(p_emb*torch.unsqueeze(pre_mask,-1),1)   #batch*emb
        total_sum_l2=v_emb*v_emb+u_emb*u_emb+u_emb_ori*u_emb_ori+torch.sum(p_emb*p_emb*torch.unsqueeze(pre_mask,-1),1)  #batch*emb
        line_result=torch.sum(total_sum,-1)
        fm_result=torch.sum(total_sum*total_sum-total_sum_l2,-1)
        '''
        vector = u_emb	#torch.cat([u_emb, p_emb], -1)
        score = torch.mul(self.item_norm(vector), v_emb)
        return torch.sigmoid(score)

    def item_match_matrix(self, v_mat, u_emb, u_emb_orin, p_emb, rej_emb):
        #print(v_mat.size())
        vector = u_emb	#torch.cat([u_emb, p_emb], -1)
        score = torch.matmul(self.item_norm(vector), v_mat.transpose(0, 1))
        return score

    def forward(self, user, sequence, seq_length, preferences, rejected, pos_item, neg_item, test=True):
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

        prefer_sequence=self.embeddings(preferences)
        reject_sequence=self.embeddings(rejected)

        #item_embs = self.embeddings.weight[:30840, :]
        pos_v_emb=self.embeddings(pos_item)
        neg_v_emb=self.embeddings(neg_item)

        #pre_mask=(preferences!=self.pad_idx4prefer).float()
        #rej_mask=(rejected!=self.pad_idx4prefer).float()
        #item_mask=(sequence!=self.pad_idx4item).float()

        #prefer emb
        pre_mask = (preferences == self.pad_idx4item).float() * -1e8
        pre_mask = torch.unsqueeze(torch.unsqueeze(pre_mask, 1), 1)
        prefer_encoded_layers = self.prefer_SAS_encoder(prefer_sequence, pre_mask,
                                          output_all_encoded_layers=True)
        prefer_emb = prefer_encoded_layers[-1][:, -1, :]

        #negative emb
        neg_mask = (rejected == self.pad_idx4item).float() * -1e8
        neg_mask = torch.unsqueeze(torch.unsqueeze(neg_mask, 1), 1)
        neg_encoded_layers = self.neg_SAS_encoder(reject_sequence, neg_mask,
                                          output_all_encoded_layers=True)
        reject_emb = neg_encoded_layers[-1][:, -1, :]


        pos_v_score=self.item_match(pos_v_emb,user_emb,user_emb_origin,prefer_emb,reject_emb)
        neg_v_score=self.item_match(neg_v_emb,user_emb,user_emb_origin,prefer_emb,reject_emb)

        v_distance=torch.sigmoid(pos_v_score-neg_v_score)
        #item_scores = self.item_match_matrix(item_embs, user_emb, prefer_emb, pre_mask, reject_emb, rej_mask)

        v_loss=self.criterion(v_distance,torch.ones_like(v_distance,dtype=torch.float32))
        #v_loss=self.cs_loss(item_scores,pos_item)

        return v_loss

    def save_model(self, filename):
        torch.save(self.state_dict(), filename)

    def load_model(self, filename):
        self.load_state_dict(torch.load(filename))

    def output(self, user, sequence, seq_length, preferences, rejected):
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

        prefer_sequence=self.embeddings(preferences)
        reject_sequence=self.embeddings(rejected)
        #pos_v_emb = self.embeddings(pos_item)
        item_embs=self.embeddings.weight[:1922,:]

        #pre_mask = (preferences != self.pad_idx4prefer).float()
        #rej_mask = (rejected!=self.pad_idx4prefer).float()
        # item_mask=(sequence!=self.pad_idx4item).float()
        # prefer emb
        pre_mask = (preferences == self.pad_idx4item).float() * -1e8
        pre_mask = torch.unsqueeze(torch.unsqueeze(pre_mask, 1), 1)
        prefer_encoded_layers = self.prefer_SAS_encoder(prefer_sequence, pre_mask,
                                                        output_all_encoded_layers=True)
        prefer_emb = prefer_encoded_layers[-1][:, -1, :]

        # negative emb
        neg_mask = (rejected == self.pad_idx4item).float() * -1e8
        neg_mask = torch.unsqueeze(torch.unsqueeze(neg_mask, 1), 1)
        neg_encoded_layers = self.neg_SAS_encoder(reject_sequence, neg_mask,
                                                  output_all_encoded_layers=True)
        reject_emb = neg_encoded_layers[-1][:, -1, :]

        item_scores = self.item_match_matrix(item_embs, user_emb, user_emb_origin, prefer_emb, reject_emb)
        return item_scores

    def pre_output(self, mask_sequence):
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

        item_matrix=self.embeddings.weight[:1922,:]

        pos_prob=self.sequence_pretrain_mat(user_emb,item_matrix)
        return torch.softmax(pos_prob, -1)

    def sequence_pretrain_mat(self, u_emb, v_mat):
        vector = u_emb	#torch.cat([u_emb, p_emb], -1)
        score = torch.matmul(self.item_norm(vector), v_mat.transpose(0, 1))
        return torch.sigmoid(score)
