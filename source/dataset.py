import json
import numpy as np
from tqdm import tqdm
import pickle as pkl
import json
from nltk import word_tokenize
import re
from torch.utils.data.dataset import Dataset
import numpy as np
from random import seed,choice,randint,random,shuffle
from copy import deepcopy
seed(1444)

class dataset4train_bert(object):
    def __init__(self,debug=False):
        self.item_dict=json.load(open('../dataset/item_dict_lastfm1.0.json',encoding='utf-8'))

        self.batch_size = 2048	#256
        self.max_length = 50
        self.items_amount=1922	#11414
        self.tag_amount=72
        self.data = []

        #temp_data=json.load(open('../dataset/temp_data.json',encoding='utf-8'))
        #conversation_data=json.load(open('../dataset/conversation.json',encoding='utf-8'))
        data = json.load(open('../dataset/train_data_lastfm2.0.json', encoding='utf-8'))
        for i, line in enumerate(data):
            if debug and i==100:
                break
            con_dict = line
            conversation = line['dialog']
            user_id=con_dict['user']
            context=con_dict['context']
            pos_item=con_dict['pos_item']
            neg_item=con_dict['neg_item']

            seq, seq_l = self.pre_padding(context)
            pos_preference=[]
            rejected=[]

            for uttr in conversation[:1]:
                if len(uttr)==2:
                    tag_id=uttr[0]+self.items_amount
                    pos_preference.append(tag_id)
                else:
                    rejected_items=uttr
                    rejected.extend(rejected_items)

                rej_items,rej_len=self.pre_padding(rejected)
                prefer,pre_len=self.pre_padding(pos_preference,max_length=10)
                self.data.append({'user':user_id, 'sequence':seq, 'seq_length':seq_l, 'rejected': rej_items, 'rej_length':rej_len,
                                  'pos_item': pos_item, 'neg_item': neg_item, 'preference': prefer, 'pre_length':pre_len})

    def pre_padding(self,sen,padding=0,max_length=50):
        if len(sen)>max_length:
            return sen[-max_length:],max_length
        else:
            return (max_length-len(sen))*[padding]+sen,len(sen)

    def mask_repadding(self,sen,padding=0,mask=1922,max_length=50,mask_ratio=0.2,prefer_replace=0.5):
        sequence=[]
        neg_sequence=[]
        prefer_sequence=[]
        neg_prefer_sequence=[]
        for item in sen[:-1]:
            if item==padding:
                sequence.append(item)
                neg_sequence.append(item)
                prefer,_=self.pre_padding([],max_length=10)
                prefer_sequence.append(prefer)
                neg_prefer_sequence.append(prefer)
                continue
            rand=random()
            if rand<mask_ratio:
                sequence.append(mask)
                neg_item=randint(1,self.items_amount-1)
                neg_sequence.append(neg_item)
                try:
                    preferences=self.item_dict[str(item)]
                    selected_count=randint(1,len(preferences))
                    shuffle(preferences)
                    selected_prefer=preferences[:selected_count]

                    neg_preference_set=list(set(range(1,self.tag_amount-1))-set(preferences))
                    neg_preference=[]
                    for prefer in selected_prefer:
                        rand=random()
                        if rand<2/len(selected_prefer):
                            neg_preference.append(choice(neg_preference_set))
                        else:
                            neg_preference.append(prefer)

                    prefer,_=self.pre_padding([tag_id+self.items_amount for tag_id in selected_prefer],max_length=10)
                    neg_prefer,_=self.pre_padding([tag_id+self.items_amount for tag_id in neg_preference],max_length=10)
                except:
                    #print(item)
                    prefer, _ = self.pre_padding([], max_length=10)
                    neg_prefer,_=self.pre_padding([],max_length=10)

                prefer_sequence.append(prefer)
                neg_prefer_sequence.append(neg_prefer)

            else:
                sequence.append(item)
                neg_sequence.append(item)
                prefer,_=self.pre_padding([],max_length=10)
                prefer_sequence.append(prefer)
                neg_prefer,_=self.pre_padding([],max_length=10)
                neg_prefer_sequence.append(neg_prefer)

        sequence.append(mask)
        neg_item = randint(1, self.items_amount - 1)
        neg_sequence.append(neg_item)
        try:
            preferences = self.item_dict[str(sen[-1])]

            selected_count = randint(1, len(preferences))
            shuffle(preferences)
            selected_prefer = preferences[:selected_count]

            neg_preference_set = list(set(range(1, self.tag_amount - 1)) - set(preferences))
            neg_preference = []
            for prefer in selected_prefer:
                rand = random()
                if rand < 2 / len(selected_prefer):
                    neg_preference.append(choice(neg_preference_set))
                else:
                    neg_preference.append(prefer)

            prefer, _ = self.pre_padding([tag_id+self.items_amount for tag_id in selected_prefer], max_length=10)
            neg_prefer, _ = self.pre_padding([tag_id+self.items_amount for tag_id in neg_preference], max_length=10)
        except:
            #print(sen[-1])
            prefer, _ = self.pre_padding([], max_length=10)
            neg_prefer,_=self.pre_padding([],max_length=10)

        prefer_sequence.append(prefer)
        neg_prefer_sequence.append(neg_prefer)

        return sequence,neg_sequence,prefer_sequence,neg_prefer_sequence

    def generate_batch(self):
        #self.data_proprecoss()
        train_data=[]
        for i,line in tqdm(enumerate(self.data)):
            sequence,neg_sequence,prefer_seq,neg_prefer_seq=self.mask_repadding(line['sequence'])

            new_line=line['user'],np.array(line['sequence']),np.array(sequence),np.array(neg_sequence),\
                     np.array(prefer_seq),np.array(neg_prefer_seq),line['pos_item'],line['neg_item'],\
                     np.array(line['rejected']),line['rej_length']
            train_data.append(new_line)
        return train_data

class dataset4train(object):
    def __init__(self,debug=False):
        #self.item2index=json.load(open('item2index_new.json'))
        self.item_dict=json.load(open('../dataset/item_dict_lastfm1.0.json',encoding='utf-8'))
        #self.tag2index=json.load(open('tag2index_new.json',encoding='utf-8'))

        self.batch_size = 2048	#256
        self.max_length = 50
        self.items_amount=1922	#11414
        self.tag_amount=72	#301
        self.data = []

        #temp_data=json.load(open('../dataset/temp_data.json',encoding='utf-8'))
        #conversation_data=json.load(open('../dataset/conversation.json',encoding='utf-8'))
        data = json.load(open('../dataset/train_data_lastfm2.0.json', encoding='utf-8'))
        for i, line in enumerate(data):
            if debug and i==100:
                break
            con_dict = line
            conversation = line['dialog']
            user_id=con_dict['user']
            context=con_dict['context']
            pos_item=con_dict['pos_item']
            neg_item=con_dict['neg_item']

            seq, seq_l = self.pre_padding(context)
            pos_preference=[]
            rejected=[]


            for uttr in conversation:
                if len(uttr)==2:
                    tag_id=uttr[0]+self.items_amount
                    pos_preference.append(tag_id)
                else:
                    rejected_items=uttr
                    rejected.extend(rejected_items)

                rej_items,rej_len=self.pre_padding(rejected)
                prefer,pre_len=self.pre_padding(pos_preference,max_length=10)
                self.data.append({'user':user_id, 'sequence':seq, 'seq_length':seq_l, 'rejected': rej_items, 'rej_length':rej_len,
                                  'pos_item': pos_item, 'neg_item': neg_item, 'preference': prefer, 'pre_length':pre_len})

    def data_proprecoss(self):
        self.pre_data=[]
        for line in tqdm(self.data):
            preference=set(line['pos_preference'])
            for pos_pre,neg_pre in zip(line['pos_preference'],line['neg_preference']):
                prefer,pre_len=self.pre_padding(list(preference-set([pos_pre])))

                new_line={'sequence':line['sequence'],'seq_length':line['seq_length'],'pos_v':line['pos_item'],'neg_v':line['neg_item'],'pre_l':pre_len,
                          'preference':prefer,'pos_p':pos_pre,'neg_p':neg_pre}
                self.pre_data.append(new_line)

    def pre_padding(self,sen,padding=0,max_length=50):
        if len(sen)>max_length:
            return sen[-max_length:],max_length
        else:
            return (max_length-len(sen))*[padding]+sen,len(sen)

    def generate_batch(self):
        #self.data_proprecoss()
        train_data=[]
        for i,line in tqdm(enumerate(self.data)):
            new_line=line['user'],np.array(line['sequence']),line['seq_length'],\
                     np.array(line['preference']),line['pre_length'],\
                     line['pos_item'],line['neg_item'],np.array(line['rejected']),line['rej_length']
            train_data.append(new_line)
        return train_data

class dataset4test(object):
    def __init__(self, test=False):
        #self.item2index=json.load(open('item2index_new.json'))
        #self.tag2index=json.load(open('tag2index_new.json',encoding='utf-8'))

        self.tag2item=json.load(open('../dataset/tag2item_lastfm1.0.json'))
        self.tag2items={tag:set(self.tag2item[tag]) for tag in self.tag2item}

        self.batch_size = 2048	#256
        self.max_length = 50
        self.items_amount=1922	#11414
        self.tag_amount=72	#301
        self.data = []
        self.candidate_data=[]

        #temp_data=json.load(open('../dataset/temp_data_test.json',encoding='utf-8'))
        #conversation_data=json.load(open('../dataset/conversation_test.json',encoding='utf-8'))
        if test==True:
            data = json.load(open('../dataset/test_lastfm.json', encoding='utf-8'))
        else:
            data = json.load(open('../dataset/val_lastfm.json', encoding='utf-8'))
        for i, line in enumerate(data):
            con_dict = line
            conversation = line['dialog']
            user_id=con_dict['user']
            context=con_dict['context']
            pos_item=con_dict['pos_item']

            if len(context)==0:
                continue

            seq, seq_l = self.pre_padding(context)
            pos_preference=[]
            rejected=[]


            for uttr in conversation:
                if len(uttr)==2:
                    tag_id=uttr[0]+self.items_amount
                    pos_preference.append(tag_id)
                else:
                    rejected_items=uttr
                    rejected.extend(rejected_items)


                rej_items,rej_len=self.pre_padding(rejected)
                prefer,pre_len=self.pre_padding(pos_preference,max_length=10)
                try:
                    candidates=self.tag2items[str(conversation[0][0])]
                    candidates.remove(pos_item)
                except:
                    continue
                self.data.append({'user': user_id, 'sequence':seq, 'seq_length':seq_l, 'rejected': rej_items, 'rej_length': rej_len,
                                  'item': pos_item, 'preference': prefer, 'pre_length': pre_len})
                self.candidate_data.append(candidates)
        print(len(self.data))
        return

    def pre_padding(self,sen,padding=0,max_length=50):
        if len(sen)>max_length:
            return sen[-max_length:],max_length
        else:
            return (max_length-len(sen))*[padding]+sen,len(sen)

    def generate_batch(self):
        #self.data_proprecoss()
        train_data=[]
        for i,line in tqdm(enumerate(self.data)):
            new_line=line['user'],np.array(line['sequence']),line['seq_length'],\
                     np.array(line['preference']),line['pre_length'],\
                     line['item'],np.array(line['rejected']),line['rej_length']
            train_data.append(new_line)
        return train_data

class CRSdataset4train(Dataset):
    def __init__(self, dataset):
        self.data=dataset.generate_batch()

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

class CRSdataset4test(Dataset):
    def __init__(self, dataset):
        self.data = dataset.generate_batch()

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
