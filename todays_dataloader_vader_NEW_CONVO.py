
import itertools
import pickle
from utils import preprocess_text
import torch
import os
import random
from transformers import AutoTokenizer
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from vader import get_score
import numpy as np
from sentence_transformers import SentenceTransformer


torch.manual_seed(400)
np.random.seed(400)

torch.cuda.set_device(2)
model=SentenceTransformer('stsb-roberta-base')
os.environ['TOKENIZERS_PARALLELISM']='False'

class Data(Dataset):
    def __init__(self,ids,papers,labels):

        self.ids=ids
        self.papers=papers
        self.labels=labels
        self.max_len=50
        self.size=len(ids)

    @classmethod
    def getReader(cls,low,up,conf_name):
        with open("input_files/paper_review_phrases_"+conf_name+".pickle",'rb') as out:
            paper_files=pickle.load(out)
            a=list(paper_files.keys())[low:up]
            b=list(paper_files.values())[low:up]
            
            paper_files={k:v for k,v in zip(a,b)}
            paper_ids=list(paper_files.keys())
        
        with open("input_files/paper_decision_"+conf_name+".pickle",'rb') as out:
            paper_decision=pickle.load(out)
            d=[paper_decision[k] for k in paper_ids]

            print("paper accepted-",d.count('accept'))
            print("paper rejected-",d.count('reject'))
            print()
            
        return cls(paper_ids,paper_files, paper_decision)

    def __getitem__(self,idx):
        var=150
        aspect={'motivation':0,'clarity':1,'substance':2,'soundness':3,'meaningful_comparison':4,'originality':5,'replicability':6}
        sentiment={'positive':0,'negative':1}
        deci={'accept':1,'reject':0}

        p_id=self.ids[idx]
        r_sen=list((self.papers[p_id]).values())
        embed=torch.zeros(7,2,16,768)
        vader_scores=torch.zeros(7,2,32)
        
        des=self.labels[p_id]

        desc=torch.zeros(1)
        
        desc[0]=deci[des]
        r_sen_new={}
        r_sen_senti={}

        for i,k in enumerate(r_sen):
            for name,asp in k.items():
               for senti,value in asp.items():
                   if(name not in r_sen_new.keys()):
                       r_sen_new[name]={}

                   if(senti not in r_sen_new[name].keys()):
                       r_sen_new[name][senti]=[]

                   r_sen_new[name][senti]+=value
        
        for name,asp in r_sen_new.items():
            for senti,value in asp.items():
                value=list(map(preprocess_text,value))
                value_score=[get_score(v,senti) for v in value]

                if(len(value_score))<32:
                    value_score=value_score+[0]*(32-len(value_score))
                else:
                    value_score=value_score[0:32]
                
                if(len(value)<16):
                    value=value+[""]*abs(16-len(value))
                else:
                    value=value[0:16]
                    
                input=torch.from_numpy(model.encode(value))
                #print(input.size())
                value_score=torch.tensor(value_score).squeeze(-1)
                embed[aspect[name]][sentiment[senti]]=input
                vader_scores[aspect[name]][sentiment[senti]]=value_score

        
        return {"ids":embed,"ids_senti":vader_scores,'targets':desc}

    def __len__(self):
        return self.size


def getLoaders(batch_size,conf_name):
    print('Reading the training Dataset...')
    print()
    train_dataset = Data.getReader(0,8000,conf_name) #19200 #21216
    
    print('Reading the validation Dataset...')
    print()
    valid_dataset = Data.getReader(8000,8800,conf_name) #23200 #25216

    
    trainloader = DataLoader(dataset=train_dataset, batch_size = batch_size, num_workers=0,shuffle=True)
    validloader = DataLoader(dataset=valid_dataset, batch_size = batch_size, num_workers=0,shuffle=True)
   
    return trainloader, validloader

