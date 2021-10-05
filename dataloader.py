
import itertools
import pickle
from utils import preprocess_text
import torch
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader

model_name='bert-base-uncased'

tokenizer=AutoTokenizer.from_pretrained(model_name)

class Data(Dataset):
    def __init__(self,ids,papers,labels):

        self.ids=ids
        self.papers=papers
        self.labels=labels
        self.max_len=50
        self.size=len(ids)

    @classmethod
    def getReader(cls,low,up):
        with open("input_files/paper_review_phrases.pickle",'rb') as out:
            paper_files=pickle.load(out)
            paper_files=dict(itertools.islice(paper_files.items(),up-low))
            paper_ids=list(paper_files.keys())

        with open("input_files/paper_decision.pickle",'rb') as out:
            paper_decision=pickle.load(out)
            
        
        return cls(paper_ids,paper_files, paper_decision)

    def __getitem__(self,idx):
        var=150
        aspect={'motivation':0,'clarity':1,'substance':2,'soundness':3,'meaningful_comparison':4,'originality':5,'replicability':6}
        sentiment={'positive':0,'negative':1}
        deci={'accept':0,'reject':1}

        p_id=self.ids[idx]
        r_sen=list((self.papers[p_id]).values())[0:3]
        #print(len(r_sen))
        des=self.labels[p_id]

        input_ids=torch.zeros(4,7,2,var)
        mask=torch.zeros(4,7,2,var)
        token_type_ids=torch.zeros(4,7,2,var)
        desc=torch.zeros(2)
        
        desc[deci[des]]=1

        for i,k in enumerate(r_sen):
            for name,asp in k.items():
                for senti,value in asp.items():
                    value=list(map(preprocess_text,value))
                    inputs = tokenizer(value,add_special_tokens=True,max_length=50,return_token_type_ids=True,return_length = True,truncation=True)

                    ids1= torch.tensor(sum(inputs['input_ids'],[])).view(-1)
                    mask1= torch.tensor(sum(inputs['attention_mask'],[])).view(-1)
                    token_type_ids1= torch.tensor(sum(inputs['token_type_ids'],[])).view(-1)
                    
                    if(ids1.size(0)<var):
                        ids1= torch.cat((ids1,torch.zeros(var-ids1.size(0))),dim=0)
                        mask1=torch.cat((mask1,torch.zeros(var-mask1.size(0))),dim=0)
                        token_type_ids1= torch.cat((token_type_ids1,torch.zeros(var-token_type_ids1.size(0))),dim=0)

                    elif(ids1.size(0)>var):
                        ids1=ids1[0:var]
                        mask1=mask1[0:var]
                        token_type_ids1=token_type_ids1[0:var]
                        
                    # print("sizes:")
                    # print(ids1.size())
                    # print(mask1.size())
                    # print(token_type_ids1.size())

                    input_ids[i][aspect[name]][sentiment[senti]]=ids1
                    mask[i][aspect[name]][sentiment[senti]]=mask1
                    token_type_ids[i][aspect[name]][sentiment[senti]]=token_type_ids1

        return {
            'ids':input_ids,
            'mask':mask,
            'token_type_ids':token_type_ids,
            'targets':desc}


    def __len__(self):
        return self.size



def getLoaders(batch_size):
    print('Reading the training Dataset...')
    print()
    train_dataset = Data.getReader(0,4100) #19200 #21216
    
    print('Reading the validation Dataset...')
    print()
    valid_dataset = Data.getReader(4100,5100) #23200 #25216

    print('Reading the test Dataset...')
    print()
    test_dataset = Data.getReader(5, 6) #23200:25248
    
    trainloader = DataLoader(dataset=train_dataset, batch_size = batch_size, num_workers=4,shuffle=True)
    validloader = DataLoader(dataset=valid_dataset, batch_size = batch_size, num_workers=4,shuffle=True)
    testloader = DataLoader(dataset=test_dataset, batch_size = batch_size, num_workers=4)
    
    return trainloader, validloader, testloader

