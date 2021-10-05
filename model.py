import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoModel


#If there's a GPU available...
if torch.cuda.is_available():    

    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda:4")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

else:
  print('No GPU available, using the CPU instead.')
  device = torch.device("cpu")

model_name='bert-base-uncased'

class AADP(nn.Module):
    def __init__(self):

        super(AADP, self).__init__()
        self.model=AutoModel.from_pretrained(model_name)
        
        self.flatten=nn.Flatten()
        self.lstm_1 = nn.LSTM(768, 512//2, batch_first=True, bidirectional=True) #bidirectional=True
    
        self.linear1=     nn.Linear(512*4,512*2)
        self.linear2=     nn.Linear(512*2,256)
        self.linear3=     nn.Linear(256,64)
        self.last_dense = nn.Linear(64,2)

        self.dropout1=nn.Dropout(p=0.5)
        
        self.relu = nn.ReLU()
        self.sigmoid=nn.Sigmoid()
        self.tanh=nn.Tanh()

        aspect_imp = torch.rand(512, 1,requires_grad=True)  #(512,1)
        nn.init.xavier_normal_(aspect_imp)
        
        aspect_polarity = torch.rand(512, 1,requires_grad=True)  #(512,1)
        nn.init.xavier_normal_(aspect_polarity)
        
        self.aspect_imp=aspect_imp.to(device)
        self.aspect_polarity=aspect_polarity.to(device)


    def forward(self, t1):
        #torch.cuda.empty_cache()
        #print("hi")
        ids1, mask1, token_type_ids1 = t1
        del t1
        
        ids=ids1.view(-1,ids1.size(4))      #(1,4,7,2,200)
        mask1=mask1.view(-1,mask1.size(4))
        token_type_ids1=token_type_ids1.view(-1,token_type_ids1.size(4))

        # print(ids.size())
        # print(mask1.size())
        # print(token_type_ids1.size())

        encoded_layers = self.model(ids, attention_mask = mask1, token_type_ids = token_type_ids1,return_dict=False)[1]
        del ids,mask1,token_type_ids1
        #print(type(encoded_layers))
        #print(encoded_layers.size())

        s_e = encoded_layers.view(ids1.size(0),-1,768)
        del encoded_layers
        #print("hello")
        h0 = torch.zeros(2, s_e.size(0), 512 // 2)
        c0 = torch.zeros(2, s_e.size(0), 512 // 2)
        h0, c0 = h0.to(device), c0.to(device)
        s_e, (hn, cn) = self.lstm_1(s_e, (h0, c0))         #(1,4,7,2,512)
        s_e=s_e.view(ids1.size(0),ids1.size(1),7,2,-1)
        #print(ids.size())
        del hn,cn
        del h0,c0
        #print(s_e.size())
        #print("hello")
        ap=((self.aspect_polarity.unsqueeze(0)).unsqueeze(0)).unsqueeze(0)    #(1,1,1,512,1)
        #print("hello-====================================")
        #print(ap.size())  
        comp_p=torch.matmul(s_e,ap)                          #(4,4,7,2,1)
        #print(comp_p.size())
        wts_p = F.softmax(comp_p, dim=3)                      #(4,4,7,2,1)
        #print(wts_p.size())
        s_e_p=s_e.permute(0,1,2,4,3)                          #(4,4,7,512,2)
        #print(s_e_p.size())
        s_e= torch.matmul(s_e_p,wts_p).squeeze(-1)                 #(4,4,7,512)

        del ap,comp_p,s_e_p

        ai=((self.aspect_imp.unsqueeze(0)).unsqueeze(0))  #(1,1,512,1)
        comp1= s_e.permute(0,1,3,2)                       #(4,4,512,7)
         
        #print(s_e.size())
        #print(ai.size())
        comp=torch.matmul(s_e,ai)                        #(4,4,7,1)
        wts = F.softmax(comp, dim=2)                      #(4,4,7,1)
    
        e=torch.matmul(comp1,wts).squeeze(-1)             #(4,4,512)  +(4,4,512) +(4,4,512)
        #print(e.size())
 
        del ai,comp1,s_e,comp

        l = torch.reshape(e, (ids1.size(0), 512*4))        #(4,512*4)

        l = self.relu(self.linear1(l))                    #(4,512*2)
        l = self.dropout1(l)                              
        l = self.relu(self.linear2(l))                    #(4,256)
        l = self.dropout1(l)                              
        l = self.relu(self.linear3(l))                    #(4,64)

        model_output = self.sigmoid(self.last_dense(l))
        
        del l
        return model_output, wts,wts_p