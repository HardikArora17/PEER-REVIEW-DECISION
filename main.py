
# #Creating the input data
# from create_data import create_data
# create_data()

#getting the dataloaders for the data 

import matplotlib.pyplot as plt
from dataloader import getLoaders
batch_size=1

train_loader,val_loader,test_loader=getLoaders(batch_size)

print("Length of TrainLoader:",len(train_loader))
print("Length of ValidLoader:",len(val_loader))
print("Length of TestLoader:",len(test_loader))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model import AADP
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm.autonotebook import tqdm


if torch.cuda.is_available():      
    device = torch.device("cuda:4")
else:
  print('No GPU available, using the CPU instead.')
  device = torch.device("cpu")

text_model=AADP()
text_model.to(device)
criterion = nn.BCELoss()

no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in text_model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in text_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.01}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=3e-5)

scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=1000)

text_model.train()
result=[]
EPOCH=15

train_out = []
val_out = []
train_true = []
val_true = []
test_out = []
test_true = []
attn_train = []
attn_val = []
attn_test = []
attn_test_senti=[]
test_out_senti=[]
test_true_senti=[]
loss_log1 = []
loss_log2 = []


for epoch in range(EPOCH):

  final_train_loss=0.0
  final_val_loss=0.0
  l1 = []
  text_model.train()

  for idx,data in tqdm(enumerate(train_loader),desc="Train epoch {}/{}".format(epoch + 1, EPOCH)):

    ids = data['ids'].to(device,dtype = torch.long)
    mask = data['mask'].to(device,dtype = torch.long)
    token_type_ids = data['token_type_ids'].to(device,dtype = torch.long)
    targets = data['targets'].to(device,dtype = torch.float)

    t1 = (ids,mask,token_type_ids)
    
    optimizer.zero_grad()
    out, attn_t,out_sen = text_model(t1)

    del t1
    del ids
    del mask
    del token_type_ids

    # print("output-1",out.size())
    # print("output-2",attn_t.size())
    # print("output-3",out_sen.size())

    # if (epoch+1 == EPOCH):
    #   train_out.append((torch.transpose(out,0,1)).detach().cpu())
    #   train_true.append((torch.transpose(targets,0,1)).detach().cpu())

    loss =criterion(out, targets)
    
    l1.append(loss.item())
    final_train_loss +=loss.item()
    loss.backward()
    optimizer.step()

    if idx % 40 == 0:
      scheduler.step()

  loss_log1.append(np.average(l1))
  
  with torch.no_grad():
    text_model.eval()
    l2 = []

    for data in tqdm(val_loader,desc="Valid epoch {}/{}".format(epoch + 1, EPOCH)):
      ids = data['ids'].to(device,dtype = torch.long)
      mask = data['mask'].to(device,dtype = torch.long)
      token_type_ids = data['token_type_ids'].to(device,dtype = torch.long)
      targets = data['targets'].to(device,dtype = torch.float)
      
      t1 = (ids,mask,token_type_ids)
      
      out_val, attn_v ,out_val_senti= text_model(t1)

      loss = criterion(out_val, targets)
      l2.append(loss.item())
      final_val_loss+=loss.item()

    loss_log2.append(np.average(l2))
    curr_lr = optimizer.param_groups[0]['lr']

  print("Epoch {}, loss: {}, val_loss: {}".format(epoch+1, final_train_loss/len(train_loader) ,final_val_loss/len(val_loader)))


plt.plot(range(len(loss_log1)), loss_log1)
plt.plot(range(len(loss_log2)), loss_log2)
plt.savefig('loss_multi.png')

torch.save(text_model.state_dict(), "decision_model.pt")

