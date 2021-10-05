
# #Creating the input data
# from create_data import create_data
# create_data()

#getting the dataloaders for the data 


import matplotlib.pyplot as plt
from dataloader import getLoaders
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model import AADP
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm.autonotebook import tqdm
import os
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

if torch.cuda.is_available():      
    device = torch.device("cuda:4")
else:
  print('No GPU available, using the CPU instead.')
  device = torch.device("cpu")

n_gpus = torch.cuda.device_count()
assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
world_size = n_gpus

#============================================================================================================================================
batch_size=1

train_loader,val_loader,test_loader=getLoaders(batch_size)

print("Length of TrainLoader:",len(train_loader))
print("Length of ValidLoader:",len(val_loader))
print("Length of TestLoader:",len(test_loader))

text_model1=AADP()

def setup(rank, world_size):
    torch.distributed.init_process_group(backend='nccl', world_size=world_size)


def cleanup():
    dist.destroy_process_group()

def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)


def demo_basic(rank, world_size):
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)

    # create model and move it to GPU with id rank
    model = text_model1().to(rank)
    text_model = DDP(model, device_ids=[rank])

    criterion= nn.BCELoss()
    
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

    cleanup()


run_demo(demo_basic, world_size)




