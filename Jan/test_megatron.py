
import torch.nn.utils as nn_utils
import math
import torch
from dataset import TimeSeriesDataset
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import numpy as np
from model import MegaTron


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(folder="saved_data",postfix="train"):
    filenames = [
        "train_set_branch_y.npy", "train_set_branch_t.npy", "train_set_trunk.npy",
        "branch_mask.npy", "test_truth.npy","stats.npy","norm_props.npy","samples.npy","samples_noisy.npy"

    ]
    arrays = [np.load(os.path.join(folder+"_"+postfix, filename), allow_pickle=True) for filename in filenames]
    print("Data loaded successfully.")
    return arrays

grid_size = 640
temporal_size = 20

class InverseSquareRootLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, init_lr, min_lr=1e-9, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.init_lr = init_lr
        self.min_lr = min_lr
        super(InverseSquareRootLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1
        if step < self.warmup_steps:
            # Linear warmup
            lr = self.init_lr * (step / self.warmup_steps)
        else:
            # Inverse square root decay
            lr = self.init_lr * math.sqrt(self.warmup_steps / step)

        # Ensure learning rate doesn't go below minimum
     #   print(lr)
        lr = max(lr, self.min_lr)

        return [lr for _ in self.base_lrs]
    

(train_set_branch_y, train_set_branch_t, train_set_trunk_t,
 branch_mask, test_truth,stats,norm_param,samples,samples_noisy) =  load_data(postfix="train_minmax_big")
n = 20000
dataset = TimeSeriesDataset(
    train_set_branch_y.reshape(-1,5,grid_size//5),
    train_set_branch_t.reshape(-1,5,grid_size//5),
    train_set_trunk_t.reshape(-1,5,grid_size//5),
    branch_mask.reshape(-1,5,grid_size//5),
    test_truth.reshape(-1,5,grid_size//5),
    stats.reshape(-1,5,9),
    norm_param.reshape(-1,5,4),
    samples.reshape(-1,5,grid_size//5),
)
epochs = 25
# Initialize the DataLoader
dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
model = MegaTron(d_model=256,heads=4)
model.load_state_dict((torch.load("model_megatron.pth")))

model = torch.compile(model)

model.to(device)
model.train()

print(f"Params: {sum(p.numel() if p.requires_grad else 0 for p in model.parameters())}")



optim = torch.optim.AdamW(model.parameters(),lr=1e-4)
lr_scheduler = InverseSquareRootLR(optim,warmup_steps=0,init_lr=1e-4,min_lr=1e-4)
scaler = torch.GradScaler()

loss = 0
alpha = 0.1
for epoch in range(epochs):
    losses = []
    for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}, lr: {lr_scheduler.get_lr()[0]}", leave=False):
        train_set_branch_y = batch["train_set_branch_y"].to('cuda')#[0]
        train_set_branch_t = batch["train_set_branch_t"].to('cuda')#[0]
        train_set_trunk_t = batch["train_set_trunk"].to('cuda')#[0]
        branch_mask = batch["branch_mask"].to('cuda')#[0]
        test_truth = batch["test_truth"].to('cuda')# [0]
        stats = batch["stats"].to('cuda')# [0]
        samples = batch["samples"].to('cuda')

        out,cosine_sim = model(train_set_branch_y,train_set_branch_t,train_set_trunk_t,branch_mask,stats)

       
        loss = ((out-samples[:,-1])**2).mean() + alpha*(torch.abs(cosine_sim - 1)).mean() #mse
        if torch.isnan(loss).any():
            raise Exception("NaN loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        losses.append(loss.item())

       
        optim.step()

        lr_scheduler.step()
        optim.zero_grad()
   #     print(loss)"""
        
    

    torch.save(model._orig_mod.state_dict(), 'model_megatron.pth')
    
    print(f"Epoch {epoch+1}/{epochs}, Loss: {np.mean(np.array(losses))}")

        #break
    #break
    


