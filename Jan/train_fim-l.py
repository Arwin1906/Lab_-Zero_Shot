
import torch.nn.utils as nn_utils
import math
import torch
from dataset import TimeSeriesDataset
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import numpy as np
from model import FIML


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
        lr = max(lr, self.min_lr)

        return [lr for _ in self.base_lrs]
    

(train_set_branch_y, train_set_branch_t, train_set_trunk_t,
 branch_mask, test_truth,stats,norm_props,samples,samples_noisy) = load_data(postfix="train_minmax_t1")
print(train_set_branch_y.shape)
dataset = TimeSeriesDataset(
    train_set_branch_y,
    train_set_branch_t,
    train_set_trunk_t,
    branch_mask,
    test_truth,
    stats,
    norm_props,
    samples
)
epochs = 20
# Initialize the DataLoader
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
model = FIML(d_model=256,heads=4)
#model.load_state_dict((torch.load("model_fim_l_2.pth")))

model = torch.compile(model)

model.to(device)
model.train()

print(f"Params: {sum(p.numel() for p in model.parameters())}")



optim = torch.optim.AdamW(model.parameters(),lr=1e-3)
lr_scheduler = InverseSquareRootLR(optim,warmup_steps=0,init_lr=1e-3,min_lr=1e-4)
scaler = torch.GradScaler()

loss = 0
for epoch in range(epochs):
    for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
        train_set_branch_y = batch["train_set_branch_y"].to('cuda')#[0]
        train_set_branch_t = batch["train_set_branch_t"].to('cuda')#[0]
        train_set_trunk_t = batch["train_set_trunk"].to('cuda')#[0]

        branch_mask = batch["branch_mask"].to('cuda')#[0]
        
        test_truth = batch["test_truth"].to('cuda')# [0]
        out,_ = model(train_set_branch_y,train_set_branch_t,train_set_trunk_t,branch_mask)
        loss = ((out-test_truth)**2).mean() #mse
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optim.step()

        lr_scheduler.step()
        optim.zero_grad()
   #     print(loss)

    
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

        #break
    #break
    

    torch.save(model._orig_mod.state_dict(), 'model_fim_l.pth')

        