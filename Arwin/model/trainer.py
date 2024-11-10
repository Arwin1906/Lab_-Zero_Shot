import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

from Arwin.src.utils import save_model
from Arwin.src.plot_utils import plot_progress

def warmup_lr_lambda(epoch, warmup_epochs):
    """ Learning rate scheduler with linear warmup """
    if epoch < warmup_epochs:
        # Linear warmup
        return epoch/warmup_epochs
    else:
        # No scaling after warmup
        return 1.0
    
class CombinedScheduler(torch.optim.lr_scheduler._LRScheduler):
    """ Combined scheduler for warmup and main scheduler """
    def __init__(self, optimizer, warmup_scheduler, main_scheduler, warmup_epochs):
        self.warmup_scheduler = warmup_scheduler
        self.main_scheduler = main_scheduler
        self.warmup_epochs = warmup_epochs
        self.transitioned = False  # To track if we have transitioned
        super(CombinedScheduler, self).__init__(optimizer)

    def get_last_lr(self):
        if not self.transitioned:
            return self.warmup_scheduler.get_last_lr()
        else:
            return self.main_scheduler.get_last_lr()

    def step(self):
        if self.last_epoch < self.warmup_epochs:
            self.warmup_scheduler.step()
        else:
            # Only transition once to avoid double step at boundary
            if not self.transitioned:
                # Adjust main_scheduler epoch by subtracting warmup epochs to sync the steps
                main_epoch = self.last_epoch - self.warmup_epochs
                self.main_scheduler.last_epoch = main_epoch
                self.transitioned = True

            # Step main scheduler normally after warmup
            self.main_scheduler.step(self.last_epoch - self.warmup_epochs)

        self.last_epoch += 1
    
class Trainer:
    """
    Class for training and validating a transformer model
    """
    def __init__(self, model, criterion, train_loader, valid_loader, modelname, n_iters=2, writer=None, inital_iter=0, schedule=True):
        """ Trainer initializer """
        self.model = model
        self.writer = writer
        self.criterion = criterion
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.modelname = modelname
        self.inital_iter= inital_iter
        self.n_iters = int(n_iters)
        self.schedule = schedule

        lr_branch = 1e-4
        lr_trunk = 1e-4

        # Optimizers
        self.optim_branch = torch.optim.Adam(model.branch.parameters(), lr=lr_branch)
        self.optim_trunk = torch.optim.Adam(model.trunk.parameters(), lr=lr_trunk)

        # Schedulers
        if self.schedule:

            scheduler_lambda_branch = torch.optim.lr_scheduler.LambdaLR(self.optim_branch, lr_lambda=lambda epoch: warmup_lr_lambda(epoch, 200))
            scheduler_step_branch = torch.optim.lr_scheduler.StepLR(self.optim_branch, step_size=50, gamma=0.99)
            self.scheduler_branch = CombinedScheduler(self.optim_branch, scheduler_lambda_branch, scheduler_step_branch, 200)

            scheduler_lambda_trunk = torch.optim.lr_scheduler.LambdaLR(self.optim_trunk, lr_lambda=lambda epoch: warmup_lr_lambda(epoch, 200))
            scheduler_step_trunk = torch.optim.lr_scheduler.StepLR(self.optim_trunk, step_size=50, gamma=0.99)
            self.scheduler_trunk = CombinedScheduler(self.optim_trunk, scheduler_lambda_trunk, scheduler_step_trunk, 200)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        self.train_loss = []
        self.valid_loss = []
        return
    
    @torch.no_grad()
    def valid_step(self, val_iters=50):
        """ Some validation iterations """
        self.model.eval()
        cur_losses = []
        for i, (true_values, observations) in enumerate(self.valid_loader):   
            # setting inputs to GPU
            values, times = observations
            true_values, values, times = true_values.to(self.device), values.to(self.device), times.to(self.device)

            # === Step 1: Forward pass for branch and trunk ===
            branch_output = self.model.branch(values, times)

            trunk_output = self.model.trunk(self.model.fine_grid_points_batch.unsqueeze(-1))

            final_output = torch.sum(trunk_output * branch_output.unsqueeze(-1), dim=-1)

            # Compute loss
            loss = self.criterion(final_output, true_values)

            cur_losses.append(loss)
            
            if(i >= val_iters):
                break
    
        self.valid_loss += cur_losses
        self.model.train()
        
        return cur_losses
    
    def train_one_step(self, true_values, values, times):
        """ One training step """
        self.model.train()
        # === Step 1: Forward pass for branch and trunk ===
        branch_output = self.model.branch(values, times)

        trunk_output = self.model.trunk(self.model.fine_grid_points_batch.unsqueeze(-1))

        final_output = torch.sum(trunk_output * branch_output.unsqueeze(-1), dim=-1)

        # Compute loss
        loss = self.criterion(final_output, true_values)
        
        # === Step 2: Backward pass ===

        self.optim_branch.zero_grad()
        self.optim_trunk.zero_grad()

        loss.backward()

        self.optim_branch.step()
        self.optim_trunk.step()

        if self.schedule:
            self.scheduler_branch.step()
            self.scheduler_trunk.step()          
            
        return loss

    def fit(self):
        """ Train/Validation loop """
        self.iter_ = self.inital_iter
        torch.autograd.set_detect_anomaly(True)
        
        for ep in range(self.inital_iter, self.n_iters + self.inital_iter):
            progress_bar = tqdm(enumerate(self.train_loader), total=(len(self.train_loader)), initial=0)
            for i, (true_values, observations) in progress_bar:     
                # setting inputs to GPU
                values, times = observations
                true_values, values, times = true_values.to(self.device), values.to(self.device), times.to(self.device)
                # forward pass and loss
                mse_loss = self.train_one_step(true_values, values, times).item()

                self.train_loss.append(mse_loss)
            
                # updating progress bar
                progress_bar.set_description(f"Ep {ep} Iter {i+1}: Loss={round(mse_loss,5)}")

                #adding loss to tensorboard
                self.writer.add_scalars("Loss/Train", {
                    'MSE': mse_loss,
                }, self.iter_)

                # adding lr to tensorboard
                if self.schedule:
                    self.writer.add_scalar("LR", self.scheduler_branch.get_last_lr()[0], self.iter_)
                else:
                    self.writer.add_scalar("LR", 1e-4, self.iter_)
                
                # doing some validation every once in a while
                if(self.iter_ % 250 == 0):
                    cur_losses = self.valid_step()
                    print(f"Valid loss @ iteration {self.iter_}: Loss={np.mean(cur_losses)}")
                    # plot depth images
                    self.writer.add_scalar("Loss/Valid", np.mean(cur_losses), self.iter_)

                    # plot function predictions
                    fig = plot_progress(self.valid_loader, self.model)
                    self.writer.add_figure('Predictions', fig, global_step=self.iter_)

                    # save model
                    stats = {
                        "train_loss": self.train_loss,
                        "valid_loss": self.valid_loss
                    }
                    optimizers = {
                        "branch": self.optim_branch,
                        "trunk": self.optim_trunk,
                    }
                    save_model(self.model, optimizers, self.iter_, stats, self.modelname)
                
                self.iter_ = self.iter_+1 
                if(self.iter_ >= (self.n_iters - self.inital_iter)*len(self.train_loader)):
                    break
            if(self.iter_ >= (self.n_iters - self.inital_iter)*len(self.train_loader)):
                break
            
        return