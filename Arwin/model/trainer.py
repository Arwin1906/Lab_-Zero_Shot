import torch
import math
import optuna
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
    
class Trainer:
    """
    Class for training and validating a transformer model
    """
    def __init__(
            self, model, criterion, train_loader, valid_loader, modelname,
            epochs=2, writer=None, inital_epoch=0, schedule=True,
            optuna_trial=None, optuna_model=False, verbose=True
        ):
        """ Trainer initializer """
        self.model = model
        self.writer = writer
        self.criterion = criterion
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.modelname = modelname
        self.inital_epoch= inital_epoch
        self.epochs = int(epochs)
        self.schedule = schedule
        self.trial = optuna_trial
        self.optuna_model = optuna_model
        self.verbose = verbose

        """
        Args:
            model: (torch.nn.Module)                        
                The model to train
            ========================================================
            criterion: (torch.nn.Module)
                The loss function to use
            ========================================================
            train_loader: (torch.utils.data.DataLoader)
                The training data loader
            ========================================================
            valid_loader: (torch.utils.data.DataLoader)
                The validation data loader
            ========================================================
            modelname: (str)
                The name of the model under which to save it (models are not sdaved if optuna trial is used)
            ========================================================
            epochs: (int)
                The number of epochs to train, default = 2
            ========================================================
            writer: (torch.utils.tensorboard.SummaryWriter)
                The tensorboard writer for logging, default = None
            ========================================================
            inital_epoch: int
                The epoch to start from, default = 0
            ========================================================
            schedule: bool
                Whether to use a learning rate scheduler, default = True
            ========================================================
            optuna_trial: optuna.Trial
                The optuna trial for lr/optimizer/scheduler optimization, default = None
            ========================================================
            optuna_model: bool
                Whether to use a model with owith optuna chosen hyperparameters, default = False
            ========================================================
            verbose: bool
                Whether to print progress, default = True
        """

        # Optimizers
        if self.trial is not None and not self.optuna_model:
            self.optim = getattr(torch.optim, self.trial.params["optimizer"])(model.parameters(), lr=self.trial.params["lr"])
        else:
            lr = 4.6e-4
            self.optim = torch.optim.AdamW(model.parameters(), lr=lr)

        # Schedulers
        if self.schedule:
            if self.trial is not None and not self.optuna_model:
                if self.trial.params["scheduler"] == "InverseSquareRootLR":
                    self.scheduler = InverseSquareRootLR(self.optim, warmup_steps=200, init_lr=self.trial.params["lr"], min_lr=self.trial.params["min_lr"])
                else:
                    scheduler_lambda = torch.optim.lr_scheduler.LambdaLR(self.optim, lr_lambda=lambda epoch: warmup_lr_lambda(epoch, 200))
                    scheduler_step = torch.optim.lr_scheduler.StepLR(self.optim, step_size=50, gamma=0.99)
                    self.scheduler = CombinedScheduler(self.optim, scheduler_lambda, scheduler_step, 200)

            else:
                self.scheduler = InverseSquareRootLR(self.optim, warmup_steps=200, init_lr=lr, min_lr=1.916e-5)
        
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

            # === Forward pass for branch and trunk ===
            eval_grid_points = torch.linspace(0, 1, 128, device=self.device)
            out = self.model(values, times, eval_grid_points)

            # Compute loss
            loss = self.criterion(out, true_values).item()

            cur_losses.append(loss)
            
            if(i >= val_iters):
                break
    
        self.valid_loss += cur_losses
        self.model.train()
        
        return cur_losses
    
    def train_one_step(self, true_values, values, times):
        """ One training step """
        self.model.train()
        # === Forward pass for branch and trunk ===

        eval_grid_points = torch.linspace(0, 1, 128, device=self.device)
        out = self.model(values, times, eval_grid_points)

        # Compute loss
        loss = self.criterion(out, true_values)
        
        # === Backward pass ===

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        if self.schedule:
            self.scheduler.step()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)      
            
        return loss

    def fit(self):
        """ Train/Validation loop """
        self.iter_ = self.inital_epoch
        torch.autograd.set_detect_anomaly(True)
        
        for ep in range(self.inital_epoch, self.epochs + self.inital_epoch):
            progress_bar = tqdm(enumerate(self.train_loader), total=(len(self.train_loader)), initial=0)
            for i, (true_values, observations) in progress_bar:     
                # setting inputs to GPU
                values, times = observations
                true_values, values, times = true_values.to(self.device), values.to(self.device), times.to(self.device)
                # forward pass and loss
                mse_loss = self.train_one_step(true_values, values, times).item()
                self.train_loss.append(mse_loss)

                # updating progress bar
                if self.verbose:
                    progress_bar.set_description(f"Ep {ep} Iter {i+1}: Loss={round(mse_loss,5)}")

                if self.writer is not None:
                    #adding loss to tensorboard
                    self.writer.add_scalar("MSE/Train", mse_loss, self.iter_)
                    self.writer.add_scalar("RMSE/Train", np.sqrt(mse_loss), self.iter_)

                    # adding lr to tensorboard
                    if self.schedule:
                        self.writer.add_scalar("LR", self.scheduler.get_last_lr()[0], self.iter_)
                    else:
                        self.writer.add_scalar("LR", 1e-4, self.iter_)
                
                # doing some validation every once in a while
                if(self.iter_ % 50 == 0):
                    cur_losses = self.valid_step()
                    # Optuna pruning
                    if self.trial is not None or self.optuna_model:
                        self.trial.report(np.mean(cur_losses), step=self.iter_)
                        if self.trial.should_prune():
                            raise optuna.TrialPruned()
                        
                    if self.verbose:
                        print(f"Valid loss @ iteration {self.iter_}: Loss={np.mean(cur_losses)}")
                        
                    # plot depth images
                    if (self.writer is not None): # only log into tensorboard and save if not optuna trial
                        self.writer.add_scalar("MSE/Valid", np.mean(cur_losses), self.iter_)
                        self.writer.add_scalar("RMSE/Valid", np.sqrt(np.mean(cur_losses)), self.iter_)

                        # plot function predictions
                        fig = plot_progress(self.valid_loader, self.model, self.device)
                        self.writer.add_figure('Predictions', fig, global_step=self.iter_)

                    # save model
                    if (self.trial is None and not self.optuna_model):
                        stats = {
                            "train_loss": self.train_loss,
                            "valid_loss": self.valid_loss
                        }
                        optimizers = {
                            "model": self.optim,
                            #"trunk": self.optim_trunk,
                        }
                        save_model(self.model, optimizers, self.iter_, stats, self.modelname)
                
                self.iter_ = self.iter_+1 
                if(self.iter_ >= (self.epochs - self.inital_epoch)*len(self.train_loader)):
                    break
            if(self.iter_ >= (self.epochs - self.inital_epoch)*len(self.train_loader)):
                break
        
        if self.trial is not None:
            return np.mean(cur_losses)
        
        return