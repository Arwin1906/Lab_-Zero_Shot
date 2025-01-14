import torch
import math
import optuna
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

from Arwin.src.utils import save_model
from Arwin.src.plot_utils import plot_progress_forecast as plot_progress

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
    
class TrainerIM2:
    """
    Class for training and validating a transformer model
    """
    def __init__(
            self, deeponet, model, criterion, train_loader, valid_loader, modelname,
            epochs=2, writer=None, inital_epoch=0, schedule=True,
            optuna_trial=None, optuna_model=False, verbose=True
        ):
        """ Trainer initializer """
        self.deeponet = deeponet
        for param in self.deeponet.parameters():
                param.requires_grad = False

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
        for i, (y_value_windows, observation_windows, mask_windows, scale_windows) in enumerate(self.valid_loader):   
            # setting inputs to GPU
            y_observation_windows, t_observation_windows = observation_windows
            y_value_windows, y_observation_windows, t_observation_windows, mask_windows = y_value_windows.to(self.device), y_observation_windows.to(self.device), t_observation_windows.to(self.device), mask_windows.to(self.device)

            # === Forward pass for branch and trunk ===

            eval_grid_points = torch.linspace(0, 1, 128, device=self.device)
            # Flatten Windows to be of shape (batch_size * num_windows, window_size)
            y_values = y_value_windows.view(-1, y_value_windows.size(2))
            y_observations = y_observation_windows.view(-1, y_observation_windows.size(2))
            t_observations = t_observation_windows.view(-1, t_observation_windows.size(2))
            masks = mask_windows.view(-1, mask_windows.size(2))
            scales = torch.stack([tensor.to(self.device) for sublist in scale_windows for tensor in sublist])

            h_b = self.deeponet(y_observations, t_observations, eval_grid_points, masks, embedd_only=True)

            # Select indices that are not every 5th element
            indices_to_keep = torch.arange(scales.size(0)) % 5 != 0

            # Compute forecast
            u_b = self.model(h_b[indices_to_keep], scales[indices_to_keep])

            # Select every 5th element
            h_b_last_window = h_b[~indices_to_keep]

            # Compute loss on embedding
            #embedding_loss = torch.nn.CosineEmbeddingLoss()(u_b, h_b_last_window, torch.ones(h_b_last_window.size(0)).to(self.device))
            #embedding_loss = self.criterion(u_b, h_b_last_window)

            # compute loss on interpolation
            y_observation_last, t_observation_last, mask_last = y_observations[~indices_to_keep], t_observations[~indices_to_keep], masks[~indices_to_keep]
            prediction = self.deeponet(y_observation_last, t_observation_last, eval_grid_points, mask_last, embedd_only=False, embedding=u_b)
            
            true_values = y_values[~indices_to_keep]
            true_scales = scales[~indices_to_keep]
            true_values = true_values * true_scales[:,2].view(-1, 1) + true_scales[:,0].view(-1, 1)

            interpolation_loss = self.criterion(prediction, true_values)

            loss = interpolation_loss.item()
            cur_losses.append(loss)
            
            if(i >= val_iters):
                break
    
        self.valid_loss += cur_losses
        self.model.train()
        
        return cur_losses
    
    def train_one_step(self, y_value_windows, y_observation_windows, t_observation_windows, mask_windows, scale_windows):
        """ One training step """
        self.model.train()
        # === Forward pass for branch and trunk ===

        eval_grid_points = torch.linspace(0, 1, 128, device=self.device)
        # Flatten Windows to be of shape (batch_size * num_windows, window_size)
        y_values = y_value_windows.view(-1, y_value_windows.size(2))
        y_observations = y_observation_windows.view(-1, y_observation_windows.size(2))
        t_observations = t_observation_windows.view(-1, t_observation_windows.size(2))
        masks = mask_windows.view(-1, mask_windows.size(2))
        scales = torch.stack([tensor.to(self.device) for sublist in scale_windows for tensor in sublist])

        h_b = self.deeponet(y_observations, t_observations, eval_grid_points, masks, embedd_only=True)

        # Select indices that are not every 5th element
        indices_to_keep = torch.arange(scales.size(0)) % 5 != 0

        # Compute forecast
        u_b = self.model(h_b[indices_to_keep], scales[indices_to_keep])

        # Select every 5th element
        h_b_last_window = h_b[~indices_to_keep]

        # Compute loss on embedding
        #embedding_loss = torch.nn.CosineEmbeddingLoss()(u_b, h_b_last_window, torch.ones(h_b_last_window.size(0)).to(self.device))
        #embedding_loss = self.criterion(u_b, h_b_last_window)

        # compute loss on interpolation
        y_observation_last, t_observation_last, mask_last = y_observations[~indices_to_keep], t_observations[~indices_to_keep], masks[~indices_to_keep]
        prediction = self.deeponet(y_observation_last, t_observation_last, eval_grid_points, mask_last, embedd_only=False, embedding=u_b)
        
        true_values = y_values[~indices_to_keep]
        true_scales = scales[~indices_to_keep]
        true_values = true_values * true_scales[:,2].view(-1, 1) + true_scales[:,0].view(-1, 1)

        interpolation_loss = self.criterion(prediction, true_values)
        
        # === Backward pass ===

        self.optim.zero_grad()
        #(embedding_loss + interpolation_loss).backward()
        interpolation_loss.backward()
        self.optim.step()

        if self.schedule:
            self.scheduler.step()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)      
            
        return interpolation_loss.item()

    def fit(self):
        """ Train/Validation loop """
        self.iter_ = self.inital_epoch
        torch.autograd.set_detect_anomaly(True)
        
        for ep in range(self.inital_epoch, self.epochs + self.inital_epoch):
            progress_bar = tqdm(enumerate(self.train_loader), total=(len(self.train_loader)), initial=0)
            for i, (y_value_windows, observation_windows,  mask_windows, scale_windows) in progress_bar:     
                # setting inputs to GPU
                y_observation_windows, t_observation_windows = observation_windows
                y_value_windows, y_observation_windows, t_observation_windows,  mask_windows = y_value_windows.to(self.device), y_observation_windows.to(self.device), t_observation_windows.to(self.device),  mask_windows.to(self.device)
                # forward pass and loss
                #embedding_loss, interpolation_loss = self.train_one_step(y_value_windows, y_observation_windows, t_observation_windows,  mask_windows, scale_windows)
                #mse_loss = embedding_loss + interpolation_loss
                interpolation_loss = self.train_one_step(y_value_windows, y_observation_windows, t_observation_windows,  mask_windows, scale_windows)
                mse_loss = interpolation_loss
                self.train_loss.append(mse_loss)

                # updating progress bar
                if self.verbose:
                    progress_bar.set_description(f"Ep {ep} Iter {i+1}: Loss={round(mse_loss,5)}")

                if self.writer is not None:
                    #adding loss to tensorboard
                    self.writer.add_scalars("MSE", {"Train": mse_loss}, self.iter_)
                    self.writer.add_scalars("RMSE", {"Train": np.sqrt(mse_loss)}, self.iter_)
                    #self.writer.add_scalars("Embedding Loss", {"Train": embedding_loss}, self.iter_)
                    self.writer.add_scalars("Interpolation Loss", {"Train": interpolation_loss}, self.iter_)
                    self.writer.add_scalar("MSE/Train", mse_loss, self.iter_)
                    self.writer.add_scalar("RMSE/Train", np.sqrt(mse_loss), self.iter_)
                    #self.writer.add_scalar("Embedding Loss/Train", embedding_loss, self.iter_)
                    self.writer.add_scalar("Interpolation Loss/Train", interpolation_loss, self.iter_)

                    # adding lr to tensorboard
                    if self.schedule:
                        self.writer.add_scalar("LR", self.scheduler.get_last_lr()[0], self.iter_)
                    else:
                        self.writer.add_scalar("LR", 1e-4, self.iter_)
                
                # doing some validation every once in a while
                if(self.iter_ % 100 == 0):
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
                        self.writer.add_scalars("MSE", {"Valid": np.mean(cur_losses)}, self.iter_)
                        self.writer.add_scalars("RMSE", {"Valid": np.sqrt(np.mean(cur_losses))}, self.iter_)
                        self.writer.add_scalar("MSE/Valid", np.mean(cur_losses), self.iter_)
                        self.writer.add_scalar("RMSE/Valid", np.sqrt(np.mean(cur_losses)), self.iter_)

                        # plot function predictions
                        fig = plot_progress(self.valid_loader, self.deeponet, self.model, self.device)
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