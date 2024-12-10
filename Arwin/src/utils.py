import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from scipy.interpolate import UnivariateSpline

def collate_fn_windows(batch):
    """
    Collate function for a batch of windows.
    Calls collate_fn_fixed for each window and preserves the structure.
    """
    y_value_windows, y_observation_windows, t_observation_windows, mask_windows, s_values = zip(*batch)
    # Process each set of windows in the batch using collate_fn_fixed
    collated_y_values = []
    collated_observations = []
    collated_masks = []
    
    for i in range(len(y_value_windows)):  # Loop over functions in the batch
        function_values = y_value_windows[i]
        observations = list(zip(y_observation_windows[i], t_observation_windows[i]))
        masks = mask_windows[i]
        # Combine into a batch and process with collate_fn_fixed
        combined_batch = list(zip(function_values, observations, masks))
        processed_values, processed_observations, processed_masks = collate_fn_fixed(combined_batch)
        
        collated_y_values.append(processed_values)
        collated_observations.append(processed_observations)
        collated_masks.append(processed_masks)
    
    # Stack across functions
    collated_y_values = torch.stack(collated_y_values)
    collated_observations = (torch.stack([obs[0] for obs in collated_observations]),
                             torch.stack([obs[1] for obs in collated_observations]))
    collated_masks = torch.stack(collated_masks)
    
    return collated_y_values, collated_observations, collated_masks, s_values

def collate_fn_fixed(batch):
    """
    Collate function for the synthetic dataset.
    Pads each sequence in the batch to a fixed length of 128.
    """
    # Define the fixed length
    max_len = 128

    # Unpack the batch into function values, observations and masks
    function_values, observations, masks = zip(*batch)
    
    # Convert function values to tensors (in case they’re numpy arrays)
    function_values = torch.stack([torch.tensor(f, dtype=torch.float32) for f in function_values])
    
    # Separate values and times in observations, and convert to tensors
    values, times = zip(*observations)
    values = [torch.tensor(v, dtype=torch.float32) for v in values]
    times = [torch.tensor(t, dtype=torch.float32) for t in times]
    masks = [torch.tensor(m, dtype=torch.float32) for m in masks]
    
    # Manually pad values, times and masks sequences to a fixed length of 128
    padded_values = [F.pad(v, (0, max_len - len(v)), value=0) if len(v) < max_len else v[:max_len] for v in values]
    padded_times = [F.pad(t, (0, max_len - len(t)), value=0) if len(t) < max_len else t[:max_len] for t in times]
    padded_masks = [F.pad(m, (0, max_len - len(m)), value=0) if len(m) < max_len else m[:max_len] for m in masks]
    
    # Stack the padded sequences into tensors
    padded_values = torch.stack(padded_values, dim=0)  # Shape: [batch, 128]
    padded_times = torch.stack(padded_times, dim=0)    # Shape: [batch, 128]
    padded_masks = torch.stack(padded_masks, dim=0)    # Shape: [batch, 128]
    
    return function_values, (padded_values, padded_times), padded_masks

def collate_fn(batch):
    """
    Collate function for the synthetic dataset.
    Pads each sequence in the batch dynamically to the longest sequence in the current batch.
    """

    # Unpack the batch into function values, observations and masks
    function_values, observations, masks = zip(*batch)
    
    # Convert function values to tensors (in case they’re numpy arrays)
    function_values = torch.stack([torch.tensor(f, dtype=torch.float32) for f in function_values])
    
    # Separate values and times in observations, and convert to tensors
    values, times = zip(*observations)
    values = [torch.tensor(v, dtype=torch.float32) for v in values]
    times = [torch.tensor(t, dtype=torch.float32) for t in times]
    masks = [torch.tensor(m, dtype=torch.float32) for m in masks]
    
    # Dynamically pad values, times and masks sequences to the longest sequence in the current batch
    padded_values = pad_sequence(values, batch_first=True, padding_value=0)
    padded_times = pad_sequence(times, batch_first=True, padding_value=0)
    padded_masks = pad_sequence(masks, batch_first=True, padding_value=0)
    
    return function_values, (padded_values, padded_times), padded_masks

def interpolate(y,t,mask):
    # Suppose x_total and y_total represent the full set of 128 points
    i = int(mask.sum())
    t = t[:i]
    y = y[:i]
    sorted_indices = np.argsort(t)

    x_sampled = t[sorted_indices]
    y_sampled = y[sorted_indices]

    # Create the smoothing spline interpolation function
    # The smoothing factor 's' can be adjusted based on the desired smoothness
    smoothing_factor = 1.75  # Adjust this parameter as needed
    spline = UnivariateSpline(x_sampled, y_sampled, s=smoothing_factor)

    # Generate 128 evenly spaced points within the range of x_sampled
    x_new = np.linspace(x_sampled.min(), x_sampled.max(), 128)
    y_new = spline(x_new)

    return x_new,y_new

def save_model(model, optimizers, epoch, stats, modelname):
    """ Saving model checkpoint """
    
    if(not os.path.exists(f"./Task_1/checkpoints/{modelname}")):
        os.makedirs(f"./Task_1/checkpoints/{modelname}")
    savepath = f"./Task_1/checkpoints/{modelname}/checkpoint_epoch_{epoch}_{modelname}.pth"

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optim_state_dict': optimizers["model"].state_dict(),
        'stats': stats
    }, savepath)
    return


def load_model(model, optimizers, savepath):
    """ Loading pretrained checkpoint """
    
    checkpoint = torch.load(savepath, map_location="cpu")
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizers["model"].load_state_dict(checkpoint['optim_state_dict'])
    epoch = checkpoint["epoch"]
    stats = checkpoint["stats"]
    
    return model, optimizers, epoch, stats


def count_model_params(model):
    """ Counting the number of learnable parameters in a nn.Module """
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_params
