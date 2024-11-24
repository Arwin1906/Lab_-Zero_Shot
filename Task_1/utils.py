import os
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

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

def save_model(model, optimizers, epoch, stats, modelname):
    """ Saving model checkpoint """
    
    if(not os.path.exists(f"./Arwin/checkpoints/{modelname}")):
        os.makedirs(f"./Arwin/checkpoints/{modelname}")
    savepath = f"./Arwin/checkpoints/{modelname}/checkpoint_epoch_{epoch}_{modelname}.pth"

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
