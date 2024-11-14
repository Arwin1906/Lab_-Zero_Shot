import os
import torch

def save_model(model, optimizers, epoch, stats, modelname):
    """ Saving model checkpoint """
    
    if(not os.path.exists(f"./Arwin/checkpoints/{modelname}")):
        os.makedirs(f"./Arwin/checkpoints/{modelname}")
    savepath = f"./Arwin/checkpoints/{modelname}/checkpoint_epoch_{epoch}_{modelname}.pth"

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        #'optim_branch_state_dict': optimizers["branch"].state_dict(),
        #'optim_trunk_state_dict': optimizers["trunk"].state_dict(),
        'optim_state_dict': optimizers["model"].state_dict(),
        'stats': stats
    }, savepath)
    return


def load_model(model, optimizers, savepath):
    """ Loading pretrained checkpoint """
    
    checkpoint = torch.load(savepath, map_location="cpu")
    model.load_state_dict(checkpoint['model_state_dict'])
    #optimizers["branch"].load_state_dict(checkpoint['optim_branch_state_dict'])
    #optimizers["trunk"].load_state_dict(checkpoint['optim_trunk_state_dict'])
    optimizers["model"].load_state_dict(checkpoint['optim_state_dict'])
    epoch = checkpoint["epoch"]
    stats = checkpoint["stats"]
    
    return model, optimizers, epoch, stats


def count_model_params(model):
    """ Counting the number of learnable parameters in a nn.Module """
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_params
