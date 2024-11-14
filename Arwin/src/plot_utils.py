import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

import torch

@torch.no_grad()
def plot_progress(data_loader, model, device):

    fig, ax = plt.subplots(4, 3, figsize=(15,15))
    X = np.linspace(0, 1, 128)

    matplotlib.use('Agg')  # Use Agg backend for rendering images without displaying them

    for i, (function_values, observations) in enumerate(data_loader):
        
        values, times = observations
        values, times = values.to(device), times.to(device)
        prediction = model(values, times).detach().cpu().numpy()
        values, times = values.detach().cpu().numpy(), times.detach().cpu().numpy()
        ground_truth = function_values.detach().cpu().numpy()

        for j in range(len(ground_truth[0])):
            ax[i, 0].plot(X, ground_truth[j])
            ax[i, 0].plot(X, prediction[j])
            mask = values[j]!=0
            ax[i, 0].plot(times[j][mask], values[j][mask], marker='.', color='red', linestyle='None')

            ax[i, 1].plot(X, ground_truth[j+1])
            ax[i, 1].plot(X, prediction[j+1])
            mask = values[j+1]!=0
            ax[i, 1].plot(times[j+1][mask], values[j+1][mask], marker='.', color='red', linestyle='None')

            ax[i, 2].plot(X, ground_truth[j+2])
            ax[i, 2].plot(X, prediction[j+2])
            mask = values[j+2]!=0
            ax[i, 2].plot(times[j+2][mask], values[j+2][mask], marker='.', color='red', linestyle='None')

            break
            
        if i == 3:
            break

    # Create custom legend handles
    ground_truth_handle = mlines.Line2D([], [], color='blue', label='Ground truth')
    prediction_handle = mlines.Line2D([], [], color='orange', label='Prediction')
    observation_handle = mlines.Line2D([], [], color='red', label='Observations', marker='.', linestyle='None')

    # Add the custom legend to the figure
    fig.legend(handles=[ground_truth_handle, prediction_handle, observation_handle], loc='upper right',fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    return fig

def smooth(f, K=5):
    """ Smoothing a function using a low-pass filter (mean) of size K """
    kernel = np.ones(K) / K
    f = np.concatenate([f[:int(K//2)], f, f[int(-K//2):]])  # to account for boundaries
    smooth_f = np.convolve(f, kernel, mode="same")
    smooth_f = smooth_f[K//2: -K//2]  # removing boundary-fixes
    return smooth_f

def visualize_progress(train_loss, val_loss, start=0):
    """ Visualizing loss and accuracy """
    fig, ax = plt.subplots(1,3)
    fig.set_size_inches(24,5)

    smooth_train = smooth(train_loss, 31)
    ax[0].plot(train_loss, c="blue", label="Loss", linewidth=3, alpha=0.5)
    ax[0].plot(smooth_train, c="red", label="Smoothed Loss", linewidth=3, alpha=1)
    ax[0].legend(loc="best")
    ax[0].set_xlabel("Iteration")
    ax[0].set_ylabel("CE Loss")
    ax[0].set_yscale("linear")
    ax[0].set_title("Training Progress (linear)")
    
    ax[1].plot(train_loss, c="blue", label="Loss", linewidth=3, alpha=0.5)
    ax[1].plot(smooth_train, c="red", label="Smoothed Loss", linewidth=3, alpha=1)
    ax[1].legend(loc="best")
    ax[1].set_xlabel("Iteration")
    ax[1].set_ylabel("CE Loss")
    ax[1].set_yscale("log")
    ax[1].set_title("Training Progress (log)")

    smooth_val = smooth(val_loss, 31)
    N_ITERS = len(val_loss)
    ax[2].plot(np.arange(start, N_ITERS)+start, val_loss[start:], c="blue", label="Loss", linewidth=3, alpha=0.5)
    ax[2].plot(np.arange(start, N_ITERS)+start, smooth_val[start:], c="red", label="Smoothed Loss", linewidth=3, alpha=1)
    ax[2].legend(loc="best")
    ax[2].set_xlabel("Iteration")
    ax[2].set_ylabel("CE Loss")
    ax[2].set_yscale("log")
    ax[2].set_title(f"Valid Progress")

    return