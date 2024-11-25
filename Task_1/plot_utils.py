import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from Task_1.utils import interpolate
import torch

@torch.no_grad()
def plot_progress(data_loader, model, device):

    fig, ax = plt.subplots(4, 3, figsize=(15,15))
    fig.set_dpi(200)
    X = np.linspace(0, 1, 128)

    matplotlib.use('Agg')  # Use Agg backend for rendering images without displaying them

    for i, (function_values, observations, masks) in enumerate(data_loader):
        
        values, times = observations
        values, times, masks = values.to(device), times.to(device), masks.to(device)
        eval_grid_points = torch.linspace(0, 1, 128, device=device)
        prediction = model(values, times, eval_grid_points, masks).detach().cpu().numpy()
        values, times, masks = values.detach().cpu().numpy(), times.detach().cpu().numpy(), masks.detach().cpu().numpy()
        ground_truth = function_values.detach().cpu().numpy()

        for j in range(len(ground_truth[0])):

            model_rmse = np.sqrt(np.mean((ground_truth[j] - prediction[j])**2))
            x_inter, y_inter = interpolate(values[j], times[j], masks[j])
            inter_rmse = np.sqrt(np.mean((ground_truth[j] - y_inter)**2))

            ax[i, 0].plot(X, ground_truth[j], color='blue')
            ax[i, 0].plot(X, prediction[j], color='green')
            mask = values[j]!=0
            ax[i, 0].plot(times[j][mask], values[j][mask], marker='.', color='red', linestyle='None')
            ax[i, 0].set_title(f" Model-RMSE: {round(model_rmse,3)}")

            ax[i, 1].plot(X, ground_truth[j], color='blue')
            ax[i, 1].plot(X, y_inter, color='orange')
            mask = values[j]!=0
            ax[i, 1].plot(times[j][mask], values[j][mask], marker='.', color='red', linestyle='None')
            ax[i, 1].set_title(f" Interpolation-RMSE: {round(inter_rmse,3)}")

            ax[i, 2].plot(X, y_inter, color='orange')
            ax[i, 2].plot(X, prediction[j], color='green')
            mask = values[j]!=0
            ax[i, 2].plot(times[j][mask], values[j][mask], marker='.', color='red', linestyle='None')

            break
            
        if i == 3:
            break

    # Create custom legend handles
    ground_truth_handle = mlines.Line2D([], [], color='blue', label='Ground truth')
    prediction_handle = mlines.Line2D([], [], color='green', label='Prediction')
    observation_handle = mlines.Line2D([], [], color='red', label='Observations', marker='.', linestyle='None')
    interpolation_handle = mlines.Line2D([], [], color='orange', label='Interpolated Values')

    # Add the custom legend to the figure
    fig.legend(handles=[ground_truth_handle, prediction_handle, observation_handle, interpolation_handle], loc='upper right',fontsize=20)
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