import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import beta as beta_dist  # Import with a different name
from torch.utils.data import Dataset, DataLoader
from dataset import TimeSeriesDataset
from model import DeepONet
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def prepare_data(grid_size,temporal_size):
    # Parameters
    num_functions = 10000    # Total number of functions to generate
    batch_size = 1000         # Batch size for processing
    n_batches = num_functions // batch_size


    alpha = 2.0
    beta = 5.0

    # Gaussian parameters for noise std


    def rademacher_mask(n):
        rademacher_values = np.random.choice([-1, 1], size=n)    
        mask = (rademacher_values + 1) // 2  # This will map -1 to 0 and +1 to 1
        return mask

    def rbf_kernel_matrix(X, length_scale=1.0, variance=1.0):
        """
        Compute the RBF kernel matrix for input array X.
        """
        # Compute pairwise squared distances
        X = X.reshape(-1, 1)
        sqdist = np.sum(X**2, axis=1).reshape(-1, 1) + \
                np.sum(X**2, axis=1) - 2 * np.dot(X, X.T)
        K = variance * np.exp(-0.5 * sqdist / length_scale**2)
        return K

    variance = 1.0           # Variance parameter σ²
    alpha, beta_param = 6.0, 7.0
    # Create input grid
    X = np.linspace(0, 1, grid_size)  # Points in [0, 1]


    train_set_branch_y_collection = []
    train_set_branch_t_collection = []

    train_set_trunk_collection = []
    branch_mask_collection = []
    trunk_mask_collection = []
    test_truth_collection = []

    for i in range(num_functions):
        if i % 1000 == 0:
            print(i)
        l = beta_dist.rvs(alpha, beta_param)

        K = rbf_kernel_matrix(X, length_scale=l, variance=variance)

        mean = np.zeros(grid_size)
        noise_std = np.abs(np.random.normal(0, 0.1))

        samples = np.random.multivariate_normal(mean, K, size=1).reshape(-1)
        samples_noisy = samples + np.random.normal(0, noise_std, size=grid_size)

        samples = (samples - samples.mean()) / samples.std() #z-scoring
        samples_noisy = (samples_noisy - samples_noisy.mean()) / samples_noisy.std() #z-scoring

        test_indices = np.random.choice(grid_size, temporal_size, replace=False)
        train_indices = np.setdiff1d(np.arange(grid_size), test_indices)

        train_set_branch_y = samples_noisy[train_indices] #'TODDO NOISY'
        train_set_branch_t = X[train_indices]

        train_set_trunk = X[test_indices]

        branch_mask = rademacher_mask(len(train_set_branch_y))
        trunk_mask = rademacher_mask(len(train_set_trunk))

        test_truth = samples[test_indices]

        train_set_branch_y_collection.append(train_set_branch_y)
        train_set_branch_t_collection.append(train_set_branch_t)

        train_set_trunk_collection.append(train_set_trunk)
        branch_mask_collection.append(branch_mask)
        trunk_mask_collection.append(trunk_mask)
        test_truth_collection.append(test_truth)

    return (train_set_branch_y_collection,train_set_branch_t_collection, train_set_trunk_collection, branch_mask_collection, trunk_mask_collection, test_truth_collection)

grid_size = 128
temporal_size = 20
(train_set_branch_y,
 train_set_branch_t,
 train_set_trunk,
 branch_mask,
 trunk_mask,
 test_truth) = prepare_data(grid_size,temporal_size)

dataset = TimeSeriesDataset(
    train_set_branch_y,
    train_set_branch_t,
    train_set_trunk,
    branch_mask,
    trunk_mask,
    test_truth
)
epochs = 10
# Initialize the DataLoader
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
model = DeepONet(grid_size-temporal_size,temporal_size,d_model=256,heads=1)

model.to(device)
model.train()

optim = torch.optim.AdamW(model.parameters(),lr=0.00001)
loss = 0
"""for epoch in range(epochs):
    for batch in dataloader:
        train_set_branch_y = batch["train_set_branch_y"].to(device)
        train_set_branch_t = batch["train_set_branch_y"].to(device)

        train_set_trunk = batch["train_set_trunk"].to(device)

        branch_mask = batch["branch_mask"].to(device)
        trunk_mask = batch["trunk_mask"].to(device)


        test_truth = batch["test_truth"].to(device)
       
        out = model(train_set_branch_y,train_set_branch_t,train_set_trunk,branch_mask,trunk_mask)
        loss = ((out-test_truth)**2).mean() #mse
        loss.backward()
        optim.step()
        optim.zero_grad()
    print(loss) 
"""
model.eval()
for  batch in dataloader:
    train_set_branch_y = batch["train_set_branch_y"].to(device)
    train_set_branch_t = batch["train_set_branch_y"].to(device)

    train_set_trunk = batch["train_set_trunk"].to(device)

    branch_mask = batch["branch_mask"].to(device)
    trunk_mask = batch["trunk_mask"].to(device)


    test_truth = batch["test_truth"].to(device)



    
    out = model(train_set_branch_y,train_set_branch_t,train_set_trunk,branch_mask,trunk_mask)
    #print(out)
    print(test_truth.shape,trunk_mask.shape)
    break