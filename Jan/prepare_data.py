import numpy as np
import os

def save_data(*arrays, folder="saved_data",postfix="train"):
    os.makedirs(folder, exist_ok=True)  # Create the folder if it doesn't exist
    filenames = [
        "train_set_branch_y.npy", "train_set_branch_t.npy", "train_set_trunk.npy",
        "branch_mask.npy", "test_truth.npy","stats.npy","norm_props.npy","samples.npy","samples_noisy.npy"
    ]
    for array, filename in zip(arrays, filenames):
        np.save(os.path.join(folder+"_"+postfix, filename), array)
    print("Data saved successfully.")

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

import random
import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import beta as beta_dist  # Import with a different name
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def prepare_data(num_functions,grid_size,train=True):
    # Parameters
      # Total number of functions to generate
    batch_size = 1        # Batch size for processing
    
    def periodic_kernel_matrix(X, length_scale=1.0, variance=1.0, p=0.1):
  
        X = np.asarray(X).reshape(-1, 1)
        abs_diff = np.abs(X - X.T)
        sin_squared = np.sin(np.pi * abs_diff / p) ** 2
        K = variance * np.exp(-2 * sin_squared / length_scale**2)
        return K

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
    def locally_periodic_kernel_matrix(X, length_scale=1.0, variance=1.0, p=0.1):

        X = np.asarray(X).reshape(-1, 1)    
        abs_diff = np.abs(X - X.T)    
        sin_squared = np.sin(np.pi * abs_diff / p) ** 2    
        k_per = variance * np.exp(-2 * sin_squared / length_scale**2)
        
        sq_diff = (X - X.T) ** 2
        

        k_se = np.exp(-sq_diff / (2 * length_scale**2))
        
        K = k_per * k_se
        
        return K

    def linear_kernel_matrix(X, variance=1.0):
        X = X.reshape(-1, 1)
        K = variance * np.dot(X, X.T)
        return K
    
    def linear_plus_periodic_kernel_matrix(X, lin_variance=1.0, per_length_scale=1.0, per_variance=1.0, p=0.1):
        K_lin = linear_kernel_matrix(X, variance=lin_variance)
        K_per = periodic_kernel_matrix(X, length_scale=per_length_scale, variance=per_variance, p=p)
        return K_lin + K_per 
    def get_kernel(X,l,v,p):
        import random
        r = random.uniform(0, 1)
        if r <=0.3:
            return periodic_kernel_matrix(X,l,v,p)
        elif r > .3 and r <= .6:
            return locally_periodic_kernel_matrix(X,l,v,p)
        else:
            return linear_plus_periodic_kernel_matrix(X,v,l,v,p)
    


    variance = 1.0           # Variance parameter σ²
    min = 50
    max = 90
    # Create input grid
    X = np.linspace(0, 1, grid_size)  # Points in [0, 1]


    train_set_branch_y_collection = []
    train_set_branch_t_collection = []

    train_set_trunk_collection = []
    branch_mask_collection = []
    test_truth_collection = []
    stats_collection = []
    norm_props_collection = []

    samples_ = []
    samples_noisy_ = []

    i = 0
    train_dist = [1.0,2.0,5.0]
    test_dist = [1.5,3.0,7.0]
    dist = train_dist if train else test_dist

    for j in tqdm(range(num_functions), desc=f"NUM {i+1}/{100000}", leave=False):
        try:
            alpha, beta_param = np.random.choice(dist, 1,replace=False)[0],np.random.choice(dist, 1,replace=False)[0]
            i = j
            l,p = beta_dist.rvs(alpha, beta_param),beta_dist.rvs(alpha, beta_param)
            K = get_kernel(X,l,variance,p)

            mean = np.zeros(grid_size)
            noise_std = np.abs(np.random.normal(0, 0.1))

            samples = np.random.multivariate_normal(mean, K, size=1).reshape(-1)
            samples_noisy = samples.copy() + np.random.normal(0, noise_std, size=grid_size)
            
            samples = samples.reshape(5,-1)
            samples_noisy = samples_noisy.reshape(5,-1)

            train_set_branch_y = np.zeros(shape=(5, grid_size//5))
            train_set_branch_t = np.zeros(shape=(5, grid_size//5))
            test_truth = np.zeros(shape=(5, grid_size//5))

            stats = np.zeros(shape=(5, 9))
            norm_props = np.zeros(shape=(5, 4))


            branch_mask = np.zeros(shape=(5, grid_size//5))

            train_set_trunk_t = X.copy().reshape(5,grid_size//5)

            shape = samples_noisy.shape[0]
           
            for i in range(0,shape):
                r = random.randint(min, max)
               
                if j < num_functions/2: 
                    indices = np.sort(np.random.choice(np.arange(0, grid_size//5), r, replace=False))
                else:
                    indices = np.arange(0, grid_size//5)[::random.randint(2, 4)]
               
                y_observed = samples_noisy[i][indices]
                y_truth = samples[i]
                t_truth =X.reshape(5,grid_size//5)[i]
                t_observed = X.reshape(5,grid_size//5)[i][indices]
           
                
                y_min_truth,y_max_truth = (np.min(y_truth),np.max(y_truth))
                y_min_noise,y_max_noise = (np.min(y_observed),np.max(y_observed))
            #    y_mean_truth,y_std_truth = (np.mean(y_truth),np.std(y_truth))     
             #   y_mean_noise,y_std_noise = (np.mean(y_observed),np.std(y_observed))
        
             

                y_observed_norm = (y_observed - y_min_noise)/(y_max_noise-y_min_noise)
                y_truth =  (y_truth - y_min_truth)/(y_max_truth-y_min_truth)
                t_observed_norm = (t_observed - np.min(t_observed))/(np.max(t_observed) - np.min(t_observed))
                t_truth = (t_truth - np.min(t_truth))/(np.max(t_truth)-np.min(t_truth))
                

                train_set_branch_y[i] = np.append(y_observed_norm,np.zeros(grid_size//5 - len(indices)))         
                train_set_branch_t[i] = np.append(t_observed_norm,np.zeros(grid_size//5 - len(indices)))
                train_set_trunk_t[i] = t_truth
                branch_mask[i]  = np.append(np.ones(len(indices)), np.zeros(grid_size//5 - len(indices)))
                test_truth[i] = y_truth
                stats[i] =np.array([y_min_noise,y_max_noise,
                                    y_max_noise - y_min_noise,
                                    y_observed[0],y_observed[-1],
                                    y_observed[-1] - y_observed[0],
                                    t_observed[0],t_observed[-1],
                                    t_observed[-1] - t_observed[0]])
                norm_props[i] = np.array([y_min_noise,y_max_noise,np.min(t_observed),np.max(t_observed)])
           
                                     #np.array([y_min_noise,y_max_noise,y_range,y_f,y_l,y_d,t_min,t_max,t_range,t_f,t_l,t_d])
            #    raise Exception()
            train_set_branch_y_collection.append(train_set_branch_y)
            train_set_branch_t_collection.append(train_set_branch_t)
            train_set_trunk_collection.append(train_set_trunk_t)
            branch_mask_collection.append(branch_mask)
            test_truth_collection.append(test_truth)
            stats_collection.append(stats)
            norm_props_collection.append(norm_props)

            
            samples_.append(samples)
            samples_noisy_.append(samples_noisy)
            
        except :

            continue
    train_set_branch_y_collection = (np.asarray(train_set_branch_y_collection, dtype=np.float32)).reshape(-1,grid_size//5)
    train_set_branch_t_collection = (np.asarray(train_set_branch_t_collection, dtype=np.float32)).reshape(-1,grid_size//5)
    train_set_trunk_collection = (np.asarray(train_set_trunk_collection, dtype=np.float32)).reshape(-1,grid_size//5)
    branch_mask_collection = (np.asarray(branch_mask_collection, dtype=np.float32)).reshape(-1,grid_size//5)
    test_truth_collection = (np.asarray(test_truth_collection, dtype=np.float32).reshape(-1,grid_size//5))
    stats_collection = (np.asarray(stats_collection, dtype=np.float32)).reshape(-1,9)
    norm_props_collection = (np.asarray(norm_props_collection, dtype=np.float32)).reshape(-1,4)


    samples_ = (np.asarray(samples_, dtype=np.float32)).reshape(-1,grid_size//5)
    samples_noisy_ = (np.asarray(samples_noisy_, dtype=np.float32)).reshape(-1,grid_size//5)


    return (train_set_branch_y_collection,
            train_set_branch_t_collection,
            train_set_trunk_collection,
            branch_mask_collection,
            test_truth_collection,
            stats_collection,
            norm_props_collection,
            samples_,samples_noisy_)





grid_size = 640
num_f = 100000
data = prepare_data(num_f,grid_size)
save_data(*data,postfix="train_minmax_big")
