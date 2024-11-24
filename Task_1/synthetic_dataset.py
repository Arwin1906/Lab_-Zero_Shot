import numpy as np
from math import exp
from tqdm import tqdm
from itertools import permutations 

class SyntheticDataset:
    """
    A class to generate a synthetic dataset of GP functions. 
    """

    def __init__(self, num_functions, num_points, min_ones=10, max_ones=100, padding=True, verbose=False, test=False):
        """
        Initializes the synthetic dataset.
        Args:
            num_functions: Number of functions to sample.
            num_points: Number of total points in the dataset.
            min_ones: Minimum number of 1s in the bitmask for sampled points.
            max_ones: Maximum number of 1s in the bitmask for sampled points.
            padding: Whether to pad the observations to num_points length.
            verbose: Whether to display progress bars.
            test: Whether to use the test dataset with different varaince.
        """
        self.num_functions = num_functions
        self.num_points = num_points
        self.min_ones = min_ones
        self.max_ones = max_ones
        self.padding = padding
        self.verbose = verbose
        self.test = test

        self.X = np.linspace(0, 1, self.num_points)
        self.functions = []
        self.observations = []
        self.masks = []
        self.size = 1 # size of each beta distribution

        # Sample functions with different smoothness
        sampled_functions = self.batch_sample_from_beta([1, 2, 5])

        # Add progess bar for generating observations
        if self.verbose:
            progress_bar = tqdm(range(self.num_functions), total=self.num_functions)
            progress_bar.set_description("Generating Observations")
        else:
            progress_bar = range(self.num_functions)

        j = 1
        for i in progress_bar:
            # # Normalize the function values and add Gaussian noise
            y_noisy = self.add_gaussian_noise(sampled_functions[i][0])
            y_normalized = self.instance_normalize(sampled_functions[i][0])
            y_noisy_normalized = self.instance_normalize(y_noisy)

            if i < j*self.size//2:
                # Generate a random bitmask for irregularly sampled points
                indices = self.generate_bitmask_irregular(size=self.num_points, min_ones=self.min_ones, max_ones=self.max_ones)
                observation_time = self.X[indices]
                observation_value = y_noisy_normalized[indices]
            else:
                # Generate a bitmask for regularly sampled points
                indices = self.generate_indices_regular(self.X, min_ones=self.min_ones, max_ones=self.max_ones)
                observation_time = self.X[indices]
                observation_value = y_noisy_normalized[indices]

            if i == self.size*j:
                j += 1

            if self.padding:
                # Pad observation_time and observation_value to the fixed length (num_points)
                observation_time = np.pad(observation_time, (0, self.num_points - len(observation_time)), 'constant', constant_values=0)
                observation_value = np.pad(observation_value, (0, self.num_points - len(observation_value)), 'constant', constant_values=0)
                indices = np.pad(indices, (0, self.num_points - len(indices)), 'constant', constant_values=0)
            
            # Add the padded observations to the dataset
            self.observations.append((observation_value, observation_time))
            self.masks.append(indices)
            self.functions.append(y_normalized)
    
        return
    
    def cov_matrix(self, X, scale, variance=1.0):
        """
        Creates a covariance matrix for the input data x using the RBF kernel.
        """
        sq_dists = np.subtract.outer(X, X)**2
        if self.test:
            variance = 1.15
        return variance * np.exp(-sq_dists / (2 * scale**2))
    
    def sample_functions(self, X, scale, number_of_functions=1):
        """
        Sample functions from the prior distribution
        """
        sigma = self.cov_matrix(X, scale)
        # Assume a mean of 0 for simplicity
        mean = np.zeros(X.shape[0])
        ys = np.random.multivariate_normal(
            mean=mean, cov=sigma, 
            size=number_of_functions)
        
        return ys
    
    def batch_sample_from_beta(self, beta_params):
        """Batch sample kernel sizes from beta distributions for multiple smoothnesses."""
        rng = np.random.default_rng()
        sampled_functions = []
        beta_params_permutations = list(permutations(beta_params, 2))
        # get even split of functions for each beta distribution
        sizes = self.get_split(self.num_functions, len(beta_params_permutations))
        self.size = sizes[0]

        for i, (params) in enumerate(beta_params_permutations):
            a, b = params
            scales = rng.beta(a, b, sizes[i])
            # Add progress bar for sampling functions
            if self.verbose:
                progress_bar = tqdm(scales, total=len(scales))
                progress_bar.set_description(f"Sampling Functions from Beta Distribution {i+1}/{len(beta_params_permutations)} with a={a}, b={b}")
            else:
                progress_bar = scales

            for scale in progress_bar:   
                sampled_functions.append(self.sample_functions(self.X, scale))

        return sampled_functions
    
    def get_split(self, total, param_len):
        # Ensure param_len is valid
        if param_len <= 0:
            raise ValueError("param_len must be a positive integer.")

        # Base size of each interval
        size = total // param_len
        remainder = total % param_len

        # Generate the list of intervals
        intervals = [size] * param_len

        # Adjust the last interval to account for the remainder
        intervals[-1] += remainder

        return intervals
    
    def generate_bitmask_irregular(self, size, min_ones, max_ones):
        """
        Generate a random bitmask of 0s and 1s with a fixed min and max number of 1s.
        """
        # Determine the number of 1s (between min_ones and max_ones)
        num_ones = np.random.randint(min_ones, max_ones + 1)
        
        # Choose unique random indices to set to 1
        indices = np.random.choice(size, num_ones, replace=False)
        
        return indices
    
    def generate_indices_regular(self, X, min_ones, max_ones):
        """
        Generate indices for regularly sampled points.
        """
        # Determine the number of 1s (between min_ones and max_ones)
        num_ones = np.random.randint(min_ones, max_ones + 1)

        # Regularly sample points from X
        indices = np.linspace(0, len(X) - 1, num_ones, dtype=int)

        return indices
    
    def instance_normalize(self, y_values, epsilon=1e-8):
        """
        Normalize the values of y_values using z-score normalization.
        """
        # Calculate mean and variance
        mean_y = np.mean(y_values)
        var_y = np.var(y_values)
        
        # Normalize each value in y_values
        y_normalized = (y_values - mean_y) / np.sqrt(var_y + epsilon)
        return y_normalized
    
    def add_gaussian_noise(self, y_values, noise_mean=0, noise_std_dev=0.1):
        """
        Add Gaussian noise to the y_values.
        """
        # Step 1: Sample noise standard deviation for this function
        rng = np.random.default_rng()
        noise_std = abs(rng.normal(loc=noise_mean, scale=noise_std_dev))
        
        # Step 2: Generate Gaussian noise with the sampled standard deviation
        noise = rng.normal(loc=0, scale=noise_std, size=y_values.shape)
        
        # Step 3: Add the noise to the function values
        y_noisy = y_values + noise
        return y_noisy
    
    def __len__(self):
        """
        Returns the number of functions in the dataset.
        """
        return len(self.functions)
    
    def __getitem__(self, idx):
        """
        Returns the function values and observations for the function at index idx.
        """
        return self.functions[idx], self.observations[idx], self.masks[idx]