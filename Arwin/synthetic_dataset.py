import numpy as np
from math import exp

class SyntheticDataset:
    """
    A class to generate a synthetic dataset of GP functions. 
    """

    def __init__(self, num_functions, num_points, min_ones=10):
        """
        Initializes the synthetic dataset.
        Args:
            num_functions: Number of functions to sample.
            num_points: Number of total points in the dataset.
            min_ones: Minimum number of 1s in the bitmask for sampled points.
        """
        self.num_functions = num_functions
        self.num_points = num_points
        self.min_ones = min_ones

        self.X = np.linspace(0, 1, self.num_points)
        self.functions = []
        self.observations = []

        # Sample functions with different smoothness
        third = self.num_functions // 3
        sampled_functions = self.batch_sample_from_beta([(2, 5, third), (3, 3, third), (5, 2, self.num_functions - 2 * third)])
        self.functions = sampled_functions

        for i in range(self.num_functions):

            # Normalize the function values
            y_normalized = self.instance_normalize(self.functions[i][0])
            self.functions[i] = y_normalized
            # Add Gaussian noise
            y_noisy = self.add_gaussian_noise(y_normalized)

            if i < self.num_functions//2:
                # Generate a random bitmask for irregularly sampled points
                bitmask = self.generate_bitmask_irregular(min_ones=self.min_ones)
                observation_time = self.X[bitmask==1]
                observation_value = y_noisy[bitmask==1]
                self.observations.append((observation_time, observation_value))
            else:
                # Generate a bitmask for regularly sampled points
                bitmask, _ = self.generate_bitmask_regular(self.X, min_ones=self.min_ones)
                observation_time = self.X[bitmask==1]
                observation_value = y_noisy[bitmask==1]
                self.observations.append((observation_time, observation_value))
    
        return
    
    def cov_matrix(self, x, scale):
        """
        Creates a covariance matrix for the input data x using the RBF kernel.
        """
        sq_dists = np.subtract.outer(x, x)**2
        return np.exp(-sq_dists / (2 * scale**2))
    
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

        for a, b, size in beta_params:
            scales = rng.beta(a, b, size)
            for scale in scales:
                sampled_functions.append(self.sample_functions(self.X, scale))

        return sampled_functions
    
    def generate_bitmask_irregular(self, size=128, min_ones=10, max_ones=128):
        """
        Generate a random bitmask of 0s and 1s with a fixed min and max number of 1s.
        """
        # Determine the number of 1s (between min_ones and max_ones)
        num_ones = np.random.randint(min_ones, max_ones + 1)
        
        # Initialize a mask of zeros
        bitmask = np.zeros(size, dtype=int)
        
        # Choose unique random indices to set to 1
        ones_indices = np.random.choice(size, num_ones, replace=False)
        
        # Set these indices to 1
        bitmask[ones_indices] = 1
        
        return bitmask
    
    def generate_bitmask_regular(self, X, min_ones=10, max_ones=128):
        """
        Generate a bitmask for regularly sampled points.
        """
        # Determine the number of 1s (between min_ones and max_ones)
        num_ones = np.random.randint(min_ones, max_ones + 1)

        # Regularly sample points from X
        indices = np.linspace(0, len(X) - 1, num_ones, dtype=int)
        sampled_points = X[indices]

        bitmask = np.zeros(len(X), dtype=int)
        # Set mask entries to 1 where points in sampled_points are found in X
        indices = np.isin(X, sampled_points)
        bitmask[indices] = 1

        return bitmask, sampled_points
    
    def instance_normalize(self, y_values, epsilon=1e-8):
        """
        Normalize the values of y_values to have a mean of 0 and a variance of 1.
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
        return self.functions[idx], self.observations[idx]