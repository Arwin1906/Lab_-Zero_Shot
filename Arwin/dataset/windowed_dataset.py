import numpy as np
import torch

class WindowedDataset(torch.utils.data.Dataset):
    def __init__(self, functions, observations, masks, n_windows=5, window_size=128, shuffle=False, eval=False):
        """
        Initialize the dataset with functions, observations, masks, and windowing parameters.
        """
        self.functions = functions  # List of function values
        self.observations = observations  # List of (y_observation, t_observation) pairs
        self.masks = masks  # List of masks
        self.n_windows = n_windows
        self.window_size = window_size
        self.shuffle = shuffle # Whether to shuffle the observations in each window
        self.eval = eval # Whether this is an evaluation dataset (affects normalization)
        self.X = np.linspace(0, 1, n_windows * window_size)  # Fine grid for splitting

    def normalize_window(self, t_values, y_values, t_observation, y_observation, epsilon=1e-8):
        """ Normalize the window of the time series and return the normalized values """

        # Calculate mean and variance
        if self.eval:
            mean_y = np.mean(y_observation)
            var_y = np.var(y_observation)
        else:
            mean_y = np.mean(y_values)
            var_y = np.var(y_values)
            
        # Normalize each value in y_values
        y_normalized = (y_values - mean_y) / np.sqrt(var_y + epsilon)
        y_observation_normalized = (y_observation - mean_y) / np.sqrt(var_y + epsilon)

        # Normalize time values
        if self.eval:
            t_min = np.min(t_observation)
            t_max = np.max(t_observation)
            t_diff = t_max - t_min
        else:
            t_min = np.min(t_values)
            t_max = np.max(t_values)
            t_diff = t_max - t_min

        t_normalized = (t_values - t_min) / t_diff
        t_observation_normalized = (t_observation - t_min) / t_diff

        s = torch.tensor([mean_y, var_y, t_min, t_max, t_diff], dtype=torch.float32)

        return t_normalized, y_normalized, t_observation_normalized, y_observation_normalized, s

    def __len__(self):
        return len(self.functions)

    def __getitem__(self, idx):
        """
        Returns the windowed function values, observations, and masks for the function at index idx.
        """
        y_values = self.functions[idx]
        y_observation, t_observation = self.observations[idx]
        masks = self.masks[idx]

        # Sort observations by time
        sorted_indices = np.argsort(t_observation)
        y_observation = y_observation[sorted_indices]
        t_observation = t_observation[sorted_indices]
        masks = masks[sorted_indices]

        # Split into windows
        y_value_windows, t_value_windows = [], []
        y_observation_windows, t_observation_windows = [], []
        mask_windows = []
        s_values = []

        start_index = 0
        for i in range(self.n_windows):
            # Find closest observation to last time point in window
            target = self.X[i * self.window_size : (i + 1) * self.window_size][-1]
            closest_index = np.argmin(np.abs(t_observation - target))

            # Normalize the window
            (
                t_normalized,
                y_normalized,
                t_observation_normalized,
                y_observation_normalized,
                s,
            ) = self.normalize_window(
                self.X[i * self.window_size : (i + 1) * self.window_size],
                y_values[i * self.window_size : (i + 1) * self.window_size],
                t_observation[start_index:closest_index],
                y_observation[start_index:closest_index],
            )

            y_value_windows.append(y_normalized)
            t_value_windows.append(t_normalized)

            if self.shuffle:
                # Generate a random permutation of indices
                perm = np.random.permutation(y_observation_normalized.shape[0])
                # Apply the permutation to the observations
                y_observation_normalized = y_observation_normalized[perm]
                t_observation_normalized = t_observation_normalized[perm]
                masks[start_index:closest_index] = masks[start_index:closest_index][perm]

            y_observation_windows.append(y_observation_normalized)
            t_observation_windows.append(t_observation_normalized)
            mask_windows.append(masks[start_index:closest_index])

            s_values.append(s)

            start_index = closest_index

        return y_value_windows, y_observation_windows, t_observation_windows, mask_windows, s_values