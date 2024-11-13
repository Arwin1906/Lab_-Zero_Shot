import torch
from torch.utils.data import Dataset, DataLoader
class TimeSeriesDataset(Dataset):
    def __init__(self,train_set_branch_y_collection,
                train_set_branch_t_collection,
                train_set_trunk_collection, 
                branch_mask_collection,
                test_truth_collection):
        self.train_set_branch_y_collection = train_set_branch_y_collection
        self.train_set_branch_t_collection = train_set_branch_t_collection

        self.train_set_trunk_collection = train_set_trunk_collection
        self.branch_mask_collection = branch_mask_collection
        self.test_truth_collection = test_truth_collection

    def __len__(self):
        # Return the total number of samples
        return len(self.train_set_branch_y_collection)

    def __getitem__(self, idx):
        # For each index, return a dictionary with each part of the data
        train_set_branch_y = torch.tensor(self.train_set_branch_y_collection[idx], dtype=torch.float32)
        train_set_branch_t = torch.tensor(self.train_set_branch_t_collection[idx], dtype=torch.float32)

        train_set_trunk = torch.tensor(self.train_set_trunk_collection[idx], dtype=torch.float32)
        branch_mask = torch.tensor(self.branch_mask_collection[idx], dtype=torch.float32)
        test_truth = torch.tensor(self.test_truth_collection[idx], dtype=torch.float32)
        
        return {
            "train_set_branch_y": train_set_branch_y,
            "train_set_branch_t": train_set_branch_t,
            "train_set_trunk": train_set_trunk,
            "branch_mask": branch_mask,
            "test_truth": test_truth
        }
