import os
import pandas as pd
import torch
from torch.utils.data import Dataset

class TwoPatternsDataset(Dataset):
    def __init__(self, tsv_path):
        self.tsv = pd.read_csv(tsv_path, sep = "\t")
    def __len__(self):
        return len(self.tsv)
    def __getitem__(self, idx):
        row = self.tsv.iloc[idx, 1:]
        label = self.tsv.iloc[idx, 0]
        # Return shape (C, T) - where C=1, T=sequence length
        return torch.tensor(row.values, dtype=torch.float32).unsqueeze(0), torch.tensor(label - 1, dtype=torch.long)

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    training_data = TwoPatternsDataset("./TwoPatterns/TwoPatterns_TRAIN.tsv")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    train_dataloader = DataLoader(training_data)
    train_feature, train_label = next(iter(train_dataloader))
    print(f"Feature Shape: {train_feature.shape}")
    print(f"Label Shape: {train_label.shape}")
