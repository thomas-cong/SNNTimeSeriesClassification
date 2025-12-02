import os
import arff
import torch
from torch.utils.data import Dataset

class BinaryHeartBeatDataset(Dataset):
    def __init__(self, arff_path):
        # arff.load returns a dict; the "data" field is a list of rows
        dataset = arff.load(open(arff_path))
        data = dataset['data']
        # encode the normal as 0, abnormal as 1
        encoded = [row[:-1] + [1 if row[-1].strip().lower() == "abnormal" else 0] for row in data]
        # turn into a torch tensor
        data_tensor = torch.tensor(encoded, dtype=torch.float32)
        self.features = data_tensor[:, :-1]
        self.labels = data_tensor[:, -1].long()  # or .float() if regression

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    arff_path = "/home/tcong13/949Final/Binary Heartbeat/BinaryHeartbeat_TRAIN.arff"
    training_data = BinaryHeartBeatDataset(arff_path)
    training_loader = DataLoader(training_data)
    feature, label = next(iter(training_loader))
    print(f"Feature Shape {feature.shape}")
    print(f"Label Shape {label.shape}")
    breakpoint()
