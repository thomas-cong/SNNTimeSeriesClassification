import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import argparse
from tqdm import tqdm

from sklearn.metrics import accuracy_score, balanced_accuracy_score

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="mlp")
    parser.add_argument("--dataset", type = str, default = "heartbeat")
    parser.add_argument("--batch_size", type = int, default = 32)
    parser.add_argument("--epochs", type = int, default = 300)
    parser.add_argument("--freeze_reservoir", type = bool, default = True)
    return parser.parse_args()
def get_model(args):
    assert type(args.seq_len) == int, "Seq length has to be int"
    assert type(args.classes) == int, "need integer classes"
    if args.model == "mlp":
        from models import MLPClassifier
        return MLPClassifier(args.seq_len, args.classes)
    elif args.model == "transformer":
        from models import TransformerClassifier
        return TransformerClassifier(args.seq_len, args.classes)
    elif args.model == "lifstatic":
        from models import ReservoirClassifier
        from SNN import LIFReservoir
        reservoir = LIFReservoir(n_in = 1, n_reservoir = 200)
        return ReservoirClassifier(classes = args.classes, reservoir = reservoir)
    elif args.model == "lifstdp":
        from models import ReservoirClassifier
        from SNN import STDPReservoir
        reservoir = STDPReservoir(n_in = 1, n_reservoir = 200)
        return ReservoirClassifier(classes = args.classes, reservoir = reservoir)
def get_dataset(args):
    if args.dataset == "heartbeat":
        from BinaryHeartBeatDataset import BinaryHeartBeatDataset
        train_data = BinaryHeartBeatDataset("/home/tcong13/949Final/Binary Heartbeat/BinaryHeartbeat_TRAIN.arff")
        test_data = BinaryHeartBeatDataset("/home/tcong13/949Final/Binary Heartbeat/BinaryHeartbeat_TEST.arff")
        args.seq_len = train_data[0][0].shape[0]
        args.classes = 2
    elif args.dataset == "twopattern":
        from TwoPatternsDataset import TwoPatternsDataset
        train_data = TwoPatternsDataset("/home/tcong13/949Final/TwoPatterns/TwoPatterns_TRAIN.tsv")
        test_data = TwoPatternsDataset("/home/tcong13/949Final/TwoPatterns/TwoPatterns_TEST.tsv")
        args.seq_len = train_data[0][0].shape[1]
        args.classes = 4
    else:
        raise ValueError("Unknown dataset: {}".format(args.dataset))
    return train_data, test_data
def encode_spikes(x, threshold=0.2, tau_m=0.9, tau_s=0.5, refractory=3):
    B, C, T = x.shape
    x_norm = (x - x.mean(dim=2, keepdim=True)) / (x.std(dim=2, keepdim=True) + 1e-8)
    
    membrane = torch.zeros(B, C, device=x.device)
    spikes = torch.zeros(B, C, T, device=x.device)
    refrac_count = torch.zeros(B, C, device=x.device)
    
    for t in range(T):
        membrane = tau_m * membrane
        membrane += tau_s * x_norm[:, :, t]
        
        can_spike = refrac_count <= 0
        is_spike = (membrane > threshold) & can_spike
        
        spikes[:, :, t] = is_spike.float()
        membrane[is_spike] = 0.0
        refrac_count[is_spike] = refractory
        refrac_count = torch.clamp(refrac_count - 1, min=0)
    
    return spikes


def main(args):
    print("Args: ", args)
    if "stdp" in args.model:
        args.batch_size = 1 # stdp should only consider one sample at a time
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device)
    train, test = get_dataset(args)
    print("Train size: ", len(train))
    print("Test size: ", len(test))
    train_labels = torch.tensor([y for _, y in train])
    test_labels = torch.tensor([y for _, y in test])
    print("Train label counts:", torch.bincount(train_labels))
    print("Test label counts:", torch.bincount(test_labels))
    class_counts = torch.bincount(train_labels)
    class_weights = (class_counts.sum()/ class_counts).to(device)
    loss_fn = nn.CrossEntropyLoss(weight = class_weights)
    train_loader = DataLoader(train, batch_size = args.batch_size)
    model = get_model(args)
    model.to(device)
    print("Model:", model)
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

    # train
    for epoch in range(args.epochs):
        pbar = tqdm(train_loader)
        for batch in pbar:
            features, labels = batch
            features, labels = features.to(device), labels.to(device)
            if "lif" in args.model:
                features = encode_spikes(features)
            predicted = model(features)
            loss = loss_fn(predicted, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print("Epoch: {}, Loss: {}".format(epoch, loss.item()))

    # test
    test_loader = DataLoader(test, batch_size = args.batch_size)
    all_preds = []
    all_labels = []
    test_losses = []
    for batch in test_loader:
        features, labels = batch
        features, labels = features.to(device), labels.to(device)
        if "lif" in args.model:
            features = encode_spikes(features)
        with torch.no_grad():
            predicted = model(features)
            loss = loss_fn(predicted, labels)
        test_losses.append(loss.item())
        preds_cls = torch.argmax(predicted, dim=1).cpu().numpy()
        labels_np = labels.cpu().numpy()
        all_preds.extend(preds_cls)
        all_labels.extend(labels_np)
    avg_test_loss = sum(test_losses) / len(test_losses)
    acc = accuracy_score(all_labels, all_preds)
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    print("Test Loss: {:.4f}, Accuracy: {:.4f}, Balanced Accuracy: {:.4f}".format(avg_test_loss, acc, bal_acc))
    
if __name__ == "__main__":
    args = parse_arguments()
    main(args)
    

