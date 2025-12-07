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
    parser.add_argument("--stdp_passes", type = int, default = 3)
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
    elif args.model == "lifstatic" or args.model == "lifstdp":
        from models import ReservoirClassifier
        from SNN import STDPReservoir
        reservoir = STDPReservoir(n_in = 10, n_reservoir = 200)  # 10 receptors per input channel
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
def encode_spikes(x, n_receptors=10):
    """Population encoding: Encode inputs using multiple receptors with different sensitivities"""
    B, C, T = x.shape
    
    # Normalize input to zero mean and unit variance
    x_norm = (x - x.mean(dim=2, keepdim=True)) / (x.std(dim=2, keepdim=True) + 1e-8)
    
    # Create receptors with different tuning curves (-2 to 2 range)
    receptors = torch.linspace(-2, 2, n_receptors).reshape(1, 1, 1, n_receptors).to(x.device)
    
    # Expand input for broadcasting
    x_expanded = x_norm.unsqueeze(-1)  # [B, C, T, 1]
    
    # Calculate receptor responses using Gaussian tuning curves
    variance = 0.4
    responses = torch.exp(-(x_expanded - receptors)**2 / variance)  # [B, C, T, n_receptors]
    
    # Convert analog responses to spike trains
    spikes = torch.zeros(B, C * n_receptors, T, device=x.device)
    
    for t in range(T):
        # Generate spikes probabilistically based on receptor response
        rand_samples = torch.rand(B, C, n_receptors, device=x.device)
        spike_probs = responses[:, :, t, :] * 0.6
        poisson_spikes = (rand_samples < spike_probs).float()
        
        # Reshape and assign
        reshaped_spikes = poisson_spikes.reshape(B, C * n_receptors)
        spikes[:, :, t] = reshaped_spikes
    
    return spikes

def stdp_train(model, data, freeze = True):
    device = next(model.parameters()).device
    assert model.reservoir.stdp_update, "no STDP functionality available for the model"
    print("Training model via STDP")
    for batch in data:
        features, labels = batch
        features, labels = features.to(device), labels.to(device)
        features = encode_spikes(features)
        model(features, use_stdp = True)
    if freeze:
        for param in model.reservoir.parameters():
            param.requires_grad = False
        print("Model parameter require_grad: ", [f"{name}: {param.requires_grad}" for name, param in model.reservoir.named_parameters()])
    return model

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
    best_loss = float('inf')
    best_state = None
    for epoch in range(args.epochs):
        pbar = tqdm(train_loader)
        if epoch == 0 and "stdp" in args.model:
            for i in range(args.stdp_passes):
                model = stdp_train(model, pbar, freeze = (i == args.stdp_passes - 1))
        epoch_losses = []
        for batch in pbar:
            features, labels = batch
            features, labels = features.to(device), labels.to(device)
            if "lif" in args.model:
                features = encode_spikes(features)
            predicted = model(features, use_stdp = False)
            loss = loss_fn(predicted, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_losses.append(loss.item())
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        print("Epoch: {}, Loss: {:.4f}".format(epoch, avg_loss))

    # load best checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)
        print("Loaded best checkpoint with loss: {:.4f}".format(best_loss))

    # test
    test_loader = DataLoader(test, batch_size = args.batch_size)
    all_preds = []
    all_labels = []
    test_losses = []
    tpbar = tqdm(test_loader)
    for batch in tpbar:
        features, labels = batch
        features, labels = features.to(device), labels.to(device)
        if "lif" in args.model:
            features = encode_spikes(features)
        with torch.no_grad():
            predicted = model(features, use_stdp = False)
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
    

