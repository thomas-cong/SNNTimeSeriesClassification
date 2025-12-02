import torch
import torch.nn as nn
import argparse
from models import get_model
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score, balanced_accuracy_score

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="mlp")
    parser.add_argument("--dataset", type = str, default = "heartbeat")
    parser.add_argument("--batch_size", type = int, default = 32)
    parser.add_argument("--epochs", type = int, default = 300)
    return parser.parse_args()

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
        args.seq_len = train_data[0][0].shape[0]
        args.classes = 4
    else:
        raise ValueError("Unknown dataset: {}".format(args.dataset))
    return train_data, test_data

def main():
    args = parse_arguments()
    print("Args: ", args)
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
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

    # train
    for epoch in range(args.epochs):
        for batch in train_loader:
            features, labels = batch
            features, labels = features.to(device), labels.to(device)
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
    main()
    

