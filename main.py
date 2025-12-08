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
    elif args.model == "lifstatic":
        from models import ReservoirClassifier
        from SNN import LIFReservoir
        reservoir = LIFReservoir(n_in = 10, n_reservoir = 200)  # 10 receptors per input channel
        return ReservoirClassifier(classes = args.classes, reservoir = reservoir)
    elif args.model == "lifstdp":
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
    assert hasattr(model.reservoir, "stdp_update"), "no STDP functionality available for the model"
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


def reset_snn_stats(model): #zeros out spike counts that has "spike_count"
    for module in model.modules():
        if hasattr(module, "spike_count"):
            module.spike_count.zero_()


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
        if epoch == 0 and "stdp" in args.model:
            for i in range(args.stdp_passes):
                stdp_train(model, train_loader, freeze = (i == args.stdp_passes - 1))
        pbar = tqdm(train_loader)

        epoch_losses = []
        for features,labels in pbar:
            features, labels = features.to(device), labels.to(device)
            if "lif" in args.model:
                features = encode_spikes(features)
            if "lif" in args.model:
                predicted = model(features, use_stdp = False)
            else:
                predicted = model(features)
            loss = loss_fn(predicted, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_losses.append(loss.item())
            pbar.set_description(f"Epoch {epoch} Loss {loss.item():.4f}")
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
    input_spike_total = 0.0
    if "lif" in args.model: # counts input spikes and resets SNN stats
        reset_snn_stats(model)
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
            input_spike_total += features.sum().item() #counts input spikes for SNN models
        with torch.no_grad():
            if "lif" in args.model:
                predicted = model(features, use_stdp = False)
            else:
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

    # SNN statistics
    if "lif" in args.model:
        n_test = len(test)
        n_res = model.reservoir.n_reservoir
        sparsity = model.reservoir.sparsity
        total_res_spikes = model.reservoir.spike_count.item()

        syn0ps_in = input_spike_total * n_res
        syn0ps_rec = total_res_spikes * n_res * sparsity
        syn0ps_total = syn0ps_in + syn0ps_rec
        syn0ps_per_sample = syn0ps_total / n_test
        print("SNN Statistics:")
        print(f" Total input spikes: {input_spike_total:.2f}")
        print(f" Total reservoir spikes: {total_res_spikes:.2f}")
        print(f" Total synaptic operations: {syn0ps_total:.2f}")
        print(f"Input spikes per sample: {input_spike_total / n_test:.2f}")
        print(f"Reservoir spikes per sample: {total_res_spikes / n_test:.2f}")
        print(f"SynOps per sample (total):{syn0ps_total / n_test:.2e}")
        print(f" Synaptic operations per sample: {syn0ps_per_sample:.2f}")
        T = args.seq_len
        firing_rate = total_res_spikes / (n_test * n_res * (T))
        print(f"Firing rate: {firing_rate:.6f}")

        mac_readout = (n_res * 128 + 128 * 64 + 64 * args.classes)
        print(f"MACs for readout layer: {mac_readout}")
        energy_snn_per_sample = mac_readout + 0.2 * syn0ps_per_sample
        print(f"Estimated energy per sample (in arbitrary units): {energy_snn_per_sample:.2f}")
        energy_snn_per_accuracy = energy_snn_per_sample / acc
        print(f"Estimated energy per accuracy (in arbitrary units): {energy_snn_per_accuracy:.2f}")


        #MLP stats
        if args.model == "mlp":
            D = args.seq_len
            mac_mlp = (D * 512 + 512 * 256 + 256 * 128 + 128 * 64 + 64 * args.classes)
            print(f"MACs for MLP model: {mac_mlp}")
            energy_mlp_per_sample = mac_mlp
            print(f"Estimated energy per sample for MLP (in arbitrary units): {energy_mlp_per_sample:.2f}")
            energy_mlp_per_accuracy = energy_mlp_per_sample / acc
            print(f"Estimated energy per accuracy for MLP (in arbitrary units): {energy_mlp_per_accuracy:.2f}")

        #transformer stats
        if args.model == "transformer":
            T_seq = args.seq_len
            d_model = 128
            dim_feedforward = 128
            num_layers = 2
            mac_layer = (3*T_seq*d_model*d_model + 2*T_seq * T_seq*d_model + T_seq*d_model*d_model + 2*dim_feedforward*d_model)
            mac_input_project = T_seq * 1 * d_model
            mac_classifier = d_model * args.classes
            mac_transformer = num_layers * mac_layer + mac_input_project + mac_classifier
            print(f"MACs for Transformer model: {mac_transformer}")
            energy_transformer_per_sample = mac_transformer
            print(f"Estimated energy per sample for Transformer (in arbitrary units): {energy_transformer_per_sample:.2f}")
            energy_transformer_per_accuracy = energy_transformer_per_sample / acc
            print(f"Estimated energy per accuracy for Transformer (in arbitrary units): {energy_transformer_per_accuracy:.2f}")



if __name__ == "__main__":
    args = parse_arguments()
    main(args)
