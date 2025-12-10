import torch
import os
import sys
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from argparse import Namespace
from tqdm import tqdm

sys.path.insert(0, '/home/tcong13/949Final')

reservoir_path = "/home/tcong13/949Final/models/200_fixed_reservoir.pt"
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

    # print("Mean input firing rate:",
    #   spikes.sum().item() / (spikes.numel()))

    return spikes
def load_or_create_reservoir(path, n_in, n_reservoir):
    from SNN import STDPReservoir
    if path and os.path.exists(path):
        return torch.load(path, weights_only = False)
    reservoir = STDPReservoir(n_in=n_in, n_reservoir=n_reservoir)
    for param in reservoir.parameters():
        param.requires_grad = False
    return reservoir

def run_reservoir(reservoir, spikes):
    """Run reservoir on spike input and return all spike outputs [B, N, T]"""
    B, N_in, T = spikes.shape
    reservoir.v = None
    reservoir.prev_spikes = None
    outputs = []
    for t in range(T):
        out = reservoir(spikes[:, :, t])
        outputs.append(out)
    return torch.stack(outputs, dim=2)

def main():
    args = Namespace(dataset="twopattern")
    train_data, test_data = get_dataset(args)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Use random reservoir instead of STDP-trained one
    from SNN import STDPReservoir
    reservoir = STDPReservoir(n_in=10, n_reservoir=200)
    for param in reservoir.parameters():
        param.requires_grad = False
    reservoir = reservoir.to(device)
    reservoir.eval()
    
    # Get color map for classes (high contrast)
    unique_classes = sorted(set(y.item() if isinstance(y, torch.Tensor) else y for _, y in train_data))
    high_contrast_colors = ['#FF0000', '#0000FF', '#00CC00', '#FF8000', '#CC00CC', '#00CCCC', '#FFCC00', '#8000FF']
    class_to_color = {c: high_contrast_colors[i % len(high_contrast_colors)] for i, c in enumerate(unique_classes)}
    
    # Plot 1: Separate spike raster per class (grid layout)
    samples_by_class = {c: None for c in unique_classes}
    for x, y in train_data:
        label = y.item() if isinstance(y, torch.Tensor) else y
        if samples_by_class[label] is None:
            samples_by_class[label] = x
        if all(v is not None for v in samples_by_class.values()):
            break
    
    fig, axes = plt.subplots(4, 2, figsize=(20, 16))
    
    # Layout: 4 rows x 2 cols
    # Row 0: Class 0 Input | Class 1 Input
    # Row 1: Class 0 Reservoir | Class 1 Reservoir
    # Row 2: Class 2 Input | Class 3 Input
    # Row 3: Class 2 Reservoir | Class 3 Reservoir
    
    for i, c in enumerate(unique_classes):
        row_base = (i // 2) * 2  # 0 for classes 0,1; 2 for classes 2,3
        col = i % 2              # 0 for classes 0,2; 1 for classes 1,3
        
        x = samples_by_class[c].unsqueeze(0).to(device)
        spikes_in = encode_spikes(x, n_receptors=10)
        with torch.no_grad():
            reservoir.v = None
            reservoir.prev_spikes = None
            spikes_out = run_reservoir(reservoir, spikes_in)
        
        spike_times_in = spikes_in[0].cpu().nonzero(as_tuple=False)
        axes[row_base, col].scatter(spike_times_in[:, 1].numpy(), spike_times_in[:, 0].numpy(), 
                                    s=1, c=class_to_color[c])
        axes[row_base, col].set_title(f'Class {int(c)} Input', color=class_to_color[c], fontweight='bold')
        axes[row_base, col].set_ylabel('Input Neuron')
        
        spike_times_out = spikes_out[0].cpu().nonzero(as_tuple=False)
        axes[row_base+1, col].scatter(spike_times_out[:, 1].numpy(), spike_times_out[:, 0].numpy(),
                                      s=1, c=class_to_color[c])
        axes[row_base+1, col].set_title(f'Class {int(c)} Reservoir', color=class_to_color[c], fontweight='bold')
        axes[row_base+1, col].set_ylabel('Reservoir Neuron')
        axes[row_base+1, col].set_xlabel('Time Step')
    
    plt.suptitle('Spike Rasters by Class', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/home/tcong13/949Final/vis/random_spike_raster.png', dpi=300)
    plt.show()
    
    # Train logistic classifier to test separability
    train_reps, train_labels = [], []
    for x, y in tqdm(train_data):
        x = x.unsqueeze(0).to(device)
        spikes_in = encode_spikes(x, n_receptors=10)
        with torch.no_grad():
            reservoir.v = None
            reservoir.prev_spikes = None
            spikes_out = run_reservoir(reservoir, spikes_in)
        rep = spikes_out[0].cpu().mean(dim=1).numpy()
        train_reps.append(rep)
        train_labels.append(y.item() if isinstance(y, torch.Tensor) else y)
    
    test_reps, test_labels = [], []
    for x, y in tqdm(test_data):
        x = x.unsqueeze(0).to(device)
        spikes_in = encode_spikes(x, n_receptors=10)
        with torch.no_grad():
            reservoir.v = None
            reservoir.prev_spikes = None
            spikes_out = run_reservoir(reservoir, spikes_in)
        rep = spikes_out[0].cpu().mean(dim=1).numpy()
        test_reps.append(rep)
        test_labels.append(y.item() if isinstance(y, torch.Tensor) else y)
    
    train_reps = np.array(train_reps)
    train_labels = np.array(train_labels)
    test_reps = np.array(test_reps)
    test_labels = np.array(test_labels)
    
    clf = LogisticRegression(max_iter=1000)
    clf.fit(train_reps, train_labels)
    
    train_pred = clf.predict(train_reps)
    test_pred = clf.predict(test_reps)
    
    print(f"Train Accuracy: {accuracy_score(train_labels, train_pred):.4f}")
    print(f"Test Accuracy: {accuracy_score(test_labels, test_pred):.4f}")
    print("\nClassification Report (Test):")
    print(classification_report(test_labels, test_pred))

if __name__ == "__main__":
    main()