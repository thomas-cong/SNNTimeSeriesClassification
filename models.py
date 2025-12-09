import torch
import torch.nn as nn
from SNN import LIFReservoir, STDPLayer

class MLPClassifier(nn.Module):
    def __init__(self, T, classes):
        # T is input length
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(T, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Linear(64, classes)
        )
    def forward(self, x):
        # Input: (B, C, T) -> reshape to (B, T) for MLP
        batch_size = x.shape[0]
        # Reshape: flatten all channels into one dimension
        x = x.view(batch_size, -1)
        return self.net(x)
class TransformerClassifier(nn.Module):
    def __init__(self, T, classes, d_model=128, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.T = T
        self.d_model = d_model

        # Updated to handle C channels
        self.input_proj = nn.Linear(1, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, T, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls = nn.Linear(d_model, classes)

    def forward(self, x):
        # x: [batch, C, T] -> transpose to [batch, T, C]
        x = x.transpose(1, 2)                        # [batch, T, C]
        x = self.input_proj(x)                        # [batch, T, d_model]
        x = x + self.pos_embedding[:, :x.size(1), :]  # [batch, T, d_model]
        h = self.encoder(x)                           # [batch, T, d_model]
        h = h.mean(dim=1)                             # [batch, d_model]
        out = self.cls(h)                             # [batch, classes]
        return out
        
class ReservoirClassifier(nn.Module):
    def __init__(self, classes, reservoir, data_channels = 1):
        super().__init__()
        self.reservoir = reservoir
        self.classifier = nn.Sequential(
            nn.Linear(self.reservoir.n_reservoir, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, classes)
        )
    def forward(self, x, reservoir_stdp = False):
        B,C, T = x.shape
        all_spikes = []
        for t in range(T):
            x_t = x[:,:,t] # move over channels simultaneously
            spike_t = self.reservoir(x_t, use_stdp = reservoir_stdp) # input as C, 1
            all_spikes.append(spike_t)
        post_spike = torch.stack(all_spikes, dim=1)  # [B, T, reservoir_size]
        post_spike = torch.mean(post_spike, dim=1)  # [B, reservoir_size] mean pooling over time
        predicted = self.classifier(post_spike)
        return predicted
class ReservoirSTDPReadout(nn.Module):
    def __init__(self, classes, reservoir, data_channels = 1):
        super().__init__()
        self.reservoir = reservoir
        self.classifier = STDPLayer(reservoir.n_reservoir, classes)
    def forward(self, x, labels = None, reservoir_stdp = False, classifier_stdp = False):
        B, C, T = x.shape
        output_spikes = []
        for t in range(T):
            x_t = x[:,:, t]
            reservoir_spike = self.reservoir(x_t, use_stdp = reservoir_stdp)
            classifier_spike = self.classifier(reservoir_spike, labels, use_stdp = classifier_stdp)
            output_spikes.append(classifier_spike)
        output_spikes = torch.stack(output_spikes, dim=1)  # [B, T, n_classes]
        spike_counts = output_spikes.sum(dim=1)  # [B, n_classes] - total spikes per class
        return spike_counts