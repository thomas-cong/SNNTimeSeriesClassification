import torch
import torch.nn as nn


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
        return self.net(x)
class TransformerClassifier(nn.Module):
    def __init__(self, T, classes, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.T = T
        self.d_model = d_model

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
        # x: [batch, T]
        x = x.unsqueeze(-1)                            # [batch, T, 1]
        x = self.input_proj(x)                        # [batch, T, d_model]
        x = x + self.pos_embedding[:, :x.size(1), :]  # [batch, T, d_model]
        h = self.encoder(x)                           # [batch, T, d_model]
        h = h.mean(dim=1)                             # [batch, d_model]
        out = self.cls(h)                             # [batch, classes]
        return out

def get_model(args):
    assert type(args.seq_len) == int, "Seq length has to be int"
    assert type(args.classes) == int, "need integer classes"
    if args.model == "mlp":
        return MLPClassifier(args.seq_len, args.classes)
    elif args.model == "transformer":
        return TransformerClassifier(args.seq_len, args.classes)
        
