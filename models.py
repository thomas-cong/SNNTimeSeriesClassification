import torch



class MLPClassifier(nn.Module):
    def __init__(self, T):
        # T is input length
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(T, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLu()
            nn.Linear(256, 64),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x)

def get_model(model_type = "mlp", seq_len):
    if model_type = "mlp":
        return MLPClassifier(seq_len)
