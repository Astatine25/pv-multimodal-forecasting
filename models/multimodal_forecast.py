import torch.nn as nn

HORIZON = 24
FEATURE_DIM = 128


class MultimodalForecastModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(HORIZON * FEATURE_DIM, 256),
            nn.ReLU(),
            nn.Linear(256, HORIZON)
        )

    def forward(self, x):
        return self.net(x)
