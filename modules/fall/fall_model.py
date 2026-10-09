"""
CampusGuard Vision - Fall Detection
The LSTM classifier. Shared by train_model.py and fall_detector.py so the two
can never drift apart.

Save as: modules/fall/fall_model.py   (NEW file)
"""

import torch.nn as nn


class FallLSTM(nn.Module):
    def __init__(self, input_size=37, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])
