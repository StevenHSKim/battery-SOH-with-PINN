import torch.nn as nn

class ForecastingModel(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 layers = 1,
                 dropout = 0.0,
                 backbone = 'lstm',
                 ):
        super().__init__()
        if backbone == 'lstm':
            self.backbone = nn.LSTM(
                input_size = input_size,
                hidden_size = hidden_size,
                num_layers = layers,
                dropout = dropout,
                batch_first = True
            )
        elif backbone == 'rnn':
            self.backbone = nn.RNN(
                input_size = input_size,
                hidden_size = hidden_size,
                num_layers = layers,
                nonlinearity = "tanh", 
                dropout = dropout, 
                batch_first = True
            )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.backbone(x)
        last = out[:, -1, :]
        return self.head(last)