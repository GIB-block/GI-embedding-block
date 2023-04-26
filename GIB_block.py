import torch.nn as nn
class Global_Information_emBedding(nn.Module):
    def __init__(self, in_channel, in_shape):
        super(Global_Information_emBedding, self).__init__()
        d = max(in_shape // 2, 1)

        self.fc1 = nn.Sequential(nn.Linear(in_shape, d),
                                 nn.BatchNorm1d(in_channel),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(d, in_shape),
                                 nn.BatchNorm1d(in_channel),
                                 nn.Sigmoid(),

                                 )
        self.fc2 = nn.Sequential(nn.Linear(in_shape, d),
                                 nn.BatchNorm1d(in_channel),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(d, in_shape),
                                 nn.BatchNorm1d(in_channel),
                                 nn.Sigmoid(),
                                 )

        self.pool_h = nn.AdaptiveMaxPool2d((1, None))
        self.pool_w = nn.AdaptiveMaxPool2d((None, 1))

    def forward(self, X):
        X_h = self.pool_h(X).squeeze(2)
        X_w = self.pool_w(X).squeeze(3)
        X_h = self.fc1(X_h)
        X_w = self.fc2(X_w)#
        X_h = X_h.unsqueeze(dim=2)
        X_w = X_w.unsqueeze(dim=3)
        X_h = X_h / 2
        X_w = X_w / 2
        out = X_h + X_w
        return out * X