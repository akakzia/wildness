import torch
import torch.nn as nn
import torch.nn.functional as F


# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class TNetworkSpatialFlat(nn.Module):
    """ Only considers spatial coordinates x, y, z """
    def __init__(self, hidden_dim):
        self.input_dim = 5 + 3 # (body coord + gripper state) + (obj coord)
        super(TNetworkSpatialFlat, self).__init__()
        self.linear1 = nn.Linear(self.input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state):
        x1 = F.relu(self.linear1(state))
        x1 = self.linear2(x1)
        x1 = torch.sigmoid(self.linear3(x1))

        return x1