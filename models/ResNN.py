import torch
from torch import nn
from torch.nn import functional as F


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2., reduction='none'):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input_tensor, target_tensor):
        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1-prob)**self.gamma) * log_prob,
            target_tensor, weight=self.weight, reduction=self.reduction
        ).mean()
    

class ClassificationModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(.5),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        out = self.fc_layers(x)
        return out
    

class ResidualBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out += residual
        out = self.relu(out)
        return out
    

class ClassificationModel2(nn.Module):
    '''
    ResNN (develop by Ph.D pant, Under Review)
    '''
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.residual_block = ResidualBlock(input_dim, hidden_dim)
        self.clf = ClassificationModel(input_dim, hidden_dim, num_classes)

    def forward(self, x):
        x = self.residual_block(x)
        return self.clf(x)