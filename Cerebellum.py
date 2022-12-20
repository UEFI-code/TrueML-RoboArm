import torch.nn as nn
import torch.nn.functional as F

class Predictor(nn.Module):
    def __init__(self, servoNum):
        super(Predictor, self).__init__()
        self.linear1 = nn.Linear(servoNum, 64)
        self.linear2 = nn.Linear(64, 64)
        self.linear3 = nn.Linear(64, 64)
        self.linear4 = nn.Linear(64, 3)
    
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x

class Decider(nn.Module):
    def __init__(self, servoNum):
        super(Decider, self).__init__()
        self.linear1 = nn.Linear(3, 64)
        self.linear2 = nn.Linear(64, 64)
        self.linear3 = nn.Linear(64, 64)
        self.linear4 = nn.Linear(64, servoNum)
    
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x
        