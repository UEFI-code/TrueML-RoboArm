import torch
import torch.nn as nn
import torch.nn.functional as F

class Predictor(nn.Module):
    def __init__(self, servoNum):
        super(Predictor, self).__init__()
        self.linear1 = nn.Linear(servoNum, 128)
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, 128)
        self.linear4 = nn.Linear(128, 3)
    
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x

class Decider(nn.Module):
    def __init__(self, servoNum):
        super(Decider, self).__init__()
        self.linear1 = nn.Linear(3, 128)
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, 128)
        self.linear4 = nn.Linear(128, servoNum)
    
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x

class Decider2(nn.Module):
    def __init__(self, servoNum):
        super(Decider2, self).__init__()
        self.linear1 = nn.Linear(3 + 64, 128)
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, 128)
        self.linear4 = nn.Linear(128, servoNum)
    
    def forward(self, x, noise):
        x = F.relu(self.linear1(torch.cat((x, noise), dim = 1)))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x

def theRealTricker(goal, Decider, Predictor, device = 'cpu'):
    angles = Decider(goal)
    # angles = torch.rand(goal.size(0), 4, device = device)

    angles = torch.nn.Parameter(angles)
    losser = torch.nn.L1Loss()
    for i in range(2000):
        y = Predictor(angles)
        loss = losser(y, goal)
        print(loss)
        loss.backward()
        angles.data -= 0.05 * angles.grad.data
        angles.grad.data.zero_()
    return angles.data