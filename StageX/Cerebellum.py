import torch
import torch.nn as nn
import torch.nn.functional as F

class Predictor(nn.Module):
    def __init__(self, servoNum, SampleNum):
        super(Predictor, self).__init__()
        self.linear1 = nn.Linear(servoNum + SampleNum * (servoNum + 3), 1024)
        self.linear2 = nn.Linear(1024, 1024)
        self.linear3 = nn.Linear(1024, 1024)
        self.linear4 = nn.Linear(1024, 3)
    
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x

class Decider(nn.Module):
    def __init__(self, servoNum, SampleNum):
        super(Decider, self).__init__()
        self.linear1 = nn.Linear(3 + SampleNum * (servoNum + 3), 1024)
        self.linear2 = nn.Linear(1024, 1024)
        self.linear3 = nn.Linear(1024, 1024)
        self.linear4 = nn.Linear(1024, servoNum)
    
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x

def theRealTricker(goal, theSample, Decider, Predictor, device = 'cpu'):
    angles = torch.nn.Parameter(torch.rand(goal.shape[0], 4, device = device))
    angles.requires_grad = True
    losser = torch.nn.L1Loss()
    for i in range(20000):
        y = Predictor(torch.cat([angles, theSample], 1))
        loss = losser(y, goal)
        print(loss)
        loss.backward()
        angles.data -= 0.05 * angles.grad.data
        angles.grad.data.zero_()
    return angles.data