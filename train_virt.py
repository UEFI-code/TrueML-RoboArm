import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import Cerebellum
import virtualKinematic

myServoDrv = virtualKinematic.theVirtualArm()

Motors = 4
BatchSize = 16
Epochs = 1000

# Initialize cerebellum
thePredictor = Cerebellum.Predictor(Motors)
theDecider = Cerebellum.Decider(Motors)

# Initialize optimizer
optimPredictor = optim.Adam(thePredictor.parameters(), lr=0.001)
optimDecider = optim.Adam(theDecider.parameters(), lr=0.001)

# Initialize loss function
lossFunc = nn.MSELoss()

# Initialize data
def getMiniBatch(batchSize, motors, servoObj):
    outputX = []
    outputY = []
    #motorData = [0.0] * motors
    for _ in range(batchSize):
        # generate suitable motor data
        motorData = np.random.rand(motors)
        servoObj.servoAngles = motorData * 180
        # get 3D position
        realPos = servoObj.calc3DPos()
        realPos = np.array(realPos)
        realPos = realPos / sum(servoObj.armLength)
        # append data
        outputX.append(motorData)
        outputY.append(realPos)
    return torch.tensor(outputX, dtype=torch.float), torch.tensor(outputY, dtype=torch.float)
    
def trainPredictor(batchSize, motors, servoObj, predictor, optimizer, lossFunc, epochs):
    for epoch in range(epochs):
        # get data
        print("Getting data...")
        x, y = getMiniBatch(batchSize, motors, servoObj)
        # train
        print("Training...thePredictor")
        optimizer.zero_grad()
        output = predictor(x)
        loss = lossFunc(output, y)
        loss.backward()
        optimizer.step()
        # print info
        print("The Predictor train Epoch: " + str(epoch) + " | Loss: " + str(loss.item()))

def trainDecider(batchSize, motors, servoObj, decider, optimizer, lossFunc, epochs):
    for epoch in range(epochs):
        # get data
        print("Getting data...")
        x, y = getMiniBatch(batchSize, motors, servoObj)
        # train
        print("Training...theDecider")
        optimizer.zero_grad()
        output = decider(y)
        loss = lossFunc(output, x)
        loss.backward()
        optimizer.step()
        # print info
        print("The Decider train Epoch: " + str(epoch) + " | Loss: " + str(loss.item()))

def teachDecider(batchSize, predictor, decider, optimizerDecider, lossFunc, epochs):
    for epoch in range(epochs):
        # generate dummy targets
        print("Generating dummy targets...")
        targets = torch.rand(batchSize, 3)
        optimizerDecider.zero_grad()
        kasoX = decider(targets)
        kasoY = predictor(kasoX)
        # We belive predictor more strongly than decider
        loss = lossFunc(kasoY, targets)
        loss.backward()
        optimizerDecider.step()
        # print info
        print("The Decider teach Epoch: " + str(epoch) + " | Loss: " + str(loss.item()))

trainPredictor(BatchSize, Motors, myServoDrv, thePredictor, optimPredictor, lossFunc, Epochs)
torch.save(thePredictor.state_dict(), "thePredictor.pth")
trainDecider(BatchSize, Motors, myServoDrv, theDecider, optimDecider, lossFunc, Epochs * 10)
torch.save(theDecider.state_dict(), "theDecider.pth")
# teachDecider(BatchSize, thePredictor, theDecider, optimDecider, lossFunc, Epochs)
# torch.save(theDecider.state_dict(), "theDecider-finetuned.pth")