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

def getVirtExperimentResult(myServoObj, batch):
    outputY = []
    for i in batch:
        motorData = i.cpu().detach() * 180
        myServoObj.servoAngles = motorData
        realPos = myServoObj.calc3DPos()
        realPos = np.array(realPos)
        realPos = realPos / sum(myServoObj.armLength)
        outputY.append(realPos)
    return torch.tensor(outputY, dtype=torch.float)
    
def trainPredictor(batchSize, motors, servoObj, predictor, optimizer, lossFunc, epochs):
    for epoch in range(epochs):
        optimizer.zero_grad()
        # get data
        print("Getting data...")
        x, y = getMiniBatch(batchSize, motors, servoObj)
        # train
        print("Training...thePredictor")
        output = predictor(x)
        loss = lossFunc(output, y)
        loss.backward()
        optimizer.step()
        # print info
        print("The Predictor train Epoch: " + str(epoch) + " | Loss: " + str(loss.item()))

def trainDecider(batchSize, motors, servoObj, decider, optimizer, lossFunc, epochs):
    for epoch in range(epochs):
        optimizer.zero_grad()
        # get data
        print("Getting data...")
        x, y = getMiniBatch(batchSize, motors, servoObj)
        # train
        print("Training...theDecider")
        action = decider(y)
        loss = lossFunc(action, x)
        loss.backward()
        optimizer.step()
        # print info
        print("The Decider train Epoch: " + str(epoch) + " | Loss: " + str(loss.item()))

def teachDecider(batchSize, predictor, decider, optimizerDecider, lossFunc, epochs):
    for epoch in range(epochs):
        optimizerDecider.zero_grad()
        # generate dummy targets
        print("Generating dummy targets...")
        targets = torch.rand(batchSize, 3)
        kasoX = decider(targets)
        kasoY = predictor(kasoX)
        # We belive predictor more strongly than decider
        loss = lossFunc(kasoY, targets)
        loss.backward()
        optimizerDecider.step()
        # print info
        print("The Decider teach Epoch: " + str(epoch) + " | Loss: " + str(loss.item()))

def testPredictor(batchSize, motors, servoObj, predictor):
    x, y = getMiniBatch(batchSize, motors, servoObj)
    output = predictor(x)
    simliar = nn.CosineSimilarity(dim=1, eps=1e-6)(output, y)
    print("The Predictor test result: " + str(simliar.mean().item() * 100) + "%")

def testDecider(batchSize, servoObj, decider):
    y = torch.rand(batchSize, 3)
    action = decider(y)
    target = getVirtExperimentResult(servoObj, action)
    simliar = nn.CosineSimilarity(dim=1, eps=1e-6)(target, y)
    print("The Decider test result: " + str(simliar.mean().item() * 100) + "%")

trainPredictor(BatchSize, Motors, myServoDrv, thePredictor, optimPredictor, lossFunc, Epochs)
torch.save(thePredictor.state_dict(), "thePredictor.pth")
trainDecider(BatchSize, Motors, myServoDrv, theDecider, optimDecider, lossFunc, Epochs * 5)
torch.save(theDecider.state_dict(), "theDecider.pth")
teachDecider(BatchSize, thePredictor, theDecider, optimDecider, lossFunc, Epochs)
# torch.save(theDecider.state_dict(), "theDecider-finetuned.pth")
# testPredictor(BatchSize, Motors, myServoDrv, thePredictor)
testDecider(BatchSize, myServoDrv, theDecider)