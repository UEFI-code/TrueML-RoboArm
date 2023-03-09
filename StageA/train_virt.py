import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import Cerebellum
import virtualKinematic
import time

myServoDrv = virtualKinematic.theVirtualArm()

Motors = 4
BatchSize = 512
Epochs = 5000

# Initialize cerebellum
thePredictor = Cerebellum.Predictor(Motors)
theDecider = Cerebellum.Decider(Motors)
theDecider2 = Cerebellum.Decider2(Motors)

# Initialize optimizer
optimPredictor = optim.Adam(thePredictor.parameters(), lr=0.001)
optimDecider = optim.Adam(theDecider.parameters(), lr=0.001)
optimDecider2 = optim.Adam(theDecider2.parameters(), lr=0.001)

# Initialize loss function
lossFunc = nn.L1Loss()

trainingDevice = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Training device: " + str(trainingDevice))
thePredictor.to(trainingDevice)
theDecider.to(trainingDevice)
theDecider2.to(trainingDevice)

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
    
def trainPredictor(batchSize, motors, servoObj, predictor, optimizer, lossFunc, epochs, trainingDevice='cpu'):
    for epoch in range(epochs):
        optimizer.zero_grad()
        # get data
        print("Getting data...")
        x, y = getMiniBatch(batchSize, motors, servoObj)
        if trainingDevice != 'cpu':
            x = x.to(trainingDevice)
            y = y.to(trainingDevice)
        # train
        print("Training...thePredictor")
        output = predictor(x)
        loss = lossFunc(output, y)
        loss.backward()
        optimizer.step()
        # print info
        print("The Predictor train Epoch: " + str(epoch) + " | Loss: " + str(loss.item()))

def trainDecider(batchSize, motors, servoObj, decider, optimizer, lossFunc, epochs, trainingDevice = 'cpu'):
    for epoch in range(epochs):
        optimizer.zero_grad()
        # get data
        print("Getting data...")
        x, y = getMiniBatch(batchSize, motors, servoObj)
        if trainingDevice != 'cpu':
            x = x.to(trainingDevice)
            y = y.to(trainingDevice)
        # train
        print("Training...theDecider")
        action = decider(y)
        loss = lossFunc(action, x)
        loss.backward()
        optimizer.step()
        # print info
        print("The Decider train Epoch: " + str(epoch) + " | Loss: " + str(loss.item()))

def trainDecider2(batchSize, motors, servoObj, decider, optimizer, lossFunc, epochs, trainingDevice = 'cpu'):
    for epoch in range(epochs):
        optimizer.zero_grad()
        # get data
        print("Getting data...")
        x, y = getMiniBatch(batchSize, motors, servoObj)
        if trainingDevice != 'cpu':
            x = x.to(trainingDevice)
            y = y.to(trainingDevice)
        # train
        print("Training...theDecider")
        noise = torch.rand(batchSize, 64).to(trainingDevice)
        action = decider(y, noise)
        loss = lossFunc(action, x)
        loss.backward()
        optimizer.step()
        # print info
        print("The Decider train Epoch: " + str(epoch) + " | Loss: " + str(loss.item()))

def teachDecider(batchSize, Motors, predictor, decider, optimizerDecider, lossFunc, epochs, trainingDevice = 'cpu'):
    for epoch in range(epochs):
        optimizerDecider.zero_grad()
        # generate dummy targets
        print("Generating dummy targets...")
        #targets = torch.rand(batchSize, 3)
        _, targets = getMiniBatch(batchSize, Motors, myServoDrv)
        if trainingDevice != 'cpu':
            targets = targets.to(trainingDevice)
        kasoX = decider(targets)
        kasoY = predictor(kasoX)
        # We belive predictor more strongly than decider
        loss = lossFunc(kasoY, targets)
        loss.backward()
        optimizerDecider.step()
        # print info
        print("The Decider teach Epoch: " + str(epoch) + " | Loss: " + str(loss.item()))

def teachDecider2(batchSize, Motors, predictor, decider, optimizerDecider, lossFunc, epochs, trainingDevice = 'cpu'):
    for epoch in range(epochs):
        optimizerDecider.zero_grad()
        # generate dummy targets
        print("Generating dummy targets...")
        #targets = torch.rand(batchSize, 3)
        _, targets = getMiniBatch(batchSize, Motors, myServoDrv)
        if trainingDevice != 'cpu':
            targets = targets.to(trainingDevice)
        noise = torch.rand(batchSize, 64).to(trainingDevice)
        kasoX = decider(targets, noise)
        kasoY = predictor(kasoX)
        # We belive predictor more strongly than decider
        loss = lossFunc(kasoY, targets)
        loss.backward()
        optimizerDecider.step()
        # print info
        print("The Decider teach Epoch: " + str(epoch) + " | Loss: " + str(loss.item()))

def testPredictor(batchSize, motors, servoObj, predictor, trainingDevice = 'cpu'):
    x, y = getMiniBatch(batchSize, motors, servoObj)
    if trainingDevice != 'cpu':
        x = x.to(trainingDevice)
        y = y.to(trainingDevice)
    output = predictor(x)
    simliar = 1 - nn.L1Loss()(output, y).abs() /  torch.cat((output, y), dim = 0).abs().mean()
    print(output)
    print(y)
    print("The Predictor test result: " + str(simliar.item() * 100) + "%")

def testDecider(batchSize, motors, servoObj, decider, trainingDevice = 'cpu'):
    #y = torch.rand(batchSize, 3)
    _, y = getMiniBatch(batchSize, motors, servoObj)
    print(y)
    if trainingDevice != 'cpu':
        y = y.to(trainingDevice)
    action = decider(y)
    target = getVirtExperimentResult(servoObj, action)
    print(target)
    if trainingDevice != 'cpu':
        target = target.to(trainingDevice)
    simliar = 1 - nn.L1Loss()(target, y).abs() /  torch.cat((target, y), dim = 0).abs().mean()
    print("The Decider test result: " + str(simliar.item() * 100) + "%")

def testDecider2(batchSize, motors, servoObj, decider, trainingDevice = 'cpu'):
    #y = torch.rand(batchSize, 3)
    _, y = getMiniBatch(batchSize, motors, servoObj)
    print(y)
    if trainingDevice != 'cpu':
        y = y.to(trainingDevice)
    noise = torch.rand(batchSize, 64).to(trainingDevice)
    action = decider(y, noise)
    target = getVirtExperimentResult(servoObj, action)
    print(target)
    if trainingDevice != 'cpu':
        target = target.to(trainingDevice)
    simliar = 1 - nn.L1Loss()(target, y).abs() /  torch.cat((target, y), dim = 0).abs().mean()
    print("The Decider test result: " + str(simliar.item() * 100) + "%")

def testTricker(batchSize, motors, servoObj, decider, predictor, testingDevice = 'cpu'):
    #goal = torch.rand(batchSize, 3)
    _, goal = getMiniBatch(batchSize, motors, servoObj)
    if testingDevice != 'cpu':
        goal = goal.to(testingDevice)
    action = Cerebellum.theRealTricker(goal, decider, predictor, testingDevice)
    target = getVirtExperimentResult(servoObj, action)
    if testingDevice != 'cpu':
        target = target.to(testingDevice)
    simliar = 1 - nn.L1Loss()(target, goal).abs() / torch.cat((target, goal), dim = 0).abs().mean()
    print(target)
    print(goal)
    print("The Tricker test result: " + str(simliar.item() * 100) + "%")
        
# trainPredictor(BatchSize, Motors, myServoDrv, thePredictor, optimPredictor, lossFunc, Epochs, trainingDevice)
# torch.save(thePredictor.state_dict(), "thePredictor.pth")
thePredictor.load_state_dict(torch.load("pths/thePredictor_baseline.pth"))
testPredictor(BatchSize, Motors, myServoDrv, thePredictor, trainingDevice)

#trainDecider(BatchSize, Motors, myServoDrv, theDecider, optimDecider, lossFunc, Epochs * 2, trainingDevice)
#torch.save(theDecider.state_dict(), "theDecider.pth")
theDecider.load_state_dict(torch.load("pths/theDecider_baseline.pth"))
testDecider(BatchSize, Motors, myServoDrv, theDecider, trainingDevice)
time.sleep(5)

#teachDecider(BatchSize, Motors, thePredictor, theDecider, optimDecider, lossFunc, Epochs * 2, trainingDevice)
#torch.save(theDecider.state_dict(), "theDecider-finetuned.pth")
theDecider.load_state_dict(torch.load("pths/theDecider-finetuned.pth"))
testDecider(BatchSize, Motors, myServoDrv, theDecider, trainingDevice)
time.sleep(5)

#trainDecider2(BatchSize, Motors, myServoDrv, theDecider2, optimDecider2, lossFunc, Epochs * 2, trainingDevice)
#torch.save(theDecider2.state_dict(), "theDecider2.pth")
# theDecider2.load_state_dict(torch.load("pths/theDecider2_baseline.pth"))
# testDecider2(BatchSize, myServoDrv, theDecider2, trainingDevice)

# teachDecider2(BatchSize, thePredictor, theDecider2, optimDecider2, lossFunc, Epochs * 2, trainingDevice)
# torch.save(theDecider2.state_dict(), "theDecider2-finetuned.pth")
# testDecider2(BatchSize, myServoDrv, theDecider2, trainingDevice)

#testTricker(100, Motors, myServoDrv, theDecider, thePredictor, trainingDevice)
