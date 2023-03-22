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
SampleNum = 6

# Initialize cerebellum
thePredictor = Cerebellum.Predictor(Motors, SampleNum)
theDecider = Cerebellum.Decider(Motors, SampleNum)

# Initialize optimizer
optimPredictor = optim.Adam(thePredictor.parameters(), lr=0.001)
optimDecider = optim.Adam(theDecider.parameters(), lr=0.001)

# Initialize loss function
lossFunc = nn.L1Loss()

trainingDevice = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Training device: " + str(trainingDevice))
thePredictor.to(trainingDevice)
theDecider.to(trainingDevice)

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
        # make arm length 0.3 ~ 1.0
        servoObj.armLength = np.random.rand(4) * 0.7 + 0.3
        print("Arm length: " + str(servoObj.armLength))
        # get data
        print("Getting data...")
        x, y = getMiniBatch(batchSize, motors, servoObj)
        for _ in range(SampleNum):
            s_x, s_y = getMiniBatch(batchSize, motors, servoObj)
            x = torch.cat((x, s_x), 1)
            x = torch.cat((x, s_y), 1)
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
        # make arm length 0.3 ~ 1.0
        servoObj.armLength = np.random.rand(4) * 0.7 + 0.3
        print("Arm length: " + str(servoObj.armLength))
        # get data
        print("Getting data...")
        x, y = getMiniBatch(batchSize, motors, servoObj)
        for _ in range(SampleNum):
            s_x, s_y = getMiniBatch(batchSize, motors, servoObj)
            y = torch.cat((y, s_x), 1)
            y = torch.cat((y, s_y), 1)
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

def teachDecider(batchSize, Motors, predictor, decider, optimizerDecider, lossFunc, epochs, trainingDevice = 'cpu'):
    for epoch in range(epochs):
        optimizerDecider.zero_grad()
        # make arm length 0.3 ~ 1.0
        servoObj.armLength = np.random.rand(4) * 0.7 + 0.3
        print("Arm length: " + str(servoObj.armLength))
        # generate dummy targets
        print("Generating dummy targets...")
        #targets = torch.rand(batchSize, 3)
        _, targets = getMiniBatch(batchSize, Motors, myServoDrv)
        theSample = None
        for _ in range(SampleNum):
            s_x, s_y = getMiniBatch(batchSize, motors, servoObj)
            if theSample == None:
                theSample = s_x
            else:
                theSample = torch.cat([theSample, s_x], 1)
            theSample = torch.cat([theSample, s_y], 1)
        if trainingDevice != 'cpu':
            targets = targets.to(trainingDevice)
            theSample = theSample.to(trainingDevice)
        kasoX = decider(torch.cat([targets, theSample], 1))
        kasoY = predictor(torch.cat([kasoX, theSample], 1))
        # We belive predictor more strongly than decider
        loss = lossFunc(kasoY, targets)
        loss.backward()
        optimizerDecider.step()
        # print info
        print("The Decider teach Epoch: " + str(epoch) + " | Loss: " + str(loss.item()))

def testPredictor(batchSize, motors, servoObj, predictor, trainingDevice = 'cpu'):
    score_sum = 0
    for _ in range(128):
        # make the length at 0.3 ~ 1.0
        servoObj.armLength = np.random.rand(4) * 0.7 + 0.3
        print("Arm length: " + str(servoObj.armLength))
        x, y = getMiniBatch(batchSize, motors, servoObj)
        for _ in range(SampleNum):
            s_x, s_y = getMiniBatch(batchSize, motors, servoObj)
            x = torch.cat((x, s_x), 1)
            x = torch.cat((x, s_y), 1)
        if trainingDevice != 'cpu':
            x = x.to(trainingDevice)
            y = y.to(trainingDevice)
        output = predictor(x)
        simliar = 1 - nn.L1Loss()(output, y).abs() /  torch.cat((output, y), dim = 0).abs().mean()
        print(output)
        print(y)
        print("The Predictor test result: " + str(simliar.item() * 100) + "%")
        score_sum += simliar.item()
    print("The Avg Predictor test result: " + str(100 * score_sum / 128) + "%")

def testDecider(batchSize, motors, servoObj, decider, trainingDevice = 'cpu'):
    score_sum = 0
    for _ in range(128):
        #y = torch.rand(batchSize, 3)
        # make arm length 0.3 ~ 1.0
        servoObj.armLength = np.random.rand(4) * 0.7 + 0.3
        print("Arm length: " + str(servoObj.armLength))
        _, y = getMiniBatch(batchSize, motors, servoObj)
        print(y)
        theSample = None
        for _ in range(SampleNum):
            s_x, s_y = getMiniBatch(batchSize, motors, servoObj)
            if theSample == None:
                theSample = s_x
            else:
                theSample = torch.cat([theSample, s_x], 1)
            theSample = torch.cat([theSample, s_y], 1)
        if trainingDevice != 'cpu':
            y = y.to(trainingDevice)
            theSample = theSample.to(trainingDevice)
        action = decider(torch.cat([y, theSample], 1))
        target = getVirtExperimentResult(servoObj, action)
        print(target)
        if trainingDevice != 'cpu':
            target = target.to(trainingDevice)
        simliar = 1 - nn.L1Loss()(target, y).abs() /  torch.cat((target, y), dim = 0).abs().mean()
        print("The Decider test result: " + str(simliar.item() * 100) + "%")
        score_sum += simliar.item()
    print("The Avg Decider test result: " + str(100 * score_sum / 128) + "%")

def testTricker(batchSize, motors, servoObj, decider, predictor, testingDevice = 'cpu'):
    servoObj.armLength = np.random.rand(4) * 0.7 + 0.3
    print("Arm length: " + str(servoObj.armLength))
    #goal = torch.rand(batchSize, 3)
    _, goal = getMiniBatch(batchSize, motors, servoObj)
    theSample = None
    for _ in range(SampleNum):
        s_x, s_y = getMiniBatch(batchSize, motors, servoObj)
        if theSample == None:
            theSample = s_x
        else:
            theSample = torch.cat([theSample, s_x], 1)
        theSample = torch.cat([theSample, s_y], 1)
    if testingDevice != 'cpu':
        goal = goal.to(testingDevice)
        theSample = theSample.to(testingDevice)
    
    action = Cerebellum.theRealTricker(goal, theSample, decider, predictor, testingDevice)
    target = getVirtExperimentResult(servoObj, action)
    if testingDevice != 'cpu':
        target = target.to(testingDevice)
    simliar = 1 - nn.L1Loss()(target, goal).abs() / torch.cat((target, goal), dim = 0).abs().mean()
    print(target)
    print(goal)
    print("The Tricker test result: " + str(simliar.item() * 100) + "%")
        
#trainPredictor(BatchSize, Motors, myServoDrv, thePredictor, optimPredictor, lossFunc, Epochs, trainingDevice)
#torch.save(thePredictor.state_dict(), "thePredictor.pth")
thePredictor.load_state_dict(torch.load("pths/thePredictor_baseline.pth"))
testPredictor(BatchSize, Motors, myServoDrv, thePredictor, trainingDevice)

#trainDecider(BatchSize, Motors, myServoDrv, theDecider, optimDecider, lossFunc, Epochs * 2, trainingDevice)
#torch.save(theDecider.state_dict(), "theDecider.pth")
#theDecider.load_state_dict(torch.load("pths/theDecider_baseline.pth"))
#testDecider(BatchSize, Motors, myServoDrv, theDecider, trainingDevice)
#time.sleep(5)

#teachDecider(BatchSize, Motors, thePredictor, theDecider, optimDecider, lossFunc, Epochs * 2, trainingDevice)
#torch.save(theDecider.state_dict(), "theDecider-finetuned.pth")
#theDecider.load_state_dict(torch.load("pths/theDecider-finetuned.pth"))
#testDecider(BatchSize, Motors, myServoDrv, theDecider, trainingDevice)
#time.sleep(5)

testTricker(100, Motors, myServoDrv, theDecider, thePredictor, trainingDevice)

