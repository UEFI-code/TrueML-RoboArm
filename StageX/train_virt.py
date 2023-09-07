import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import Cerebellum
import virtualKinematic
import time
import cv2

myServoDrv = virtualKinematic.theVirtualArm()

Motors = 4
BatchSize = 512
Epochs = 50000
SampleNum = 6

# Initialize cerebellum
thePredictor = Cerebellum.Predictor(Motors, SampleNum)
theDecider = Cerebellum.Decider(Motors, SampleNum)

# Initialize optimizer
optimPredictor = optim.Adam(thePredictor.parameters(), lr=0.0001)
optimDecider = optim.Adam(theDecider.parameters(), lr=0.0001)

# Initialize loss function
lossFunc = nn.L1Loss()

trainingDevice = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Training device: " + str(trainingDevice))
thePredictor.to(trainingDevice)
theDecider.to(trainingDevice)

def initVirtualArmHyperParam(servoObj):
    # make arm length 0.3 ~ 1.0
    servoObj.armLength = np.random.rand(4) * 0.7 + 0.3
    print("Arm length: " + str(servoObj.armLength))
    src_points = np.float32([[0, 0], [1, 0], [0, 1], [1, 1]])
    dst_offset1 = np.float32(np.random.rand(4, 2) * 0.6 - 0.3)
    dst_points1 = src_points + dst_offset1
    servoObj.camTrans1 = cv2.getPerspectiveTransform(src_points, dst_points1)
    print("Camera Transform1:\n" + str(servoObj.camTrans1))
    dst_offset2 = np.float32(np.random.rand(4, 2) * 0.6 - 0.3)
    dst_points2 = src_points + dst_offset2
    servoObj.camTrans2 = cv2.getPerspectiveTransform(src_points, dst_points2)
    print("Camera Transform2:\n" + str(servoObj.camTrans2))

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
        initVirtualArmHyperParam(servoObj)
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
        initVirtualArmHyperParam(servoObj)
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
        initVirtualArmHyperParam(myServoDrv)
        # generate dummy targets
        print("Generating dummy targets...")
        #targets = torch.rand(batchSize, 3)
        _, targets = getMiniBatch(batchSize, Motors, myServoDrv)
        theSample = None
        for _ in range(SampleNum):
            s_x, s_y = getMiniBatch(batchSize, Motors, myServoDrv)
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

try:
    thePredictor.load_state_dict(torch.load("thePredictor.pth"))
    print("thePredictor.pth loaded")
    theDecider.load_state_dict(torch.load("theDecider.pth"))
    print("theDecider.pth loaded")
except Exception as e:
    print(e)

trainPredictor(BatchSize, Motors, myServoDrv, thePredictor, optimPredictor, lossFunc, Epochs, trainingDevice)
torch.save(thePredictor.state_dict(), "thePredictor.pth")

trainDecider(BatchSize, Motors, myServoDrv, theDecider, optimDecider, lossFunc, Epochs, trainingDevice)
torch.save(theDecider.state_dict(), "theDecider.pth")

teachDecider(BatchSize, Motors, thePredictor, theDecider, optimDecider, lossFunc, Epochs, trainingDevice)
torch.save(theDecider.state_dict(), "theDecider-finetuned.pth")

