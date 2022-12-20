import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import ServoDrv
import CVModule
import Cerebellum

Motors = 6
MotorMaxTrys = 10

theServoDrv = ServoDrv.ServoDrv(Motors)
theCVModule = CVModule.myCV(0, 1, "NiceTryA", "NiceTryB")

# Initialize cerebellum
thePredictor = Cerebellum.Predictor(Motors)
theDecider = Cerebellum.Decider(Motors)

# Initialize optimizer
optimPredictor = optim.Adam(thePredictor.parameters(), lr=0.001)
optimDecider = optim.Adam(theDecider.parameters(), lr=0.001)

# Initialize loss function
lossFunc = nn.MSELoss()

# Initialize data
def getMiniBatch(batchSize, motors, servoObj, cvObj):
    outputX = []
    outputY = []
    motorData = [0.0] * motors
    for _ in range(batchSize):
        # generate suitable motor data
        for n in range(MotorMaxTrys):
            # generate random motor data
            print("Generating random motor data... times: " + str(n))
            for i in range(motors):
                motorData[i] = random.random()
            # try to set motor data
            if servoObj.setServoGroupRatio(motorData):
                print('Great! Motor data is suitable.')
                break
            # check if it is the last try
            if n == MotorMaxTrys - 1:
                print("Error Critical: ServoDrv.setServoGroup() failed. Please check the hardware connection.")
                exit(1)
        # get 3D position
        ret, realPos = cvObj.getQRCode3DPos()
        if not ret:
            print("Error Critical: CVModule.getQRCode3DPos() failed. Please check the camera connection.")
            exit(1)
        # append data
        outputX.append(motorData)
        outputY.append(realPos)
    return torch.tensor(outputX, dtype=torch.float), torch.tensor(outputY, dtype=torch.float)
    
def trainPredictor(batchSize, motors, servoObj, cvObj, predictor, optimizer, lossFunc, epochs):
    for epoch in range(epochs):
        # get data
        print("Getting data...")
        x, y = getMiniBatch(batchSize, motors, servoObj, cvObj)
        # train
        print("Training...thePredictor")
        optimizer.zero_grad()
        output = predictor(x)
        loss = lossFunc(output, y)
        loss.backward()
        optimizer.step()
        # print info
        print("The Predictor train Epoch: " + str(epoch) + " | Loss: " + str(loss.item()))

def trainDecider(batchSize, motors, servoObj, cvObj, decider, optimizer, lossFunc, epochs):
    for epoch in range(epochs):
        # get data
        print("Getting data...")
        x, y = getMiniBatch(batchSize, motors, servoObj, cvObj)
        # train
        print("Training...theDecider")
        optimizer.zero_grad()
        output = decider(y)
        loss = lossFunc(output, x)
        loss.backward()
        optimizer.step()
        # print info
        print("The Decider train Epoch: " + str(epoch) + " | Loss: " + str(loss.item()))