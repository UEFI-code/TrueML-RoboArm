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
optimPredictor = optim.Adam(thePredictor.parameters(), lr=0.001)
optimDecider = optim.Adam(theDecider.parameters(), lr=0.001)

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

def testPredictor(batchSize, motors, servoObj, predictor, trainingDevice = 'cpu'):
    score_sum = 0
    for _ in range(128):
        initVirtualArmHyperParam(servoObj)
        x, y = getMiniBatch(batchSize, motors, servoObj)
        for _ in range(SampleNum):
            s_x, s_y = getMiniBatch(batchSize, motors, servoObj)
            x = torch.cat((x, s_x, s_y), dim = 1)
        if trainingDevice != 'cpu':
            x = x.to(trainingDevice)
            y = y.to(trainingDevice)
        output = predictor(x)
        simliar = 1 - nn.L1Loss()(output, y).abs() /  torch.cat((output, y), dim = 0).abs().mean()
        print(output)
        print(y)
        #print("The Predictor test result: " + str(simliar.item() * 100) + "%")
        score_sum += simliar.item()
    print("The Avg Predictor test result: " + str(100 * score_sum / 128) + "%")

def testDecider(batchSize, motors, servoObj, decider, trainingDevice = 'cpu'):
    score_sum = 0
    for _ in range(128):
        #y = torch.rand(batchSize, 3)
        initVirtualArmHyperParam(servoObj)
        _, y = getMiniBatch(batchSize, motors, servoObj)
        print(y)
        theSample = None
        for _ in range(SampleNum):
            s_x, s_y = getMiniBatch(batchSize, motors, servoObj)
            if theSample == None:
                theSample = torch.cat([s_x, s_y], 1)
            else:
                theSample = torch.cat([theSample, s_x, s_y], 1)
        if trainingDevice != 'cpu':
            y = y.to(trainingDevice)
            theSample = theSample.to(trainingDevice)
        action = decider(torch.cat([y, theSample], 1))
        target = getVirtExperimentResult(servoObj, action)
        print(target)
        if trainingDevice != 'cpu':
            target = target.to(trainingDevice)
        simliar = 1 - nn.L1Loss()(target, y).abs() /  torch.cat((target, y), dim = 0).abs().mean()
        #print("The Decider test result: " + str(simliar.item() * 100) + "%")
        score_sum += simliar.item()
    print("The Avg Decider test result: " + str(100 * score_sum / 128) + "%")

def testTricker(batchSize, motors, servoObj, decider, predictor, testingDevice = 'cpu'):
    initVirtualArmHyperParam(servoObj)
    _, goal = getMiniBatch(batchSize, motors, servoObj)
    theSample = None
    for _ in range(SampleNum):
        s_x, s_y = getMiniBatch(batchSize, motors, servoObj)
        if theSample == None:
            theSample = torch.cat([s_x, s_y], 1)
        else:
            theSample = torch.cat([theSample, s_x, s_y], 1)
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

thePredictor.load_state_dict(torch.load("pths/thePredictor_baseline.pth"))
testPredictor(BatchSize, Motors, myServoDrv, thePredictor, trainingDevice)
time.sleep(5)

theDecider.load_state_dict(torch.load("pths/theDecider_baseline.pth"))
testDecider(BatchSize, Motors, myServoDrv, theDecider, trainingDevice)
time.sleep(5)

theDecider.load_state_dict(torch.load("pths/theDecider-finetuned.pth"))
testDecider(BatchSize, Motors, myServoDrv, theDecider, trainingDevice)
time.sleep(5)

#testTricker(100, Motors, myServoDrv, theDecider, thePredictor, trainingDevice)
