import ServoDrv
import Cerebellum
import torch
import CVModule
import virtualKinematic
import numpy as np
from autoFetch import fetch

myServoDrv = ServoDrv.ServoDrv(4)
myCV = CVModule.myScaner()
virtualArm = virtualKinematic.theVirtualArm()
# Create a decider

decider = Cerebellum.Decider(4)
decider.load_state_dict(torch.load('pths/theDecider_baseline.pth'))
decider.cuda()

def getMiniBatch(batchSize, motors, servoObj):
    output = []
    #motorData = [0.0] * motors
    for _ in range(batchSize):
        # generate suitable motor data
        # alpha should be 45 to 90
        alpha = (np.pi / 4) + np.random.rand() * (np.pi / 4)
        theta = np.random.rand() * np.pi
        x = np.cos(theta) * np.cos(alpha)
        y = np.sin(theta) * np.cos(alpha)
        z = np.sin(alpha)
        output.append((x, y, z))
    return torch.tensor(output, dtype=torch.float)

def getHardExperimentResult(hardServoObj, cvObj, actions):
    targets = []
    for i in actions:
        res = hardServoObj.setServoGroupAngleInternal(i)
        if res: 
            targets.append(cvObj.searchScanAngle())
        else:
            targets.append([0,0,0])
    return torch.tensor(targets, dtype=torch.float)


def testDecider(batchSize, motors, virtServoObj, hardServoObj, cvObj, decider, trainingDevice = 'cpu'):
    #y = torch.rand(batchSize, 3)
    goals = getMiniBatch(batchSize, motors, virtServoObj)
    print(goals)
    open('testDecider_hardware_goals.txt', 'w').write(str(goals.tolist()))
    if trainingDevice != 'cpu':
        goals = goals.to(trainingDevice)
    actions = decider(goals) * 180
    print(actions)
    open('testDecider_hardware_actions.txt', 'w').write(str(actions.tolist()))
    targets = getHardExperimentResult(hardServoObj, cvObj, actions)
    print(targets)
    open('testDecider_hardware_targets.txt', 'w').write(str(targets.tolist()))
    if trainingDevice != 'cpu':
        targets = targets.to(trainingDevice)
    simliar = 1 - nn.L1Loss()(targets, goals).abs() /  torch.cat((targets, goals), dim = 0).abs().mean()
    print("The Decider test result: " + str(simliar.item() * 100) + "%")

if __name__ == '__main__':
    fetch(myServoDrv.arm_device)
    testDecider(128, 4, virtualArm, myServoDrv, myCV, decider, 'cuda')