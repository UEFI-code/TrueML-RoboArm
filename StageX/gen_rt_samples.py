import virtualKinematic
import numpy as np
import torch

myServoDrv = virtualKinematic.theVirtualArm()

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

mySamples = None
for _ in range(6):
    s_x, s_y = getMiniBatch(1, 4, myServoDrv)
    if mySamples is None:
        mySamples = torch.cat((s_x, s_y), 1)
    else:
        mySamples = torch.cat((mySamples, s_x, s_y), 1)
print(mySamples)
torch.save(mySamples, 'HardwareSamples.pkl')
