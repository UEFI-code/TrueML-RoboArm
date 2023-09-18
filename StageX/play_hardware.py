import ServoDrv
import Cerebellum
import CVModule
import torch
import random
import numpy as np
import os
import autoFetch
import time

myServoDrv = ServoDrv.ServoDrv(4)
myCV = CVModule.myScaner()

# Create a decider
decider = Cerebellum.Decider(4, 6)
decider.load_state_dict(torch.load('pths/theDecider-finetuned.pth'))
decider.cuda()

def getSamples(cvObj, servoObj, num):
    Samples = []
    for i in range(num):
        pos = (0, 0, 0)
        while pos == (0, 0, 0):
            # while(not myServoDrv.setServoGroupAngle(np.random.randint(low=0, high=180, size=4))):
            #     pass
            servoObj.arm_device.Arm_serial_set_torque(0)
            servoObj.arm_device.Arm_Buzzer_On(5)
            print('Please move the arm manially...')
            input('Press Enter to continue...')
            servoObj.getHardwareAngles()
            print('Current Angle: ', servoObj.servoAngles)
            pos = cvObj.searchScanAngle()
        servoObj.arm_device.Arm_Buzzer_On(3)
        servoObj.arm_device.Arm_Buzzer_On(3)
        print('Get %d Position: ' % i, pos)
        for m in servoObj.servoAngles:
            Samples.append(m)
        for m in pos:
            Samples.append(m)
    return torch.tensor(Samples).cuda()

if os.path.exists('HardwareSamples.pkl'):
    mySamples = torch.load('HardwareSamples.pkl')
    print('Load Samples from file.')
else:
    autoFetch.fetch(myServoDrv.arm_device)
    time.sleep(1)
    mySamples = getSamples(myCV, myServoDrv, 6).unsqueeze_(0)
    torch.save(mySamples, 'HardwareSamples.pkl')
print('Samples: ', mySamples)

while True:
    x,y,z = input('Please input the goal: ').split()
    x = float(x) / 4
    y = float(y) / 4
    z = float(z) / 4
    goal = torch.tensor([x,y,z]).cuda()
    goal.unsqueeze_(0)
    print('Goal: ', goal)
    action = decider(torch.cat((goal, mySamples), 1))
    print('Action: ', action)
    # PredictedResult = predictor(action)
    # print('Predicted Result: ', PredictedResult)
    controlData = action[0] * 180
    print('Control Data: ', controlData)
    myServoDrv.setServoGroupAngleInternal(controlData)
    