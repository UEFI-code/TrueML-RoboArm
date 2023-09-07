import ServoDrv
import Cerebellum
import CVModule
import torch
import random

myServoDrv = ServoDrv.ServoDrv(4)
myCV = CVModule.myScaner()

# Create a decider
decider = Cerebellum.Decider(4)
decider.load_state_dict(torch.load('pths/theDecider_baseline.pth'))
decider.cuda()

def getSamples(cvObj, servoObj, num):
    Samples = []
    for i in range(num):
        pos = (0, 0, 0)
        while pos == (0, 0, 0):
            while(not myServoDrv.moveServoAngleInternal(random.randint(0, 3), random.randint(0, 90) - 45, 1024)):
                pass
            print('Current Angle: ', servoObj.servoAngles)
            pos = cvObj.searchScanAngle()
        print('Get %d Position: ' % i, pos)
        for m in servoObj.servoAngles:
            Samples.append(m)
        for m in pos:
            Samples.append(m)
    return torch.tensor(Samples).cuda()

mySamples = getSamples(myCV, myServoDrv, 6).unsqueeze_(0)

while True:
    x,y,z = input('Please input the goal: ').split()
    x = float(x) / 4
    y = float(y) / 4
    z = float(z) / 4
    goal = torch.tensor([x,y,z]).cuda()
    goal.unsqueeze_(0)
    action = decider(torch.cat((goal, mySamples), 1))
    print('Action: ', action)
    # PredictedResult = predictor(action)
    # print('Predicted Result: ', PredictedResult)
    controlData = action[0] * 180
    print('Control Data: ', controlData)
    myServoDrv.setServoGroupAngleInternal(controlData)
    