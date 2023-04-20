import ServoDrv
import Cerebellum
import torch
import virtualKinematic

myServoDrv = ServoDrv.ServoDrv(4)

# Create a decider

decider = Cerebellum.Decider(4)
decider.load_state_dict(torch.load('pths/theDecider_baseline.pth'))
decider.cuda()

# Create a predictor

# predictor = Cerebellum.Predictor(4)
# predictor.load_state_dict(torch.load('pths/thePredictor_baseline.pth'))
# predictor.cuda()

# create virtual kinematics
virtualArm = virtualKinematic.theVirtualArm()

while True:
    x,y,z = input('Please input the goal: ').split()
    x = float(x) / 4
    y = float(y) / 4
    z = float(z) / 4
    goal = torch.tensor([x,y,z]).cuda()
    goal.unsqueeze_(0)
    action = decider(goal)
    print('Action: ', action)
    # PredictedResult = predictor(action)
    # print('Predicted Result: ', PredictedResult)
    controlData = action[0].cpu().detach().numpy() * 180
    print('Control Data: ', controlData)
    virtualArm.servoAngles = controlData
    virtualResult = virtualArm.calc3DPos()
    print('Virtual Result: ', virtualResult)
    myServoDrv.setServoGroupAngleInternal(controlData)
    