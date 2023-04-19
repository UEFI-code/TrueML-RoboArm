import ServoDrv
import Cerebellum
import torch

myServoDrv = ServoDrv.ServoDrv(4)

# Create a decider

decider = Cerebellum.Decider(4)
decider.load_state_dict(torch.load('pths/theDecider-finetuned.pth'))
decider.cuda()

# Create a predictor

predictor = Cerebellum.Predictor(4)
predictor.load_state_dict(torch.load('pths/thePredictor_baseline.pth'))
predictor.cuda()

while True:
    x,y,z = input('Please input the goal: ').split()
    x = float(x)
    y = float(y)
    z = float(z)
    goal = torch.tensor([x,y,z]).cuda()
    goal.unsqueeze_(0)
    action = decider(goal)
    print('Action: ', action)
    PredictedResult = predictor(action)
    print('Predicted Result: ', PredictedResult)

    controlData = action[0].cpu().detach().numpy().tolist()
    print('Control Data: ', controlData)
    myServoDrv.setServoGroupRatio(controlData)
    