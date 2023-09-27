import torch
import numpy as np
import ServoDrv
import Cerebellum
import os

predictor = Cerebellum.Predictor(4, 6)
predictor.load_state_dict(torch.load('pths/thePredictor_baseline.pth'))

myServoDrv = ServoDrv.ServoDrv(4)
myServoDrv.arm_device.Arm_serial_set_torque(0)
myServoDrv.arm_device.Arm_Buzzer_On(5)

if os.path.exists('HardwareSamples.pkl'):
    mySamples = torch.load('HardwareSamples.pkl').cuda()
    print('Load Samples from file.')
else:
    print('No Samples found.')
    exit()

while True:
    input('Please move the arm manially...')
    myServoDrv.getHardwareAngles()
    print('Current Angle: ', myServoDrv.servoAngles)
    angles = torch.tensor(myServoDrv.servoAngles.copy()).float().unsqueeze_(0) / 180
    angles = angles.cuda()
    target_predicted = predictor(torch.cat((angles, mySamples), 1))
    print('Predicted Position: ', target_predicted[0])