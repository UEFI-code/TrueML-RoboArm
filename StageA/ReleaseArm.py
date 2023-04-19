from Arm_Lib import Arm_Device
myArm = Arm_Device()
myArm.Arm_serial_set_torque(0)
print('Now the arm is released!')