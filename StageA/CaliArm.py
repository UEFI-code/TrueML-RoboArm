import time
from Arm_Lib import Arm_Device
myArm = Arm_Device()
myArm.Arm_serial_set_torque(1)
testAngles = [180, 0, 0, 0, 0, 0]
myArm.Arm_serial_servo_write6_array(testAngles, 3000)