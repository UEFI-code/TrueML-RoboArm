from Arm_Lib import Arm_Device
import time
myArm = Arm_Device()
myArm.Arm_serial_set_torque(0)
#myArm.Arm_serial_servo_write6_array([0] * 6, 3000)
#myArm.Arm_serial_set_torque(0)
print('Now the arm is released!')
while True:
    for i in range(1, 7):
        print('Servo %d: %d' % (i, myArm.Arm_serial_servo_read(i)))
    print('----------------------------------------')
    time.sleep(1)