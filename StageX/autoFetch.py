from Arm_Lib import Arm_Device
import time as time_lib

def fetch(devObj):
    devObj.Arm_serial_set_torque(1)
    time_lib.sleep(0.3)
    readAngle = devObj.Arm_serial_servo_read(6)
    if readAngle < 160:
        devObj.Arm_serial_servo_write(6, readAngle + 5, 10)
        time_lib.sleep(0.3)
        if devObj.Arm_serial_servo_read(6) < readAngle + 3:
            print('Already fetched!')
            return
    devObj.Arm_serial_servo_write(6, 0, 1024)
    time_lib.sleep(1.0)
    print('Now put the squre QR code!')
    time_lib.sleep(3.0)
    for i in range(0,180,2):
        devObj.Arm_serial_servo_write(6, i, 10)
        time_lib.sleep(50 / 1000.0)
        readAngle = devObj.Arm_serial_servo_read(6)
        if abs(readAngle - i) > 5.0:
            print('Fetched at %d' % i)
            break

if __name__ == '__main__':
    devObj = Arm_Device()
    fetch(devObj)