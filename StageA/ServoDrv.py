from Arm_Lib import Arm_Device
import time as time_lib
class ServoDrv():
    def __init__(self, servoNum):
        self.servoNum = servoNum
        self.servoAngles = [0.0] * servoNum
        self.arm_device = Arm_Device()
        self.arm_device.Arm_serial_set_torque(1)
        # Initialize hardware
        for i in range(1, 5):
            self.servoAngles[i - 1] = self.arm_device.Arm_serial_servo_read(i)
    
    def hardwareSync(self, id, angle, time):
        # Wait for hardware response
        self.arm_device.Arm_serial_servo_write(id + 1, angle, time)
        time_lib.sleep(time / 1000.0)
        return True

    def setServoAngle(self, id, angle, time = 1024):
        if id < 0 or id >= self.servoNum:
            return False
        if angle < 0.0 or angle > 180.0:
            return False
        if time <= 0:
            return False
        if self.hardwareSync(id, angle, time):
            self.servoAngles[id] = angle
            return True
        else:
            return False

    def setServoAngleInternal(self, id, angle, time):
        # Wait for hardware response
        if self.hardwareSync(id, angle, time):
            self.servoAngles[id] = angle
            return True
        else:
            return False

    def moveServoAngle(self, id, angle, time):
        if id < 0 or id >= self.servoNum:
            return False
        if time <= 0:
            return False
        currentAngle = self.servoAngles[id]
        if currentAngle + angle < 0.0 or currentAngle + angle > 180.0:
            return False
        # Wait for hardware response
        if self.hardwareSync(id, currentAngle + angle, time):
            self.servoAngles[id] += angle
            return True
        else:
            return False

    def moveServoAngleInternal(self, id, angle, time):
        # Wait for hardware response
        if self.hardwareSync(id, self.servoAngles[id] + angle, time):
            self.servoAngles[id] += angle
            return True
        else:
            return False
    
    def setServoGroupAngle(self, angleData, time = 1024):
        if len(angleData) != self.servoNum:
            return False
        if time <= 0:
            return False
        for i in range(self.servoNum):
            if angleData[i] < 0.0 or angleData[i] > 180.0:
                return False
            res = self.setServoAngleInternal(i, angleData[i], time)
            if not res:
                return False
    
    def setServoGroupAngleInternal(self, angleData, time = 1024):
        # Wait for hardware response
        for i in range(self.servoNum):
            if self.hardwareSync(i, angleData[i], time):
                self.servoAngles[i] = angleData[i]
            else:
                return False
        return True
    
    def setServoGroupRatio(self, ratioData, time = 1024):
        if len(ratioData) != self.servoNum:
            return False
        if time <= 0:
            return False
        for i in range(self.servoNum):
            if ratioData[i] < 0.0 or ratioData[i] > 1.0:
                return False
            res = self.setServoAngleInternal(i, ratioData[i] * 180, time)
            if not res:
                return False
    
    def setServoGroupRatioInternal(self, ratioData, time = 1024):
        # Wait for hardware response
        for i in range(self.servoNum):
            if self.hardwareSync(i, ratioData[i] * 180, time):
                self.servoAngles[i] = ratioData[i] * 180
            else:
                return False
        return True

if __name__ == '__main__':
    myServoDrv = ServoDrv(4)
    myServoDrv.setServoAngle(0, 90)
