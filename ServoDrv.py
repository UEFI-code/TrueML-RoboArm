from Arm_Lib import Arm_Device
class ServoDrv():
    def __init__(self, servoNum):
        self.servoNum = servoNum
        self.servoAngles = [0.0] * servoNum
        self.arm_device = Arm_Device()
        # Initialize hardware
    
    def hardwareSync(self, id, angle, time):
        # Wait for hardware response
        self.arm_device.Arm_serial_servo_write(id, angle, time)
        return True

    def setServoAngle(self, id, angle, time):
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
    
    def setServoGroupAngle(self, angleData, time):
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
    
    def setServoGroupRatio(self, ratioData, time):
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