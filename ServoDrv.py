class ServoDrv():
    def __init__(self, servoNum):
        self.servoNum = servoNum
        self.servoAngles = [0.0] * servoNum
        # Initialize hardware
    
    def setServoAngle(self, id, angle, time):
        if id < 0 or id >= self.servoNum:
            return False
        if angle < 0.0 or angle > 180.0:
            return False
        if time <= 0:
            return False
        # Wait for hardware response
        self.servoAngles[id] = angle
        return True

    def setServoAngleInternal(self, id, angle, time):
        # Wait for hardware response
        self.servoAngles[id] = angle
        return True

    def moveServo(self, id, angle, time):
        if id < 0 or id >= self.servoNum:
            return False
        if time <= 0:
            return False
        currentAngle = self.servoAngles[id]
        if currentAngle + angle < 0.0 or currentAngle + angle > 180.0:
            return False
        # Wait for hardware response
        self.servoAngles[id] += angle
        return True
    
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
