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
