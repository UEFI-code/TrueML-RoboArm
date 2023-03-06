import math
class theVirtualArm():
    def __init__(self) -> None:
        self.servoAngles = [0.0] * 4
        self.armLength = [1.0] * 4
        self.cos = lambda x: math.cos(x * 3.14 / 180.0)
        self.sin = lambda x: math.sin(x * 3.14 / 180.0)
    
    def calc3DPos(self):
        # Look at the first servo
        lastX = 0.0
        lastY = 0.0
        lastZ = self.armLength[0]
        lastAlpha = 0.0
        lastTheta = self.servoAngles[0]

        # Look at the second servo
        lastAlpha += self.servoAngles[1]
        lastTheta += 0
        deltaX = self.armLength[1] * self.cos(lastTheta) * self.cos(lastAlpha)
        deltaY = self.armLength[1] * self.sin(lastTheta) * self.cos(lastAlpha)
        deltaZ = self.armLength[1] * self.sin(lastAlpha)
        lastX += deltaX
        lastY += deltaY
        lastZ += deltaZ

        # Look at the third servo
        lastAlpha += self.servoAngles[2]
        lastTheta += 0
        deltaX = self.armLength[2] * self.cos(lastTheta) * self.cos(lastAlpha)
        deltaY = self.armLength[2] * self.sin(lastTheta) * self.cos(lastAlpha)
        deltaZ = self.armLength[2] * self.sin(lastAlpha)
        lastX += deltaX
        lastY += deltaY
        lastZ += deltaZ

        # Look at the fourth servo
        lastAlpha += self.servoAngles[3]
        lastTheta += 0
        deltaX = self.armLength[3] * self.cos(lastTheta) * self.cos(lastAlpha)
        deltaY = self.armLength[3] * self.sin(lastTheta) * self.cos(lastAlpha)
        deltaZ = self.armLength[3] * self.sin(lastAlpha)
        lastX += deltaX
        lastY += deltaY
        lastZ += deltaZ

        return (lastX, lastY, lastZ)

if __name__ == "__main__":
    myobj = theVirtualArm()
    myobj.servoAngles = [90.0, 0.0, 0.0, 0.0]
    print(myobj.calc3DPos())
