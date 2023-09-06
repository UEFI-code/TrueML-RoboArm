import math
import cv2
import numpy as np

class theVirtualArm():
    def __init__(self) -> None:
        self.servoAngles = [0.0] * 4
        self.armLength = [1.0] * 4
        # Add Rotation Matrix and Zoom Matrix
        self.camTrans1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.camTrans2 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.cos = lambda x: math.cos(x * 3.14 / 180.0)
        self.sin = lambda x: math.sin(x * 3.14 / 180.0)
        self.drawPoints = [(0,0,0)] * 5
    
    def calc3DPos(self):
        # Look at the first servo
        self.drawPoints[0] = (0,0,0)
        lastX = 0.0
        lastY = 0.0
        lastZ = self.armLength[0]
        lastAlpha = 0.0
        lastTheta = self.servoAngles[0]

        # Look at the second servo
        self.drawPoints[1] = (lastX, lastY, lastZ)
        lastAlpha += self.servoAngles[1]
        lastTheta += 0
        deltaX = self.armLength[1] * self.cos(lastTheta) * self.cos(lastAlpha)
        deltaY = self.armLength[1] * self.sin(lastTheta) * self.cos(lastAlpha)
        deltaZ = self.armLength[1] * self.sin(lastAlpha)
        lastX += deltaX
        lastY += deltaY
        lastZ += deltaZ

        # Look at the third servo
        self.drawPoints[2] = (lastX, lastY, lastZ)
        lastAlpha += self.servoAngles[2]
        lastTheta += 0
        deltaX = self.armLength[2] * self.cos(lastTheta) * self.cos(lastAlpha)
        deltaY = self.armLength[2] * self.sin(lastTheta) * self.cos(lastAlpha)
        deltaZ = self.armLength[2] * self.sin(lastAlpha)
        lastX += deltaX
        lastY += deltaY
        lastZ += deltaZ

        # Look at the fourth servo
        self.drawPoints[3] = (lastX, lastY, lastZ)
        lastAlpha += self.servoAngles[3]
        lastTheta += 0
        deltaX = self.armLength[3] * self.cos(lastTheta) * self.cos(lastAlpha)
        deltaY = self.armLength[3] * self.sin(lastTheta) * self.cos(lastAlpha)
        deltaZ = self.armLength[3] * self.sin(lastAlpha)
        lastX += deltaX
        lastY += deltaY
        lastZ += deltaZ

        lastPos1 = np.array([[lastX], [lastY], [1]])
        lastPos1 = np.dot(self.camTrans1, lastPos1)
        lastX = lastPos1[0][0]
        lastY1 = lastPos1[1][0]
        lastPos2 = np.array([[lastY], [lastZ], [1]])
        lastPos2 = np.dot(self.camTrans2, lastPos2)
        lastY2 = lastPos2[0][0]
        lastZ = lastPos2[1][0]
        lastY = (lastY1 + lastY2) / 2
        # Look at the end effector
        self.drawPoints[4] = (lastX, lastY, lastZ)

        return (lastX, lastY, lastZ)
    
    def draw(self):
        imgXZ = np.zeros((1024, 1024, 3), np.uint8)
        imgYZ = np.zeros((1024, 1024, 3), np.uint8)
        xzPixels = []
        yzPixels = []
        for i in self.drawPoints:
            xzPixels.append((int(i[0] * 100 + 512), int(512 - i[2] * 100)))
            yzPixels.append((int(i[1] * 100 + 512), int(512 - i[2] * 100)))
        cv2.polylines(imgXZ, [np.array(xzPixels)], False, (255, 255, 255), 2)
        cv2.polylines(imgYZ, [np.array(yzPixels)], False, (255, 255, 255), 2)
        for i, j in zip(xzPixels, yzPixels):
            cv2.circle(imgXZ, i, 15, (0, 255, 0), -1)
            cv2.circle(imgYZ, j, 15, (0, 255, 0), -1)
        cv2.imshow("XZ", imgXZ)
        cv2.imshow("YZ", imgYZ)
        cv2.waitKey(0)
            
if __name__ == "__main__":
    myobj = theVirtualArm()
    myobj.servoAngles = [00.0, 10.0, 45.0, 30.0]
    print(myobj.calc3DPos())
    myobj.draw()
