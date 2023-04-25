import cv2
from pyzbar.pyzbar import decode
from Arm_Lib import Arm_Device
import time as time_lib

class myScaner():
    def __init__(self, magicWord = '233'):
        self.magicWord = magicWord
        #self.detector = cv2.QRCodeDetector()
        sample = cv2.VideoCapture(0)
        ret, frame = sample.read()
        if ret:
            self.capWidth = int(sample.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.capHeight = int(sample.get(cv2.CAP_PROP_FRAME_HEIGHT))
            sample.release()
            print('Camera initialized!  Resolution: %d x %d' % (self.capWidth, self.capHeight))
        else:
            print('Camera error!')
            exit(0)
        self.arm_device = Arm_Device()
    
    def takePhoto(self):
        self.frames = []
        for i in range(5):
            cap = cv2.VideoCapture(i)
            ret, frame = cap.read()
            if ret:
                self.frames.append(frame)
                cap.release()
            else:
                print('Camera %d error!' % i)
                exit(0)

    def getQRCode3DPos(self):
        self.QRPos = []
        for frame in self.frames:
            result = decode(frame)
            if len(result) == 0:
                self.QRPos.append(None)
            else:
                for i in range(len(result)):
                    if result[i].data.decode('utf-8') == self.magicWord:
                        print('Found QR Code!')
                        theRect = result[i].rect
                        self.QRPos.append(((theRect.left + theRect.width / 2) / self.capWidth, (theRect.top + theRect.height / 2) / self.capHeight))
                        break
                    if i == len(result) - 1:
                        self.QRPos.append(None)

    def searchScanAngle(self):
        self.arm_device.Arm_serial_set_torque(1)
        time_lib.sleep(0.3)
        initAngle = self.arm_device.Arm_serial_servo_read(5)
        if initAngle == None:
            initAngle = 90
            self.arm_device.Arm_serial_servo_write(5, initAngle, 1500)
        print('initAngle %d' % initAngle)
        for i in range(initAngle, 180, 5):
            if i > initAngle:
                print('Turning to %d' % i)
                self.arm_device.Arm_serial_servo_write(5, i, 500)
            self.takePhoto()
            self.getQRCode3DPos()
            location = self.compute3DPos()
            if location != (0, 0, 0):
                print('Wonderful!')
                return location

        self.arm_device.Arm_serial_servo_write(5, initAngle, 500)

        for i in range(initAngle, 0, -5):
            if i < initAngle:
                print('Turning to %d' % i)
                self.arm_device.Arm_serial_servo_write(5, i, 500)
            self.takePhoto()
            self.getQRCode3DPos()
            location = self.compute3DPos()
            if location != (0, 0, 0):
                print('Wonderful!')
                return location

        return (0, 0, 0)
    
    def compute3DPos(self):
        x_cam_list = [1, 3, 4]
        y_cam_list = [0, 2, 4]
        z_cam_list = [0, 1, 2, 3]
        x, y, z = None, None, None
        for i in x_cam_list:
            if self.QRPos[i] != None:
                if i == 1 or i == 4:
                    x = self.QRPos[i][0] - 0.5
                else:
                    x = 0.5 - self.QRPos[i][0]
                break
        for i in y_cam_list:
            if self.QRPos[i] != None:
                if i == 2:
                    y = self.QRPos[i][0] - 0.5
                elif i == 4:
                    y = 0.5 - self.QRPos[i][1]
                else:
                    y = 0.5 - self.QRPos[i][0]
                break
        for i in z_cam_list:
            if self.QRPos[i] != None:
                z = 1.0 - self.QRPos[i][1]
                break
        
        if x == None or y == None or z == None:
            return (0, 0, 0)
        
        return(x, y, z)

if __name__ == '__main__':
    myScanerObj = myScaner()
    # myScanerObj.takePhoto()
    # myScanerObj.getQRCode3DPos()
    print(myScanerObj.searchScanAngle())
    #print(myScanerObj.QRPos)