import cv2
from pyzbar.pyzbar import decode

class myScaner():
    def __init__(self, magicWord = '233'):
        self.magicWord = magicWord
        self.detector = cv2.QRCodeDetector()
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

    def compute3DPos(self):
        return

if __name__ == '__main__':
    myScaner = myScaner()
    myScaner.takePhoto()
    myScaner.getQRCode3DPos()
    print(myScaner.QRPos)