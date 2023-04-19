import cv2
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
            ret, data, bbox, straight_qrcode = self.detector.detectAndDecodeMulti(frame)
            print(data)
            if bbox is None:
                self.QRPos.append(None)
            else:
                for i in range(len(data)):
                    if data[i] == self.magicWord:
                        print('QR code found!')
                        self.QRPos.append([(bbox[i][0][0] + bbox[i][2][0]) / (2 * self.capWidth), (bbox[i][0][1] + bbox[i][2][1]) / (2 * self.capHeight)])
                        break
                    if i == len(data) - 1:
                        self.QRPos.append(None)
    
    def compute3DPos(self):
        return

if __name__ == '__main__':
    myScaner = myScaner()
    myScaner.takePhoto()
    myScaner.getQRCode3DPos()
    print(myScaner.QRPos)