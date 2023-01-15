import cv2
import numpy as np
import math

class myCV():
    def __init__(self, deviceA, deviceB, magicWordA, magicWordB):
        self.capA = cv2.VideoCapture(deviceA)
        self.capB = cv2.VideoCapture(deviceB)
        self.capWidth = int(self.capA.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.capHeight = int(self.capA.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.magicWordA = magicWordA
        self.magicWordB = magicWordB
        self.detector = cv2.QRCodeDetector()
    
    def getQRCode3DPos(self):
        # Get image from camera A
        ret, frameA = self.capA.read()
        if not ret:
            return False, None

        # Get image from camera B
        ret, frameB = self.capB.read()
        if not ret:
            return False, None

        # Detect QR code in image A
        ret, dataA, bboxA, straight_qrcodeA = self.detector.detectAndDecodeMulti(frameA)
        if bboxA is None:
            return False, None
        qrID_A = 0
        for i in range(len(dataA)):
            if dataA[i] == self.magicWordA:
                qrID_A = i
                break

        # Detect QR code in image B
        ret, dataB, bboxB, straight_qrcodeB = self.detector.detectAndDecodeMulti(frameB)
        if bboxB is None:
            return False, None
        qrID_B = 0
        for i in range(len(dataB)):
            if dataB[i] == self.magicWordB:
                qrID_B = i
                break

        # Calculate 3D position and return
        centerPosA = [(bboxA[qrID_A][0][0] + bboxA[qrID_A][2][0]) / 2, (bboxA[qrID_A][0][1] + bboxA[qrID_A][2][1]) / 2]
        centerPosB = [(bboxB[qrID_B][0][0] + bboxB[qrID_B][2][0]) / 2, (bboxB[qrID_B][0][1] + bboxB[qrID_B][2][1]) / 2]
        return True, [centerPosA[0] / self.capWidth, centerPosA[1] / self.capHeight, centerPosB[1] / self.capHeight]
