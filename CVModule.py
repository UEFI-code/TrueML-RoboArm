import cv2
import numpy as np
import math

class myCV():
    def __init__(self, deviceA, deviceB, magicWordA, magicWordB):
        self.capA = cv2.VideoCapture(deviceA)
        self.capB = cv2.VideoCapture(deviceB)
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
        dataA, bboxA, straight_qrcodeA = self.detector.detectAndDecode(frameA)
        if bboxA is None:
            return False, None
        qrID_A = 0

        # Detect QR code in image B
        dataB, bboxB, straight_qrcodeB = self.detector.detectAndDecode(frameB)
        if bboxB is None:
            return False, None
        qrID_B = 0

        # Calculate 3D position and return
        centerPosA = [(bboxA[qrID_A][0][0] + bboxA[qrID_A][2][0]) / 2, (bboxA[qrID_A][0][1] + bboxA[qrID_A][2][1]) / 2]
        centerPosB = [(bboxB[qrID_B][0][0] + bboxB[qrID_B][2][0]) / 2, (bboxB[qrID_B][0][1] + bboxB[qrID_B][2][1]) / 2]
        return True, [centerPosA[0], centerPosA[1], centerPosB[2]]
