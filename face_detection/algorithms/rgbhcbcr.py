import sys
sys.path.append('algorithms//')
sys.path.append('algorithms//rgbhcbcr_files//')
from algorithm import FDAlgorithm
from rgbhcbcr_files.face_detect import *

class FdRGBHCbCr(FDAlgorithm):
    def __init__(self):
        self.faceDetector = None

    def prepareAlgorithm(self):
        self.faceDetector = Face_Detector(Skin_Detect())
        if self.faceDetector is not None:
            return True
        return False

    def detectFaces(self, picture, minFaceSize=(70,100), maxFaceSize=(200, 300)):
        detectedFaces = self.faceDetector.Detect_Face_Img(picture, minFaceSize, maxFaceSize)
        return detectedFaces