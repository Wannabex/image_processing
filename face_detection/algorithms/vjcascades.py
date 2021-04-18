import sys
sys.path.append('algorithms//')
from algorithm import FDAlgorithm
import cv2 as cv


class FdVJCascades(FDAlgorithm):
    def __init__(self):
        self.faceDetector = None

    def prepareAlgorithm(self):
        self.faceDetector = cv.CascadeClassifier()
        if self.__loadCascadeClassifier():
            return True
        return False

    def __loadCascadeClassifier(self, cascadeFile='algorithms//vjcascades_files//haarcascade_frontalface_default.xml'):
        if self.faceDetector.load(cascadeFile):
            return True
        return False

    def detectFaces(self, picture):
        grayPicture = cv.cvtColor(picture, cv.COLOR_BGR2GRAY)
        grayPicture = cv.equalizeHist(grayPicture)
        # -- Detect faces
        detectedFaces = self.faceDetector.detectMultiScale(grayPicture)
        return detectedFaces