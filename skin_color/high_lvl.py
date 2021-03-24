import numpy as np
import cv2 as cv
from face_detect import *

class MasterCamera:
    def __init__(self):
        self.faceDetector = Face_Detector(Skin_Detect())
        self.videoSensor = cv.VideoCapture(0)

    def __del__(self):
        self.videoSensor.release()
        cv.destroyAllWindows()

    def captureFrame(self):
        status, frame = self.videoSensor.read()
        if status:
            return frame
        return False

    def detectFaces(self, frame, minFaceSize=(70,100), maxFaceSize=(200, 300)):
        faces = self.faceDetector.Detect_Face_Img(frame, minFaceSize, maxFaceSize)
        return frame, faces

    def anonymiseFaces(self, frame, faces):
        for i, r in enumerate(faces):
            x, y, w, h = r
            center = (x + w // 2, y + h // 2)
            frame = cv.ellipse(frame, center, (w // 2, h // 2), 0, 0, 360, (175, 175, 175), -1)
        return frame

    def displayFrame(self, frame):
        cv.imshow('Video frame - face detection', frame)


if __name__ == '__main__':
    camera = MasterCamera()

    while True:
        videoFrame = camera.captureFrame()
        videoFrame, detectedFaces = camera.detectFaces(videoFrame)
        videoFrame = camera.anonymiseFaces(videoFrame, detectedFaces)
        camera.displayFrame(videoFrame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

