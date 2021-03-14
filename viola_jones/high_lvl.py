import numpy as np
import cv2 as cv


class MasterCamera:
    def __init__(self):
        self.faceCascade = cv.CascadeClassifier()
        self.videoSensor = cv.VideoCapture(0)

    def __del__(self):
        self.videoSensor.release()
        cv.destroyAllWindows()

    def loadCascadeClassifier(self, cascadeFileName='haarcascade_frontalface_default.xml'):
        if self.faceCascade.load(cascadeFileName):
            return True
        return False

    def captureFrame(self):
        status, frame = self.videoSensor.read()
        if status:
            return frame
        return False

    def detectFaces(self, frame):
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame_gray = cv.equalizeHist(frame_gray)
        # -- Detect faces
        faces = self.faceCascade.detectMultiScale(frame_gray)
        return frame, faces

    def anonymiseFaces(self, frame, faces):
        for (x, y, w, h) in faces:
            center = (x + w // 2, y + h // 2)
            frame = cv.ellipse(frame, center, (w // 2, h // 2), 0, 0, 360, (122, 122, 122), -1)
        return frame

    def displayFrame(self, frame):
        cv.imshow('Video frame - face detection', frame)


if __name__ == '__main__':
    camera = MasterCamera()
    camera.loadCascadeClassifier('haarcascade_frontalface_alt.xml')

    while True:
        videoFrame = camera.captureFrame()
        videoFrame, detectedFaces = camera.detectFaces(videoFrame)
        videoFrame = camera.anonymiseFaces(videoFrame, detectedFaces)
        camera.displayFrame(videoFrame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

