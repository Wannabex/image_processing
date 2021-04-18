import cv2 as cv
from algorithms.vjcascades import FdVJCascades
from algorithms.rgbhcbcr import FdRGBHCbCr
from algorithms.hog import FdHog
from algorithms.ocsvm import FdOCSvm


class MasterCamera:
    def __init__(self, fdAlgorithm):
        self.faceDetector = fdAlgorithm
        self.videoSensor = cv.VideoCapture(0)

    def __del__(self):
        self.videoSensor.release()
        cv.destroyAllWindows()

    def getFdAlgorithm(self):
        return self.faceDetector

    def getFdAlgorithm(self, newFdAlgorithm):
        self.faceDetector = newFdAlgorithm

    def prepareAlgorithm(self):
        if self.faceDetector.prepareAlgorithm():
            return True
        print('Ran into some problems while preparing the algorithm')
        return False

    def captureFrame(self):
        status, frame = self.videoSensor.read()
        if status:
            return frame
        return False

    def detectFaces(self, frame):
        faces = self.faceDetector.detectFaces(frame)
        return faces

    def anonymiseFaces(self, frame, faces):
        for (x, y, w, h) in faces:
            center = (x + w // 2, y + h // 2)
            frame = cv.ellipse(frame, center, (w // 2, h // 2), 0, 0, 360, (122, 122, 122), -1)
        return frame

    def displayFrame(self, frame):
        cv.imshow('Video frame - face detection', frame)


if __name__ == '__main__':
    algorithms = [FdVJCascades(), FdRGBHCbCr(), FdHog(), FdOCSvm()]
    camera = MasterCamera(algorithms[3])
    if camera.prepareAlgorithm():
        while True:
            videoFrame = camera.captureFrame()
            detectedFaces = camera.detectFaces(videoFrame)
            videoFrame = camera.anonymiseFaces(videoFrame, detectedFaces)
            camera.displayFrame(videoFrame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

