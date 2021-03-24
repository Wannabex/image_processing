import numpy as np
import cv2 as cv
from sklearn.svm import OneClassSVM
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA

class MasterCamera:
    CAMERA_FRAME_ROWS = 480
    CAMERA_FRAME_COLS = 640

    def __init__(self, kernel='rbf', gamma='auto', pca_components=20):
        self.videoSensor = cv.VideoCapture(0)

        self.lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=1.0)
        self.trained_kernel_rows = self.lfw_people.images.shape[1]
        self.trained_kernel_cols = self.lfw_people.images.shape[2]
        self.training_faces = self.lfw_people.data

        self.pca = PCA(n_components=pca_components, svd_solver='randomized', whiten=True)
        self.classifier = OneClassSVM(kernel=kernel, gamma=gamma)

    def __del__(self):
        self.videoSensor.release()
        cv.destroyAllWindows()

    def perform_PCA(self):
        # #############################################################################
        # Compute a PCA (eigenfaces) on the faces dataset (treated as unlabeled
        # dataset): unsupervised feature extraction / dimensionality reduction

        self.pca.fit(self.training_faces)

        # Project the input data to new basis of eigenvalues
        self.training_faces = self.pca.transform(self.training_faces)

    def train_OCSVM(self):
        self.classifier.fit(self.training_faces)

    def captureFrame(self):
        status, frame = self.videoSensor.read()
        if status:
            return frame
        return False

    def detectFaces(self, frame):
        faces = []
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame_gray = cv.equalizeHist(frame_gray)

        for current_row in range(0, self.CAMERA_FRAME_ROWS - self.trained_kernel_rows + 1, self.trained_kernel_rows):
            for current_col in range(0, self.CAMERA_FRAME_COLS - self.trained_kernel_cols + 1, self.trained_kernel_cols):
                pca_kernel = np.ndarray(shape=(1, self.trained_kernel_rows * self.trained_kernel_cols))
                # print(f'r {current_row} - {current_row+trained_kernel_rows} and c {current_col} - {current_col+trained_kernel_cols}')
                pca_kernel[0] = np.ravel(frame_gray[current_row:current_row + self.trained_kernel_rows,
                                    current_col:current_col + self.trained_kernel_cols])
                pca_kernel = self.pca.transform(pca_kernel)
                result = self.classifier.predict(pca_kernel)
                if result == 1:
                    faces.append((current_row, current_col, self.trained_kernel_rows, self.trained_kernel_cols)) #  x, y, w, h
        return faces

    def anonymiseFaces(self, frame, faces):
        for (x, y, w, h) in faces:
            center = (x + w // 2, y + h // 2)
            frame = cv.ellipse(frame, center, (w // 2, h // 2), 0, 0, 360, (122, 122, 122), -1)
        return frame

    def displayFrame(self, frame):
        cv.imshow('Video frame - face detection', frame)


if __name__ == '__main__':
    camera = MasterCamera()
    camera.perform_PCA()
    camera.train_OCSVM()


    while True:
        videoFrame = camera.captureFrame()
        detectedFaces = camera.detectFaces(videoFrame)
        videoFrame = camera.anonymiseFaces(videoFrame, detectedFaces)
        camera.displayFrame(videoFrame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

