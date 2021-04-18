import sys
sys.path.append('algorithms//')
from algorithm import FDAlgorithm
import cv2 as cv
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from skimage import transform

class FdOCSvm(FDAlgorithm):
    FRAME_ROWS = 480
    FRAME_COLS = 640
    def __init__(self):
        self.faceDetector = None
        self.pca = None
        self.patchSize = None


    def prepareAlgorithm(self):
        self.faceDetector = OneClassSVM(kernel='rbf', gamma='auto')
        self.pca = PCA(n_components=150, svd_solver='randomized', whiten=True)
        faceImages = fetch_lfw_people(resize=1.0)
        trainingFaces = faceImages.data
        self.patchSize = faceImages.images[0].shape
        eigenfaces = self.__performPCA(trainingFaces)
        self.__trainSvm(eigenfaces)
        return True

    def __performPCA(self, trainingData):
        # #############################################################################
        # Compute a PCA (eigenfaces) on the faces dataset (treated as unlabeled
        # dataset): unsupervised feature extraction / dimensionality reduction
        self.pca.fit(trainingData)

        # Project the input data to new basis of eigenvalues
        eigenData = self.pca.transform(trainingData)
        return eigenData

    def __trainSvm(self, trainingData):
        self.faceDetector.fit(trainingData)

    def detectFaces(self, picture):
        detectedFaces = []
        grayPicture = cv.cvtColor(picture, cv.COLOR_BGR2GRAY)
        indices, patches = zip(*self.__slidingWindow(grayPicture, self.patchSize, self.patchSize[0], self.patchSize[1]))
        detectionLabels = []
        for patch in patches:
            pcaPatch = self.pca.transform(np.ravel(patch).reshape(1, -1))
            detectionLabels.append(self.faceDetector.predict(pcaPatch))
        labelsCounter = 0
        for x, y in indices:
            if detectionLabels[labelsCounter] == 1:
                detectedFaces.append((x, y, self.patchSize[1], self.patchSize[0]))
            labelsCounter += 1
        return detectedFaces

    def __slidingWindow(self, img, patch_size, istep=5, jstep=5, scale=1.0):
        Ni, Nj = (int(scale * s) for s in patch_size)
        for i in range(0, img.shape[0] - Ni, istep):
            for j in range(0, img.shape[1] - Ni, jstep):
                patch = img[i:i + Ni, j:j + Nj]
                if scale != 1:
                    patch = transform.resize(patch, patch_size)
                yield (i, j), patch