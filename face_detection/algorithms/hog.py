import sys
sys.path.append('algorithms//')
from algorithm import FDAlgorithm
import os
import numpy as np
import cv2 as cv
from sklearn.datasets import fetch_lfw_people
from skimage import color, feature, transform
from sklearn.feature_extraction.image import PatchExtractor
from sklearn.svm import LinearSVC

class FdHog(FDAlgorithm):
    def __init__(self):
        self.faceDetector = None
        self.patchSize = None

    def prepareAlgorithm(self):
        self.faceDetector = LinearSVC(C=2.0, max_iter=10000)
        positivePatches, negativePatches = self.__loadSamples()
        hogFeatures = self.__combineIntoHog(positivePatches, negativePatches)
        X_train = np.array(hogFeatures)
        y_train = np.zeros(X_train.shape[0])
        y_train[:positivePatches.shape[0]] = 1
        self.__trainSVM(X_train, y_train)
        return True

    def __loadSamples(self):
        # 1. positivie training samples
        faceImages = fetch_lfw_people(resize=1.0)
        positivePatches = faceImages.images
        self.patchSize = positivePatches[0].shape

        # 2. negative training samples
        negativeImages = []
        imgsPath = 'not_faces//'
        for file in os.listdir(imgsPath):
            imgObject = cv.imread(imgsPath + str(file))
            negativeImages.append(color.rgb2gray((imgObject)))

        extractedPatches = []
        for img in negativeImages:
            for img_scale in [1.0, 1.5, 2.0]:
                extractedPatches.append(self.__extractPatches(img, self.patchSize, 1000, img_scale))
        negativePatches = np.vstack(extractedPatches)
        return positivePatches, negativePatches

    def __extractPatches(self, img, patch_size, max_patches=1000, scale=1.0):
        extracted_patch_size = tuple((scale * np.array(patch_size)).astype(int))
        extractor = PatchExtractor(patch_size=extracted_patch_size,
                                   max_patches=max_patches, random_state=0)
        patches = extractor.transform(img[np.newaxis])
        if scale != 1.0:
            patches = np.array([transform.resize(patch, patch_size)
                                for patch in patches])
        return patches

    def __combineIntoHog(self, pos_patches, neg_patches):
        hog_features = []
        for patch in pos_patches:
            hog_features.append(feature.hog(patch))
        for patch in neg_patches:
            hog_features.append(feature.hog(patch))
        return hog_features

    def __trainSVM(self, X_train, y_train):
        self.faceDetector.fit(X_train, y_train)

    def detectFaces(self, picture):
        grayPicture = cv.cvtColor(picture, cv.COLOR_BGR2GRAY)
        indices, patches = zip(*self.__slidingWindow(grayPicture, self.patchSize, self.patchSize[0], self.patchSize[1]))
        patchesHog = []
        for patch in patches:
            patchesHog.append(feature.hog(patch))
        labels = self.faceDetector.predict(patchesHog)
        detectedFaces = []
        labelsCounter = 0
        for x, y in indices:
            if labels[labelsCounter] == 1:
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