import os
import numpy as np
import cv2 as cv

from sklearn.datasets import fetch_lfw_people
from skimage import data, color, feature, transform
from sklearn.feature_extraction.image import PatchExtractor
from sklearn.svm import LinearSVC


class MasterCamera:
    def __init__(self):
        self.videoSensor = cv.VideoCapture(0)
        self.classifier = LinearSVC(C=2.0)
        self.patchSize = 0

    def __del__(self):
        self.videoSensor.release()
        cv.destroyAllWindows()

    def prepareForWork(self):
        positive_patches, negative_patches = self.__loadSamples()
        hogFeatures = self.__combineIntoHog(positive_patches, negative_patches)
        X_train = np.array(hogFeatures)
        y_train = np.zeros(X_train.shape[0])
        y_train[:positive_patches.shape[0]] = 1
        self.trainSVM(X_train, y_train)

    def __loadSamples(self):
        # 1. positivie training samples
        face_images = fetch_lfw_people(resize=1.0)
        positive_patches = face_images.images
        self.patchSize = positive_patches[0].shape

        # 2. negative training samples
        negative_images = []
        imgs_path = '..//not_faces//'
        for file in os.listdir(imgs_path):
            img_object = cv.imread(imgs_path + str(file))
            negative_images.append(color.rgb2gray((img_object)))

        extracted_patches = []
        for img in negative_images:
            for img_scale in [1.0, 1.5, 2.0]:
                extracted_patches.append(self.__extractPatches(img, self.patchSize, 1000, img_scale))
        negative_patches = np.vstack(extracted_patches)
        return positive_patches, negative_patches


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

    def trainSVM(self, X_train, y_train):
        self.classifier.fit(X_train, y_train)

    def captureFrame(self):
        status, frame = self.videoSensor.read()
        if status:
            return frame
        return False

    def detectFaces(self, frame):
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # -- Detect faces
        indices, patches = zip(*self.__slidingWindow(frame_gray, self.patchSize))
        patches_hog = []
        for patch in patches:
            patches_hog.append(feature.hog(patch))
        labels = self.classifier.predict(patches_hog)
        faces = []
        labels_counter = 0
        for x, y in indices:
            if labels[labels_counter] == 1:
                faces.append((x, y, self.patchSize[0], self.patchSize[1]))
            labels_counter += 1
        return faces

    def __slidingWindow(self, img, patch_size, istep=5, jstep=5, scale=1.0):
        Ni, Nj = (int(scale * s) for s in patch_size)
        for i in range(0, img.shape[0] - Ni, istep):
            for j in range(0, img.shape[1] - Ni, jstep):
                patch = img[i:i + Ni, j:j + Nj]
                if scale != 1:
                    patch = transform.resize(patch, patch_size)
                yield (i, j), patch

    def anonymiseFaces(self, frame, faces):
        for (x, y, w, h) in faces:
            center = (x + w // 2, y + h // 2)
            frame = cv.ellipse(frame, center, (w // 2, h // 2), 0, 0, 360, (122, 122, 122), -1)
        return frame

    def displayFrame(self, frame):
        cv.imshow('Video frame - face detection', frame)


if __name__ == '__main__':
    camera = MasterCamera()
    camera.prepareForWork()

    while True:
        videoFrame = camera.captureFrame()
        detectedFaces = camera.detectFaces(videoFrame)
        videoFrame = camera.anonymiseFaces(videoFrame, detectedFaces)
        camera.displayFrame(videoFrame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
