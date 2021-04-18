from time import time
import logging
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import OneClassSVM
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import cv2 as cv
import numpy as np

if __name__ == "__main__":
    # Download the data, if not already on disk and load it as numpy arrays
    lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.5)

    # for machine learning we use the 2 data directly (as relative pixel
    # positions info is ignored by this model)
    X = lfw_people.data
    n_features = X.shape[1]

    # the label to predict is the id of the person
    y = lfw_people.target
    target_names = lfw_people.target_names
    n_classes = target_names.shape[0]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42)

    # #############################################################################
    # Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
    # dataset): unsupervised feature extraction / dimensionality reduction
    n_components = 150

    pca = PCA(n_components=n_components, svd_solver='randomized',
              whiten=True).fit(X_train)

    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)

    #############################################################################
    # Train a SVM classification model

    clfoc = OneClassSVM(kernel='poly', gamma='auto')
    clfoc.fit(X_train_pca)
    result = clfoc.predict(X_train_pca)
    print(result)