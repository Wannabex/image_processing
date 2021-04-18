import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
from algorithms.vjcascades import FdVJCascades
from algorithms.rgbhcbcr import FdRGBHCbCr
from algorithms.hog import FdHog
from algorithms.ocsvm import FdOCSvm

VIOLA_JONES = 0
SKIN_COLOR = 1
GRADIENTS_HISTOGRAM = 2
OC_SVM = 3
ALGORITHM_NAMES = ['VIOLA_JONES', 'SKIN_COLOR', 'GRADIENTS_HISTOGRAM', 'OC-SVM']


if __name__ == '__main__':
    algorithms = [FdVJCascades(), FdRGBHCbCr()]
    #algorithms = [FdVJCascades(), FdRGBHCbCr(), FdHog(), FdOCSvm()]
    testDataPath = 'test_data/'
    testDataSize = len(os.listdir(testDataPath))
    noOfAlgorithms = len(algorithms)
    testResults = np.zeros(shape=(noOfAlgorithms, testDataSize), dtype=bool)
    print(testResults)

    algorithmCounter = 0
    for algorithm in algorithms:
        truePositives = 5
        falsePositives = 2
        falseNegatives = 9
        precision = 0.0
        recall = 0.0
        testX = np.arange(0.0, 2.0, 0.01)
        testY = 1 + np.sin((algorithmCounter + 1) * np.pi * testX)

        if algorithm.prepareAlgorithm():
            for testImg in os.listdir(testDataPath):
                imgData = matplotlib.image.imread(testDataPath + str(testImg))
                #detectedFaces = algorithm.detectFaces(imgData)
                #compare detectedFaces with ground truth data
            precision = truePositives / (truePositives + falsePositives)
            print(precision)
            recall = truePositives / (truePositives + falseNegatives)
            print(recall)

            fig, ax = plt.subplots()
            ax.plot(testX, testY)
            ax.set(xlabel='my x (N)', ylabel='my y (%)', title=ALGORITHM_NAMES[algorithmCounter])
            ax.grid()
        algorithmCounter += 1
    plt.show()






