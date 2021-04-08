import cv2 as cv
import os
import numpy as np
from sklearn.datasets import fetch_lfw_people
from skimage import data, transform, color, feature
from sklearn.feature_extraction.image import PatchExtractor
from sklearn.svm import LinearSVC


# 1. positivie training samples
faces = fetch_lfw_people().images
positive_patches = faces

negative_images = []
# 2. negative training samples
imgs_to_use = ['camera', 'moon', 'logo', 'text', 'coins', 'moon',
               'page', 'clock', 'immunohistochemistry', 'rocket',
               'brick', 'horse', 'chelsea', 'coffee', 'hubble_deep_field']

for name in imgs_to_use:
    negative_images.append(color.rgb2gray((getattr(data, name)())))

imgs_path = '..//not_faces//'
for file in os.listdir(imgs_path):
    img_object = cv.imread(imgs_path + str(file))
    negative_images.append(color.rgb2gray((img_object)))


def extract_patches(img, max_patches, scale=1.0, patch_size=positive_patches[0].shape):
    extracted_patch_size = tuple((scale * np.array(patch_size)).astype(int))
    extractor = PatchExtractor(patch_size=extracted_patch_size,
                               max_patches=max_patches, random_state=0)
    patches = extractor.transform(img[np.newaxis])
    if scale != 1.0:
        patches = np.array([transform.resize(patch, patch_size)
                            for patch in patches])
    return patches


extracted_patches = []
for im in negative_images:
    for img_scale in [0.5, 1.0, 1.5, 2.0]:
        extracted_patches.append(extract_patches(im, 1000, img_scale))
negative_patches = np.vstack(extracted_patches)

# 3. combine sets and extract hog features
hog_features = []
for im in positive_patches:
    hog_features.append(feature.hog(im))
for im in negative_patches:
    hog_features.append(feature.hog(im))
X_train = np.array(hog_features)
y_train = np.zeros(X_train.shape[0])
y_train[:positive_patches.shape[0]] = 1

# 4. training a support vector machine
# cross_val_score(GaussianNB(), X_train, y_train)
# cross validation evaluates estimator performance. Here GaussianNB performs fit

# SVC_parameters = {'C': [1.0, 2.0, 4.0, 8.0]}
# grid = GridSearchCV(LinearSVC(), SVC_parameters)
# grid searches over specified parameters for linear support vector classification. C is regularization
# grid.fit(X_train, y_train)
# model = grid.best_estimator_ # C=4.0 is the best

model = LinearSVC(C=1.0)
# model = SVC(C=4.0)
model.fit(X_train, y_train)
print("Finished SVM classifier training")


# 5. Find faces in a new image
def sliding_window(img, patch_size=positive_patches[0].shape,
                   istep=2, jstep=2, scale=1.0):
    Ni, Nj = (int(scale * s) for s in patch_size)
    for i in range(0, img.shape[0] - Ni, istep):
        for j in range(0, img.shape[1] - Ni, jstep):
            patch = img[i:i + Ni, j:j + Nj]
            if scale != 1:
                patch = transform.resize(patch, patch_size)
            yield (i, j), patch


# videoSensor = cv.VideoCapture(0)
# status, frame = videoSensor.read()
# videoSensor.release()
# frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
# frame_gray = transform.rescale(frame, 0.5)
test_image = data.astronaut()
test_image = color.rgb2gray(test_image)
test_image = transform.rescale(test_image, 0.5)
# test_image = test_image[:160, 40:180]

# indices, patches = zip(*sliding_window(test_image))
indices, patches = zip(*sliding_window(test_image))
patches_hog = []
for patch in patches:
    patches_hog.append(feature.hog(patch))
labels = model.predict(patches_hog)
labels.sum()
Ni, Nj = positive_patches[0].shape
indices = np.array(indices)

for i, j in indices[labels == 1]:
    x, y, w, h = j, i, Nj, Ni
    center = (x + w // 2, y + h // 2)
    test_image = cv.ellipse(test_image, center, (w // 2, h // 2), 0, 0, 360, (175, 175, 175), -1)
    # frame = cv.ellipse(frame, center, (w // 2, h // 2), 0, 0, 360, (175, 175, 175), -1)

cv.imshow('Video frame - face detection', test_image)
cv.waitKey(0)
cv.destroyAllWindows()




