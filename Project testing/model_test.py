import cv2
import pickle
import numpy as np

path = "D:/CMP/third_Year/first_Semester/imageProcessing and computerVision/Project/data set/"
# Load kmeans model
filename1 = 'kmeans_model.sav'
k_means = pickle.load(open(filename1, 'rb'))
# Load SVM model
filename2 = 'gestures_model.sav'
clf = pickle.load(open(filename2, 'rb'))
# Read image
img = cv2.imread(path + '2018.jpg')

hsvim = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# Lower boundary of skin color in HSV
lower = np.array([0, 48, 80], dtype="uint8")
# Upper boundary of skin color in HSV
upper = np.array([20, 255, 255], dtype="uint8")
skinMask = cv2.inRange(hsvim, lower, upper)

# Gaussian filter (blur) to remove noise
skinMask = cv2.GaussianBlur(skinMask, (17, 17), 0)

# get thresholded image
# ret, thresh1 = cv2.threshold(
# skinMask, 100, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
thresh1 = cv2.adaptiveThreshold(
    skinMask, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 355, 5)
# Feature extraction
sift = cv2.SIFT_create()
kp, descriptor = sift.detectAndCompute(thresh1, None)
# Produce "bag of words" vector
descriptor = k_means.predict(descriptor)
n_clusters = 1600
vq = [0] * n_clusters
for feature in descriptor:
    vq[feature] = vq[feature] + 1  # load the model from disk
# Predict the result
result = clf.predict([vq])
print("the result is ", result[0])
