import numpy as np
import cv2 as cv
from sklearn.preprocessing import KBinsDiscretizer
import matplotlib.pyplot as plt
# Read image
img = cv.imread('hand.jpg')
# Grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# Binary
et, thresh1 = cv.threshold(gray, 127, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
# Initialize sift
sift = cv.SIFT_create()
# Keypoints, descriptors
kp, descriptor = sift.detectAndCompute(gray, None)
# Each keypoint has a descriptor with length 128
print(len(descriptor[0]))
n_bins = 8
encoder = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="kmeans")
compressed_descriptor_kmeans = encoder.fit_transform(descriptor.reshape(-1, 1)).reshape(
    descriptor.shape
)


bin_edges = encoder.bin_edges_[0]
bin_center = bin_edges[:-1] + (bin_edges[1:] - bin_edges[:-1]) / 2

_, ax = plt.subplots()
ax[0].hist(descriptor.ravel(), bins=256)
cv.waitKey(0)
color = "tab:orange"
for center in bin_center:
    ax.axvline(center, color=color)
    ax.text(center - 10, ax.get_ybound()
            [1] + 100, f"{center:.1f}", color=color)

# cv.imshow('sift_keypoints.jpg', img)
# img = cv.drawKeypoints(thresh1, kp, img)
# cv.imshow('sift_keypoints.jpg', img)
