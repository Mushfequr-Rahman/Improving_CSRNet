# This code is based on the following article: https://www.hindawi.com/journals/jece/2017/2580860/
# Focus on this implementation is to do crowd counting without deep learning
# Technique used is counting by detection, regression, and density estimation

import cv2
import numpy as np
from sklearn.svm import SVR
from skimage.feature import local_binary_pattern # ULBP

from dsift import * # Dense SIFT
#from lbp import *  # ULBP

# Import a picture
img = cv2.imread("crowd.jpg")

# Block a picture
# Experiment with the threshold

# Loop all the training data here

# Approximate to count by doing different descriptors

# D-SIFT Feature Descriptor
# Apply SVR
#clf = SVR(C=1.0, epsilon=0.2)
#clf.fit(X, y)

# ULBP Feature Descriptor
# Apply SVR

# GIST Feature Descriptor
# Apply SVR

# Combine using voting then do density estimation

# MAE
# MRE