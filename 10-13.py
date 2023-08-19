#!/usr/bin/env python
# coding: utf-8

# In[10]:


#CANNY EDGE 
import cv2
# Load the image using cv2
image = cv2.imread('IMG2.jpg', cv2.IMREAD_GRAYSCALE)
# Apply Gaussian blur to reduce noise
blurred_img = cv2.GaussianBlur(image, (5, 5), 0)
# Apply Canny edge detection
edges = cv2.Canny(blurred_img, threshold1=50, threshold2=50)
# Display the original image and Canny edge detection result
cv2.imshow('Original Image', image)
cv2.imshow('Canny Edge Detection', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[23]:


#SOBEL EDGE DETECTION
import cv2
import matplotlib.pyplot as plt

#Read the original image
img = cv2.imread('IMG4.jpg') 

# Blur the image for better edge detection
img_blur = cv2.GaussianBlur(img,(3,3), 0)

# Sobel Edge Detection

sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)       # Sobel Edge Detection on the X axis
sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)       # Sobel Edge Detection on the Y axis
sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)      # Combined X and Y Sobel Edge Detection

# Display Sobel Edge Detection Images

plt.figure()
plt.subplot(2,2,1)
plt.imshow(sobelx),plt.title("sobel X")
plt.subplot(2,2,2)
plt.imshow(sobely),plt.title("sobel Y")
plt.subplot(2,2,3)
plt.imshow(sobelxy),plt.title("sobel X & Y")


# In[2]:


#PREWITT EDGE DETECTION
import cv2
import numpy as np
import matplotlib.pyplot as plt
image=cv2.imread("images.jpeg",cv2.IMREAD_GRAYSCALE)

gaus=cv2.GaussianBlur(image,(5,5),0)
#kernel
kernelx=np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
kernely=np.array([[-1,0,1],[-1,0,1],[-1,0,1]])

perwittx=cv2.filter2D(gaus,-1,kernelx)
prewitty=cv2.filter2D(gaus,-1,kernely)
cv2.imshow('Original Image', image)
cv2.imshow('Canny Edge Detection', perwittx)
cv2.imshow('Canny Edge tion', prewitty)
cv2.waitKey(0)
cv2.destroyAllWindows()
# #use matplotlib or cv2 for printing
# plt.figure()
# plt.subplot(2,2,1)
# plt.imshow(image)
# plt.title("orginalimage")
# plt.subplot(2,2,2)
# plt.imshow(perwittx)
# plt.title("perwitt-x")
# plt.subplot(2,2,3)
# plt.imshow(prewitty)


# In[6]:


import cv2

# Load the image using cv2
image = cv2.imread('IMG4.jpg', cv2.IMREAD_GRAYSCALE)

# Apply Laplacian edge detection
laplacian = cv2.Laplacian(image, cv2.CV_64F)

# Convert the result to 8-bit image
laplacian_8bit = cv2.convertScaleAbs(laplacian)

# Display the original image and Laplacian edge detection result
cv2.imshow('Original Image', image)
cv2.imshow('Laplacian Edge Detection', laplacian_8bit)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[3]:


#global thresholding
import cv2

# Read the image using cv2
image = cv2.imread('IMG4.jpg', cv2.IMREAD_GRAYSCALE)

# Apply global thresholding using OpenCV's threshold function
_, thresholded_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)


# Display the original and thresholded images using cv2
cv2.imshow('Original Image', image)
cv2.imshow('Thresholded Image', thresholded_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# image: The input grayscale image on which adaptive thresholding will be applied.
# 
# 255: The maximum pixel value used in the thresholding process. In this case, pixels exceeding the threshold will be set to 255 (white), and pixels below the threshold will be set to 0 (black).
# 
# cv2.ADAPTIVE_THRESH_GAUSSIAN_C: The adaptive thresholding method to use. In this case, the Gaussian-weighted mean is used to calculate the threshold value for each pixel's neighborhood.
# 
# cv2.THRESH_BINARY: The type of thresholding applied after comparing the pixel's value with the calculated threshold. In this case, it's a binary thresholding, meaning that pixels above the threshold become white and pixels below the threshold become black.
# 
# 11: The size of the neighborhood region (block size) used for adaptive thresholding. This determines the area around each pixel that is used to calculate the local threshold.
# 
# 2: The constant subtracted from the calculated mean threshold value. It helps fine-tune the thresholding.

# In[4]:


#local thresholding
import cv2

# Read the image using cv2
image = cv2.imread('IMG4.jpg', cv2.IMREAD_GRAYSCALE)

# Apply adaptive thresholding using OpenCV's adaptiveThreshold function
thresholded_image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, 11, 2)

# Save the thresholded image
cv2.imwrite('thresholded_image.jpg', thresholded_image)

# Display the original and thresholded images using cv2
cv2.imshow('Original Image', image)
cv2.imshow('Thresholded Image', thresholded_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

