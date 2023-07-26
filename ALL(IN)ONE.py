#!/usr/bin/env python
# coding: utf-8

# Program to implement various blurring in image processing

# In[ ]:


import cv2
image=cv2.imread("IMG4.jpg")
gaus=cv2.GaussianBlur(image,(7,7),0)
median=cv2.medianBlur(image,5)
bila=cv2.bilateralFilter(image,9,75,75)
cv2.imshow("orginalimage",image)
cv2.imshow("gaus",gaus)
cv2.imshow("median",median)
cv2.imshow("bilateral",bila)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Program to implement various filtering technique using opencv
# 

# In[ ]:


import cv2
import numpy as np
rectangle=np.zeros((300,300),dtype="uint8")
cv2.rectangle(rectangle,(25,25),(275,275),255,-1)
cv2.imshow("Rectangle", rectangle)


circle=np.zeros((300,300),dtype="uint8")
cv2.circle(circle,(150,150),150,255,-1)
cv2.imshow("circle", circle)

bitwiseAnd = cv2.bitwise_and(rectangle, circle)
cv2.imshow("AND", bitwiseAnd)


bitwiseOR = cv2.bitwise_or(rectangle, circle)
cv2.imshow("or", bitwiseOR)


bitwiseXor = cv2.bitwise_xor(rectangle, circle)
cv2.imshow("XOR", bitwiseXor)


# Bitwise NOT: inverts the ON and OFF pixels in an image
bitwiseNot = cv2.bitwise_not(circle)
cv2.imshow("NOT", bitwiseNot)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Program to  perform matrix transformation in opencv python

# In[3]:


import numpy as np
import cv2
m=np.random.rand(3,3)
image=cv2.imread("IMG4.jpg")
cv2.imshow("the orginal image",image)
transformed=cv2.transform(image,m,None)
cv2.imshow("t",transformed)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Program to find the negative of an image

# In[8]:


import cv2
import numpy as np
image=cv2.imread("IMG4.jpg")
max_value=255
# the neagtiive of the image can be obtaine by subtracting the image from the max value
image_neg=255-image
cv2.imshow("the orginal image",image)
cv2.imshow("the negative image",image_neg)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Program to apply the Affine Transformation To an Image

# In[12]:


# import required libraries
import cv2
import numpy as np

# read the input image
img = cv2.imread('affine.jpg')

# access the image height and width
rows,cols,_ = img.shape

# define at three point on input image
pts1 = np.float32([[50,50],[200,50],[50,200]])
# define three points corresponding location to output image
pts2 = np.float32([[10,100],[200,50],[100,250]])

# get the affine transformation Matrix
M = cv2.getAffineTransform(pts1,pts2)

# apply affine transformation on the input image
dst = cv2.warpAffine(img,M,(cols,rows))
cv2.imshow("Affine Transform", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Program to implement box filter on an image

# In[14]:


# import required libraries
import cv2
import numpy as np

# read the input image
img = cv2.imread('affine.jpg')
img_box = cv2.boxFilter(img, -1, (50,50))
 
cv2.imshow("THEIMAGE", img_box)
cv2.imshow("Model ", img)
 
cv2.waitKey(0)
cv2.destroyAllWindows()


# Program to implement the Canny edge detectionusing opencv(USE THE BELOW CODE FOR GOOD DETECTION)

# In[17]:


import cv2

# Load the image in grayscale
# "  # Replace this with the path to your image file
img = cv2.imread('fish.jpg', cv2.IMREAD_GRAYSCALE)

# Apply Gaussian blur to reduce noise
blurred_img = cv2.GaussianBlur(img, (5, 5), 0)

# Perform Canny edge detection
low_threshold = 50
high_threshold = 150
edges = cv2.Canny(blurred_img, low_threshold, high_threshold)

# Display the original and edge-detected images
cv2.imshow("Original Image", img)
cv2.imshow("Canny Edge Detection", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Program to perfrom the Thresholding

# In[19]:


# Python program to illustrate
# simple thresholding type on an image
	
# organizing imports
import cv2
import numpy as np

# path to input image is specified and
# image is loaded with imread command
image1 = cv2.imread('IMG4.jpg')

# cv2.cvtColor is applied over the
# image input with applied parameters
# to convert the image in grayscale
img = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

# applying different thresholding
# techniques on the input image
# all pixels value above 120 will
# be set to 255
ret, thresh1 = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)
ret, thresh2 = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY_INV)
ret, thresh3 = cv2.threshold(img, 120, 255, cv2.THRESH_TRUNC)
ret, thresh4 = cv2.threshold(img, 120, 255, cv2.THRESH_TOZERO)
ret, thresh5 = cv2.threshold(img, 120, 255, cv2.THRESH_TOZERO_INV)

# the window showing output images
# with the corresponding thresholding
# techniques applied to the input images
cv2.imshow('Binary Threshold', thresh1)
cv2.imshow('Binary Threshold Inverted', thresh2)
cv2.imshow('Truncated Threshold', thresh3)
cv2.imshow('Set to 0', thresh4)
cv2.imshow('Set to 0 Inverted', thresh5)
	
# De-allocate any associated memory usage
if cv2.waitKey(0) & 0xff == 27:
	cv2.destroyAllWindows()


# Program to detect countour of an image

# In[27]:


import cv2
import numpy as np
  
# Let's load a simple image with 3 black squares
image = cv2.imread('fish.jpg')
cv2.waitKey(0)
  
# Grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  
# Find Canny edges
edged = cv2.Canny(gray, 30, 200)
cv2.waitKey(0)
  
# Finding Contours
# Use a copy of the image e.g. edged.copy()
# since findContours alters the image
contours, hierarchy = cv2.findContours(edged,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
  
cv2.imshow('Canny Edges After Contouring', edged)
cv2.waitKey(0)
# Draw all contours
# -1 signifies drawing all contours
cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
  
cv2.imshow('Contours', image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Program to draw different shapes on image
# 

# In[4]:


import cv2
import numpy as np

# Load the image
# Create a black image(use large image if you are not intrested to black image)
image = np.zeros((400, 400, 3), dtype=np.uint8)

# Draw a line
start_point = (50, 50)
end_point = (200, 200)

color = (0, 0, 255) # BGR color format
thickness = 2
cv2.line(image, start_point, end_point, color, thickness)

# Draw a rectangle
top_left = (200, 100)
bottom_right = (300, 300)
color = (0, 255, 0) # BGR color format
thickness = 3
cv2.rectangle(image, top_left, bottom_right, color, thickness)

# Draw a circle
center = (300, 300)
radius = 50
color = (255, 0, 0) # BGR color format
thickness = 2
cv2.circle(image, center, radius, color, thickness)

# Display the image
cv2.imshow("Image with Drawings", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

