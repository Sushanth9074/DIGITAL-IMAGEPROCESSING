#!/usr/bin/env python
# coding: utf-8

# In[17]:


#HISTOGRAM EQUALIZATION WITHOUT FUNCTION
import numpy as np
import matplotlib.pyplot as plt
import cv2
image=cv2.imread("1-500x250-3.jpg",cv2.IMREAD_GRAYSCALE)
# cv2.imshow("o",image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
row,col=image.shape
histogram=np.zeros(256,dtype="int")
for i in range(row):
    for j in range(col):
        pixel_value=image[i,j]
        histogram[pixel_value]+=1
#determine the cdf
cdf=histogram.cumsum()
cdf_min=cdf.min()
#determine the equalized image
eq_img=((cdf[image]-cdf_min)/(row*col-cdf_min)*255).astype(np.uint8)
cv2.imshow("o",image)
cv2.imshow("h",eq_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

histogram_eq=np.zeros(256,dtype="int")
for i in range(row):
    for j in range(col):
        pixel_value=eq_img[i,j]
        histogram_eq[pixel_value]+=1

# Plot original and equalized histograms
plt.figure()
plt.subplot(2, 2, 1)
plt.bar(range(256), histogram, color='gray', width=1)
plt.title("Original Histogram")

plt.figure()
plt.subplot(2, 2, 2)
plt.bar(range(256), histogram_eq, color='gray', width=1)
plt.title("EQUALIZED")
        


# In[25]:


get_ipython().run_line_magic('pinfo', 'cv2.calcHist')


# In[61]:


#USING BUILT IN FUNCTION
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('IMG4.jpg',cv2.IMREAD_GRAYSCALE)
# Apply histogram equalization
equalized_image = cv2.equalizeHist(image)
# Calculate histograms
hist_original = cv2.calcHist(image, [0], None, [256], [0, 256])
hist_equalized = cv2.calcHist(equalized_image, [0], None, [256], [0, 256])

# Display the original and equalized images along with their histograms
plt.figure(figsize=(8,8))

# Original Image and Histogram
plt.subplot(2, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.subplot(2, 2, 2)
plt.plot(hist_original)
plt.title('Histogram of Original Image')

# Equalized Image and Histogram
plt.subplot(2, 2, 3)
plt.imshow(equalized_image,cmap="gray")
plt.title('Equalized Image')

plt.subplot(2, 2, 4)
plt.plot(hist_equalized)
plt.title("Histogram of Equalized Image")



# In[33]:


import cv2

# Load the image
image = cv2.imread("IMG4.jpg")

# Downsample the image
downsampled_image = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)

# Upsample the image
upsampled_image = cv2.resize(image, None, fx=3, fy=3, interpolation=cv2.INTER_LINEAR)

# Display the original, downsampled, and upsampled images using cv2
cv2.imshow('Original Image', image)
cv2.imshow('Downsampled Image', downsampled_image)
cv2.imshow('Upsampled Image', upsampled_image)
height1, width1, channels1 = image.shape
height2, width2, channels2 = downsampled_image.shape
height3, width3, channels3 = upsampled_image.shape
print("Orginal Image Resolution: Width = {}, Height = {}, Channels = {}".format(width1, height1, channels1))
print("downsampled_image Resolution: Width = {}, Height = {}, Channels = {}".format(width2, height2, channels2))
print("upsampled_image Resolution: Width = {}, Height = {}, Channels = {}".format(width3, height3, channels3))
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[13]:


#medianfilter
import numpy as np
import cv2
# Load the image
image = cv2.imread("IMG4.jpg", cv2.IMREAD_GRAYSCALE)
height, width = image.shape
filter_img = np.zeros((height, width), dtype=np.uint8)

# Set the kernel size
kernel_size = 4

# Calculate the padding required for the borders
padding = kernel_size // 2
print(padding)

for i in range(padding, height - padding):
    for j in range(padding, width - padding):
        # Extract the neighborhood
        neighborhood = image[i - padding:i + padding + 1, j - padding:j + padding + 1]
        median_value = np.median(neighborhood)
        filter_img[i, j] = int(median_value)

# Display the original and filtered images using cv2
cv2.imshow('Original Image', image)
cv2.imshow('Filtered Image', filter_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[21]:


#meanfilter
import numpy as np
import cv2
image=cv2.imread("IMG4.jpg",cv2.IMREAD_GRAYSCALE)
height,width=image.shape
filter_img=np.zeros((height,width),dtype=np.uint8)
kernel=3
padding=kernel//2
for i in range(padding,height-padding):
    for j in range(padding,width-padding):
        #nei
        nei=image[i-padding:i+padding+1,j-padding:j+padding+1]
        mean_val=np.mean(nei)
        filter_img[i,j]=mean_val
cv2.imshow("orginalimage",image)
cv2.imshow("filterimage",filter_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(nei)
print(median_val)


# In[64]:


#iterate through all image files and apply the transformation rotate and resize
import os
from PIL import Image
path="D:\Imageprocessingimg"
dir_list=os.listdir(path)
new_size = (20, 50)
formats=('.jpg','.png','JPEG')
imagefilelist=[file for file in dir_list if file.endswith(formats)]
for image in imagefilelist:
    image_path=os.path.join(path,image)
    image=Image.open(image_path)
    rot=image.rotate(45)
    resized_image = image.resize(new_size) 
    rot.show()
    resized_image.show()
    
print(imagefilelist)


# In[ ]:




