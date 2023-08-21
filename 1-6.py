#!/usr/bin/env python
# coding: utf-8

# 1)READ,DISPLAY AND SAVE IMAGE USING VARIOUS LIB

# In[18]:


import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

#read image using cv2
image=cv2.imread("IMG4.jpg")
#read image using PIL
image1=Image.open("IMG2.jpg")
#read using matplotlib
image2=mpimg.imread("IMG3.jpg")

# grayscale conversion
gray_image1=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
gray_image2=image1.convert('L')
gray_image3=image2.mean(axis=2) 

#display the image
cv2.imshow("usingcv2",gray_image1)
cv2.waitKey(0)
cv2.destroyAllWindows()
#using PIL 
gray_image2.show()
#using matplotlib
plt.imshow(gray_image3,cmap="gray")

# "saving"
cv2.imwrite("gray_image1.jpg",gray_image1)
gray_image2.save("gray_image2.jpg")
plt.imsave("gray_scale3.jpg",gray_image3)


# DEVELOP A PROGRAM FOR 
# A)HEIGHT AND WIDTH OF IMAGE
# B)NUMBER OF CHANNELS IN IMAGE
# C)SEPERATE RGB CHANNEL

# In[14]:


import cv2
# Read the image
img = cv2.imread("rgbimage.png")

# Get the height, width, and number of channels of the image
height, width, channels = img.shape

# Separate the RGB channels
b, g, r = cv2.split(img)

# Display the image information
print("Height:", height)
print("Width:", width)
print("Number of channels:", channels)

# Display the RGB channels
plt.subplot(1,4,1)
plt.imshow(img)
plt.title("orginalimage")

plt.subplot(1,4,2)
plt.imshow(b,cmap="gray")
plt.title("Blue channel")

plt.subplot(1,4,3)
plt.imshow(g, cmap="gray")
plt.title("Green channel")

plt.subplot(1,4,4)
plt.imshow(r, cmap="gray")
plt.title("Red channel")
plt.show()


# 3)WRITE A PROGRAM TO RESIZE THE IMAGE AND ROTATE THE ORGINAL IMAGE AND CONVERT THE ORGINAL IMAGE INTO BINARY

# In[19]:


from PIL import Image
# Resize and rotate the image.
image = Image.open("IMG4.jpg")

resized_image = image.resize((200, 200))

rotated_image = image.rotate(90)

# Convert the image to binary.
binary_image = image.convert("1")

resized_image.show()
rotated_image.show()
binary_image.show()


# 4)PROGRAM TO DISPLAY IMAGE ATTRIBUTES

# In[21]:


from PIL import Image
import cv2


# Read the image using PIL.
image = Image.open("IMG4.jpg")

# Get the width and height of the image.
width, height = image.size

# Get the number of channels in the image.
channels = image.getbands()

# Get the image type.
image_type = image.mode

# Convert the image to a NumPy array using OpenCV.
image_array = cv2.imread("IMG4.jpg")

# Get the shape of the image array.
shape = image_array.shape

# Print the image attributes.
print("Image width:", width)
print("Image height:", height)
print("Number of channels:", channels)
print("Image type:", image_type)
print("Image shape:", shape)


# 5)program to perform arithematic and logical operation on image

# In[22]:


#logical
import cv2
import numpy as np
rectangle=np.zeros((300,300),dtype=np.uint8)
cv2.rectangle(rectangle,(25,25),(275,275),255,-1)
cv2.imshow("Rectangle", rectangle)
cv2.imwrite("Rectangle.jpg",rectangle)


circle=np.zeros((300,300),dtype="uint8")
cv2.circle(circle,(150,150),150,255,-1)
cv2.imshow("circle", circle)
cv2.imwrite("circle.jpg",circle)

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


# In[27]:


#arithematic operation on image
import cv2
image1=cv2.imread("add1.jpg")
image2=cv2.imread("add2.jpg")

addition=image1+image2
substraction=image1-image2
multiplication=image1*image2
division=image1/image2

cv2.imshow("orginal image1",image1)
cv2.imshow("orginal image2",image2)
cv2.imshow("addition",addition)
cv2.imshow("substraction",substraction)
cv2.imshow("multiplctn",multiplication)
cv2.imshow("division",division)
cv2.waitKey(0)
cv2.destroyAllWindows()


# 6)PROGRAM TO PERFORM
# 
# A)IMAGE SLICING
# B)BLENDING OF IMAGES BY USING MASK,FILTER AND BLUR FUNCTIONS
# C)CROPPING
# D)NEGATIVE OF IMAGE
# E)DRAWING ON IMAGE
# D)FINDING BASIC STATISTICS OF IMAGE

# In[52]:


from PIL import Image, ImageFilter, ImageChops, ImageDraw
import numpy as np

# A) Image Slicing

image = Image.open("IMG3.jpg")
region = (100, 100, 300, 300)
sliced_image = image.crop(region)
sliced_image.show()


# PLEASE DONT READ BLENDING CODE I TOO HAVE CONFUSION

# In[32]:


# # B) Blending of Images using Mask, Filter, and Blur
# image1_path = "image1.jpg"
# image2_path = "image2.jpg"
# mask_path = "mask.png"
# blended_image_path = "blended_image.jpg"

# image1 = Image.open("1-500x250-3.jpg")
# image2 = Image.open("2-500x250-2.jpg")
# mask = Image.open("2-500x250-2.jpg").convert("L")
# blended_image1 = ImageChops.composite(image2, image1, mask)
# blended_image1.show()
# blended_image2 = blended_image.filter(ImageFilter.GaussianBlur(radius=5))
# blended_image2.show()


# In[53]:


# C) Cropping
cropped_image_path = "cropped_image.jpg"
crop_left, crop_upper, crop_right, crop_lower = 100, 100, 300, 300 #make adjustment in lab

image = Image.open("IMG4.jpg")
cropped_image = image.crop((crop_left, crop_upper, crop_right, crop_lower))
cropped_image.show()


# In[58]:


# D) Negative of Image
import cv2

image = cv2.imread("IMG4.jpg")
negative =255-image
cv2.imshow("the negative image",negative)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[64]:


# E) Drawing on Image

image = Image.open("IMG4.jpg")
draw = ImageDraw.Draw(image)
draw.rectangle([50, 50, 100, 100], outline="red", width=2)
draw.text((100, 20), "Hello, Pillow!", fill="blue")
image.show()


# In[65]:


# Finding Basic Statistics of Image

image = Image.open("IMG4.jpg")
image_array = np.array(image)
mean = np.mean(image_array)
std_dev = np.std(image_array)
min_value = np.min(image_array)
max_value = np.max(image_array)

print("Mean:", mean)
print("Standard Deviation:", std_dev)
print("Min Value:", min_value)
print("Max Value:", max_value)

