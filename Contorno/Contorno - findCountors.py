import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D 
from matplotlib.colors import hsv_to_rgb


#img = cv2.imread('./bezerro/images/bezerra6.jpg')

img = cv2.imread('./bezerro/images/bezerro6.jpg')
plt.imshow(img)
plt.show()
cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(cinza)
plt.show()
plt.imshow(cinza,cmap="gray")
plt.show()

ret, thresh = cv2.threshold(cinza,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
kernel = np.ones((5,5),np.uint8) 
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
kernel = np.ones((6,6),np.uint8) 
dilate = cv2.dilate(opening,kernel,iterations=1)
blur = cv2.blur(dilate,(15,15))
ret, thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) 
contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE) 

cnt = max(contours, key=cv2.contourArea) 

h, w = img.shape[:2] 
mask = np.zeros((h, w), np.uint8) 

cv2.drawContours(mask, [cnt],-1, 255, -1) 
res = cv2.bitwise_and(img, img, mask=mask) 
plt.imshow(res)
plt.show()
