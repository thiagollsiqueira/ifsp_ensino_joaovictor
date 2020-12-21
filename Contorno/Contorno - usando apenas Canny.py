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

cinza = cv2.bilateralFilter(cinza,11,17,17)
borda = cv2.Canny(cinza,30,200)
plt.imshow(borda,cmap="gray")
plt.show()
