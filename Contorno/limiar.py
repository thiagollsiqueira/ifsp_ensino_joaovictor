import cv2
import numpy as np
import matplotlib.pyplot as plt
import mahotas
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D 
from matplotlib.colors import hsv_to_rgb


def ContornoSobel(img):
    sobelX = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    sobelY = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    sobelX = np.uint8(np.absolute(sobelX))
    sobelY = np.uint8(np.absolute(sobelY))
    sobel = cv2.bitwise_or(sobelX, sobelY)
    contornoSobel = np.vstack([np.hstack([img,    sobelX]),np.hstack([sobelY, sobel])]) 
    return contornoSobel 


def Laplacian(img):
    lap = cv2.Laplacian(img, cv2.CV_64F)
    lap = np.uint8(np.absolute(lap))
    contornoLaplaciano = np.vstack([img, lap])
    return contornoLaplaciano

def ContornoCanny(img, suave):
    canny1 = cv2.Canny(suave, 20, 120)
    canny2 = cv2.Canny(suave, 70, 200)
    contornoCanny = np.vstack([np.hstack([img,suave]),np.hstack([canny1, canny2])]) 
    return contornoCanny

img = cv2.imread('./bezerro/images/bezerro6.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

suave = cv2.GaussianBlur(img, (7,7),0)

#Setando o intervalo de busca do threshold para encontrar tons de branco entre 240 e 255
limiar, imgLimiar = cv2.threshold(img,240,255,cv2.THRESH_BINARY)
#Setando o intervalo de busca do threshold para encontrar tons acima de cores escuras e invertendo o resultado
limiar2, imgLimiar2 = cv2.threshold(img,60,255,cv2.THRESH_BINARY_INV)

#limiar, imgLimiar = cv2.threshold(suave,240,255,cv2.THRESH_BINARY)
#limiar2, imgLimiar2 = cv2.threshold(suave,60,255,cv2.THRESH_BINARY_INV)


plt.imshow(imgLimiar, cmap='gray')
plt.show()

plt.imshow(imgLimiar2, cmap='gray')
plt.show()



limiares = cv2.addWeighted(imgLimiar,0.5,imgLimiar2,0.5,0)
plt.imshow(limiares, cmap='gray')
plt.show()

sobel = ContornoSobel(limiares)
plt.imshow(sobel, cmap='gray')
plt.show()

lap = Laplacian(limiares)
plt.imshow(lap, cmap='gray')
plt.show()

can = ContornoCanny(limiares, suave)
plt.imshow(can, cmap='gray')
plt.show()






