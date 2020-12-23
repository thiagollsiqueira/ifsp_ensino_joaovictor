import cv2
import numpy as np
import matplotlib.pyplot as plt
import mahotas
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D 
from matplotlib.colors import hsv_to_rgb

img = cv2.imread('./bezerro/images/bezerro6.jpg')
plt.imshow(img)
plt.show()

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
suave = cv2.GaussianBlur(img, (7,7),0)

#limiar comum setado em 160
(T,bin) = cv2.threshold(suave,160,255,cv2.THRESH_BINARY)
(T,binI) = cv2.threshold(suave,160,255,cv2.THRESH_BINARY_INV)
binarizacaoComum = np.vstack([np.hstack([suave, bin]), np.hstack([binI, cv2.bitwise_and(img, img, mask = binI)])])
#plt.imshow(binarizacaoComum)
#plt.show()

#limiar adaptativo
    #Não funciona como o esperado visando o objetivo
bin1 = cv2.adaptiveThreshold(suave, 255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 5)
bin2 = cv2.adaptiveThreshold(suave, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5)
limiarAdaptativo = np.vstack([np.hstack([img, suave]),np.hstack([bin1, bin2])])
plt.imshow(limiarAdaptativo,cmap='gray')
plt.show()

#limitar otsu
T = mahotas.thresholding.otsu(suave)
temp = img.copy()
temp[temp > T] = 255
temp[temp < 255] = 0
temp = cv2.bitwise_not(temp)
T = mahotas.thresholding.rc(suave)
temp2 = img.copy()
temp2[temp2 > T] = 255
temp2[temp2 < 255] = 0
temp2 = cv2.bitwise_not(temp2)
binarizacaoOtsu = np.vstack([np.hstack([img, suave]),np.hstack([temp, temp2])]) 
#plt.imshow(binarizacaoOtsu)
#plt.show()


#contornado com sobel
sobelX = cv2.Sobel(binarizacaoOtsu, cv2.CV_64F, 1, 0)
sobelY = cv2.Sobel(binarizacaoOtsu, cv2.CV_64F, 0, 1)
sobelX = np.uint8(np.absolute(sobelX))
sobelY = np.uint8(np.absolute(sobelY))
sobel = cv2.bitwise_or(sobelX, sobelY)
contornoSobel = np.vstack([np.hstack([binarizacaoOtsu,    sobelX]),np.hstack([sobelY, sobel])]) 
#plt.imshow(contornoSobel, cmap='gray')
#plt.show() 

#contorno laplaciano
lap = cv2.Laplacian(binarizacaoOtsu, cv2.CV_64F)
lap = np.uint8(np.absolute(lap))
contornoLaplaciano = np.vstack([binarizacaoOtsu, lap])
#plt.imshow(contornoLaplaciano, cmap='gray')
#plt.show() 


#contorno Canny
canny1 = cv2.Canny(suave, 20, 120)
canny2 = cv2.Canny(suave, 70, 200)
contornoCanny = np.vstack([np.hstack([img,suave ]),np.hstack([canny1, canny2])]) 
#plt.imshow(contornoCanny, cmap='gray')
#plt.show() 
