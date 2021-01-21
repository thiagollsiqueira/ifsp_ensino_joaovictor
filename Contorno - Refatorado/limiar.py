import cv2
import numpy as np
import matplotlib.pyplot as plt
import mahotas as mh

def LimiarAdaptativo(img):
    return cv2.adaptiveThreshold(img, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5)
     
def LimiarNormal(img):
    limiar, imgLimiar = cv2.threshold(img,240,255,cv2.THRESH_BINARY)
    limiar2, imgLimiar2 = cv2.threshold(img,60,255,cv2.THRESH_BINARY_INV)
    return cv2.addWeighted(imgLimiar,0.5,imgLimiar2,0.5,0)

def LimiarOtsu(img):
    T = mh.thresholding.otsu(img)
    temp = img.copy()
    temp[temp > T] = 255
    temp[temp < 255] = 0
    temp = cv2.bitwise_not(temp)
    return temp

path = "./bezerros/bezerra"

results = []

for i in range(1,7):
    b = cv2.cvtColor(cv2.imread(path + str(i) + ".jpeg"), cv2.COLOR_BGR2GRAY)
    results.append(cv2.GaussianBlur(b, (7,7),0))

i = 1
for img in results:
    fig, axs = plt.subplots(1,3,constrained_layout=True)
    fig.suptitle('Resultado da imagem ' + str(i), fontsize=16)
    axs[0].set_title('Limiar Normal')
    axs[0].imshow(LimiarNormal(img))
    axs[1].set_title('Limiar Adaptativo')
    axs[1].imshow(LimiarAdaptativo(img))
    axs[2].set_title('Limiar Otsu')
    axs[2].imshow(LimiarOtsu(img))
    plt.show()
    i+=1