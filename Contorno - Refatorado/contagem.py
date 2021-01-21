import cv2
import numpy as np
import matplotlib.pyplot as plt
import mahotas as mh

def ContornoSobel(img):
    sobelX = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    sobelY = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    sobelX = np.uint8(np.absolute(sobelX))
    sobelY = np.uint8(np.absolute(sobelY))
    sobel = cv2.bitwise_or(sobelX, sobelY)
    return np.vstack([np.hstack([img,    sobelX]),np.hstack([sobelY, sobel])])  

def Laplacian(img):
    lap = cv2.Laplacian(img, cv2.CV_32F)
    return np.uint8(np.absolute(lap))

def ContornoCanny(img):
    return cv2.Canny(img, 100, 250)

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

def Contar(img):
    ctns,_ = cv2.findContours(img, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    return ctns

path = "./bezerros/bezerra"

resultsLimiar = []
resultsLimiarAdaptativo = []
resultsLimiarOtsu = []

for i in range(1,7):
    b = cv2.cvtColor(cv2.imread(path + str(i) + ".jpeg"), cv2.COLOR_BGR2GRAY)
    suave = cv2.GaussianBlur(b, (7,7),0)
    resultsLimiar.append(LimiarNormal(suave))
    resultsLimiarAdaptativo.append(LimiarAdaptativo(suave))
    resultsLimiarOtsu.append(LimiarOtsu(suave))


for i in range(1,7):
    print('\n________________Resultados da '+str(i)+' imagem________________')
    print('\n   ~~~~~Limiarização Normal:')
    print('     |Quantidade de contornos encontrados utilizando Sobel: ' + str(len(Contar(ContornoSobel(resultsLimiar[i-1])))))
    print('     |Quantidade de contornos encontrados utilizando Laplace: ' + str(len(Contar(Laplacian(resultsLimiar[i-1])))))
    print('     |Quantidade de contornos encontrados utilizando Canny: ' + str(len(Contar(ContornoCanny(resultsLimiar[i-1])))))
    print('\n   ~~~~~Limiarização Adaptativo:')
    print('     |Quantidade de contornos encontrados utilizando Sobel: ' + str(len(Contar(ContornoSobel(resultsLimiarAdaptativo[i-1])))))
    print('     |Quantidade de contornos encontrados utilizando Laplace: ' + str(len(Contar(Laplacian(resultsLimiarAdaptativo[i-1])))))
    print('     |Quantidade de contornos encontrados utilizando Canny: ' + str(len(Contar(ContornoCanny(resultsLimiarAdaptativo[i-1])))))
    print('\n   ~~~~~Limiarização OTSU:')
    print('     |Quantidade de contornos encontrados utilizando Sobel: ' + str(len(Contar(ContornoSobel(resultsLimiarOtsu[i-1])))))
    print('     |Quantidade de contornos encontrados utilizando Laplace: ' + str(len(Contar(Laplacian(resultsLimiarOtsu[i-1])))))
    print('     |Quantidade de contornos encontrados utilizando Canny: ' + str(len(Contar(ContornoCanny(resultsLimiarOtsu[i-1]))))+'\n')