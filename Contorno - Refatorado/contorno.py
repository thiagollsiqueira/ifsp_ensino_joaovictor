import cv2
import numpy as np
import matplotlib.pyplot as plt

def ContornoSobel(img):
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
    return cv2.addWeighted(sobelx, 0.5, sobely, 0.5,0) 

def Laplacian(img):
    lap = cv2.Laplacian(img, cv2.CV_32F)
    return np.uint8(np.absolute(lap))

def ContornoCanny(img):
    return cv2.Canny(img, 100, 250)

path = "./bezerros/bezerra"

results = []

for i in range(1,7):
    b = cv2.cvtColor(cv2.imread(path + str(i) + ".jpeg"), cv2.COLOR_BGR2GRAY)
    results.append(cv2.GaussianBlur(b, (7,7),0))

i = 1
for img in results:
    fig, axs = plt.subplots(1,3,constrained_layout=True)
    fig.suptitle('Resultado da imagem ' + str(i), fontsize=16)
    axs[0].set_title('Contorno Sobel')
    axs[0].imshow(ContornoSobel(img))
    axs[1].set_title('Contorno Laplaciano')
    axs[1].imshow(Laplacian(img))
    axs[2].set_title('Contorno com Canny')
    axs[2].imshow(ContornoCanny(img))
    plt.show()
    i+=1