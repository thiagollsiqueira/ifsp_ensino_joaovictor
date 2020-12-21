import cv2
import numpy as np
import matplotlib.pyplot as plt
import mahotas
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D 
from matplotlib.colors import hsv_to_rgb


#imgColorida = cv2.imread('./bezerro/images/bezerra6.jpg')
imgColorida = cv2.imread('./bezerro/images/bezerro6.jpg')
plt.imshow(imgColorida)
plt.show()
#Convertendo para tons de cinza
imgCinza = cv2.cvtColor(imgColorida, cv2.COLOR_BGR2GRAY)
plt.imshow(imgCinza)
plt.show()
plt.imshow(imgCinza,cmap="gray")
plt.show()
#Aplicando Blur para diminuir o ruido da imagem
suavizacao = cv2.blur(imgCinza, (7,7))
plt.imshow(suavizacao,cmap="gray")
plt.show()

# Transformando a imagem em preto e branco apenas
T = mahotas.thresholding.otsu(suavizacao)
bin = suavizacao.copy()
bin[bin>T] = 255
bin[bin<255] = 0
bin = cv2.bitwise_not(bin)
plt.imshow(bin,cmap="gray")
plt.show()

#Definindo as bordas
bordas = cv2.Canny(bin,70,150)
plt.imshow(bordas,cmap="gray")
plt.show()