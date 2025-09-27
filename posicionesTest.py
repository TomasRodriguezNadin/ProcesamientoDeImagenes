import numpy as np
import matplotlib.pyplot as plt
from skimage import io, util
from skimage.color import rgb2hsv, hsv2rgb, rgb2gray
import obtenerPosicionesFiguras as op
import sacarRuido as sr
import sys


def mostrarFiguras(imagen):
    axs[0][0].set_title("Imagen")
    axs[0][0].axis("off")
    axs[0][0].imshow(imagen, clim=(0, 1))

    limitesFiguras = op.encontrarPosicionesImagenes(imagen)
    cant = min(len(limitesFiguras), 3)
    for i in range(cant):
        minX = limitesFiguras[i][0]
        maxX = limitesFiguras[i][1]
        minY = limitesFiguras[i][2]
        maxY = limitesFiguras[i][3]
        print(f"({minX}, {minY}), ({maxX}, {maxY})")
        axs[1][cant].set_title(f"Figura {i}")
        axs[1][cant].axis("off")
        axs[1][cant].imshow(imagen[minY:maxY+1, minX:maxX+1])


if __name__ == "__main__":
    imagen = util.img_as_float64(io.imread("./image.png"))
    imagen = sr.sacarNiebla(imagen)

    fig, axs = plt.subplots(2, 4, figsize=(20, 10))
    mostrarFiguras(imagen)

    plt.tight_layout()
    plt.show()
