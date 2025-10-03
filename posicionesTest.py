import matplotlib.pyplot as plt
from skimage import io, util
import obtenerPosicionesFiguras as op
import sacarRuido as sr


def mostrarFiguras(imagen):
    limitesFiguras = op.encontrarPosicionesImagenes(imagen)
    cant = min(len(limitesFiguras), 4)
    for i in range(cant):
        minX = limitesFiguras[i][0]
        maxX = limitesFiguras[i][1]
        minY = limitesFiguras[i][2]
        maxY = limitesFiguras[i][3]
        figura = imagen[minY:maxY+1, minX:maxX+1]
        print(f"({minX}, {minY}), ({maxX}, {maxY})")
        axs[0][i].set_title(f"Figura {i}")
        axs[0][i].axis("off")
        axs[0][i].imshow(figura)

        enGris = rgb2gray(figura)
        axs[1][i].set_title("Histograma del gris")
        axs[1][i].grid(True)
        axs[1][i].hist(enGris.ravel(), bins=256, histtype='step', color='black')
       


if __name__ == "__main__":
    imagen = util.img_as_float64(io.imread("./image.png"))
    imagen = sr.sacarGrises(imagen)

    fig, axs = plt.subplots(2, 4, figsize=(20, 10))
    mostrarFiguras(imagen)

    plt.tight_layout()
    plt.show()
