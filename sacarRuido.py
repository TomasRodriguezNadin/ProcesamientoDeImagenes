import numpy as np
import matplotlib.pyplot as plt
from skimage import io, util
from skimage.color import rgb2hsv, hsv2rgb, rgb2gray
from skimage.filters.rank import maximum


def pltImagenConH(imagen, imagenH, nombre, nombreH, axs, columna):
    axs[0][columna].set_title(f"Imagen {nombre}")
    axs[0][columna].axis("off")
    axs[0][columna].imshow(imagen, clim=(0, 1))

    axs[1][columna].set_title(nombreH)
    axs[1][columna].axis("off")
    axs[1][columna].imshow(imagenH, clim=(0, 1))

    imagenMascara = imagenH.copy()
    imagenMascara = rgb2hsv(imagenMascara)
    for i in range(imagenMascara.shape[0]):
        for j in range(imagenMascara.shape[1]):
            if (imagenMascara[i, j, 1] < 0.1 and imagenMascara[i, j, 2] < 0.1):
                imagenMascara[i, j, 0] = 0
                imagenMascara[i, j, 1] = 0
                imagenMascara[i, j, 2] = 0
            else:
                imagenMascara[i, j, 1] = 0
                imagenMascara[i, j, 2] = 1
    imagenMascara = hsv2rgb(imagenMascara)
    axs[2][columna].set_title("Mascara no negros")
    axs[2][columna].axis("off")
    axs[2][columna].imshow((rgb2gray(imagenMascara)), cmap='gray', clim=(0, 1))


# Promedio geometrico de Digital Image Processing (Gonzales, Woods)
# Capitulo 5, seccion de restauracion en presencia de ruido aditivo
def promedioGeometrico(imagen):
    res = np.zeros(imagen.shape)
    for i in range(2, res.shape[0] - 2):
        for j in range(2, res.shape[1] - 2):
            vecinos = imagen[i-2:i+3, j-2:j+3]
            for canal in range(3):
                res[i, j, canal] = np.power(np.prod(vecinos[:, :, canal]), 1.0/25)

    vecindario = np.ones((3, 3))
    resFiltrada = np.empty(imagen.shape)
    for canal in range(3):
        resFiltrada[:, :, canal] = maximum(res[:, :, canal], footprint=vecindario)
    return resFiltrada


def sacarGrises(imagen):
    imagenHSV = rgb2hsv(imagen)
    canalSaturacion = imagenHSV[:, :, 1]
    imagenHSV[canalSaturacion == 0] = 0
    return hsv2rgb(imagenHSV)


def max(arr):
    maximo = np.max(arr)
    if arr[2] == maximo:
        return 2
    if arr[1] == maximo:
        return 1
    return 0


def umbralizarColores(imagen):
    copia = np.copy(imagen)
    for i in range(imagen.shape[0]):
        for j in range(imagen.shape[1]):
            maximo = max(imagen[i, j])
            for canal in range(3):
                if canal != maximo:
                    copia[i, j, canal] = 0

    return copia


def sacarNiebla(imagen):
    imagenHSV = rgb2hsv(imagen)
    for i in range(imagen.shape[0]):
        for j in range(imagen.shape[1]):
            if imagenHSV[i, j, 1] == 0:
                imagenHSV[i, j] = 0
            else:
                imagenHSV[i, j, 1] = 1
    return hsv2rgb(imagenHSV)


if __name__ == "__main__":
    imagenNormal = util.img_as_float64(io.imread("./shape_dataset/image_0001.png"))
    imagenRuidosa = util.img_as_float64(io.imread("./shape_dataset/image_0009.png"))
    imagenNiebla = util.img_as_float64(io.imread("./shape_dataset/image_0011.png"))
    imagenSaltAndPepper = util.img_as_float64(io.imread("./shape_dataset/image_0012.png"))

    fig, axs = plt.subplots(3, 4, figsize=(20, 10))
    pltImagenConH(imagenNormal, sacarNiebla(imagenNormal), "Imagen Normal", "Sacar grises", axs, 0)
    pltImagenConH(imagenRuidosa, promedioGeometrico(imagenRuidosa), "Imagen Ruidosa", "Promedio Geometrico", axs, 1)
    pltImagenConH(imagenNiebla, sacarNiebla(imagenNiebla), "Imagen Niebla", "Sacar grises", axs, 2)
    pltImagenConH(imagenSaltAndPepper, sacarNiebla(imagenSaltAndPepper), "Imagen SaltAndPepper", "SacarGrises", axs, 3)

    plt.tight_layout()
    plt.show()
