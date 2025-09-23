import numpy as np
import matplotlib.pyplot as plt
from skimage import io, util
from skimage.color import rgb2hsv, hsv2rgb
from skimage.filters import gaussian
from scipy.stats import gmean


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
    axs[2][columna].set_title("mascara no negros")
    axs[2][columna].axis("off")
    axs[2][columna].imshow(imagenMascara, clim=(0, 1))


def esBlanco(pixel):
    return pixel[1] == 0 and pixel[2] == 1


def esNegro(pixel):
    return pixel[1] == 0 and pixel[2] == 0


def promedioSaltAndPepper(imagen):
    copia = rgb2hsv(imagen)

    for i in range(copia.shape[0]):
        for j in range(copia.shape[1]):
            if esBlanco(copia[i, j]):
                copia[i, j] = 0

    copia = promedioAritmetico(copia)
    return hsv2rgb(copia)


def unsharpMasking(imagen):
    gausiana = gaussian(imagen)
    return np.clip(imagen - gausiana, 0, 1)


# Promedio geometrico de Digital Image Processing (Gonzales, Woods)
# Capitulo 5, seccion de restauracion en presencia de ruido aditivo
def promedioGeometrico(imagen):
    res = np.zeros(imagen.shape)
    imagenProducto = np.power(imagen, 1.0/16)
    for i in range(2, res.shape[0] - 2):
        for j in range(2, res.shape[1] - 2):
            vecinos = imagenProducto[i-2:i+3, j-2:j+3]
            for canal in range(3):
                res[i, j, canal] = np.prod(vecinos[:, :, canal])
    return np.clip(res, 0, 1)


def promedioAritmetico(imagen):
    res = np.zeros(imagen.shape)
    imagenSuma = imagen / 9
    for i in range(1, res.shape[0] - 1):
        for j in range(1, res.shape[1] - 1):
            vecinos = imagenSuma[i-1:i+2, j-1:j+2]
            for canal in range(3):
                res[i, j, canal] = np.sum(vecinos[:, :, canal])
    return res


imagenNormal = util.img_as_float64(io.imread("./shape_dataset/image_0000.png"))
imagenRuidosa = util.img_as_float64(io.imread("./shape_dataset/image_0003.png"))
imagenNiebla = util.img_as_float64(io.imread("./shape_dataset/image_0008.png"))
imagenSaltAndPepper = util.img_as_float64(io.imread("./shape_dataset/image_0012.png"))

fig, axs = plt.subplots(3, 4, figsize=(20, 10))
pltImagenConH(imagenNormal, imagenNormal, "Imagen Normal", "Identidad", axs, 0)
pltImagenConH(imagenRuidosa, promedioGeometrico(imagenRuidosa), "Imagen Ruidosa", "Promedio Geometrico", axs, 1)
pltImagenConH(imagenNiebla, promedioGeometrico(imagenNiebla), "Imagen Niebla", "Promedio Geometrico", axs, 2)
pltImagenConH(imagenSaltAndPepper, promedioSaltAndPepper(imagenSaltAndPepper), "Imagen SaltAndPepper", "Promediar blancos por vecinos", axs, 3)

plt.tight_layout()
plt.show()
