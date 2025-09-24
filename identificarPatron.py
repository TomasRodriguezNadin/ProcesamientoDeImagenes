import numpy as np
import matplotlib.pyplot as plt
from skimage import io, util
from skimage.color import rgb2hsv, hsv2rgb, rgb2gray


def imagenConHistograma(imagen, nombre, axs, columna):
    axs[0][columna].set_title(f"Imagen {nombre}")
    axs[0][columna].axis("off")
    axs[0][columna].imshow(imagen, clim=(0, 1))

    binarizada = rgb2hsv(imagen)
    binarizada = binarizada[:, :, 2]

    axs[1][columna].set_title("Histograma por valores")
    axs[1][columna].grid(True)
    axs[1][columna].hist(binarizada.ravel(), bins=256, histtype='step', color='black')

    enGris = rgb2gray(imagen)

    axs[2][columna].set_title("Histograma del gris")
    axs[2][columna].grid(True)
    axs[2][columna].hist(enGris.ravel(), bins=256, histtype='step', color='black')

    cantBajos = np.sum(binarizada < 0.2) - np.sum(binarizada == 0)
    print(f"{nombre} cant de pixeles en el menor 1/5 {cantBajos}")


imagenNormal = util.img_as_float64(io.imread("./shape_dataset/image_0000.png"))
imagenRuidosa = util.img_as_float64(io.imread("./shape_dataset/image_0003.png"))
imagenNiebla = util.img_as_float64(io.imread("./shape_dataset/image_0008.png"))
imagenSaltAndPepper = util.img_as_float64(io.imread("./shape_dataset/image_0012.png"))

fig, axs = plt.subplots(3, 4, figsize=(20, 10))
imagenConHistograma(imagenNormal, "Imagen Normal", axs, 0)
imagenConHistograma(imagenRuidosa, "Imagen Ruidosa", axs, 1)
imagenConHistograma(imagenNiebla, "Imagen Niebla", axs, 2)
imagenConHistograma(imagenSaltAndPepper, "Imagen SaltAndPepper", axs, 3)

plt.tight_layout()
plt.show()
