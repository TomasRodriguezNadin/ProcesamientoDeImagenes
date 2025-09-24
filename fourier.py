import numpy as np
import matplotlib.pyplot as plt
from skimage import io, util
from skimage.color import rgb2hsv, hsv2rgb
from skimage.filters import gaussian
import sacarRuido as sr


def log_filter(image):
    image_float = util.img_as_float(image)
    c = 1.0 / np.log(2 + np.max(image_float))
    log_transformed_image = c * np.log(1 + image_float)
    return log_transformed_image


def limpiarImagen(imagen):
    # transformada = np.fft.fft2(imagen)
    # transformada_limpia = gaussian(transformada)
    return imagen


def mostrarFourier(imagen, nombre, axs, columna):
    axs[0][columna].set_title(f"Imagen {nombre}")
    axs[0][columna].axis("off")
    axs[0][columna].imshow(imagen, clim=(0, 1))

    transformada = np.fft.fft2(imagen)
    magnitud = np.abs(transformada)
    axs[1][columna].set_title("Magnitud")
    axs[1][columna].axis("off")
    axs[1][columna].matshow(log_filter(magnitud), cmap='gray', clim=(0, 1))

    angulo = np.angle(transformada)
    axs[2][columna].set_title("angulo")
    axs[2][columna].axis("off")
    axs[2][columna].matshow(angulo, cmap='gray', clim=(0, 1))


if __name__ == "__main__":
    imagenNormal = util.img_as_float64(io.imread("./shape_dataset/image_0000.png"))
    imagenRuidosa = util.img_as_float64(io.imread("./shape_dataset/image_0003.png"))
    imagenNiebla = util.img_as_float64(io.imread("./shape_dataset/image_0008.png"))
    imagenSaltAndPepper = util.img_as_float64(io.imread("./shape_dataset/image_0012.png"))

    fig, axs = plt.subplots(3, 4, figsize=(20, 10))
    mostrarFourier(sr.sacarGrises(imagenNormal), "Imagen Normal", axs, 0)
    mostrarFourier(sr.sacarGrises(imagenRuidosa), "Imagen Ruidosa", axs, 1)
    mostrarFourier(imagenRuidosa, "Imagen Niebla", axs, 2)
    mostrarFourier(sr.sacarGrises(imagenSaltAndPepper), "Imagen SaltAndPepper", axs, 3)

    plt.tight_layout()
    plt.show()
