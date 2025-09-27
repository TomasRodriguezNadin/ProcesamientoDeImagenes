import numpy as np
import matplotlib.pyplot as plt
from skimage import io, util
from scipy.ndimage import gaussian_filter


def log_filter(image):
    image_float = util.img_as_float(image)
    c = 1.0 / np.log(2 + np.max(image_float))
    log_transformed_image = c * np.log(1 + image_float)
    return log_transformed_image


def limpiarCanal(canal):
    transformada = np.fft.fft2(canal)
    shifted = np.fft.fftshift(transformada)
    gausiana = gaussian_filter(shifted, sigma=1)
    transformada = np.fft.ifftshift(gausiana)
    return np.real(np.fft.ifft2(transformada))


def limpiarImagen(imagen):
    copia = np.copy(imagen)
    for canal in range(3):
        copia[:, :, canal] = limpiarCanal(imagen[:, :, canal])
    return copia


def mostrarFourier(imagen, nombre, axs, fila):
    axs[fila][0].set_title(f"Imagen {nombre}")
    axs[fila][0].axis("off")
    axs[fila][0].imshow(imagen, clim=(0, 1))

    transformada = np.fft.fft2(imagen)
    magnitud = np.abs(transformada)
    axs[fila][1].set_title("Magnitud")
    axs[fila][1].axis("off")
    axs[fila][1].matshow(log_filter(magnitud), cmap='gray', clim=(0, 1))

    angulo = np.angle(transformada)
    axs[fila][2].set_title("angulo")
    axs[fila][2].axis("off")
    axs[fila][2].matshow(angulo, cmap='gray', clim=(0, 1))


if __name__ == "__main__":
    imagenNormal = util.img_as_float64(io.imread("./shape_dataset/image_0000.png"))
    imagenRuidosa = util.img_as_float64(io.imread("./shape_dataset/image_0003.png"))
    imagenNiebla = util.img_as_float64(io.imread("./shape_dataset/image_0008.png"))
    imagenSaltAndPepper = util.img_as_float64(io.imread("./shape_dataset/image_0012.png"))

    fig, axs = plt.subplots(2, 3, figsize=(20, 10))
    mostrarFourier(imagenRuidosa, "Imagen Normal", axs, 0)
    mostrarFourier(limpiarImagen(imagenRuidosa), "Imagen Ruidosa", axs, 1)

    plt.tight_layout()
    plt.show()
