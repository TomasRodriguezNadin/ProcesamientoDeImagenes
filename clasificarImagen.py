import sacarRuido as sr
import numpy as np
from skimage.color import rgb2hsv


def esRuidoGaussiano(imagen):
    imagenFiltrada = sr.sacarGrises(imagen)

    blancoYNegro = rgb2hsv(imagenFiltrada)
    blancoYNegro = blancoYNegro[:, :, 2]

    cantBajos = np.sum(blancoYNegro < 0.1) - np.sum(blancoYNegro == 0)

    return cantBajos > 2000
