import numpy as np 
import matplotlib.pyplot as plt
from skimage import io, util, feature ,filters
from skimage.color import rgb2hsv ,  hsv2rgb , rgb2gray  
from skimage.morphology import dilation, square
from skimage.transform import hough_circle, hough_circle_peaks
import obtenerPosicionesFiguras
from sacarRuido import sacarNiebla

def promedioGeometrico(self, imagen : np.ndarray) -> np.ndarray:
    res = np.zeros(imagen.shape)
    for i in range(2, res.shape[0] - 2):
        for j in range(2, res.shape[1] - 2):
            vecinos = imagen[i-2:i+3, j-2:j+3]
            for canal in range(3):
                res[i, j, canal] = np.power(np.prod(vecinos[:, :, canal]), 1.0/25)
    return res
def sacarRuidoGaussiano(imagen : np.ndarray) -> np.ndarray:
    return (promedioGeometrico(imagen))

def sacarGrises(imagen : np.ndarray) -> np.ndarray:
    imagenHSV = rgb2hsv(imagen)
    canalSaturacion = imagenHSV[:, :, 1]
    imagenHSV[canalSaturacion == 0] = 0
    return hsv2rgb(imagenHSV)
def sacarRuido(image : np.ndarray) -> np.ndarray:
    imagen = image.copy()
    #imagen = self.sacarGrises(imagen)
    #imagen = self.promedioGeometrico(imagen)
    imagenHSV = rgb2hsv(imagen.copy())
    cantidadPixelesBajos = np.sum(imagenHSV < 0.05) - np.sum(imagenHSV == 0)
    print(cantidadPixelesBajos)
    imagen = sacarGrises(imagen)
    if (cantidadPixelesBajos/(image.shape[0] * image.shape[1]) > 0.08333):
        #Es de ruido gaussiano
        print("Es gaussiana")
        return promedioGeometrico(imagen)
    return imagen

def filtroMaximo3x3( imagen : np.ndarray) -> np.ndarray:
    res = np.zeros(imagen.shape)
    for i in range(1, imagen.shape[0] - 1):
        for j in range(1, imagen.shape[1] - 1):
            for canal in range(3):
                # Me mataba los colores si no le ponia el canal dentro de los vecinos
                # No se como estaba definido el maximo antes
                # Pero no deberia estar cambiandome el color ahora?
                # Preguntando por el maximo rojo, el maximo verde y el maximo azul y mezclandolos
                # No termino de entender de que esta tomando el maximo
                vecinos = imagen[i-1:i+2, j-1:j+2, canal]
                res[i, j, canal] = np.max(vecinos)
    return res
def promedioGeometrico(imagen : np.ndarray) -> np.ndarray:
        res = np.zeros(imagen.shape)
        for i in range(2, res.shape[0] - 2):
            for j in range(2, res.shape[1] - 2):
                vecinos = imagen[i-2:i+3, j-2:j+3]
                for canal in range(3):
                    res[i, j, canal] = np.power(np.prod(vecinos[:, :, canal]), 1.0/25)
        return res


import numpy as np 
import matplotlib.pyplot as plt
from skimage import io, util, feature, filters
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.morphology import square

def main():
    imagen = util.img_as_float64(io.imread("shape_dataset/image_0029.png")) 
    imagen = sacarGrises(imagen)
    # Convertir a escala de grises
    if len(imagen.shape) == 3:
        imagen_gris = np.mean(imagen, axis=2)
    else:
        imagen_gris = imagen
    
    # Bordes Canny (igual que antes)
    bordes = feature.canny(imagen_gris, sigma=1.0)
    
    # Hough para círculos en una línea
   
    posiciones_figuras = obtenerPosicionesFiguras.encontrarPosicionesImagenes(imagen)
    radios = []
    for posicion in obtenerPosicionesFiguras.encontrarPosicionesImagenes(imagen):
        minX, maxX, minY, maxY = posicion
        ancho = maxX - minX
        alto = maxY - minY
        # Tomar el promedio entre ancho y alto como radio base
        radio_base = (ancho + alto) / 4
        # Crear un rango alrededor del radio base
        radios.extend(np.arange(radio_base - 5, radio_base + 5, 1))

    # Convertir a array de numpy y asegurar que sean enteros
    radios = np.unique(np.round(radios).astype(int))
    
    # Hough para círculos
    hough_res = hough_circle(bordes, radios)
    accum, cx, cy, rad = hough_circle_peaks(hough_res, radios, total_num_peaks=len(posiciones_figuras))
    
    print(f"Círculos detectados: {len(cx)}")
    for i in range(len(cx)):
        print(f"Círculo {i+1}: centro=({cx[i]}, {cy[i]}), radio={rad[i]}")
    
    # Mostrar resultados
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    # Imagen original
    axs[0].imshow(imagen_gris, cmap='gray')
    axs[0].set_title("Imagen Original")
    axs[0].axis('off')
    
    # Bordes Canny
    axs[1].imshow(bordes, cmap='gray')
    axs[1].set_title("Bordes Canny")
    axs[1].axis('off')
    
    # Círculos detectados
    axs[2].imshow(imagen_gris, cmap='gray')
    for center_y, center_x, radius in zip(cy, cx, rad):
        circulo = plt.Circle((center_x, center_y), radius, color='red', fill=False, linewidth=2)
        axs[2].add_patch(circulo)
    axs[2].set_title("Círculos detectados")
    axs[2].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
