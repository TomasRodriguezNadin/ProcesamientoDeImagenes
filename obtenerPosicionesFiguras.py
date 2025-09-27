import numpy as np
from skimage.color import rgb2hsv
from skimage.color import rgb2hsv


maxX = 0
minX = 0
maxY = 0
minY = 0


def esNegro(pixel):
    return pixel[1] < 0.1 and pixel[2] < 0.1


def dfs(imagen, visitados, y, x):
    global maxX, minX, maxY, minY
    maxX = x
    minX = x
    maxY = y
    minY = y

    stack = [(x, y)]

    while len(stack) != 0:
        xActual, yActual = stack.pop()
        if visitados[yActual, xActual] != 0:
            continue

        visitados[yActual, xActual] = 1

        if xActual > maxX:
            maxX = xActual
        elif xActual < minX:
            minX = xActual

        if yActual > maxY:
            maxY = yActual
        elif yActual < minY:
            minY = yActual

        for xvecino in range(max(xActual-1, 0), min(xActual+2, imagen.shape[0] - 1)):
            for yvecino in range(max(yActual-1, 0), min(yActual+2, imagen.shape[1] - 1)):
                if (not esNegro(imagen[yvecino, xvecino])) and (xvecino != xActual or yvecino != yActual):
                    stack.append((yvecino, xvecino))


def encontrarPosicionesImagenes(imagen):
    global maxX
    global minX
    global maxY
    global minY

    res = []
    imagenHSV = rgb2hsv(imagen)
    visitados = np.zeros((imagen.shape[0], imagen.shape[1]))
    for i in range(1, imagen.shape[0] - 1):
        for j in range(1, imagen.shape[1] - 1):
            if (not esNegro(imagenHSV[i, j])) and visitados[i, j] == 0:
                dfs(imagenHSV, visitados, i, j)
                res.append([minX, maxX, minY, maxY])

    return res
