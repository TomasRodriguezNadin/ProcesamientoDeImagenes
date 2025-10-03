import numpy as np
from skimage.color import rgb2hsv


maxX = 0
minX = 0
maxY = 0
minY = 0


def esNegro(pixel):
    return pixel[1] == 0 and pixel[2] == 0


def dfs(imagen, visitados, starty, startx):
    global maxX, minX, maxY, minY
    maxX = startx
    minX = startx
    maxY = starty
    minY = starty

    stack = [(startx, starty)]

    while len(stack) != 0:
        x, y = stack.pop()
        if visitados[y, x] != 0:
            continue

        visitados[y, x] = 1

        if x > maxX:
            maxX = x
        elif x < minX:
            minX = x

        if y > maxY:
            maxY = y
        elif y < minY:
            minY = y

        for xvecino in range(max(x-2, 0), min(x+3, imagen.shape[1])):
            for yvecino in range(max(y-2, 0), min(y+3, imagen.shape[0])):
                if (xvecino != x or yvecino != y) and not esNegro(imagen[yvecino, xvecino]):
                    stack.append((xvecino, yvecino))


def encontrarPosicionesImagenes(imagen):
    global maxX
    global minX
    global maxY
    global minY

    res = []
    imagenHSV = rgb2hsv(imagen)
    visitados = np.zeros((imagen.shape[0], imagen.shape[1]))
    for i in range(imagen.shape[0]):
        for j in range(imagen.shape[1]):
            if visitados[i, j] == 0 and not esNegro(imagenHSV[i, j]):
                dfs(imagenHSV, visitados, i, j)
                res.append([minX, maxX, minY, maxY])

    return res
