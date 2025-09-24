import json
from clasificarImagen import esRuidoGaussiano
from skimage import io, util


dataset = './shape_dataset/'

with open(dataset + 'annotations.json', 'r') as file:
    data = json.load(file)
    # data = json.load(data)

    imagenesTotales = len(data)
    imagenesCorrectas = 0
    fallidas = []

    print("*Empezando test*")

    for dato in data:
        imagen = util.img_as_float64(io.imread(dataset + dato["filename"]))

        res = esRuidoGaussiano(imagen)

        if (dato["effect_applied"] == "gaussian") == res:
            imagenesCorrectas += 1
        else:
            fallidas.append(dato["filename"])

    print("*Test terminado *")
    print(f"Hubo {imagenesCorrectas} de {imagenesTotales} aciertos")
    print("Las imagenes fallidas fueron:")
    for filename in fallidas:
        print(filename)
