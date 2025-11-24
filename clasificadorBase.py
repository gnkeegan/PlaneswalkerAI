# =============================================================
# Nombre del archivo: img_download.py
# Autor: Grant Nathaniel Keegan
# Descripción: Usa el modelo entrenado para clasificar las cartas
# en un directorio a su color (o colores) correspondientes.
# Dependencias: os, shutil, cv2, numpy, tensorflow, tkinter, PIL
# =============================================================

import os
import shutil
import cv2
import numpy as np
import tensorflow as tf
from tkinter import Tk
from tkinter.filedialog import askdirectory
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input

# Cargamos el modelo de la misma carpeta que este código.
model = tf.keras.models.load_model("MTGCNNBase.keras", compile=False)

# Definimos las variables de clasificación, el umbral de confianza, y las dimensiones de los datos.
color_labels = ["Colorless", "Red", "Blue", "Green", "Black", "White"]
threshold = 0.70
# Nota: Estos valores deben ser los mismos que el dataset en el que se entrenó el modelo de keras.
x_size = 224
y_size = 224

# Seleccionamos la carpeta con Tkinter.
Tk().withdraw()
input_dir = askdirectory(title="Selecciona la carpeta con las cartas")

if not input_dir:
    raise ValueError("No seleccionaste ninguna carpeta.")

print(f"\nCarpeta seleccionada:\n{input_dir}\n")

# Crea las carpetas de salida en el mismo directorio.
def crear_carpeta(nombre):
    ruta = os.path.join(input_dir, nombre)
    os.makedirs(ruta, exist_ok=True)
    return ruta

other_folder = crear_carpeta("Other")

# Preprocesamiento igual que en el colab para el entrenamiento,
def load_and_preprocess(path):
    try:
        # Abrir con PIL (mejor soporte unicode).
        img = Image.open(path).convert("RGB")
        img = np.array(img)

        # Convertir de PIL a CV2 estilo BGR a RGB consistente.
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        h, w = img.shape[:2]

        # Resize con padding (sin deformar).
        scale = min(x_size / h, y_size / w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (new_w, new_h))

        # Lienzo cuadrado.
        canvas = np.zeros((x_size, y_size, 3), dtype=np.uint8)
        pad_x = (x_size - new_h) // 2
        pad_y = (y_size - new_w) // 2
        canvas[pad_x:pad_x+new_h, pad_y:pad_y+new_w] = resized

        # Convertir a RGB.
        canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

        # Convertir a float y aplicar preprocess_input EXACTO A COLAB.
        canvas = canvas.astype("float32")
        canvas = preprocess_input(canvas)

        return canvas

    except Exception as e:
        print(f"Error con '{path}': {e}")
        return None

# Empezamos a clasificar.
print("Clasificando cartas...\n")

for file in os.listdir(input_dir):
    if not file.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    path = os.path.join(input_dir, file)
    img = load_and_preprocess(path)

    if img is None:
        shutil.move(path, os.path.join(other_folder, file))
        print(f"{file} Error de lectura. Carta mandada a Other")
        continue

    # PREDICCIÓN EXACTA A EN EL COLAB.
    pred = model.predict(img.reshape(1, x_size, y_size, 3))[0]

    # Colores detectados.
    detected_colors = [label for label, p in zip(color_labels, pred) if p >= threshold]

    # Si no detecta nada las manda a la carpeta "other".
    if len(detected_colors) == 0:
        shutil.move(path, os.path.join(other_folder, file))
        print(f"{file} a Other (sin colores detectados)")
        continue

    # Crear carpetas de colores combinados.
    folder_name = " & ".join(detected_colors)
    destino_folder = crear_carpeta(folder_name)

    shutil.move(path, os.path.join(destino_folder, file))
    print(f"{file} → {folder_name}")

print("\nClasificación completada con éxito.")