# =============================================================
# Nombre del archivo: metadata_generator.py
# Autor: Grant Nathaniel Keegan
# Descripción: Genera un nuevo .csv con los metadatos del valor
# de color en las cartas, para datos reales del entrenamiento.
# Dependencias: requests, pandas, tqdm, time
# =============================================================

import pandas as pd
import re
import unicodedata

# Congifurar archivos (CSV original y nombre del nuevo con metadatos.
input_file = "mtg_standard_cards.csv" # CSV con datos original.
output_file = "card_metadata.csv" # Archivo con metadatos para entrenar.

# Carga el dataset original.
df = pd.read_csv(input_file, encoding="utf-8", low_memory=False)

# Verifica columnas necesarias.
if "name" not in df.columns or "colors" not in df.columns:
    raise ValueError("El CSV debe contener las columnas 'name' y 'colors'.")

# Limpiamos los nombres para que no encuentre problemas la carga de metadatos.
def clean_name_for_jpg(name):
    "Limpia la columna de nombres para adaptar caracteres especiales."
    if pd.isna(name):
        return ""

    cleaned = str(name)

    # Elimina diagonales // en ciertas cartas.
    # Ejemplo: "SP//dr" a "SPdr"
    cleaned = cleaned.replace("/", "")

    # Quita caracteres ilegales, mantiene acentos y ñ.
    cleaned = re.sub(r"[\"'’´`]", "", cleaned)
    cleaned = re.sub(r"[,;:(){}\[\]]", "", cleaned)
    cleaned = re.sub(r"[^A-Za-z0-9 áéíóúÁÉÍÓÚüÜñÑ_-]", "", cleaned)

    # Limpia con strip().
    cleaned = cleaned.strip()

    return cleaned

# Creamos nuestro dataframe de nombres.
df["name"] = df["name"].apply(clean_name_for_jpg)

# Normaliza los colores para quitar espacios en las columnas.
def normalize_colors(val):
    if pd.isna(val):
        return "Colorless"
    # Elimina espacios y corchetes.
    val = str(val).replace(" ", "").replace("[", "").replace("]", "").replace("'", "")
    if val == "" or val.lower() == "colorless":
        return "Colorless"
    return val.upper()

# Creamos nuestro dataframe de colores.
df["colors"] = df["colors"].apply(normalize_colors)

# Crea columnas binarias para agregar el one-hot encoding.
color_cols = ["Colorless", "R", "U", "G", "B", "W"]

# Establece 0 como default en el one-hot encoding de las columnas.
for c in color_cols:
    df[c] = 0

# Genera un 1 en la columna si detecta que tiene en valor de su color.
# Esto asegura que los metadatos sean acertados.
for i, row in df.iterrows():
    color_list = str(row["colors"]).split(",")
    for c in color_list:
        c = c.strip().upper()
        if c == "" or c == "COLORLESS":
            df.at[i, "Colorless"] = 1
        elif c in color_cols:
            df.at[i, c] = 1

# Formatea la columna "colors" como una lista.
df["colors"] = df["colors"].apply(
    lambda x: f"[{', '.join(x.split(','))}]" if ',' in x else f"['{x}']"
)

# Selecciona las columnas y lo guarda en el dataframe.
output_df = df[["name", "colors"] + color_cols]

# Output
output_df.to_csv(output_file, index=False, encoding="utf-8-sig")

print(f"Archivo generado exitosamente: {output_file}")
print(f"Total de registros: {len(output_df)}")
print("\nVista previa:")
print(output_df.head(10))