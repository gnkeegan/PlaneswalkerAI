import os
import pandas as pd

# === CONFIGURA AQUÍ ===
BASE_DIR = "C:\Users\MSI\OneDrive\Magic The Gathering\PlaneswalkerAI\PlaneswalkerAI\mtg_standard_cards_split"
ORIGINAL_CSV = "C:\Users\MSI\OneDrive\Magic The Gathering\PlaneswalkerAI\PlaneswalkerAI\mtg_standard_cards.csv"  # tu dataset con "colors" y "image_uri"
# ======================

# Carga el dataset original
df = pd.read_csv(ORIGINAL_CSV)

# Extrae el nombre del archivo de la URL
df["filename"] = df["image_uri"].apply(lambda x: os.path.basename(str(x)) if pd.notnull(x) else None)
df = df.dropna(subset=["filename", "colors"])

print("Dataset base cargado:", len(df), "cartas")

# Función auxiliar para crear CSVs según las imágenes encontradas
def create_label_csv(split_name):
    split_path = os.path.join(BASE_DIR, split_name)
    files = os.listdir(split_path)
    subset = df[df["filename"].isin(files)][["filename", "colors"]]
    out_path = os.path.join(BASE_DIR, f"{split_name}_labels.csv")
    subset.to_csv(out_path, index=False)
    print(f"💾 {split_name}_labels.csv creado con {len(subset)} filas")

for split in ["train", "val", "test"]:
    create_label_csv(split)

print("\n¡Listo! Se generaron los archivos:")
print(os.listdir(BASE_DIR))
