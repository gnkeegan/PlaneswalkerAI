import pandas as pd
import matplotlib.pyplot as plt

# === 1. Cargar CSV ===
df = pd.read_csv("card_metadata.csv")

# === 2. Columnas ===
color_cols = ["Colorless", "R", "U", "G", "B", "W"]

# === 3. Normalizar combinaciones (orden fijo R U G B W) ===
ORDER = ["R", "U", "G", "B", "W"]

def normalized_combination(row):
    if row["Colorless"] == 1:
        return "Colorless"
    
    colors = [c for c in ORDER if row[c] == 1]
    return "".join(colors)

df["comb"] = df.apply(normalized_combination, axis=1)

# === 4. Contar ===
counts = df["comb"].value_counts()

# Ordenar por tamaño de combinación
def sort_key(x):
    if x == "Colorless":
        return (0, x)
    return (len(x), x)

counts = counts.sort_index(key=lambda x: x.map(sort_key))

# === 5. Colores para barras ===
bar_colors = []
special_colors = {
    "Colorless": "#A8A8A8",
    "R": "#FF2B2B",
    "U": "#2B6CFF",
    "G": "#2ECC40",
    "B": "#000000",
    "W": "#DADADA"
}

for name in counts.index:
    bar_colors.append(special_colors.get(name, "#87CEEB"))

# === 6. Graficar con etiquetas ===
plt.figure(figsize=(18, 8))
bars = plt.bar(counts.index, counts.values, color=bar_colors, edgecolor="black")

# Añadir etiquetas encima de cada barra
for bar in bars:
    y = bar.get_height()
    x = bar.get_x() + bar.get_width()/2
    plt.text(x, y + 2, str(int(y)), ha='center', va='bottom', fontsize=9)

plt.title("Distribución de cartas por combinación exacta de colores", fontsize=18)
plt.xlabel("Combinación exacta de colores", fontsize=14)
plt.ylabel("Cantidad de cartas", fontsize=14)
plt.xticks(rotation=90, fontsize=10)

plt.tight_layout()
plt.show()

# === 7. Mostrar conteos ===
print("\n=== Cantidades por combinación normalizada ===\n")
print(counts)
