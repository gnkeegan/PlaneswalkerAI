import os
import pandas as pd
import requests
from tqdm import tqdm
from tkinter import Tk, filedialog, messagebox

# ================================
#  Descargador de imágenes MTG 🧙‍♂️
# ================================
def sanitize_filename(name):
    """Elimina caracteres inválidos del nombre del archivo."""
    return "".join(c for c in str(name) if c.isalnum() or c in (" ", "_", "-")).rstrip()

def main():
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    # === Seleccionar archivo de entrada ===
    input_file = filedialog.askopenfilename(
        title="Selecciona tu archivo Excel o CSV",
        filetypes=[("Archivos de Excel o CSV", "*.xlsx *.xls *.csv")]
    )

    if not input_file:
        messagebox.showwarning("Cancelado", "No seleccionaste ningún archivo.")
        return

    # === Seleccionar carpeta de salida ===
    output_dir = filedialog.askdirectory(title="Selecciona la carpeta de destino para las imágenes")
    if not output_dir:
        messagebox.showwarning("Cancelado", "No seleccionaste carpeta de destino.")
        return

    # === Leer archivo ===
    ext = os.path.splitext(input_file)[1].lower()
    try:
        if ext in [".xlsx", ".xls"]:
            df = pd.read_excel(input_file)
        elif ext == ".csv":
            df = pd.read_csv(input_file, encoding="utf-8", low_memory=False)
        else:
            messagebox.showerror("Error", "Formato no soportado. Usa .xlsx o .csv")
            return
    except Exception as e:
        messagebox.showerror("Error al leer el archivo", str(e))
        return

    # === Detección automática de columnas ===
    possible_columns = [c for c in df.columns if "image" in c.lower()]
    name_columns = [c for c in df.columns if "name" in c.lower()]

    if not possible_columns:
        messagebox.showerror("Error", "No se encontró una columna que contenga 'image' en el nombre.")
        return

    image_col = possible_columns[0]
    name_col = name_columns[0] if name_columns else None

    # === Filtrar filas con URL válidas ===
    df = df.dropna(subset=[image_col])
    df = df[df[image_col].str.startswith("http", na=False)]
    total = len(df)

    if total == 0:
        messagebox.showinfo("Sin imágenes", "No se encontraron URLs válidas para descargar.")
        return

    print(f"📂 Archivo: {os.path.basename(input_file)}")
    print(f"💾 Carpeta de destino: {output_dir}")
    print(f"🧮 Total de imágenes a descargar: {total}\n")

    # === Descarga ===
    for i, row in tqdm(df.iterrows(), total=total, desc="Descargando imágenes", unit="img"):
        url = str(row[image_col]).strip()
        name = sanitize_filename(row[name_col]) if name_col else f"card_{i}"

        # Detectar extensión válida
        ext = os.path.splitext(url)[1].split("?")[0].lower()
        if ext not in [".jpg", ".jpeg", ".png"]:
            ext = ".jpg"

        save_path = os.path.join(output_dir, f"{name[:80]}{ext}")

        # Evitar duplicados
        if os.path.exists(save_path):
            continue

        try:
            r = requests.get(url, timeout=15)
            if r.status_code == 200:
                with open(save_path, "wb") as f:
                    f.write(r.content)
            else:
                print(f"⚠️ Error {r.status_code} con {name}")
        except Exception as e:
            print(f"⚠️ Error con {name}: {e}")

    print(f"\n🎉 Descarga completa. Imágenes guardadas en: {output_dir}")
    messagebox.showinfo("Completado", f"Descarga completa.\nImágenes guardadas en:\n{output_dir}")

if __name__ == "__main__":
    main()
