import os
import random
import shutil
from math import floor
from tkinter import Tk, filedialog, messagebox
from tqdm import tqdm

def main():
    # === Seleccionar carpeta con imágenes ===
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    image_dir = filedialog.askdirectory(title="Selecciona la carpeta con tus cartas (imágenes)")
    if not image_dir:
        messagebox.showwarning("Cancelado", "No seleccionaste ninguna carpeta.")
        return

    base_dir = os.path.dirname(image_dir)
    output_dir = os.path.join(base_dir, "dataset_split")

    # Crear carpetas de salida
    for folder in ["train", "val", "test"]:
        os.makedirs(os.path.join(output_dir, folder), exist_ok=True)

    # Buscar imágenes válidas
    exts = (".jpg", ".jpeg", ".png")
    all_images = [f for f in os.listdir(image_dir) if f.lower().endswith(exts)]
    total = len(all_images)

    if total == 0:
        messagebox.showerror("Error", "No se encontraron imágenes en la carpeta seleccionada.")
        return

    print(f"Total de imágenes encontradas: {total}")

    # Mezclar aleatoriamente
    random.seed(42)
    random.shuffle(all_images)

    # Calcular proporciones
    train_size = floor(total * 0.70)
    val_size = floor(total * 0.15)
    test_size = total - train_size - val_size

    train_files = all_images[:train_size]
    val_files = all_images[train_size:train_size + val_size]
    test_files = all_images[train_size + val_size:]

    print(f"Dividiendo dataset:")
    print(f"Train: {len(train_files)}")
    print(f"Val: {len(val_files)}")
    print(f"Test: {len(test_files)}")

    # === Función para copiar con barra de progreso ===
    def copy_files(files, subset):
        dest = os.path.join(output_dir, subset)
        for filename in tqdm(files, desc=f"Copiando {subset}", unit="img"):
            src = os.path.join(image_dir, filename)
            dst = os.path.join(dest, filename)
            try:
                shutil.copy2(src, dst)
            except Exception as e:
                print(f"Error copiando {filename}: {e}")

    print("\nCopiando imágenes...\n")
    copy_files(train_files, "train")
    copy_files(val_files, "val")
    copy_files(test_files, "test")

    print("\nListo! Dataset dividido en:")
    print(f"{os.path.join(output_dir, 'train')}")
    print(f"{os.path.join(output_dir, 'val')}")
    print(f"{os.path.join(output_dir, 'test')}")

    messagebox.showinfo("Completado",
                        f"Dataset dividido correctamente.\n\nUbicación:\n{output_dir}")

if __name__ == "__main__":
    main()
