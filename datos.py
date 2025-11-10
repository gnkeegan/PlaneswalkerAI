# =============================================================
# Nombre del archivo: datos.py
# Autor: Grant Nathaniel Keegan
# Fecha de creación: 2025-10-15
# Descripción: ETL para obtener las cartas en formato standard.
# Dependencias: requests, pandas, tqdm, time
# =============================================================

# Instalamos dependencias.
import requests
import pandas as pd # Para trabajar y manipular los archivos csv.
from tqdm import tqdm
import time

def get_standard_cards():
    """
    Descarga todas las cartas legales en Standard de Scryfall (últimos 3 años aprox)
    y guarda un CSV con sus atributos más útiles.
    """
    print("🔮 Descargando cartas legales en formato Standard desde la API de Scryfall...")

    # Endpoint base
    base_url = "https://api.scryfall.com/cards/search"
    # Este query se asegura que las cartas sean legales en formato standard de acuerdo a la base de datos.
    # Utiliza la variable legal:standard. Esta es actualizada automáticamente en Scryfall.
    query = "game:paper legal:standard -is:promo -is:digital"

    # Agrega cada carta a una lista.
    all_cards = [] 
    next_page = f"{base_url}?q={query}"

    while next_page:
        response = requests.get(next_page)
        if response.status_code != 200:
            print(f"Error {response.status_code}: {response.text}")
            break

        data = response.json()
        cards = data.get("data", [])

        for card in tqdm(cards, desc="Procesando cartas"):
            # Agregamos solamente columnas relevantes, ya que muchas solo estorban.
            # En especial necesitamos "colors" y "image_uri" para descargarlas.
            card_info = {
                "name": card.get("name"),
                "type_line": card.get("type_line"),
                "oracle_text": card.get("oracle_text"),
                "colors": ",".join(card.get("colors", [])) if card.get("colors") else "Colorless",
                "cmc": card.get("cmc"),
                "rarity": card.get("rarity"),
                "set_name": card.get("set_name"),
                "released_at": card.get("released_at"),
                "power": card.get("power"),
                "toughness": card.get("toughness"),
                "keywords": ",".join(card.get("keywords", [])),
                "image_uri": card.get("image_uris", {}).get("normal") if card.get("image_uris") else None,
            }
            all_cards.append(card_info)

        # Paginación
        next_page = data.get("next_page")
        time.sleep(0.1)  # para no saturar la API

    print(f"Total de cartas descargadas: {len(all_cards)}")

    # Crear dataframe y guardar CSV como "mtg_standard_cards.csv"
    df = pd.DataFrame(all_cards)
    df.to_csv("mtg_standard_cards.csv", index=False, encoding="utf-8-sig")
    print("Archivo guardado como mtg_standard_cards.csv")

    return df

if __name__ == "__main__":
    df = get_standard_cards()
    print(df.head())