from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter
import pandas as pd

# Cargar dataset descargado previamente
cards_df = pd.read_csv("mtg_standard_cards.csv")

# Crear lista con los nombres de cartas (sin duplicados)
card_names = sorted(list(set(cards_df['name'].dropna().tolist())))

# Configurar autocompletador
card_completer = WordCompleter(card_names, ignore_case=True, sentence=True)

def get_user_cards():
    print("🧙‍♂️ Bienvenido a DeckForge AI!")
    print("Introduce 5 cartas con las que quieres construir tu deck Standard.")
    print("(Usa autocompletado con TAB o escribe el nombre manualmente)\n")

    selected_cards = []
    for i in range(1, 6):
        card = prompt(f"🃏 Carta {i}: ", completer=card_completer)
        selected_cards.append(card)

    print("\n✨ Cartas seleccionadas:")
    for c in selected_cards:
        print(f"   - {c}")

    return selected_cards

if __name__ == "__main__":
    user_cards = get_user_cards()
