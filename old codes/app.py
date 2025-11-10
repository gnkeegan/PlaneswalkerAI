# DeckForge AI — GUI Recommender for MTG Standard (PySide6)
# ---------------------------------------------------------
# Features
# - Fetch/Update dataset from Scryfall API (Standard-legal cards)
# - 5 autocompleted inputs for seed cards
# - Recommend 55 compatible cards using TF‑IDF semantic similarity
# - Group results by type (Lands, Creatures, Spells, etc.) with images
# - Modern dark theme
# - Ready to package with PyInstaller into a single .exe
#
# Requirements (pip):
#   PySide6 pandas scikit-learn requests tqdm
#
# Run:
#   python app.py
#
# Build (Windows example):
#   pyinstaller --noconfirm --onefile --windowed --name "DeckForgeAI" app.py
# ---------------------------------------------------------

import os
import sys
import io
import json
import time
import threading
from dataclasses import dataclass

import pandas as pd
import requests
from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from PySide6.QtCore import Qt, QSize, QThread, Signal
from PySide6.QtGui import QAction, QIcon, QPixmap
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QCompleter, QFileDialog, QMessageBox,
    QTabWidget, QScrollArea, QFrame, QSpacerItem, QSizePolicy, QStatusBar,
    QProgressBar
)

DATA_CSV = "mtg_standard_cards.csv"
RECOMMEND_TOP_N = 55

# ----------------------- Utilities -----------------------

def nice_text(s: str) -> str:
    if not isinstance(s, str) or not s:
        return ""
    return s.replace("\n", " ").strip()


def build_text_features(row: pd.Series) -> str:
    parts = [
        str(row.get("name", "")),
        str(row.get("type_line", "")),
        str(row.get("oracle_text", "")),
        str(row.get("colors", "")),
        str(row.get("keywords", "")),
        str(row.get("set_name", "")),
        str(row.get("rarity", "")),
    ]
    parts = [nice_text(p) for p in parts if isinstance(p, str)]
    return " | ".join(parts)


def infer_primary_bucket(type_line: str) -> str:
    if not isinstance(type_line, str):
        return "Other"
    t = type_line.lower()
    if "land" in t:
        return "Lands"
    if "creature" in t:
        return "Creatures"
    if "planeswalker" in t:
        return "Planeswalkers"
    if "artifact" in t and "creature" not in t:
        return "Artifacts"
    if "enchantment" in t:
        return "Enchantments"
    if "instant" in t:
        return "Instants"
    if "sorcery" in t:
        return "Sorceries"
    return "Other"


# ----------------------- Data Fetcher -----------------------

class ScryfallFetcher(QThread):
    progress = Signal(int)
    finished_ok = Signal(pd.DataFrame)
    failed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)

    def run(self):
        try:
            base_url = "https://api.scryfall.com/cards/search"
            # Standard paper, exclude promos/digital tokens
            query = "game:paper legal:standard -is:promo -is:digital"
            next_page = f"{base_url}?q={query}"
            all_cards = []
            total = 0
            fetched = 0

            # First probe for estimated total if available
            try:
                probe = requests.get(next_page, timeout=20)
                if probe.status_code != 200:
                    self.failed.emit(f"HTTP {probe.status_code}: {probe.text}")
                    return
                first = probe.json()
                cards = first.get("data", [])
                for card in cards:
                    all_cards.append(self._extract_card(card))
                fetched += len(cards)
                total = max(fetched * (2 if first.get("has_more") else 1), fetched)
                self.progress.emit(int(100 * fetched / max(total, 1)))
                next_page = first.get("next_page")
            except Exception as e:
                self.failed.emit(str(e))
                return

            while next_page:
                r = requests.get(next_page, timeout=30)
                if r.status_code != 200:
                    self.failed.emit(f"HTTP {r.status_code}: {r.text}")
                    return
                data = r.json()
                cards = data.get("data", [])
                for card in cards:
                    all_cards.append(self._extract_card(card))
                fetched += len(cards)
                # Heuristic progress update
                if total < fetched:
                    total = fetched + 200
                self.progress.emit(min(99, int(100 * fetched / max(total, 1))))
                next_page = data.get("next_page")
                time.sleep(0.05)

            df = pd.DataFrame(all_cards)
            # Deduplicate by name + set + released_at
            if not df.empty:
                df = df.drop_duplicates(subset=["name", "set_name", "released_at"], keep="first")
            self.progress.emit(100)
            self.finished_ok.emit(df)
        except Exception as e:
            self.failed.emit(str(e))

    @staticmethod
    def _extract_card(card: dict) -> dict:
        image_uri = None
        if card.get("image_uris"):
            image_uri = card["image_uris"].get("normal") or card["image_uris"].get("large")
        elif card.get("card_faces"):
            # Double-faced cards
            faces = card.get("card_faces", [])
            if faces and faces[0].get("image_uris"):
                image_uri = faces[0]["image_uris"].get("normal")
        return {
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
            "keywords": ",".join(card.get("keywords", [])) if card.get("keywords") else "",
            "image_uri": image_uri,
        }


# ----------------------- Recommender Core -----------------------

@dataclass
class RecommenderState:
    df: pd.DataFrame
    vectorizer: TfidfVectorizer
    tfidf_matrix: any


def build_recommender(df: pd.DataFrame) -> RecommenderState:
    if df.empty:
        raise ValueError("Dataset vacío. Descarga cartas primero.")

    work = df.copy()
    work["text_features"] = work.apply(build_text_features, axis=1)

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=50000,
        ngram_range=(1, 2)
    )
    tfidf = vectorizer.fit_transform(work["text_features"].fillna(""))

    state = RecommenderState(df=work, vectorizer=vectorizer, tfidf_matrix=tfidf)
    return state


def recommend_cards(state: RecommenderState, seed_names: list[str], top_n: int = RECOMMEND_TOP_N) -> pd.DataFrame:
    df = state.df
    tfidf = state.tfidf_matrix
    vec = state.vectorizer

    # Locate seed rows
    mask = df["name"].isin(seed_names)
    if not mask.any():
        raise ValueError("Ninguna de las cartas semilla está en el dataset.")

    seed_indices = df[mask].index.tolist()
    # Aggregate seed vector (mean of seed vectors)
    seed_matrix = tfidf[seed_indices]
    seed_vec = seed_matrix.mean(axis=0)

    # Similarity to all cards
    sims = linear_kernel(seed_vec, tfidf).flatten()

    df_res = df.copy()
    df_res["similarity"] = sims

    # Exclude seeds and NaN names
    df_res = df_res[~df_res["name"].isin(seed_names) & df_res["name"].notna()]
    df_res = df_res.sort_values("similarity", ascending=False).head(top_n)
    df_res["bucket"] = df_res["type_line"].apply(infer_primary_bucket)

    return df_res


# ----------------------- UI Elements -----------------------

class CardTile(QWidget):
    def __init__(self, card_row: pd.Series, parent=None):
        super().__init__(parent)
        self.card = card_row
        self._build()

    def _build(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)

        # Image
        img_label = QLabel()
        img_label.setFixedSize(QSize(90, 126))
        img_label.setStyleSheet("background: #222; border: 1px solid #444; border-radius: 6px;")
        image_uri = self.card.get("image_uri")
        if isinstance(image_uri, str) and image_uri.startswith("http"):
            try:
                resp = requests.get(image_uri, timeout=10)
                if resp.status_code == 200:
                    pix = QPixmap()
                    pix.loadFromData(resp.content)
                    img_label.setPixmap(pix.scaled(img_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            except Exception:
                pass

        # Text block
        text_box = QVBoxLayout()
        # Mapeo de colores a emojis
        color_map = {
            "R": "🔴",
            "U": "🔵",
            "G": "🟢",
            "W": "⚪",
            "B": "⚫",
            "Colorless": "🩶"
        }

        # Obtener colores de la carta
        colors_str = str(self.card.get("colors", ""))
        colors_list = [c.strip() for c in colors_str.split(",") if c.strip()]

        # Crear la cadena de emojis según los colores
        emojis = " ".join([color_map.get(c, "") for c in colors_list])
        if not emojis:
            emojis = "🩶"  # fallback si no hay color detectado

        # Mostrar nombre con emojis
        name_lb = QLabel(f"<b>{emojis} {self.card.get('name','')}</b>")
        name_lb.setStyleSheet("color: #EEE; font-size: 14px;")
        meta = QLabel(
            f"{self.card.get('type_line','')} • CMC {self.card.get('cmc','?')} • {self.card.get('rarity','')}\n"
            f"Colors: {self.card.get('colors','')} • Set: {self.card.get('set_name','')}"
        )
        meta.setStyleSheet("color:#BBB; font-size:12px;")
        meta.setWordWrap(True)
        oracle = QLabel(nice_text(self.card.get("oracle_text", "")))
        oracle.setStyleSheet("color:#DDD; font-size:12px;")
        oracle.setWordWrap(True)

        text_box.addWidget(name_lb)
        text_box.addWidget(meta)
        text_box.addWidget(oracle)
        text_box.addStretch(1)

        layout.addWidget(img_label)
        layout.addLayout(text_box)

        self.setStyleSheet("""
            QWidget {
                background: #1A1A1A;
                border: 1px solid #2A2A2A;
                border-radius: 10px;
            }
        """)


class ScrollList(QWidget):
    def __init__(self, rows: pd.DataFrame, parent=None):
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0,0,0,0)

        area = QScrollArea()
        area.setWidgetResizable(True)
        content = QWidget()
        v = QVBoxLayout(content)
        v.setContentsMargins(8,8,8,8)
        v.setSpacing(8)

        if rows is None or rows.empty:
            v.addWidget(QLabel("No results"))
        else:
            for _, row in rows.iterrows():
                tile = CardTile(row)
                v.addWidget(tile)

        v.addSpacerItem(QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding))
        area.setWidget(content)
        outer.addWidget(area)


# ----------------------- Main Window -----------------------

class DeckForgeWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DeckForge AI — MTG Standard Recommender")
        self.resize(1200, 800)
        self.setStyleSheet(self._dark_qss())

        self.df: pd.DataFrame | None = None
        self.state: RecommenderState | None = None

        self._build_menu()
        self._build_ui()
        self._load_initial_dataset()

    # ---------- UI Build ----------
    def _build_menu(self):
        menubar = self.menuBar()
        menu_file = menubar.addMenu("&File")
        act_open = QAction("Open CSV...", self)
        act_open.triggered.connect(self._action_open_csv)
        menu_file.addAction(act_open)

        act_fetch = QAction("Fetch/Update from Scryfall", self)
        act_fetch.triggered.connect(self._action_fetch)
        menu_file.addAction(act_fetch)

        menu_file.addSeparator()
        act_exit = QAction("Exit", self)
        act_exit.triggered.connect(self.close)
        menu_file.addAction(act_exit)

        self.status = QStatusBar()
        self.setStatusBar(self.status)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(14, 10, 14, 10)
        root.setSpacing(12)

        # Title
        title = QLabel("<h2>DeckForge AI — Build Standard Decks with 5 Seeds</h2>")
        title.setStyleSheet("color:#EDEDED;")
        root.addWidget(title)

        # Input row
        self.inputs_box = QHBoxLayout()
        self.inputs_box.setSpacing(8)
        self.seed_edits: list[QLineEdit] = []
        for i in range(5):
            ed = QLineEdit()
            ed.setPlaceholderText(f"Seed card {i+1}")
            ed.setMinimumWidth(180)
            self.seed_edits.append(ed)
            self.inputs_box.addWidget(ed)
        root.addLayout(self.inputs_box)

        # Actions
        act_row = QHBoxLayout()
        self.btn_recommend = QPushButton("Recommend 55 Cards")
        self.btn_recommend.clicked.connect(self._on_recommend)
        self.btn_recommend.setEnabled(False)

        self.btn_clear = QPushButton("Clear Seeds")
        self.btn_clear.clicked.connect(self._on_clear)

        act_row.addWidget(self.btn_recommend)
        act_row.addWidget(self.btn_clear)
        act_row.addStretch(1)
        root.addLayout(act_row)

        # Progress
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        root.addWidget(self.progress)

        # Tabs for results
        self.tabs = QTabWidget()
        root.addWidget(self.tabs, 1)

    # ---------- Dataset / Vectorizer ----------
    def _load_initial_dataset(self):
        if os.path.exists(DATA_CSV):
            try:
                self.df = pd.read_csv(DATA_CSV)
                self._post_load_df()
                self.status.showMessage(f"Loaded {len(self.df)} cards from {DATA_CSV}", 5000)
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to load dataset: {e}")
        else:
            self.status.showMessage("No dataset found. Fetch from Scryfall to start.")

    def _post_load_df(self):
        # Autocomplete setup
        names = sorted(list(set(self.df["name"].dropna().tolist())))
        completer = QCompleter(names)
        completer.setCaseSensitivity(Qt.CaseInsensitive)
        completer.setFilterMode(Qt.MatchContains)
        for ed in self.seed_edits:
            ed.setCompleter(completer)
        # Build recommender state
        self.state = build_recommender(self.df)
        self.btn_recommend.setEnabled(True)

    # ---------- Actions ----------
    def _action_open_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open CSV", os.getcwd(), "CSV Files (*.csv)")
        if path:
            try:
                self.df = pd.read_csv(path)
                self._post_load_df()
                self.status.showMessage(f"Loaded {len(self.df)} cards from {os.path.basename(path)}", 5000)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Cannot load CSV: {e}")

    def _action_fetch(self):
        # Start threaded fetcher
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.status.showMessage("Fetching Standard cards from Scryfall…")
        self.fetcher = ScryfallFetcher(self)
        self.fetcher.progress.connect(self.progress.setValue)
        self.fetcher.finished_ok.connect(self._on_fetch_ok)
        self.fetcher.failed.connect(self._on_fetch_fail)
        self.fetcher.start()

    def _on_fetch_ok(self, df: pd.DataFrame):
        try:
            df.to_csv(DATA_CSV, index=False, encoding="utf-8-sig")
            self.df = df
            self._post_load_df()
            self.status.showMessage(f"Fetched and saved {len(df)} cards.")
        finally:
            self.progress.setVisible(False)

    def _on_fetch_fail(self, msg: str):
        self.progress.setVisible(False)
        QMessageBox.critical(self, "Fetch failed", msg)
        self.status.showMessage("Fetch failed")

    def _on_clear(self):
        for ed in self.seed_edits:
            ed.clear()
        self.tabs.clear()

    def _on_recommend(self):
        seeds = [ed.text().strip() for ed in self.seed_edits if ed.text().strip()]
        if len(seeds) < 1:
            QMessageBox.information(self, "Seeds required", "Enter at least 1 seed card (up to 5).")
            return
        try:
            res = recommend_cards(self.state, seeds, RECOMMEND_TOP_N)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            return
        self._render_results(res)

    def _render_results(self, df_res: pd.DataFrame):
        self.tabs.clear()
        # Buckets: Lands, Creatures, Instants, Sorceries, Enchantments, Artifacts, Planeswalkers, Other
        order = [
            "Lands", "Creatures", "Instants", "Sorceries", "Enchantments", "Artifacts", "Planeswalkers", "Other"
        ]
        for bucket in order:
            rows = df_res[df_res["bucket"] == bucket]
            widget = ScrollList(rows)
            self.tabs.addTab(widget, f"{bucket} ({len(rows)})")

    # ---------- Theme ----------
    @staticmethod
    def _dark_qss() -> str:
        return """
        QMainWindow { background-color: #0F0F10; }
        QMenuBar { background: #141416; color: #EEE; }
        QMenuBar::item:selected { background: #1F1F23; }
        QMenu { background: #141416; color: #EEE; }
        QMenu::item:selected { background: #2A2A2E; }
        QLabel { color: #DADADA; }
        QLineEdit {
            background: #1B1B1D; color: #EEE; border: 1px solid #333; border-radius: 8px; padding: 6px 8px;
        }
        QPushButton {
            background: #2C2C31; color: #EEE; border: 1px solid #3A3A3F; border-radius: 10px; padding: 8px 14px;
        }
        QPushButton:hover { background: #3A3A40; }
        QPushButton:pressed { background: #2A2A2F; }
        QTabWidget::pane { border: 1px solid #2A2A2A; top: -1px; }
        QTabBar::tab {
            background: #161618; color: #BBB; padding: 8px 12px; border-top-left-radius: 8px; border-top-right-radius: 8px;
        }
        QTabBar::tab:selected { background: #1F1F23; color: #EEE; }
        QScrollArea { border: none; }
        QStatusBar { color: #AAA; }
        QProgressBar { border: 1px solid #333; border-radius: 6px; background: #1A1A1A; }
        QProgressBar::chunk { background-color: #4B7BEC; }
        """


# ----------------------- Entry -----------------------

def main():
    app = QApplication(sys.argv)
    win = DeckForgeWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
