"""
Backend Flask — Démonstration génération mot par mot avec modèle Keras LSTM/GRU
---------------------------------------------------------------------------
Structure attendue du modèle :
  - Entrée  : séquence d'indices de mots  (shape: [1, seq_len])
  - Sortie  : distribution de probabilité sur le vocabulaire  (shape: [1, vocab_size])

Fichiers nécessaires (même dossier que app.py) :
  - model.keras   (ou model.h5)  → votre modèle entraîné
  - tokenizer.pkl               → le Tokenizer Keras sérialisé avec pickle
                                  (celui utilisé à l'entraînement)

Lancer le serveur :
  pip install flask flask-cors tensorflow numpy
  python app.py
"""

import os
import pickle
import time
import numpy as np

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

import tensorflow as tf
from tensorflow import keras
try:
    # Prefer AutoTokenizer so user can specify "camembert-base" or a local tokenizer name
    from transformers import AutoTokenizer
except Exception:
    AutoTokenizer = None

# ---------------------------------------------------------------------------
# Configuration — adaptez ces valeurs à votre modèle
# ---------------------------------------------------------------------------

MODEL_PATH     = "llm_512_epoch_05.keras"      # chemin vers le fichier .keras ou .h5
TOKENIZER_PATH = "tokenizer.pkl"    # chemin vers le tokenizer sérialisé
SEQ_LEN        = 20                 # longueur de séquence d'entrée du modèle
MAX_TOKENS     = 512                # nombre maximum de mots générés
TOP_K          = 5                  # nombre d'alternatives retournées par token
# Nom du tokenizer HuggingFace à charger directement (vide pour désactiver)
HF_TOKENIZER_NAME = ""             # ex: "camembert-base" ou "facebook/camembert-base"

# ---------------------------------------------------------------------------
# Chargement du modèle et du tokenizer
# ---------------------------------------------------------------------------

print("Chargement du modèle Keras…")
model = keras.models.load_model(MODEL_PATH)
model.summary()

print("Chargement du tokenizer…")
is_hf_tokenizer = False
tokenizer = None
index_word = {}

# Priorité : si HF_TOKENIZER_NAME est défini, charger directement le tokenizer HF
if HF_TOKENIZER_NAME:
    if AutoTokenizer is None:
        raise RuntimeError("transformers non installé — installez 'transformers' pour utiliser un tokenizer CamemBERT")
    print("Chargement du tokenizer HuggingFace (option explicite):", HF_TOKENIZER_NAME)
    tokenizer = AutoTokenizer.from_pretrained(HF_TOKENIZER_NAME)
    is_hf_tokenizer = True

    try:
        vocab_size = tokenizer.vocab_size
    except Exception:
        vocab_size = getattr(tokenizer, "get_vocab", lambda: {})()
        vocab_size = len(vocab_size) if isinstance(vocab_size, dict) else 0

    index_word = {}
    if vocab_size and hasattr(tokenizer, 'convert_ids_to_tokens'):
        ids = list(range(vocab_size))
        toks = tokenizer.convert_ids_to_tokens(ids)
        for i, t in enumerate(toks):
            index_word[i] = t

    print(f"Tokenizer HF explicite chargé (vocab_size={vocab_size})")
else:
    # Si le chemin se termine par .pkl, on tente d'abord le Tokenizer Keras sérialisé
    if TOKENIZER_PATH and TOKENIZER_PATH.lower().endswith(".pkl"):
        try:
            with open(TOKENIZER_PATH, "rb") as f:
                tokenizer = pickle.load(f)
            # Index inversé : indice → mot (Keras Tokenizer)
            index_word = getattr(tokenizer, "index_word", {})
            print("Tokenizer Keras chargé depuis", TOKENIZER_PATH)
        except Exception as e:
            print("Échec chargement tokenizer Keras:", e)

    # Si on n'a pas de tokenizer Keras, on essaye de charger un tokenizer HuggingFace
    if tokenizer is None:
        if AutoTokenizer is None:
            raise RuntimeError("transformers non installé — installez 'transformers' pour utiliser un tokenizer CamemBERT")
        # Permettre d'utiliser des alias comme 'camembert' ou 'camembert-base'
        hf_name = TOKENIZER_PATH if TOKENIZER_PATH and not TOKENIZER_PATH.lower().endswith('.pkl') else 'camembert-base'
        print("Chargement du tokenizer HuggingFace:", hf_name)
        tokenizer = AutoTokenizer.from_pretrained(hf_name)
        is_hf_tokenizer = True

        # Construire un index inversé simple id -> token
        try:
            vocab_size = tokenizer.vocab_size
        except Exception:
            vocab_size = getattr(tokenizer, "get_vocab", lambda: {})()
            vocab_size = len(vocab_size) if isinstance(vocab_size, dict) else 0

        # Convert ids to tokens for the possible id range
        index_word = {}
        if vocab_size and hasattr(tokenizer, 'convert_ids_to_tokens'):
            ids = list(range(vocab_size))
            toks = tokenizer.convert_ids_to_tokens(ids)
            for i, t in enumerate(toks):
                index_word[i] = t

        print(f"Tokenizer HF chargé (vocab_size={vocab_size})")

# ---------------------------------------------------------------------------
# Fonctions utilitaires
# ---------------------------------------------------------------------------

def sample_with_temperature(logits: np.ndarray, temperature: float) -> int:
    """Echantillonne un indice depuis les logits avec une température."""
    if temperature <= 0.01:
        return int(np.argmax(logits))
    logits = np.asarray(logits, dtype=np.float64)
    logits = np.log(logits + 1e-10) / temperature
    logits -= logits.max()                      # stabilisation numérique
    probs = np.exp(logits)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


def top_k_alternatives(probs: np.ndarray, chosen_idx: int, k: int = 5) -> list:
    """Retourne les k mots les plus probables (hors mot choisi)."""
    top_indices = np.argsort(probs)[::-1]
    alts = []
    for i in top_indices:
        if i == chosen_idx:
            continue
        word = index_word.get(int(i), "<unk>")
        if word and word != "<unk>":
            alts.append(word)
        if len(alts) >= k:
            break
    return alts


def text_to_sequence(text: str) -> list:
    """Convertit un texte en liste d'indices via le tokenizer."""
    if is_hf_tokenizer:
        # Utilise le tokenizer HuggingFace pour obtenir des ids de tokens
        # Ne pas ajouter de tokens spéciaux pour rester compatible avec le modèle
        ids = tokenizer.encode(text, add_special_tokens=False)
        return list(ids)
    else:
        seqs = tokenizer.texts_to_sequences([text])
        return seqs[0] if seqs else []


def generate_all_tokens(seed_text: str, temperature: float = 0.7) -> list:
    """
    Génère MAX_TOKENS mots à partir d'un texte amorce.
    Retourne une liste de dicts  { token, alternatives, probs }.
    """
    sequence = text_to_sequence(seed_text)
    results  = []
    t0 = time.time()

    for _ in range(MAX_TOKENS):
        # Prépare la fenêtre d'entrée (padding à gauche si nécessaire)
        padded = sequence[-SEQ_LEN:] if len(sequence) >= SEQ_LEN else \
                 [0] * (SEQ_LEN - len(sequence)) + sequence
        x = np.array([padded])                     # shape (1, SEQ_LEN)

        # Inférence
        preds = model.predict(x, verbose=0)[0]     # shape (vocab_size,)

        # Echantillonnage
        chosen_idx  = sample_with_temperature(preds, temperature)
        chosen_word = index_word.get(chosen_idx, "<unk>")

        # Alternatives
        alts = top_k_alternatives(preds, chosen_idx, k=TOP_K)

        results.append({
            "token":        " " + chosen_word,     # espace devant comme GPT
            "alternatives": [" " + a for a in alts],
            "prob":         float(preds[chosen_idx])
        })

        # Arrêt sur token de fin (si votre modèle en a un)
        if chosen_word in ("<eos>", "<end>", "."):
            break

        sequence.append(chosen_idx)

    elapsed = time.time() - t0
    print(f"Génération : {len(results)} tokens en {elapsed:.2f}s "
          f"({len(results)/elapsed:.1f} tok/s)")
    return results

# ---------------------------------------------------------------------------
# Application Flask
# ---------------------------------------------------------------------------

app = Flask(__name__)
CORS(app)   # autorise les requêtes depuis localhost (front HTML)


@app.route("/", methods=["GET"])
def root():
    """Servir le fichier index.html depuis le répertoire du script."""
    base_dir = os.path.abspath(os.path.dirname(__file__))
    return send_from_directory(base_dir, "index.html")


@app.route("/generate", methods=["POST"])
def generate():
    """
    Corps JSON attendu :
    {
      "system"      : "...",   # consignes (optionnel, ajouté au seed)
      "history"     : "...",   # historique (optionnel)
      "user"        : "...",   # demande actuelle (obligatoire)
      "temperature" : 0.7,     # float entre 0 et 1.5
      "tokens"      : []       # ignoré côté serveur (état géré par le front)
    }

    Réponse JSON :
    {
      "tokens"     : [ { "token": " mot", "alternatives": [" alt1", ...] }, … ],
      "max_tokens" : 80
    }
    """
    data = request.get_json(force=True)

    system      = data.get("system", "").strip()
    history     = data.get("history", "").strip()
    user        = data.get("user", "").strip()
    temperature = float(data.get("temperature", 0.7))

    if not user:
        return jsonify({"error": "Le champ 'user' est obligatoire."}), 400

    # Construit le texte d'amorce en concaténant les champs disponibles
    seed_parts = [p for p in [system, history, user] if p]
    seed_text  = " ".join(seed_parts)

    tokens = generate_all_tokens(seed_text, temperature=temperature)

    return jsonify({
        "tokens":     tokens,
        "max_tokens": MAX_TOKENS
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": MODEL_PATH})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
