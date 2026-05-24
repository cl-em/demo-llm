
import os
import pickle
import time
import numpy as np

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

import tensorflow as tf
from tensorflow import keras
from transformers import AutoTokenizer



MODEL_PATH     = "llm_epoch_10.keras"      # chemin vers le fichier .keras ou .h5
SEQ_LEN        = 20                 # longueur de séquence d'entrée du modèle
MAX_TOKENS     = 512                # nombre maximum de mots générés
TOP_K          = 5                  # nombre d'alternatives retournées par token




print("Chargement du modèle Keras…")
model = keras.models.load_model(MODEL_PATH)
model.summary()

# Adapter la longueur de séquence au modèle chargé (évite la mismatch shape)
try:
    model_input_shape = model.input_shape
    if isinstance(model_input_shape, (list, tuple)) and len(model_input_shape) >= 2:
        SEQ_LEN = int(model_input_shape[1])
        print(f"SEQ_LEN ajusté à {SEQ_LEN} d'après le modèle chargé.")
except Exception:
    pass

print("Chargement du tokenizer…")
is_hf_tokenizer = False
tokenizer = None
index_word = {}


tokenizer = AutoTokenizer.from_pretrained("camembert-base")
is_hf_tokenizer = True

# Construire la table id -> token pour pouvoir décoder les indices produits
index_word = {}
try:
    # get_vocab renvoie {token: id}
    vocab = tokenizer.get_vocab()
    for tok, idx in vocab.items():
        # convert_ids_to_tokens gère les marqueurs SentencePiece/BPE (ex: '▁')
        index_word[int(idx)] = tokenizer.convert_ids_to_tokens([int(idx)])[0]
except Exception:
    index_word = {}


def normalize_token(tok: str) -> str:
    """Transforme un token HF en forme lisible (remplace les marqueurs par un espace)."""
    if tok is None:
        return "<unk>"
    # remplace les marqueurs courants de tokenisation par un espace
    return tok.replace('▁', ' ').replace('Ġ', ' ')

def sample_with_temperature(logits: np.ndarray, temperature: float) -> int:
    """Echantillonne un indice depuis les logits avec une température."""
    if temperature <= 0.01:
        return int(np.argmax(logits))
    logits = np.asarray(logits, dtype=np.float64)
    logits = np.log(logits + 1e-10) / temperature
    logits -= logits.max()                  
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
            alts.append(normalize_token(word))
        if len(alts) >= k:
            break
    return alts


def text_to_sequence(text: str) -> list:
    """Convertit un texte en liste d'indices via le tokenizer."""
    if is_hf_tokenizer:
        ids = tokenizer.encode(text, add_special_tokens=False)
        return list(ids)
    else:
        seqs = tokenizer.texts_to_sequences([text])
        return seqs[0] if seqs else []


def generate_all_tokens(seed_text: str, temperature: float = 0.7, repetition_penalty: float = 1.0) -> list:
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
        preds = model.predict(x, verbose=0)[0]     # shape possible (SEQ_LEN, vocab_size)
        # Si le modèle renvoie une distribution pour chaque position, on prend la
        # distribution du dernier pas de temps (prédiction du token suivant).
        preds = np.asarray(preds)
        if preds.ndim == 2:
            preds = preds[-1]
        preds = preds.ravel()

        # Appliquer une pénalité de répétition : réduire la probabilité
        # des tokens déjà présents dans la séquence.
        if repetition_penalty is not None and repetition_penalty > 1.0:
            try:
                seen = set(sequence)
                for sid in seen:
                    if 0 <= int(sid) < preds.shape[0]:
                        preds[int(sid)] = preds[int(sid)] / float(repetition_penalty)
            except Exception:
                pass

        # Echantillonnage
        chosen_idx  = sample_with_temperature(preds, temperature)
        chosen_word = normalize_token(index_word.get(chosen_idx, "<unk>"))

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


app = Flask(__name__)
CORS(app)   


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
    repetition_penalty = float(data.get("repetition_penalty", 1.0))

    if not user:
        return jsonify({"error": "Le champ 'user' est obligatoire."}), 400





    seed_text = "<s>[INST]"


    if system:
        seed_text += f" Consigne : {system} "

    if user:
        seed_text += f" Consigne : {user} "
    
    seed_text += "[/INST]"

   

    tokens = generate_all_tokens(seed_text, temperature=temperature,
                                  repetition_penalty=repetition_penalty)

    return jsonify({
        "tokens":     tokens,
        "max_tokens": MAX_TOKENS
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": MODEL_PATH})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
