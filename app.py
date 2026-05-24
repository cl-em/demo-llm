
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
    Génère des tokens de la même manière que `chat_avec_llm_sans_repetition`
    Retourne une liste de dicts { token, alternatives, prob }.
    """
    # Construire le prompt et obtenir les ids d'entrée
    BOS_TOKEN = "<s>"
    INST_START = "[INST]"
    INST_END = "[/INST]"

    instruction = seed_text
    prompt = f"{BOS_TOKEN}{INST_START} {instruction} {INST_END}"

    input_ids = tokenizer(
        prompt,
        return_tensors="np",
        truncation=True,
        add_special_tokens=False
    )["input_ids"][0].tolist()

    generated_tokens = input_ids.copy()
    maxlen = 100
    length = MAX_TOKENS if MAX_TOKENS is not None else 40

    for _ in range(length):
        current_tokens = generated_tokens[-maxlen:]
        x_pad = tf.keras.preprocessing.sequence.pad_sequences(
            [current_tokens], maxlen=maxlen, padding='post'
        )

        preds = model.predict(x_pad, verbose=0)
        last_idx = min(len(current_tokens) - 1, maxlen - 1)
        next_token_probs = preds[0, last_idx, :].astype("float64")

        # Passage en logits
        next_token_logits = np.log(next_token_probs + 1e-7)

        # Pénalité de répétition : on diminue la probabilité des tokens déjà vus
        for token in set(generated_tokens):
            if token < len(next_token_logits):
                if next_token_logits[token] < 0:
                    next_token_logits[token] *= repetition_penalty
                else:
                    next_token_logits[token] /= repetition_penalty

        # Température
        next_token_logits /= max(temperature, 1e-8)

        # Softmax
        next_token_logits -= next_token_logits.max()
        exp_preds = np.exp(next_token_logits)
        next_token_probs = exp_preds / np.sum(exp_preds)

        next_token = np.random.choice(len(next_token_probs), p=next_token_probs)

        if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
            if next_token == tokenizer.eos_token_id:
                break

        generated_tokens.append(int(next_token))

    # Construire la sortie sous forme de tokens individuels (compatible API)
    results = []
    # On transforme uniquement les nouveaux tokens générés (après l'amorce)
    for tid in generated_tokens[len(input_ids):]:
        try:
            if is_hf_tokenizer:
                word = tokenizer.convert_ids_to_tokens([int(tid)])[0]
            else:
                word = index_word.get(int(tid), str(int(tid)))
        except Exception:
            word = index_word.get(int(tid), "<unk>")

        word = normalize_token(word)
        results.append({
            "token": " " + word,
            "alternatives": [],
            "prob": 0.0
        })

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
