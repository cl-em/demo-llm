import os
from collections import Counter

import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

import tensorflow as tf
from tensorflow import keras
from transformers import AutoTokenizer

tf.keras.backend.set_floatx('float32')

MODEL_PATH = "llm_model_latest.keras"
MAX_TOKENS = 512
TOP_K = 5
GEN_LENGTH = 95
MAXLEN = 100

DEFAULT_TEMPERATURE = 0.8
DEFAULT_REPETITION_PENALTY = 1.3
DEFAULT_NO_REPEAT_NGRAM = 3
DEFAULT_FREQ_PENALTY = 0.4
DEFAULT_TOP_P = 0.92
DEFAULT_TOP_K_SAMPLING = 50
DEFAULT_PENALTY_WINDOW = 64


print("Chargement du modele Keras...")
model = keras.models.load_model(MODEL_PATH)
model.summary()

try:
    model_input_shape = model.input_shape
    if isinstance(model_input_shape, (list, tuple)) and len(model_input_shape) >= 2:
        inferred = int(model_input_shape[1])
        if inferred and inferred > 0:
            MAXLEN = inferred
            print(f"MAXLEN ajuste a {MAXLEN} d'apres le modele charge.")
except Exception:
    pass


print("Chargement du tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("camembert-base")
vocab_size = len(tokenizer)
print(f"Taille du vocabulaire: {vocab_size} tokens.")

index_word = {}
try:
    vocab = tokenizer.get_vocab()
    for tok, idx in vocab.items():
        index_word[int(idx)] = tokenizer.convert_ids_to_tokens([int(idx)])[0]
except Exception:
    index_word = {}


def get_special_ids():
    ids = set()
    for name in ("pad_token_id", "unk_token_id", "bos_token_id", "eos_token_id",
                 "sep_token_id", "cls_token_id", "mask_token_id"):
        v = getattr(tokenizer, name, None)
        if v is not None:
            ids.add(int(v))
    return ids


SPECIAL_IDS = get_special_ids()
EOS_ID = getattr(tokenizer, "eos_token_id", None)
BOS_ID = getattr(tokenizer, "bos_token_id", None)


def normalize_token(tok):
    if tok is None:
        return "<unk>"
    return tok.replace('\u2581', ' ').replace('\u0120', ' ')


def apply_repetition_penalty(logits, generated_tokens, penalty, window):
    if penalty == 1.0 or not generated_tokens:
        return logits
    recent = set(generated_tokens[-window:])
    for tok in recent:
        if 0 <= tok < len(logits):
            if logits[tok] > 0:
                logits[tok] = logits[tok] / penalty
            else:
                logits[tok] = logits[tok] * penalty
    return logits


def apply_frequency_penalty(logits, generated_tokens, alpha, window):
    if alpha <= 0.0 or not generated_tokens:
        return logits
    counts = Counter(generated_tokens[-window:])
    for tok, c in counts.items():
        if 0 <= tok < len(logits):
            logits[tok] -= alpha * c
    return logits


def block_repeated_ngrams(logits, generated_tokens, ngram_size):
    if ngram_size <= 0 or len(generated_tokens) < ngram_size:
        return logits
    prefix = tuple(generated_tokens[-(ngram_size - 1):]) if ngram_size > 1 else tuple()
    banned = set()
    for i in range(len(generated_tokens) - ngram_size + 1):
        ngram = tuple(generated_tokens[i:i + ngram_size])
        if ngram[:-1] == prefix:
            banned.add(ngram[-1])
    for tok in banned:
        if 0 <= tok < len(logits):
            logits[tok] = -1e10
    return logits


def mask_special_tokens(logits, allow_eos=True):
    for tok in SPECIAL_IDS:
        if allow_eos and EOS_ID is not None and tok == EOS_ID:
            continue
        if 0 <= tok < len(logits):
            logits[tok] = -1e10
    return logits


def top_k_top_p_filter(logits, top_k, top_p):
    logits = logits.copy()
    if top_k is not None and top_k > 0:
        k = min(top_k, logits.shape[-1])
        threshold = np.partition(logits, -k)[-k]
        logits[logits < threshold] = -1e10

    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_idx = np.argsort(logits)[::-1]
        sorted_logits = logits[sorted_idx]
        max_logit = np.max(sorted_logits)
        exp_logits = np.exp(sorted_logits - max_logit)
        probs = exp_logits / np.sum(exp_logits)
        cumulative = np.cumsum(probs)
        cutoff = np.searchsorted(cumulative, top_p) + 1
        to_remove = sorted_idx[cutoff:]
        logits[to_remove] = -1e10

    return logits


def softmax_stable(logits):
    logits = logits.astype('float64')
    max_logit = np.max(logits)
    if not np.isfinite(max_logit):
        max_logit = 0.0
    exp_logits = np.exp(logits - max_logit)
    s = np.sum(exp_logits)
    if s <= 0 or not np.isfinite(s):
        probs = np.zeros_like(exp_logits)
        probs[int(np.argmax(logits))] = 1.0
        return probs
    return exp_logits / s


def sample_from_logits(logits, temperature):
    if temperature <= 0:
        return int(np.argmax(logits))
    scaled = logits / max(temperature, 1e-6)
    probs = softmax_stable(scaled)
    try:
        return int(np.random.choice(len(probs), p=probs))
    except ValueError:
        return int(np.argmax(probs))


def top_k_alternatives(probs, chosen_idx, k=5):
    top_indices = np.argsort(probs)[::-1]
    alts = []
    for i in top_indices:
        i = int(i)
        if i == chosen_idx or i in SPECIAL_IDS:
            continue
        word = index_word.get(i, "<unk>")
        if word and word != "<unk>":
            alts.append(normalize_token(word))
        if len(alts) >= k:
            break
    return alts


def generate_all_tokens(seed_text,
                        temperature=DEFAULT_TEMPERATURE,
                        repetition_penalty=DEFAULT_REPETITION_PENALTY,
                        no_repeat_ngram_size=DEFAULT_NO_REPEAT_NGRAM,
                        freq_penalty=DEFAULT_FREQ_PENALTY,
                        top_p=DEFAULT_TOP_P,
                        top_k=DEFAULT_TOP_K_SAMPLING,
                        penalty_window=DEFAULT_PENALTY_WINDOW,
                        max_new_tokens=GEN_LENGTH):

    input_ids = tokenizer(seed_text, return_tensors="np",
                          add_special_tokens=False)["input_ids"][0].tolist()

    bos = [BOS_ID] if BOS_ID is not None else []
    generated_tokens = bos + list(input_ids)
    generated_steps = []

    for step_num in range(max_new_tokens):
        current_tokens = generated_tokens[-MAXLEN:]
        x_pad = tf.keras.preprocessing.sequence.pad_sequences(
            [current_tokens], maxlen=MAXLEN, padding='post'
        )

        preds = model.predict(x_pad, verbose=0)
        last_idx = min(len(current_tokens) - 1, MAXLEN - 1)
        raw = preds[0, last_idx, :].astype("float64")

        if np.allclose(np.sum(raw), 1.0, atol=1e-3) and np.all(raw >= 0):
            logits = np.log(np.clip(raw, 1e-12, 1.0))
        else:
            logits = raw

        logits = mask_special_tokens(logits, allow_eos=True)
        logits = apply_repetition_penalty(logits, generated_tokens,
                                          repetition_penalty, penalty_window)
        logits = apply_frequency_penalty(logits, generated_tokens,
                                         freq_penalty, penalty_window)
        logits = block_repeated_ngrams(logits, generated_tokens, no_repeat_ngram_size)

        filtered = top_k_top_p_filter(logits, top_k, top_p)
        next_token = sample_from_logits(filtered, temperature)

        if EOS_ID is not None and next_token == EOS_ID:
            print(f"EOS detecte au step {step_num}, arret.")
            break

        full_probs = softmax_stable(logits)
        prob = float(full_probs[next_token]) if next_token < len(full_probs) else 0.0
        alts = top_k_alternatives(full_probs, next_token, k=TOP_K)

        generated_tokens.append(int(next_token))
        generated_steps.append({"id": int(next_token), "alternatives": alts, "prob": prob})

    print(f"Generation terminee. Tokens generes: {len(generated_steps)}")

    try:
        decoded_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        print(f"Texte decode: {decoded_text}")
    except Exception as e:
        print(f"Erreur decodage: {e}")

    results = []
    for step in generated_steps:
        tid = step["id"]
        try:
            word = tokenizer.convert_ids_to_tokens([int(tid)])[0]
        except Exception:
            word = index_word.get(int(tid), "<unk>")
        word = normalize_token(word)
        alts = [" " + a for a in step.get("alternatives", [])]
        results.append({
            "token": " " + word,
            "alternatives": alts,
            "prob": float(step.get("prob", 0.0))
        })
    return results


app = Flask(__name__)
CORS(app)


@app.route("/", methods=["GET"])
def root():
    base_dir = os.path.abspath(os.path.dirname(__file__))
    return send_from_directory(base_dir, "index.html")


@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json(force=True)

    user = (data.get("user") or "").strip()
    if not user:
        return jsonify({"error": "Le champ 'user' est obligatoire."}), 400

    temperature = float(data.get("temperature", DEFAULT_TEMPERATURE))
    repetition_penalty = float(data.get("repetition_penalty", DEFAULT_REPETITION_PENALTY))
    no_repeat_ngram_size = int(data.get("no_repeat_ngram_size", DEFAULT_NO_REPEAT_NGRAM))
    freq_penalty = float(data.get("freq_penalty", DEFAULT_FREQ_PENALTY))
    top_p = float(data.get("top_p", DEFAULT_TOP_P))
    top_k = int(data.get("top_k", DEFAULT_TOP_K_SAMPLING))
    penalty_window = int(data.get("penalty_window", DEFAULT_PENALTY_WINDOW))

    seed_text = user

    tokens = generate_all_tokens(
        seed_text,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        freq_penalty=freq_penalty,
        top_p=top_p,
        top_k=top_k,
        penalty_window=penalty_window,
    )

    return jsonify({"tokens": tokens, "max_tokens": MAX_TOKENS})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": MODEL_PATH})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)