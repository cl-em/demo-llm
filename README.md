# Démonstration de génération de texte token par token — Modèle Keras LSTM/GRU

Cette application web propose une interface de démonstration interactive permettant de visualiser le processus de génération de texte mot par mot à l'aide d'un modèle séquentiel Keras (LSTM ou GRU). Elle expose le comportement du modèle en temps réel : construction progressive de la réponse, alternatives possibles à chaque étape et métriques de génération.

---

## Prérequis

- Python 3.10 ou supérieur (vérifier la compatibilité avec la version de TensorFlow utilisée)
- Un environnement virtuel est fortement recommandé

---

## Installation des dépendances

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note GPU :** pour bénéficier de l'accélération matérielle, installer la variante TensorFlow adaptée à votre configuration GPU en suivant la [documentation officielle de TensorFlow](https://www.tensorflow.org/install/pip).

---

## Lancement de l'application

```bash
python app.py
```

L'interface est ensuite accessible à l'adresse suivante :

```
http://127.0.0.1:5000
```

---

## Configuration du modèle

Le modèle utilisé par l'application est spécifié par la variable `MODEL_PATH` dans `app.py`. Pour substituer un modèle, modifier cette variable en indiquant le chemin vers le fichier `.keras` souhaité.

Plusieurs modèles entraînés sont disponibles dans le répertoire du projet. Le modèle offrant les meilleures performances est `llm_epoch_10_better.keras`.

Pour obtenir des résultats cohérents et de qualité, il est conseillé de formuler une **instruction explicite** en tant que consigne système (champ *Consignes données au modèle*) avant de soumettre une demande.

---

## Structure du projet

```
keras-demo/
├── app.py              # Serveur Flask — chargement du modèle et endpoint /generate
├── index.html          # Interface web de démonstration
├── requirements.txt    # Dépendances Python
├── tokenizer.pkl       # Tokenizer sérialisé (utilisé à l'entraînement)
└── *.keras             # Fichiers de modèles Keras entraînés
```

---

## Format de l'API

Le serveur expose un endpoint `POST /generate` qui accepte la structure JSON suivante :

```json
{
  "system":      "Consignes données au modèle (optionnel)",
  "history":     "Historique de conversation (optionnel)",
  "user":        "Demande de l'utilisateur (obligatoire)",
  "temperature": 0.7
}
```

La réponse retournée est de la forme :

```json
{
  "tokens": [
    { "token": " mot", "alternatives": [" alt1", " alt2"] }
  ],
  "max_tokens": 80
}
```

Un endpoint `GET /health` permet de vérifier la disponibilité du serveur.