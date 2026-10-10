"""
SenaGistics - the two ML models.

1. ForecastModel   : LSTM regressor. Predicts tomorrow's fuel/ammo consumption
                     from the last WINDOW days, and doubles as the anomaly
                     detector (large residual vs. prediction = anomaly).
2. IntentClassifier: small Embedding+LSTM text classifier ("micro-LLM" stand-in).
                     Reads an operator's message and classifies its intent.

Both are deliberately small (a few thousand parameters) -- fast to train,
fast to run, and small enough to be deployed on-prem/offline, which is the
whole design point of SenaGistics.
"""

import json
import os
import pickle

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

WINDOW = 14
VOCAB_SIZE = 500
MAX_LEN = 12

MODEL_DIR = "saved_models"
os.makedirs(MODEL_DIR, exist_ok=True)

INTENT_CLASSES = ["STATUS_CHECK", "PROCUREMENT_REQUEST", "ANOMALY_REPORT", "RESUPPLY_ALERT"]

INTENT_TRAINING_DATA = [
    ("what is the current fuel level at depot alpha", "STATUS_CHECK"),
    ("show me today's ammunition stock", "STATUS_CHECK"),
    ("how much diesel is left in bravo depot", "STATUS_CHECK"),
    ("check current consumables inventory", "STATUS_CHECK"),
    ("give me the stock report for this week", "STATUS_CHECK"),
    ("what is today's fuel reading at charlie", "STATUS_CHECK"),
    ("display current ammunition levels for all depots", "STATUS_CHECK"),
    ("how many rounds are currently in storage", "STATUS_CHECK"),
    ("send me the latest consumption summary", "STATUS_CHECK"),
    ("what was yesterday's fuel usage figure", "STATUS_CHECK"),

    ("we need to order more diesel fuel urgently", "PROCUREMENT_REQUEST"),
    ("please raise a purchase order for 5.56mm rounds", "PROCUREMENT_REQUEST"),
    ("request additional ammunition supply for next month", "PROCUREMENT_REQUEST"),
    ("submit a resupply order for depot charlie", "PROCUREMENT_REQUEST"),
    ("we require more fuel before the exercise begins", "PROCUREMENT_REQUEST"),
    ("please arrange procurement of diesel for the convoy", "PROCUREMENT_REQUEST"),
    ("initiate a purchase request for additional rounds", "PROCUREMENT_REQUEST"),
    ("we would like to place an order for more fuel", "PROCUREMENT_REQUEST"),
    ("kindly process a procurement request for ammunition", "PROCUREMENT_REQUEST"),
    ("can you order extra diesel for the training exercise", "PROCUREMENT_REQUEST"),

    ("fuel consumption spiked suddenly today at depot alpha", "ANOMALY_REPORT"),
    ("there is a mismatch in the ammunition count", "ANOMALY_REPORT"),
    ("unexpected drop in recorded diesel stock overnight", "ANOMALY_REPORT"),
    ("possible leakage detected in the fuel tank", "ANOMALY_REPORT"),
    ("consumption today looks abnormally high compared to usual", "ANOMALY_REPORT"),
    ("something is wrong with today's fuel numbers", "ANOMALY_REPORT"),
    ("ammunition usage today is far higher than normal", "ANOMALY_REPORT"),
    ("we suspect a data entry error in the fuel log", "ANOMALY_REPORT"),
    ("today's readings do not match the expected pattern", "ANOMALY_REPORT"),
    ("please verify the sudden irregular spike in stock usage", "ANOMALY_REPORT"),

    ("fuel stock will run out in three days at this rate", "RESUPPLY_ALERT"),
    ("ammunition reserves are below the safety threshold", "RESUPPLY_ALERT"),
    ("depot bravo will be out of diesel by next week", "RESUPPLY_ALERT"),
    ("critical low stock warning for 7.62mm rounds", "RESUPPLY_ALERT"),
    ("projected depletion date for fuel is in five days", "RESUPPLY_ALERT"),
    ("fuel levels are approaching the critical minimum", "RESUPPLY_ALERT"),
    ("stock will be depleted very soon at the current rate", "RESUPPLY_ALERT"),
    ("we are running low on ammunition reserves", "RESUPPLY_ALERT"),
    ("diesel supply is nearly exhausted at this depot", "RESUPPLY_ALERT"),
    ("current stock is below the minimum safety level", "RESUPPLY_ALERT"),
]


# ---------------------------------------------------------------------------
# Forecasting model
# ---------------------------------------------------------------------------

def make_windows(arr, window):
    X, y = [], []
    for i in range(len(arr) - window):
        X.append(arr[i:i + window])
        y.append(arr[i + window])
    return np.array(X), np.array(y)


def build_forecast_model():
    model = models.Sequential([
        layers.Input(shape=(WINDOW, 1)),
        layers.LSTM(32, return_sequences=False),
        layers.Dense(16, activation="relu"),
        layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def train_forecast_model(series: np.ndarray, epochs=40):
    mean, std = series.mean(), series.std()
    series_norm = (series - mean) / std

    X, y = make_windows(series_norm, WINDOW)
    X = X[..., np.newaxis]

    split = int(len(X) * 0.85)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = build_forecast_model()
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                         epochs=epochs, batch_size=8, verbose=0)

    preds_norm = model.predict(X_test, verbose=0).flatten()
    preds = preds_norm * std + mean
    actual = y_test * std + mean
    mae = float(np.mean(np.abs(actual - preds)))

    stats = {"mean": float(mean), "std": float(std), "mae": mae}
    return model, stats, history


def save_forecast_model(model, stats, path_prefix=os.path.join(MODEL_DIR, "forecast")):
    model.save(path_prefix + ".keras")
    with open(path_prefix + "_stats.json", "w") as f:
        json.dump(stats, f)


def load_forecast_model(path_prefix=os.path.join(MODEL_DIR, "forecast")):
    model = tf.keras.models.load_model(path_prefix + ".keras")
    with open(path_prefix + "_stats.json") as f:
        stats = json.load(f)
    return model, stats


def predict_next_day(model, stats, recent_window: np.ndarray) -> float:
    norm = (recent_window - stats["mean"]) / stats["std"]
    norm = norm.reshape(1, WINDOW, 1)
    pred_norm = model.predict(norm, verbose=0)[0][0]
    return float(pred_norm * stats["std"] + stats["mean"])


def is_anomalous(actual_value: float, predicted_value: float, stats: dict, k: float = 2.0) -> bool:
    return abs(actual_value - predicted_value) > k * stats["mae"]


# ---------------------------------------------------------------------------
# Intent classifier ("micro-LLM")
# ---------------------------------------------------------------------------

def build_intent_model():
    model = models.Sequential([
        layers.Input(shape=(MAX_LEN,)),
        layers.Embedding(input_dim=VOCAB_SIZE, output_dim=16),
        layers.LSTM(24),
        layers.Dense(16, activation="relu"),
        layers.Dense(len(INTENT_CLASSES), activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def train_intent_model(epochs=150):
    texts = [t for t, _ in INTENT_TRAINING_DATA]
    labels = [l for _, l in INTENT_TRAINING_DATA]
    label_to_id = {c: i for i, c in enumerate(INTENT_CLASSES)}
    y = np.array([label_to_id[l] for l in labels])

    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=MAX_LEN, padding="post")

    model = build_intent_model()
    history = model.fit(padded, y, epochs=epochs, verbose=0)

    return model, tokenizer, history


def save_intent_model(model, tokenizer, path_prefix=os.path.join(MODEL_DIR, "intent")):
    model.save(path_prefix + ".keras")
    with open(path_prefix + "_tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)


def load_intent_model(path_prefix=os.path.join(MODEL_DIR, "intent")):
    model = tf.keras.models.load_model(path_prefix + ".keras")
    with open(path_prefix + "_tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer


def classify_intent(model, tokenizer, message: str):
    seq = tokenizer.texts_to_sequences([message])
    pad = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=MAX_LEN, padding="post")
    probs = model.predict(pad, verbose=0)[0]
    idx = int(np.argmax(probs))
    return INTENT_CLASSES[idx], float(probs[idx])


# ---------------------------------------------------------------------------
# Train-and-save-everything entry point
# ---------------------------------------------------------------------------

def train_and_save_all(verbose=True):
    from data_sim import simulate_all_depots

    data = simulate_all_depots()
    series = data[data.depot == "Depot-Alpha"]["fuel_litres"].values.astype("float32")

    if verbose:
        print("Training forecast model...")
    fmodel, fstats, _ = train_forecast_model(series)
    save_forecast_model(fmodel, fstats)
    if verbose:
        print(f"  Forecast model saved. Test MAE: {fstats['mae']:.1f} litres/day")

    if verbose:
        print("Training intent classifier...")
    imodel, tok, ihist = train_intent_model()
    save_intent_model(imodel, tok)
    if verbose:
        print(f"  Intent model saved. Final train accuracy: {ihist.history['accuracy'][-1]*100:.1f}%")

    data.to_csv("depot_consumption.csv", index=False)
    if verbose:
        print("Synthetic data saved to depot_consumption.csv")


if __name__ == "__main__":
    train_and_save_all()
