"""
backend.py
==========
Wildfire "Major Incident" Risk Prediction — Backend / ML Engine

Kya karta hai:
    1. California_Fire_Incidents.csv load karta hai
    2. Features engineer karta hai (date, county, resources, damage stats)
    3. Char models train karta hai: SVM, Random Forest, ANN, CNN (CNN optional,
       sirf tab agar TensorFlow installed ho — warna baqi 3 models normally chalte hain)
    4. Har model ka Accuracy, Precision, Recall, F1-score, Confusion Matrix nikalta hai
    5. Naye / custom incident input par prediction bhi deta hai

Standalone chalane ke liye (terminal me):
    python backend.py

Ye file khud mukammal hai — Streamlit app (app.py) isi file ke functions
import karke use karta hai, taake training logic sirf aik jagah likha ho.
"""

import os
import json
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# ---------------------------------------------------------------------------
# CNN (TensorFlow) is OPTIONAL. Agar deployment environment me TensorFlow
# install/available nahi hota to app crash nahi karega — sirf CNN skip ho
# jayega aur SVM / Random Forest / ANN normally kaam karte rahenge.
# ---------------------------------------------------------------------------
try:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import tensorflow as tf
    from tensorflow.keras import layers, models as keras_models

    tf.get_logger().setLevel("ERROR")
    TENSORFLOW_AVAILABLE = True
except Exception:
    TENSORFLOW_AVAILABLE = False


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_FILENAME = "California_Fire_Incidents.csv"
MODEL_DIR = "models"
TOP_N_COUNTIES = 12
RANDOM_STATE = 42
TEST_SIZE = 0.2

TARGET = "MajorIncident"

NUMERIC_FEATURES = [
    "Latitude",
    "Longitude",
    "AcresBurned",
    "PercentContained",
    "CrewsInvolved",
    "Engines",
    "Dozers",
    "Helicopters",
    "AirTankers",
    "WaterTenders",
    "PersonnelInvolved",
    "StructuresThreatened",
    "StructuresDestroyed",
    "StructuresDamaged",
    "Injuries",
    "Fatalities",
    "Month",
]

FILL_ZERO_COLS = [
    "AirTankers",
    "CrewsInvolved",
    "Dozers",
    "Engines",
    "Fatalities",
    "Helicopters",
    "Injuries",
    "PersonnelInvolved",
    "StructuresDamaged",
    "StructuresDestroyed",
    "StructuresThreatened",
    "WaterTenders",
]

MODEL_DISPLAY_NAMES = ["SVM", "Random Forest", "ANN", "CNN"]


# ---------------------------------------------------------------------------
# Data loading & feature engineering
# ---------------------------------------------------------------------------
def find_data_path():
    """Repo root, script folder, ya current working dir — jahan bhi CSV mile."""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        DATA_FILENAME,
        os.path.join(here, DATA_FILENAME),
        os.path.join(os.getcwd(), DATA_FILENAME),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(
        f"'{DATA_FILENAME}' nahi mili. Is file ko backend.py / app.py ke "
        f"sath usi GitHub repo folder me rakhein."
    )


def load_raw_data(path=None):
    path = path or find_data_path()
    df = pd.read_csv(path)
    return df


def engineer_features(df, top_counties=None):
    """Raw CSV -> model-ready feature matrix X aur target y."""
    df = df.copy()

    # Tareekh se Month nikalna (wildfire season ka asar capture karne ke liye)
    df["Started"] = pd.to_datetime(df["Started"], errors="coerce", utc=True)
    df["Month"] = df["Started"].dt.month.fillna(0).astype(int)

    # Resource/Impact columns: NaN ka matlab "deploy nahi hua / report nahi hua" -> 0
    for c in FILL_ZERO_COLS:
        df[c] = df[c].fillna(0)

    df["AcresBurned"] = df["AcresBurned"].fillna(df["AcresBurned"].median())
    df["PercentContained"] = df["PercentContained"].fillna(0)

    # Multi-county string se pehla/primary county nikalna
    df["PrimaryCounty"] = (
        df["Counties"].fillna("Unknown").astype(str).str.split(",").str[0].str.strip()
    )

    if top_counties is None:
        top_counties = (
            df["PrimaryCounty"].value_counts().head(TOP_N_COUNTIES).index.tolist()
        )
    df["CountyGroup"] = df["PrimaryCounty"].where(
        df["PrimaryCounty"].isin(top_counties), "Other"
    )

    county_dummies = pd.get_dummies(df["CountyGroup"], prefix="County")

    X = pd.concat([df[NUMERIC_FEATURES].reset_index(drop=True),
                   county_dummies.reset_index(drop=True)], axis=1)
    y = df[TARGET].astype(int).reset_index(drop=True)

    feature_columns = X.columns.tolist()
    return X, y, top_counties, feature_columns


def split_and_scale(X, y, test_size=TEST_SIZE):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train.values, y_test.values, scaler


# ---------------------------------------------------------------------------
# Model building / training
# ---------------------------------------------------------------------------
def build_cnn(input_dim):
    """Tabular data par 1D-CNN — features ko ek 'signal' ki tarah treat karte
    hain. Sklearn models jitna natural nahi (CNN images ke liye design hua
    tha) lekin comparison ke liye kaam karta hai."""
    model = keras_models.Sequential(
        [
            layers.Input(shape=(input_dim, 1)),
            layers.Conv1D(32, 3, activation="relu", padding="same"),
            layers.MaxPooling1D(2),
            layers.Conv1D(64, 3, activation="relu", padding="same"),
            layers.GlobalAveragePooling1D(),
            layers.Dense(32, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train_all_models(X_train, y_train):
    """SVM, Random Forest, ANN (hamesha) + CNN (agar TensorFlow available ho)."""
    models = {}

    models["SVM"] = SVC(kernel="rbf", C=2.0, gamma="scale",
                         probability=True, random_state=RANDOM_STATE)
    models["SVM"].fit(X_train, y_train)

    models["Random Forest"] = RandomForestClassifier(
        n_estimators=300, max_depth=14, random_state=RANDOM_STATE, n_jobs=-1
    )
    models["Random Forest"].fit(X_train, y_train)

    models["ANN"] = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        max_iter=600,
        random_state=RANDOM_STATE,
    )
    models["ANN"].fit(X_train, y_train)

    if TENSORFLOW_AVAILABLE:
        cnn = build_cnn(X_train.shape[1])
        X_train_cnn = X_train.reshape(-1, X_train.shape[1], 1)
        cnn.fit(X_train_cnn, y_train, epochs=25, batch_size=32, verbose=0)
        models["CNN"] = cnn

    return models


def predict_with_model(name, model, X):
    """Har model type ke liye yaksaan (uniform) predict + probability interface."""
    if name == "CNN":
        X_cnn = X.reshape(-1, X.shape[1], 1)
        proba = model.predict(X_cnn, verbose=0).ravel()
        preds = (proba >= 0.5).astype(int)
    else:
        proba = model.predict_proba(X)[:, 1]
        preds = model.predict(X)
    return preds, proba


def evaluate_models(models, X_test, y_test):
    """Accuracy, Precision, Recall, F1, Confusion Matrix — har model ke liye."""
    results = {}
    for name, model in models.items():
        preds, proba = predict_with_model(name, model, X_test)
        results[name] = {
            "accuracy": float(accuracy_score(y_test, preds)),
            "precision": float(precision_score(y_test, preds, zero_division=0)),
            "recall": float(recall_score(y_test, preds, zero_division=0)),
            "f1": float(f1_score(y_test, preds, zero_division=0)),
            "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
        }
    return results


# ---------------------------------------------------------------------------
# Single-incident prediction (naya / custom input)
# ---------------------------------------------------------------------------
def prepare_single_input(raw_input, top_counties, feature_columns):
    """
    raw_input: dict, e.g.
        {
            "Latitude": 38.5, "Longitude": -120.8, "AcresBurned": 500,
            "PercentContained": 40, "CrewsInvolved": 10, "Engines": 5,
            "Dozers": 1, "Helicopters": 2, "AirTankers": 0, "WaterTenders": 1,
            "PersonnelInvolved": 150, "StructuresThreatened": 20,
            "StructuresDestroyed": 0, "StructuresDamaged": 0, "Injuries": 0,
            "Fatalities": 0, "Month": 8, "County": "Riverside",
        }
    Return: (1, n_features) numpy array, feature_columns ke order me.
    """
    row = {col: 0.0 for col in feature_columns}

    for key in NUMERIC_FEATURES:
        if key in raw_input:
            row[key] = float(raw_input[key])

    county = raw_input.get("County", "Other")
    county_col = f"County_{county}" if county in top_counties else "County_Other"
    if county_col in row:
        row[county_col] = 1.0

    ordered = [row[col] for col in feature_columns]
    return np.array(ordered, dtype=float).reshape(1, -1)


def predict_incident(model_name, models, scaler, raw_input, top_counties, feature_columns):
    """Naye incident ke liye prediction: (label:int, probability:float)"""
    if model_name not in models:
        raise ValueError(f"Model '{model_name}' available nahi hai.")
    X_raw = prepare_single_input(raw_input, top_counties, feature_columns)
    X_scaled = scaler.transform(X_raw)
    preds, proba = predict_with_model(model_name, models[model_name], X_scaled)
    return int(preds[0]), float(proba[0])


# ---------------------------------------------------------------------------
# Save / load trained artifacts (optional — standalone run ke liye)
# ---------------------------------------------------------------------------
def save_artifacts(models, scaler, top_counties, feature_columns, results):
    os.makedirs(MODEL_DIR, exist_ok=True)
    for name, model in models.items():
        if name == "CNN":
            model.save(os.path.join(MODEL_DIR, "cnn_model.keras"))
        else:
            fname = name.replace(" ", "_").lower() + ".joblib"
            joblib.dump(model, os.path.join(MODEL_DIR, fname))

    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
    joblib.dump(
        {"top_counties": top_counties, "feature_columns": feature_columns},
        os.path.join(MODEL_DIR, "meta.joblib"),
    )
    with open(os.path.join(MODEL_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)


# ---------------------------------------------------------------------------
# Full pipeline — ek hi function jo Streamlit app aur standalone run dono
# use karte hain (taake logic duplicate na ho)
# ---------------------------------------------------------------------------
def run_pipeline(data_path=None):
    df = load_raw_data(data_path)
    X, y, top_counties, feature_columns = engineer_features(df)
    X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y)
    models = train_all_models(X_train, y_train)
    results = evaluate_models(models, X_test, y_test)

    return {
        "df": df,
        "X": X,
        "y": y,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "models": models,
        "results": results,
        "top_counties": top_counties,
        "feature_columns": feature_columns,
        "tensorflow_available": TENSORFLOW_AVAILABLE,
    }


# ---------------------------------------------------------------------------
# Standalone run: `python backend.py`
# ---------------------------------------------------------------------------
def _print_comparison_table(results):
    rows = []
    for name, m in results.items():
        rows.append(
            {
                "Model": name,
                "Accuracy": round(m["accuracy"], 4),
                "Precision": round(m["precision"], 4),
                "Recall": round(m["recall"], 4),
                "F1-Score": round(m["f1"], 4),
            }
        )
    table = pd.DataFrame(rows).set_index("Model")
    print("\n========== MODEL COMPARISON ==========")
    print(table.to_string())
    best = table["F1-Score"].idxmax()
    print(f"\nBest model (by F1-Score): {best}")


def _print_confusion_matrices(results):
    print("\n========== CONFUSION MATRICES ==========")
    for name, m in results.items():
        cm = np.array(m["confusion_matrix"])
        print(f"\n{name}:")
        print("              Pred: Not-Major   Pred: Major")
        print(f"Actual Not-Major     {cm[0][0]:<14d} {cm[0][1]:<12d}")
        print(f"Actual Major         {cm[1][0]:<14d} {cm[1][1]:<12d}")


def main():
    print("Wildfire Major-Incident Risk Prediction — Backend\n")
    print("Dataset load aur features engineer ho rahe hain...")
    pipeline = run_pipeline()

    print(f"Total incidents: {len(pipeline['df'])}")
    print(f"Features used: {len(pipeline['feature_columns'])}")
    print(f"TensorFlow available (CNN enabled): {pipeline['tensorflow_available']}")

    _print_comparison_table(pipeline["results"])
    _print_confusion_matrices(pipeline["results"])

    # Demo prediction — ek sample custom incident
    sample_input = {
        "Latitude": 38.6,
        "Longitude": -120.9,
        "AcresBurned": 1200,
        "PercentContained": 35,
        "CrewsInvolved": 12,
        "Engines": 8,
        "Dozers": 2,
        "Helicopters": 3,
        "AirTankers": 1,
        "WaterTenders": 2,
        "PersonnelInvolved": 220,
        "StructuresThreatened": 50,
        "StructuresDestroyed": 0,
        "StructuresDamaged": 0,
        "Injuries": 0,
        "Fatalities": 0,
        "Month": 8,
        "County": "Shasta",
    }

    print("\n========== SAMPLE PREDICTION ==========")
    print("Input:", sample_input)
    for model_name in pipeline["models"].keys():
        label, prob = predict_incident(
            model_name,
            pipeline["models"],
            pipeline["scaler"],
            sample_input,
            pipeline["top_counties"],
            pipeline["feature_columns"],
        )
        verdict = "MAJOR INCIDENT" if label == 1 else "Not Major"
        print(f"{model_name:>15}: {verdict}  (probability = {prob:.3f})")

    print("\nTrained models 'models/' folder me save ho rahe hain...")
    save_artifacts(
        pipeline["models"],
        pipeline["scaler"],
        pipeline["top_counties"],
        pipeline["feature_columns"],
        pipeline["results"],
    )
    print("Done. ✔")


if __name__ == "__main__":
    main()
