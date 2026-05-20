import pandas as pd
import numpy as np
import pickle
import os
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from app.config import settings

FEATURES = ["air_temp", "process_temp", "rpm", "torque", "tool_wear", "type_H", "type_L", "type_M", "temp_diff", "power"]
FAILURE_COLS = ["TWF", "HDF", "PWF", "OSF", "RNF"]
FAILURE_NAMES = {
    "TWF": "공구 마모 고장 (Tool Wear Failure)",
    "HDF": "열 방산 고장 (Heat Dissipation Failure)",
    "PWF": "전력 고장 (Power Failure)",
    "OSF": "과부하 고장 (Overstrain Failure)",
    "RNF": "무작위 고장 (Random Failure)",
}


def _preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().replace("﻿", "") for c in df.columns]

    rename = {
        "Air temperature [K]": "air_temp",
        "Process temperature [K]": "process_temp",
        "Rotational speed [rpm]": "rpm",
        "Torque [Nm]": "torque",
        "Tool wear [min]": "tool_wear",
        "Type": "type",
    }
    df = df.rename(columns=rename)

    df["type_H"] = (df["type"] == "H").astype(int)
    df["type_L"] = (df["type"] == "L").astype(int)
    df["type_M"] = (df["type"] == "M").astype(int)
    df["temp_diff"] = df["process_temp"] - df["air_temp"]
    df["power"] = df["torque"] * df["rpm"] * (2 * np.pi / 60)

    return df


def train():
    df = pd.read_csv(settings.data_path)
    df = _preprocess(df)

    X = df[FEATURES]
    y_failure = df["Machine failure"]
    y_types = df[FAILURE_COLS]

    X_train, X_test, y_train, y_test = train_test_split(X, y_failure, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                          scale_pos_weight=10, random_state=42, eval_metric="logloss")
    model.fit(X_train_scaled, y_train)

    type_models = {}
    for col in FAILURE_COLS:
        m = XGBClassifier(n_estimators=100, max_depth=4, random_state=42,
                          scale_pos_weight=20, eval_metric="logloss")
        m.fit(X_train_scaled, df.loc[X_train.index, col])
        type_models[col] = m

    y_pred = model.predict(X_test_scaled)
    print(classification_report(y_test, y_pred))

    # PCA: 전체 데이터로 학습해서 배경 분포 표현
    X_all_scaled = scaler.transform(X)
    pca = PCA(n_components=2, random_state=42)
    pca.fit(X_all_scaled)

    os.makedirs("models", exist_ok=True)
    with open(settings.model_path, "wb") as f:
        pickle.dump({"model": model, "scaler": scaler, "type_models": type_models, "pca": pca}, f)

    print(f"모델 저장 완료: {settings.model_path}")


def predict(sensor_input: dict) -> dict:
    with open(settings.model_path, "rb") as f:
        bundle = pickle.load(f)

    model = bundle["model"]
    scaler = bundle["scaler"]
    type_models = bundle["type_models"]

    row = {
        "air_temp": sensor_input["air_temperature"],
        "process_temp": sensor_input["process_temperature"],
        "rpm": sensor_input["rotational_speed"],
        "torque": sensor_input["torque"],
        "tool_wear": sensor_input["tool_wear"],
        "type_H": 1 if sensor_input["machine_type"] == "H" else 0,
        "type_L": 1 if sensor_input["machine_type"] == "L" else 0,
        "type_M": 1 if sensor_input["machine_type"] == "M" else 0,
        "temp_diff": sensor_input["process_temperature"] - sensor_input["air_temperature"],
        "power": sensor_input["torque"] * sensor_input["rotational_speed"] * (2 * 3.14159 / 60),
    }

    X = pd.DataFrame([row])[FEATURES]
    X_scaled = scaler.transform(X)

    prob = float(model.predict_proba(X_scaled)[0][1])
    failure = prob >= 0.3

    failure_types = []
    for col in FAILURE_COLS:
        p = float(type_models[col].predict_proba(X_scaled)[0][1])
        if p >= 0.3:
            failure_types.append(FAILURE_NAMES[col])

    pca = bundle.get("pca")
    pca_coords = None
    if pca is not None:
        coords = pca.transform(X_scaled)[0]
        pca_coords = {"x": round(float(coords[0]), 4), "y": round(float(coords[1]), 4)}

    return {
        "failure_predicted": failure,
        "failure_probability": round(prob * 100, 1),
        "failure_types": failure_types,
        "features": row,
        "pca": pca_coords,
    }
