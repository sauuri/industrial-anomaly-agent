import pandas as pd
import numpy as np
import os
from app.config import settings

_df = None
_idx = 0


def _load():
    global _df
    df = pd.read_csv(settings.data_path)
    df.columns = [c.strip().replace("﻿", "") for c in df.columns]
    df = df.rename(columns={
        "Air temperature [K]": "air_temperature",
        "Process temperature [K]": "process_temperature",
        "Rotational speed [rpm]": "rotational_speed",
        "Torque [Nm]": "torque",
        "Tool wear [min]": "tool_wear",
        "Type": "machine_type",
        "Machine failure": "machine_failure",
    })
    # 정상 800개 + 고장 200개 섞기
    normal = df[df["machine_failure"] == 0].sample(800, random_state=42)
    failure = df[df["machine_failure"] == 1].sample(min(200, df["machine_failure"].sum()), random_state=42)
    mixed = pd.concat([normal, failure]).sample(frac=1, random_state=42).reset_index(drop=True)
    _df = mixed


def next_row() -> dict:
    global _idx
    if _df is None:
        _load()
    row = _df.iloc[_idx % len(_df)]
    _idx += 1
    return {
        "air_temperature": float(row["air_temperature"]),
        "process_temperature": float(row["process_temperature"]),
        "rotational_speed": float(row["rotational_speed"]),
        "torque": float(row["torque"]),
        "tool_wear": float(row["tool_wear"]),
        "machine_type": str(row["machine_type"]),
    }
