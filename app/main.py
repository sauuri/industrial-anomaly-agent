import asyncio
import json
from datetime import datetime
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.responses import JSONResponse
from app.schemas import SensorInput, PredictionResult
from app.model import predict
from app.agent import explain
from app.simulator import next_row
import os

app = FastAPI(title="Industrial Anomaly Agent", version="0.1.0")

static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
def root():
    return FileResponse(os.path.join(static_dir, "index.html"))


@app.get("/dashboard")
def dashboard():
    return FileResponse(os.path.join(static_dir, "dashboard.html"))


@app.get("/api/pca-background")
def pca_background():
    import pickle, pandas as pd
    from app.config import settings
    from app.model import _preprocess, FEATURES, FAILURE_COLS

    with open(settings.model_path, "rb") as f:
        bundle = pickle.load(f)
    scaler, pca = bundle["scaler"], bundle.get("pca")
    if pca is None:
        return JSONResponse({"normal": [], "failure": []})

    df = pd.read_csv(settings.data_path)
    df = _preprocess(df)
    X = df[FEATURES]
    y = df["Machine failure"]

    X_scaled = scaler.transform(X)
    coords = pca.transform(X_scaled)

    # 정상 500개, 고장 전체 샘플링
    normal_idx  = y[y == 0].sample(500, random_state=42).index
    failure_idx = y[y == 1].index

    normal_pts  = [{"x": round(float(coords[i][0]),3), "y": round(float(coords[i][1]),3)} for i in normal_idx]
    failure_pts = [{"x": round(float(coords[i][0]),3), "y": round(float(coords[i][1]),3)} for i in failure_idx]

    return JSONResponse({"normal": normal_pts, "failure": failure_pts})


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResult)
def predict_failure(data: SensorInput):
    try:
        prediction = predict(data.model_dump())
        llm_result = explain(prediction)

        return PredictionResult(
            failure_predicted=prediction["failure_predicted"],
            failure_probability=prediction["failure_probability"],
            failure_types=prediction["failure_types"],
            explanation=llm_result["explanation"],
            recommendations=llm_result["recommendations"],
        )
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="모델이 학습되지 않았습니다. /train을 먼저 실행하세요.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train")
def train_model():
    from app.model import train
    try:
        train()
        return {"message": "모델 학습 완료"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/stream")
async def stream(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            sensor = next_row()
            prediction = predict(sensor)

            await websocket.send_text(json.dumps({
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "sensors": sensor,
                "failure_probability": prediction["failure_probability"],
                "failure_predicted": prediction["failure_predicted"],
                "failure_types": prediction["failure_types"],
                "pca": prediction.get("pca"),
            }))
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
