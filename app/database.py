import json
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, Float, Boolean, String, DateTime, text
from sqlalchemy.orm import declarative_base, sessionmaker
from app.config import settings

is_sqlite = settings.database_url.startswith("sqlite")
engine = create_engine(
    settings.database_url,
    pool_pre_ping=not is_sqlite,
    connect_args={"check_same_thread": False} if is_sqlite else {},
)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class SensorLog(Base):
    __tablename__ = "sensor_logs"

    id                  = Column(Integer, primary_key=True, index=True)
    ts                  = Column(DateTime, default=datetime.utcnow, index=True)
    air_temperature     = Column(Float)
    process_temperature = Column(Float)
    rotational_speed    = Column(Float)
    torque              = Column(Float)
    tool_wear           = Column(Float)
    machine_type        = Column(String(1))
    failure_probability = Column(Float)
    failure_predicted   = Column(Boolean)
    failure_types       = Column(String)  # JSON 문자열


def init_db():
    Base.metadata.create_all(bind=engine)


def save_log(sensors: dict, prediction: dict):
    db = SessionLocal()
    try:
        log = SensorLog(
            air_temperature     = sensors["air_temperature"],
            process_temperature = sensors["process_temperature"],
            rotational_speed    = sensors["rotational_speed"],
            torque              = sensors["torque"],
            tool_wear           = sensors["tool_wear"],
            machine_type        = sensors["machine_type"],
            failure_probability = prediction["failure_probability"],
            failure_predicted   = prediction["failure_predicted"],
            failure_types       = json.dumps(prediction["failure_types"], ensure_ascii=False),
        )
        db.add(log)
        db.commit()
    finally:
        db.close()


def get_history(limit: int = 100):
    db = SessionLocal()
    try:
        rows = db.query(SensorLog).order_by(SensorLog.ts.desc()).limit(limit).all()
        return [
            {
                "id": r.id,
                "ts": r.ts.strftime("%Y-%m-%d %H:%M:%S"),
                "air_temperature": r.air_temperature,
                "process_temperature": r.process_temperature,
                "rotational_speed": r.rotational_speed,
                "torque": r.torque,
                "tool_wear": r.tool_wear,
                "machine_type": r.machine_type,
                "failure_probability": r.failure_probability,
                "failure_predicted": r.failure_predicted,
                "failure_types": json.loads(r.failure_types or "[]"),
            }
            for r in rows
        ]
    finally:
        db.close()


def get_stats():
    db = SessionLocal()
    try:
        today = datetime.utcnow().date()
        if is_sqlite:
            result = db.execute(text("""
                SELECT
                    COUNT(*) AS total,
                    SUM(CASE WHEN failure_predicted THEN 1 ELSE 0 END) AS failures,
                    ROUND(AVG(failure_probability), 1) AS avg_prob
                FROM sensor_logs
                WHERE DATE(ts) = :today
            """), {"today": str(today)}).fetchone()
        else:
            result = db.execute(text("""
                SELECT
                    COUNT(*) AS total,
                    SUM(CASE WHEN failure_predicted THEN 1 ELSE 0 END) AS failures,
                    ROUND(AVG(failure_probability)::numeric, 1) AS avg_prob
                FROM sensor_logs
                WHERE ts::date = :today
            """), {"today": today}).fetchone()
        return {
            "total": result.total or 0,
            "failures": result.failures or 0,
            "avg_prob": float(result.avg_prob or 0),
        }
    finally:
        db.close()
