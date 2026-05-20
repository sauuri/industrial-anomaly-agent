from pydantic import BaseModel


class SensorInput(BaseModel):
    air_temperature: float
    process_temperature: float
    rotational_speed: float
    torque: float
    tool_wear: float
    machine_type: str = "M"


class PredictionResult(BaseModel):
    failure_predicted: bool
    failure_probability: float
    failure_types: list[str]
    explanation: str
    recommendations: str
