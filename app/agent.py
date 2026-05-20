from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from app.config import settings

PROMPT = """당신은 산업 설비 전문가 AI입니다.
아래 센서 데이터와 ML 모델 예측 결과를 바탕으로 현장 엔지니어에게 상황을 설명하고 조치를 안내하세요.

센서 데이터:
- 공기 온도: {air_temp:.1f} K
- 공정 온도: {process_temp:.1f} K
- 온도 차이: {temp_diff:.1f} K
- 회전 속도: {rpm:.0f} RPM
- 토크: {torque:.1f} Nm
- 공구 마모: {tool_wear:.0f} min
- 출력: {power:.0f} W

예측 결과:
- 고장 확률: {failure_probability:.1f}%
- 예측 고장 유형: {failure_types}

다음 두 가지를 간결하게 작성하세요:

[상황 분석] (2~3문장으로 현재 설비 상태 설명)
[조치 권고] (구체적인 점검/조치 사항 3가지 이내)"""


def explain(prediction: dict) -> dict:
    llm = ChatOpenAI(
        model=settings.llm_model,
        openai_api_key=settings.openai_api_key,
        temperature=0.3,
    )

    prompt = PromptTemplate(
        template=PROMPT,
        input_variables=["air_temp", "process_temp", "temp_diff", "rpm",
                         "torque", "tool_wear", "power", "failure_probability", "failure_types"]
    )

    chain = prompt | llm
    result = chain.invoke({
        **prediction["features"],
        "failure_probability": prediction["failure_probability"],
        "failure_types": ", ".join(prediction["failure_types"]) if prediction["failure_types"] else "없음",
    })

    text = result.content
    explanation = ""
    recommendations = ""

    if "[상황 분석]" in text and "[조치 권고]" in text:
        explanation = text.split("[상황 분석]")[1].split("[조치 권고]")[0].strip()
        recommendations = text.split("[조치 권고]")[1].strip()
    else:
        explanation = text
        recommendations = ""

    return {"explanation": explanation, "recommendations": recommendations}
