import os
import streamlit as st
import pandas as pd
from pycaret.regression import load_model, predict_model
from dotenv import load_dotenv
import instructor
from pydantic import BaseModel, Field
from typing import Optional, Literal
from langfuse import observe
from langfuse.openai import OpenAI


load_dotenv()

MODEL_NAME = "halfmarathon_model"

class RunnerData(BaseModel):
    """Dane biegacza wyciaga z opisu w języku naturalnym."""

    plec: Optional[Literal["M", "K"]] = Field(
        default=None,
        description="Płeć biegacza: 'M' dla mężczyzny, 'K' dla kobiety.",
    )
    wiek: Optional[int] = Field(
        default=None,
        description="Wiek biegacza w pełnych latach (np. 20).",
    )
    czas_5km_sekundy: Optional[int] = Field(
        default=None,
        description=(
            "Czas biegacza na 5 km wyrażony w sekundach. "
            "Np. '25:30' = 1530 sekund, '23 minuty' = 1380 sekund."
        ),
    )


def get_openai_client():
    api_key = st.session_state.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
    return OpenAI(api_key=api_key)

@observe(name="extract_runner_data")
def retrieve_structure(user_text: str) -> RunnerData:
    client = instructor.from_openai(get_openai_client())
    return client.chat.completions.create(
        model="gpt-5-mini",
        response_model=RunnerData,
        messages=[
            {
                "role": "system",
                "content": (
                    "Jesteś asystentem, który z opisu biegacza w języku polskim "
                    "wyciąga jego płeć (M lub K), wiek (lata) oraz czas na 5 km "
                    "(w sekundach). Jeśli któraś informacja nie została podana, "
                    "ustaw to pole na null. Nie zgaduj brakujących danych."
                ),
            },
            {"role": "user", "content": user_text},
        ],
    )


def validate(data: RunnerData) -> list[str]:
    missing = []
    if data.plec is None:
        missing.append("**płeć** (mężczyzna / kobieta)")
    if data.wiek is None:
        missing.append("**wiek** (w latach)")
    if data.czas_5km_sekundy is None:
        missing.append("**czas na 5 km**")
    return missing


def seconds_to_hms(seconds: float) -> str:
    seconds = int(round(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


@st.cache_resource
def load_pycaret_model():
    return load_model(MODEL_NAME)


def predict_finish_time(data: RunnerData) -> float:
    model = load_pycaret_model()
    df = pd.DataFrame(
        [
            {
                "Płeć": data.plec,
                "5 km Czas": float(data.czas_5km_sekundy),
                "Wiek": int(data.wiek),
            }
        ]
    )
    prediction = predict_model(model, data=df)
    return float(prediction["prediction_label"].iloc[0])


# ---------- App UI
st.set_page_config(
    page_title="Półmaraton — predyktor czasu",
    page_icon="🏃",
    layout="centered",
)

st.markdown(
    """
    <style>
    div.stButton > button[kind="primary"] {
        background-color: #1e6934;
        border-color: #1e6934;
        color: white;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #237a3c;
        border-color: #237a3c;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🏃 Oszacuję Twój czas półmaratonu")
st.caption(
    "Napisz mi coś o sobie — podaj **płeć**, **wiek** i **czas na 5 km**, "
    "a ja określę, w jakim czasie ukończysz półmaraton."
)

# OpenAI API key handling
if not os.getenv("OPENAI_API_KEY") and "openai_api_key" not in st.session_state:
    api_key = st.text_input("🔑 Klucz OpenAI API", type="password")
    if api_key:
        st.session_state["openai_api_key"] = api_key
        st.rerun()
    st.stop()

user_text = st.text_area(
    "Opisz się:",
    placeholder=(
        "Np. Cześć, jestem Marcin, mam 49 lat, jestem mężczyzną, "
        "a 5 km biegam w 30 minut i 30 sekund."
    ),
    height=120,
)

if st.button("Oszacuj czas", type="primary", use_container_width=True):
    if not user_text.strip():
        st.warning("Najpierw napisz proszę coś o sobie 🙂")
        st.stop()

    with st.spinner("Analizuję Twój opis..."):
        try:
            data = retrieve_structure(user_text)
        except Exception as e:
            st.error(f"Nie udało się przetworzyć opisu: {e}")
            st.stop()

    missing = validate(data)
    if missing:
        st.warning(
            "Brakuje mi części informacji. Dopisz proszę: "
            + ", ".join(missing)
            + "."
        )
        with st.expander("Co udało mi się odczytać"):
            st.json(data.model_dump())
        st.stop()

    if data.wiek > 100 or data.wiek < 13:
        st.error(
            f"Podany wiek ({data.wiek}) jest poza dopuszczalnym zakresem. "
            "Wpisz wiek z przedziału 13–100 lat."
        )
        st.stop()

    czas_5km_hms = seconds_to_hms(data.czas_5km_sekundy)
    st.markdown(
        f"<p style='font-size:1.4rem;'><b>Odczytane dane:</b> "
        f"płeć: {data.plec} &nbsp;|&nbsp; wiek: {data.wiek} &nbsp;|&nbsp; "
        f"czas 5 km: {czas_5km_hms}</p>",
        unsafe_allow_html=True,
    )

    with st.spinner("Przewiduję czas..."):
        try:
            seconds = predict_finish_time(data)
        except Exception as e:
            st.error(f"Błąd predykcji: {e}")
            st.stop()

    st.success(f"### Przewidywany czas półmaratonu: **{seconds_to_hms(seconds)}**")
    st.caption(
        f"(Inaczej mówiąc ~{seconds/60:.1f} min, Twoje tempo ~{seconds/21.0975/60:.2f} min/km)"
    )
