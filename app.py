"""
SMS spam classifier UI — calls remote HTTP inference only (no local model).
"""

from __future__ import annotations

import os
import json
import time
from typing import Any

import requests
import streamlit as st

DEFAULT_ENDPOINT = os.environ.get(
    "SMS_SPAM_ENDPOINT",
    "http://5.42.111.215:8082/serve/sms_spam",
)

EXAMPLE_SPAM = (
    "URGENT! You won $5000 in our lottery. Click http://bit.ly/fake-claim now "
    "or reply YES to claim. Limited time — fees may apply."
)

EXAMPLE_HAM = (
    "Hey, are we still on for coffee at 3pm tomorrow? "
    "Let me know if you need to reschedule."
)

TEXT_KEY = "sms_body"
LAST_OK_KEY = "last_ok_prediction"


def ui_label_from_class_name(class_name: object) -> str:
    """Map API class_name to user-facing label (avoid jargon 'ham')."""
    low = str(class_name).strip().lower()
    if low == "spam":
        return "SPAM"
    if low == "ham":
        return "NOT SPAM"
    return str(class_name).strip().upper()


def inject_styles() -> None:
    st.markdown(
        """
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=JetBrains+Mono:wght@400;500&display=swap');

  html, body, [class*="css"]  {
    font-family: "DM Sans", system-ui, sans-serif;
  }

  .block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 720px;
  }

  /* Main surface */
  div[data-testid="stAppViewContainer"] {
    background: radial-gradient(1200px 600px at 10% -10%, rgba(99, 102, 241, 0.18), transparent 55%),
                radial-gradient(900px 500px at 100% 0%, rgba(236, 72, 153, 0.12), transparent 50%),
                linear-gradient(180deg, #0c0f14 0%, #0a0b10 40%, #08090d 100%);
  }

  section[data-testid="stSidebar"] {
    background: linear-gradient(165deg, rgba(18, 20, 28, 0.98) 0%, rgba(12, 14, 22, 0.99) 100%);
    border-right: 1px solid rgba(255, 255, 255, 0.06);
  }

  section[data-testid="stSidebar"] .stMarkdown { color: rgba(245, 245, 250, 0.75); }

  .hero-wrap {
    text-align: center;
    margin-bottom: 1.75rem;
  }

  .hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.35rem 0.85rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    color: #c7d2fe;
    background: rgba(99, 102, 241, 0.2);
    border: 1px solid rgba(129, 140, 248, 0.35);
    margin-bottom: 0.85rem;
  }

  .hero-title {
    font-size: clamp(1.75rem, 4vw, 2.35rem);
    font-weight: 700;
    line-height: 1.15;
    margin: 0 0 0.5rem 0;
    background: linear-gradient(120deg, #f8fafc 0%, #e2e8f0 35%, #a5b4fc 90%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }

  .hero-sub {
    color: rgba(226, 232, 240, 0.65);
    font-size: 0.95rem;
    max-width: 34rem;
    margin: 0 auto;
    line-height: 1.5;
  }

  .hero-note {
    color: rgba(251, 191, 36, 0.9);
    font-size: 0.88rem;
    max-width: 34rem;
    margin: 0.85rem auto 0;
    line-height: 1.45;
    text-align: center;
  }

  .card {
    background: rgba(15, 17, 24, 0.72);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 1.25rem 1.35rem 1.1rem;
    margin-bottom: 1.25rem;
    box-shadow: 0 24px 48px rgba(0, 0, 0, 0.35);
    backdrop-filter: blur(10px);
  }

  .card-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: rgba(148, 163, 184, 0.85);
    margin-bottom: 0.65rem;
    font-weight: 600;
  }

  div[data-testid="stTextArea"] textarea {
    font-family: "JetBrains Mono", ui-monospace, monospace;
    font-size: 0.88rem !important;
    border-radius: 12px !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    background: rgba(8, 10, 16, 0.65) !important;
    color: #e2e8f0 !important;
    min-height: 140px;
  }

  div[data-testid="stTextArea"] textarea:focus {
    border-color: rgba(129, 140, 248, 0.55) !important;
    box-shadow: 0 0 0 1px rgba(99, 102, 241, 0.25);
  }

  .quick-row { gap: 0.5rem; flex-wrap: wrap; }

  div[data-testid="stHorizontalBlock"] button {
    border-radius: 999px !important;
    font-weight: 600 !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    background: rgba(30, 32, 42, 0.9) !important;
    color: #e2e8f0 !important;
    transition: transform 0.12s ease, border-color 0.12s ease, background 0.12s ease;
  }

  div[data-testid="stHorizontalBlock"] button:hover {
    border-color: rgba(129, 140, 248, 0.45) !important;
    background: rgba(49, 46, 129, 0.35) !important;
  }

  button[kind="primary"] {
    border-radius: 12px !important;
    font-weight: 700 !important;
    padding: 0.55rem 1.25rem !important;
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%) !important;
    border: none !important;
    box-shadow: 0 8px 24px rgba(99, 102, 241, 0.35);
  }

  button[kind="primary"]:hover {
    box-shadow: 0 12px 28px rgba(139, 92, 246, 0.45);
  }

  .result-spam {
    border-left: 4px solid #f43f5e;
    padding-left: 1rem;
    margin-top: 0.5rem;
  }

  .result-not-spam {
    border-left: 4px solid #34d399;
    padding-left: 1rem;
    margin-top: 0.5rem;
  }

  div[data-testid="stMetricValue"] {
    font-variant-numeric: tabular-nums;
  }

  div[data-testid="stExpander"] {
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 12px;
    overflow: hidden;
    background: rgba(10, 12, 18, 0.5);
  }

  div[data-testid="stProgress"] > div > div > div > div {
    background: linear-gradient(90deg, #f43f5e, #fb7185, #fbbf24) !important;
  }
</style>
        """,
        unsafe_allow_html=True,
    )


def predict(
    endpoint: str, text: str, timeout_s: float = 30.0
) -> tuple[dict[str, Any] | None, float | None, str | None]:
    """POST to serve endpoint; returns (json_body, latency_seconds, error_message)."""
    t0 = time.perf_counter()
    try:
        r = requests.post(
            endpoint,
            json={"text": text},
            headers={"Content-Type": "application/json"},
            timeout=timeout_s,
        )
        latency = time.perf_counter() - t0
        r.raise_for_status()
        return r.json(), latency, None
    except requests.exceptions.Timeout:
        return None, time.perf_counter() - t0, "Превышено время ожидания ответа (timeout)."
    except requests.exceptions.ConnectionError as e:
        return None, time.perf_counter() - t0, f"Нет соединения с endpoint: {e}"
    except requests.exceptions.HTTPError as e:
        return None, time.perf_counter() - t0, (
            f"HTTP ошибка: {e} (тело: {getattr(e.response, 'text', '')[:500]})"
        )
    except requests.exceptions.RequestException as e:
        return None, time.perf_counter() - t0, f"Ошибка запроса: {e}"
    except ValueError as e:
        return None, time.perf_counter() - t0, f"Некорректный JSON в ответе: {e}"


def render_prediction(data: dict[str, Any], latency: float) -> None:
    preds = (data or {}).get("predictions") or []
    if not preds:
        st.error("В ответе нет поля predictions или оно пустое.")
        st.json(data or {})
        return

    p0 = preds[0]
    label_display = ui_label_from_class_name(p0.get("class_name", "—"))
    prob_spam = p0.get("probability_spam")
    prob_ham = p0.get("probability_ham")

    is_spam = label_display == "SPAM"
    result_class = "result-spam" if is_spam else "result-not-spam"
    accent = "#fda4af" if is_spam else "#6ee7b7"

    with st.container(border=True):
        st.markdown(
            f'<div class="{result_class}">'
            f'<p class="card-label" style="color:{accent};margin-top:0">Результат</p></div>',
            unsafe_allow_html=True,
        )

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Класс", label_display)
        with c2:
            st.metric("Задержка", f"{latency * 1000:.1f} ms")

        if prob_spam is not None and prob_ham is not None:
            st.progress(
                float(prob_spam),
                text=f"spam: {prob_spam:.4f}  ·  not spam: {prob_ham:.4f}",
            )

        c_dl, _ = st.columns([1, 2])
        with c_dl:
            st.download_button(
                label="Скачать JSON ответа",
                data=json.dumps(data, ensure_ascii=False, indent=2),
                file_name="sms_spam_response.json",
                mime="application/json",
                use_container_width=True,
            )

        with st.expander("Сырой ответ API"):
            st.json(data)


def main() -> None:
    st.set_page_config(
        page_title="SMS Spam — классификатор",
        layout="centered",
        initial_sidebar_state="expanded",
    )
    inject_styles()

    if TEXT_KEY not in st.session_state:
        st.session_state[TEXT_KEY] = ""

    with st.sidebar:
        st.markdown("### Подключение")
        endpoint = st.text_input(
            "Endpoint URL",
            value=DEFAULT_ENDPOINT,
            help='POST JSON: {"text": "..."}',
        )
        timeout = st.number_input(
            "Timeout (сек)",
            min_value=1.0,
            max_value=120.0,
            value=30.0,
            step=1.0,
        )
        st.caption("Модель не загружается локально — только HTTP-запрос к сервису.")
        if st.button("Сбросить последний результат", use_container_width=True):
            st.session_state.pop(LAST_OK_KEY, None)
            st.rerun()

    st.markdown(
        """
<div class="hero-wrap">
  <div class="hero-badge">● Remote inference</div>
  <p class="hero-title">SMS Spam classifier</p>
  <p class="hero-sub">Вставьте текст SMS или выберите готовый пример — сервер вернёт метку и вероятности spam / not spam.</p>
  <p class="hero-note"><strong>Note:</strong> This classifier is intended for <strong>English</strong> SMS only; scores for other languages may be unreliable.</p>
</div>
        """,
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        st.markdown(
            '<p class="card-label" style="margin-bottom:0.75rem">Сообщение</p>',
            unsafe_allow_html=True,
        )

        q1, q2 = st.columns(2, gap="small")
        with q1:
            if st.button(
                "Авто: спам",
                use_container_width=True,
                help="Подставить типичный спам-текст",
            ):
                st.session_state[TEXT_KEY] = EXAMPLE_SPAM
                st.rerun()
        with q2:
            if st.button(
                "Авто: не спам",
                use_container_width=True,
                help="Подставить обычное личное сообщение",
            ):
                st.session_state[TEXT_KEY] = EXAMPLE_HAM
                st.rerun()

        text = st.text_area(
            "Текст",
            height=150,
            placeholder="e.g. Call me after the meeting…",
            key=TEXT_KEY,
            label_visibility="collapsed",
        )

    predict_btn = st.button("Отправить на классификацию", type="primary", use_container_width=True)

    if predict_btn:
        if not (text or "").strip():
            st.warning("Введите текст или нажмите «Авто: спам» / «Авто: не спам».")
        else:
            with st.spinner("Запрос к серверу…"):
                data, latency, err = predict(
                    endpoint.strip(), text.strip(), timeout_s=float(timeout)
                )

            if err is not None:
                st.error(err)
                if latency is not None:
                    st.metric("Latency (до ошибки)", f"{latency * 1000:.1f} ms")
            else:
                preds = (data or {}).get("predictions") or []
                if not preds:
                    st.error("В ответе нет поля predictions или оно пустое.")
                    st.json(data or {})
                else:
                    st.session_state[LAST_OK_KEY] = {
                        "data": data,
                        "latency": float(latency or 0.0),
                    }

    bundle = st.session_state.get(LAST_OK_KEY)
    if bundle and isinstance(bundle.get("data"), dict):
        render_prediction(bundle["data"], float(bundle.get("latency", 0.0)))


if __name__ == "__main__":
    main()
