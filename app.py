"""
SMS spam classifier — Gradio UI, calls remote HTTP endpoint only.
"""

import os
import time

import gradio as gr
import requests

ENDPOINT = os.environ.get(
    "SMS_SPAM_ENDPOINT",
    "http://5.42.111.215:8082/serve/sms_spam",
)


def ui_label_from_class_name(class_name: object) -> str:
    low = str(class_name).strip().lower()
    if low == "spam":
        return "SPAM"
    if low == "ham":
        return "NOT SPAM"
    return str(class_name).strip().upper()


EXAMPLE_SPAM = (
    "URGENT! You won $5000 in our lottery. Click http://bit.ly/fake-claim now "
    "or reply YES to claim. Limited time — fees may apply."
)

EXAMPLE_HAM = (
    "Hey, are we still on for coffee at 3pm tomorrow? "
    "Let me know if you need to reschedule."
)


def predict(text: str) -> tuple[str, str]:
    if not text.strip():
        return "—", "Введите текст"

    t0 = time.perf_counter()
    try:
        r = requests.post(
            ENDPOINT,
            json={"text": text.strip()},
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        r.raise_for_status()
        data = r.json()
    except requests.exceptions.Timeout:
        return "ERROR", "Timeout — сервер не ответил за 30 секунд"
    except requests.exceptions.ConnectionError as e:
        return "ERROR", f"Нет соединения: {e}"
    except requests.exceptions.HTTPError as e:
        return "ERROR", f"HTTP {e.response.status_code}: {e.response.text[:300]}"
    except Exception as e:
        return "ERROR", f"Ошибка: {e}"

    preds = data.get("predictions") or []
    if not preds:
        return "ERROR", f"Нет predictions в ответе: {data}"

    p = preds[0]
    display = ui_label_from_class_name(p.get("class_name", "?"))
    prob_spam = p.get("probability_spam", 0)
    prob_ham = p.get("probability_ham", 0)

    emoji = "🔴" if display == "SPAM" else "🟢"
    result_label = f"{emoji} {display}"
    result_info = (
        f"Endpoint: {ENDPOINT}\n"
        f"Latency: {latency_ms:.1f} ms\n"
        f"P(spam): {prob_spam:.4f}\n"
        f"P(not spam): {prob_ham:.4f}"
    )
    return result_label, result_info


theme = gr.themes.Soft(
    primary_hue="violet",
    secondary_hue="indigo",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("DM Sans"), "ui-sans-serif", "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
).set(
    body_background_fill_dark="*neutral_950",
    block_background_fill_dark="*neutral_900",
    block_border_width="1px",
    block_title_text_weight="600",
)

with gr.Blocks(title="SMS Spam Classifier") as demo:
    gr.Markdown(
        "## SMS Spam classifier\n\n"
        "**Note:** This classifier is intended for **English** SMS only; "
        "scores for other languages may be unreliable."
    )

    with gr.Row():
        txt = gr.Textbox(
            label="Текст сообщения",
            placeholder="e.g. Call me after the meeting…",
            lines=5,
            scale=4,
        )

    with gr.Row():
        b_spam = gr.Button("Авто: спам", variant="secondary", scale=1)
        b_ham = gr.Button("Авто: не спам", variant="secondary", scale=1)
        btn = gr.Button("Классифицировать", variant="primary", scale=2)

    b_spam.click(lambda: EXAMPLE_SPAM, outputs=txt)
    b_ham.click(lambda: EXAMPLE_HAM, outputs=txt)

    with gr.Row():
        label_out = gr.Textbox(label="Метка", interactive=False)
        info_out = gr.Textbox(
            label="Задержка и вероятности",
            interactive=False,
            lines=4,
        )

    btn.click(fn=predict, inputs=txt, outputs=[label_out, info_out])
    txt.submit(fn=predict, inputs=txt, outputs=[label_out, info_out])


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=80,
        show_error=True,
        theme=theme,
        css="""
        .gradio-container { max-width: 52rem !important; margin: auto !important; }
        footer {visibility: hidden}
        """,
    )
