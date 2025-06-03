import gradio as gr
import requests
from pathlib import Path

gr.set_static_paths(paths=[Path.cwd().absolute()/"assets"])

"""
Applikation för att ställa frågor om Street Fighter VI med hjälp av RAG-modellen.
"""

"""
Global variabel för RAG-modellen.
"""
rag_model = None

def answer_question_with_history(message, history):
    try:
        response = requests.post(
            "http://localhost:8000/answer",
            json={"message": message, "history": history}
        )
        if response.status_code == 200:
            return response.json()["answer"]
        else:
            return "Fel från backend: " + response.text
    except Exception as e:
        return f"Kunde inte kontakta backend: {e}"

def build_ui():
    """
    Bygger användargränssnittet med Gradio.
    """

    with gr.Blocks() as demo:
        gr.Markdown("# Street Fighter VI chatbot")
        gr.Markdown("### Fråga om Street Fighter VI")
        chatbot = gr.Chatbot(label="Dialog", type="messages")
        question = gr.Textbox(label="Ställ din fråga", placeholder="Skriv din fråga här...")
        send_btn = gr.Button("Skicka")


        state = gr.State([])  # Sparar historiken

        def user_asks(message, history):
            answer = answer_question_with_history(message, history)
            history = history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": answer}]
            return history, history, ""

        send_btn.click(
            user_asks,
            inputs=[question, state],
            outputs=[chatbot, state, question]
        )
        gr.Markdown(
            "<hr><div style='text-align:center; font-size: 0.9em; color: #888;'>"
            "Denna chatbot är ett lärandeprojekt i min data science-utbildning vid EC Utbildning AB. Den har ingen koppling till Capcom eller Street Fighter VI.<br>"
            "Ingen personlig data sparas.<br>"
            "Sissel Nevestveit, vårterminen 2025 "
            "Koden finns på GitHub: https://github.com/SisselN/street_fighter_VI_chatbot"
            "</div>"
        )

    return demo

if __name__ == "__main__":
    ui = build_ui()
    ui.launch()