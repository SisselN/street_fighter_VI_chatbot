import gradio as gr
import json
from rag.rag_pipeline import SimpleRAG
from rag.rag_with_llm import generate_answer
from pathlib import Path

gr.set_static_paths(paths=[Path.cwd().absolute()/"assets"])

"""
Applikation för att ställa frågor om Street Fighter VI med hjälp av RAG-modellen.
"""

"""
Global variabel för RAG-modellen.
"""
rag_model = None

def load_best_chunks():
    """
    Laddar in chunks och metod från best_chunking.json
    """
    with open("chunking/best_chunking.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["chunks"]

def setup_rag():
    """
    Initierar RAG-modellen med de bästa chunkarna
    """
    global rag_model
    chunks = load_best_chunks()
    rag_model = SimpleRAG(chunks)

def answer_question_with_history(message, history):
    """
    Svarar på en fråga och uppdaterar historiken
    """
    if rag_model is None:
        return "Modellen är inte redo än."

    context = rag_model.query(message, top_k=3)
    answer = generate_answer(message, context)
    return answer

def build_ui():
    """
    Bygger användargränssnittet med Gradio.
    """
    setup_rag()

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
            return history, history

        send_btn.click(
            user_asks,
            inputs=[question, state],
            outputs=[chatbot, state]
        )
        gr.Markdown(
            "<hr><div style='text-align:center; font-size: 0.9em; color: #888;'>"
            "Denna chatbot är ett lärandeprojekt i min data science-utbildning vid EC Utbildning AB. Den har ingen koppling till Capcom eller Street Fighter VI.<br>"
            "© 2025 Sissel Nevestveit"
            "</div>"
        )

    return demo

if __name__ == "__main__":
    ui = build_ui()
    ui.launch()