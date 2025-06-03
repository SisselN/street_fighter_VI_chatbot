from transformers import pipeline
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
   
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

"""
RAG-modell med LLM för att svara på frågor om Street Fighter VI
"""

# Kallar på generative AI-modellen från Google Gemini
app_model = genai.GenerativeModel("gemini-1.5-flash")


# Pipeline för en enkel sekundär modell som används för att generera frågor under utvärdering
eval_pipeline = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_new_tokens=64
)

"""
Genererar svar på frågor med hjälp av RAG-modellen och LLM.
Använder en kontext från RAG-modellen för att ge svar.
"""
def generate_answer(question, context):
    short_context = context[:500] if context else ""
    prompt = (
        "Du är en expert på Street Fighter VI. "
        "Du kan härleda alla frågor till fakta om spelet. "
        "Du svarar alltid på svenska. "
        "Om frågan inte handlar om Street Fighter VI så får du den att handla om det oavsett om det är långsökt'\n\n"
        f"Kontext:\n{short_context}\n\nFråga: {question}\nSvar:"
    )
    try:
        response = app_model.generate_content(prompt)
        text = response.candidates[0].content.parts[0].text
        return text.strip()
    except Exception as e:
        print("Fel vid anrop eller tolkning av Gemini:", e)
        return f"Fel vid anrop eller tolkning av Gemini: {e}"