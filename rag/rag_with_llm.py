from transformers import pipeline
import google.generativeai as genai
import os
# Sätt din Google Generative AI API-nyckel   
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# genai.configure(api_key=GEMINI_API_KEY)

genai.configure(api_key="AIzaSyAMDLeIrFH1iwtdb0W4UyL0Dv2QvohDdCw")

"""
RAG-modell med LLM för att svara på frågor om Street Fighter VI
"""
#gemini_model = genai.Model("gemini-1.5-flash")
#genai.configure(api_key=GEMINI_API_KEY)

# Pipeline för primär modell.
app_model = genai.GenerativeModel("gemini-1.5-flash")
#app_model =pipeline("text-generation", model=model_name, max_new_tokens=256)


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
    # Begränsa kontexten till de 500 första tecknen
    short_context = context[:500] if context else ""
    prompt = f"Fråga: {question}\nSvar:"
    try:
        response = app_model.generate_content(prompt)
        # Hämta svaret från Gemini-svaret
        text = response.candidates[0].content.parts[0].text
        return text.strip()
    except Exception as e:
        print("Fel vid anrop eller tolkning av Gemini:", e)
        return f"Fel vid anrop eller tolkning av Gemini: {e}"