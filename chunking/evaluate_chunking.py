import json
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from chunking import chunk_sentences, chunk_characters, chunk_semantic
from rag.rag_pipeline import SimpleRAG
from rag.rag_with_llm import eval_pipeline as question_model
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

"""
Utvärderar olika chunkingmetoder för att hitta den bästa för RAG-modellen.
"""

def generate_questions(text, n=3):
    """
    Genererar frågor för utvärdering av chunkingmetoder.
    """
    prompt = f"Generate {n} relevant questions that someone might ask based on the following text:\n\n{text[:100]}..."
    print("🔹 Genererar frågor med prompt:")
    print(prompt)
    try:
        result = question_model(prompt)[0]['generated_text']
        print("🔹 Frågeresultat:")
        print(result)
        questions = [q.strip() for q in result.split('\n') if q.strip()]
        return questions
    except Exception as e:
        print("Fel vid frågegeneration:", e)
        return []

def generate_chunks(text):
    """
    Genererar olika chunkingmetoder med olika parametrar."""
    chunks_dict = {}

    for n_sent in [3, 10, 15, 20]:
        key = f"Sentences_{n_sent}"
        chunks_dict[key] = {
            "chunks": chunk_sentences(text, max_sentences=n_sent),
            "param": n_sent,
            "method": "Sentences"
        }

    for n_char in [250, 500, 750, 1000]:
        key = f"Characters_{n_char}"
        chunks_dict[key] = {
            "chunks": chunk_characters(text, chunk_size=n_char),
            "param": n_char,
            "method": "Characters"
        }

    
    chunks_dict["Semantic"] = {
        "chunks": chunk_semantic(text),
        "param": None,
        "method": "Semantic"
    }

    return chunks_dict

def select_best_chunking_method(chunks_dict, questions):
    """
    Utvärderar olika chunkingmetoder och väljer den bästa baserat på medelpoäng.
    """
    best_method = None
    best_chunks = []
    best_param = None
    highest_score = -1

    for method, data in chunks_dict.items():
        chunks = data["chunks"]
        rag = SimpleRAG(chunks)
        vec = TfidfVectorizer()
        chunk_vectors = vec.fit_transform(chunks)

        score_sum = 0
    
        print(f"Utvärderar metod: {method}")
        for question in tqdm(questions, desc=f"Utvärderar {method}"):
            q_vec = vec.transform([question])
            scores = cosine_similarity(q_vec, chunk_vectors).flatten()
            score_sum += max(scores)

        avg_score = score_sum / len(questions)
        print(f"Medelpoäng för {method}: {avg_score:.4f}")

        if avg_score > highest_score:
            highest_score = avg_score
            best_method = method
            best_chunks = chunks
            best_param = data["param"]

    return best_method, best_param, best_chunks


"""
Huvudfunktion för att köra utvärderingen och spara resultatet i en JSON-fil.
"""
if __name__ == "__main__":
    with open("scraping/scraped_text.txt", "r", encoding="utf-8") as f:
        text = f.read()

    print("Genererar frågor...")
    questions = generate_questions(text, n=3)

    print("Genererar chunks...")
    all_chunks = generate_chunks(text)

    print("Utvärderar...")
    best_method, best_param, best_chunks = select_best_chunking_method(all_chunks, questions)

    print(f"Bästa metod: {best_method}")

    output = {
        "method": best_method,
        "param": best_param,
        "chunks": best_chunks
    }

    with open("chunking/best_chunking.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("Sparat i best_chunking.json")
