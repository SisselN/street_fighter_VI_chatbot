from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

"""
Klass för att hämta och bearbeta textdata och generera svar med hjälp av RAG-modellen.
"""

class SimpleRAG:
    def __init__(self, chunks):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer()
        self.chunk_vectors = self.vectorizer.fit_transform(chunks)

    def query(self, question, top_k=3):
        question_vector = self.vectorizer.transform([question])
        scores = cosine_similarity(question_vector, self.chunk_vectors).flatten()
        top_indices = scores.argsort()[::-1][:top_k]
        retrieved_chunks = [self.chunks[i] for i in top_indices]

        context = "\n\n".join(retrieved_chunks)
        return context