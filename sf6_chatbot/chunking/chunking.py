from sentence_transformers import SentenceTransformer, util
from nltk.tokenize import sent_tokenize
import torch

"""
Chunking-funktioner för att dela upp text i mindre delar.
"""

def chunk_sentences(text, max_sentences=3):
    """
    Delar upp texten i chunks där varje chunk består av 'max_sentences' meningar.
    """
    sentences = sent_tokenize(text)
    chunks = [
        ' '.join(sentences[i:i + max_sentences])
        for i in range(0, len(sentences), max_sentences)
    ]
    return chunks

def chunk_characters(text, chunk_size=500):
    """
    Delar upp texten i chunks om 'chunk_size' tecken vardera.
    """
    return [
        text[i:i + chunk_size]
        for i in range(0, len(text), chunk_size)
    ]

def chunk_semantic(text, similarity_threshold=0.75, max_chunk_size=5):
    """
    Delar upp texten i semantiska chunks med SentenceTransformer som beräknar embeddings och likhet.
    """

    model = SentenceTransformer("all-MiniLM-L6-v2")
    sentences = sent_tokenize(text)
    embeddings = model.encode(sentences, convert_to_tensor=True)

    chunks = []
    current_chunk = [sentences[0]]
    current_embedding = embeddings[0].unsqueeze(0)

    for i in range(1, len(sentences)):
        similarity = util.cos_sim(embeddings[i], current_embedding.mean(dim=0)).item()

        if similarity > similarity_threshold and len(current_chunk) < max_chunk_size:
            current_chunk.append(sentences[i])
            current_embedding = embeddings[i].unsqueeze(0) if len(current_chunk) == 1 else torch.cat((current_embedding, embeddings[i].unsqueeze(0)), dim=0)
        else:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentences[i]]
            current_embedding = embeddings[i].unsqueeze(0)

    # Sista chunken
    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks