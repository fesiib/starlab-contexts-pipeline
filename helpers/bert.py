from sentence_transformers import SentenceTransformer
from numpy import zeros

def bert_embed_text(text):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    if text == "":
        return zeros(384)
    embeddings = model.encode([text])
    return embeddings[0]