from sentence_transformers import SentenceTransformer
import numpy as np

# Load model once globally (fastest small one)
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embeddings(chunks):
    embeddings = model.encode(chunks)
    return embeddings
