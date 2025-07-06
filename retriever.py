import faiss
import numpy as np

class FAISSRetriever:
    def __init__(self, dim):
        self.index = faiss.IndexFlatL2(dim)
        self.chunk_store = []  # Keep original chunks for display

    def add(self, embeddings, chunks):
        self.index.add(np.array(embeddings).astype('float32'))
        self.chunk_store.extend(chunks)

    def search(self, query_embedding, top_k=3):
        D, I = self.index.search(np.array([query_embedding]).astype('float32'), top_k)
        return [self.chunk_store[i] for i in I[0]]
