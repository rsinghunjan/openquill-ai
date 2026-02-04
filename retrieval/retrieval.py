"""
Simple retrieval wrapper using Sentence-Transformers + FAISS.
Use this to build a RAG pipeline: retrieve top-k chunks, then include into prompt.
"""
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List

class Retriever:
    def __init__(self, embed_model_name="all-MiniLM-L6-v2", index_path=None):
        self.model = SentenceTransformer(embed_model_name)
        self.index = None
        self.index_path = index_path
        self.id_to_text = {}

        if index_path and os.path.exists(index_path + ".meta.npy"):
            self.index = faiss.read_index(index_path)
            try:
                self.id_to_text = np.load(index_path + ".meta.npy", allow_pickle=True).item()
            except:
                self.id_to_text = {}

    def build_index(self, texts: List[str], index_path="indexes/faiss_index.idx"):
        os.makedirs(os.path.dirname(index_path) or ".", exist_ok=True)
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.asarray(embeddings))
        faiss.write_index(self.index, index_path)
        # store texts map
        id_map = {i: texts[i] for i in range(len(texts))}
        np.save(index_path + ".meta.npy", id_map)
        self.index_path = index_path
        self.id_to_text = id_map

    def query(self, q: str, k: int = 5):
        if self.index is None:
            raise RuntimeError("Index not built or loaded.")
        q_emb = self.model.encode([q], convert_to_numpy=True)
        D, I = self.index.search(q_emb, k)
        ids = I[0].tolist()
        results = [self.id_to_text.get(i, "") for i in ids]
        return results, D[0].tolist()
