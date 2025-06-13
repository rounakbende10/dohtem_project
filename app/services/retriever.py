import asyncio
from typing import List, Dict, Any, Tuple
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from langchain.schema import Document
from app.services.vector_store import VectorStore

class HybridRetriever:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.bm25 = None
        self.documents = []
        self.document_texts = []
        
    async def initialize_bm25(self):
        self.documents = await self.vector_store.get_all_documents()
        self.document_texts = [doc.page_content for doc in self.documents]
        
        if self.document_texts:
            tokenized_docs = [doc.split() for doc in self.document_texts]
            self.bm25 = BM25Okapi(tokenized_docs)
    
    async def bm25_search(self, query: str, top_k: int = 10) -> List[Tuple[Document, float]]:
        if not self.bm25 or not self.documents:
            await self.initialize_bm25()
        
        if not self.bm25:
            return []
        
        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)
        
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((self.documents[idx], float(scores[idx])))
        
        return results
    
    async def vector_search(self, query: str, top_k: int = 10) -> List[Tuple[Document, float]]:
        results = await self.vector_store.similarity_search_with_scores(query, k=top_k)
        return [(doc, float(score)) for doc, score in results]
    
    async def hybrid_search(
        self, 
        query: str, 
        top_k: int = 5, 
        vector_weight: float = 0.6, 
        bm25_weight: float = 0.4
    ) -> List[Tuple[Document, float]]:
        
        vector_results = await self.vector_search(query, top_k * 2)
        bm25_results = await self.bm25_search(query, top_k * 2)
        
        vector_scores = {doc.page_content: score for doc, score in vector_results}
        bm25_scores = {doc.page_content: score for doc, score in bm25_results}
        
        if vector_scores:
            max_vector_score = max(vector_scores.values())
            min_vector_score = min(vector_scores.values())
            if max_vector_score != min_vector_score:
                vector_scores = {
                    text: (score - min_vector_score) / (max_vector_score - min_vector_score)
                    for text, score in vector_scores.items()
                }
        
        if bm25_scores:
            max_bm25_score = max(bm25_scores.values())
            min_bm25_score = min(bm25_scores.values())
            if max_bm25_score != min_bm25_score:
                bm25_scores = {
                    text: (score - min_bm25_score) / (max_bm25_score - min_bm25_score)
                    for text, score in bm25_scores.items()
                }
        
        combined_results = {}
        all_docs = {doc.page_content: doc for doc, _ in vector_results + bm25_results}
        
        for text, doc in all_docs.items():
            vector_score = vector_scores.get(text, 0)
            bm25_score = bm25_scores.get(text, 0)
            
            hybrid_score = vector_weight * vector_score + bm25_weight * bm25_score
            combined_results[text] = (doc, hybrid_score)
        
        sorted_results = sorted(
            combined_results.values(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return sorted_results[:top_k]
    
    async def search(
        self, 
        query: str, 
        top_k: int = 5, 
        use_hybrid: bool = True
    ) -> List[Tuple[Document, float]]:
        if use_hybrid:
            return await self.hybrid_search(query, top_k)
        else:
            return await self.vector_search(query, top_k)