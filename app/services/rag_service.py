import time
from typing import List, Dict, Any
from openai import OpenAI

from langchain.schema import Document
from app.services.vector_store import VectorStore
from app.services.retriever import HybridRetriever
from app.services.document_processor import DocumentProcessor
from app.models.schemas import QueryResponse, RetrievedDocument
from app.config import settings

class RAGService:
    def __init__(self):
        self.vector_store = VectorStore()
        self.retriever = HybridRetriever(self.vector_store)
        self.document_processor = DocumentProcessor()
        self.openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
        
    async def process_documents(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        # Extract and chunk documents
        chunks = await self.document_processor.process_uploaded_file(file_content, filename)
        
        # Add to vector store
        await self.vector_store.add_documents(chunks)
        
        # Reinitialize BM25 with new documents
        await self.retriever.initialize_bm25()
        
        return {
            "message": f"Successfully processed {filename}",
            "document_count": 1,
            "chunks_created": len(chunks)
        }
    
    async def query(
        self, 
        query: str, 
        top_k: int = 5, 
        use_hybrid: bool = True
    ) -> QueryResponse:
        start_time = time.time()
        
        # Retrieve relevant documents
        retrieved_docs = await self.retriever.search(
            query=query,
            top_k=top_k,
            use_hybrid=use_hybrid
        )
        
        # Prepare context for LLM
        context_docs = []
        retrieved_documents = []
        
        for doc, score in retrieved_docs:
            context_docs.append(doc.page_content)
            retrieved_documents.append(
                RetrievedDocument(
                    content=doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                    metadata=doc.metadata,
                    score=score,
                    source=doc.metadata.get("source", "unknown")
                )
            )
        
        # Generate answer using OpenAI
        context = "\n\n".join(context_docs)
        answer = await self._generate_answer(query, context)
        
        processing_time = time.time() - start_time
        
        return QueryResponse(
            answer=answer,
            retrieved_documents=retrieved_documents,
            query=query,
            processing_time=processing_time
        )
    
    async def _generate_answer(self, query: str, context: str) -> str:
        prompt = f'''You are a helpful assistant that answers questions based on the provided context.
        
Context:
{context}

Question: {query}

Please provide a comprehensive answer based on the context provided. If the context doesn't contain enough information to answer the question, please say so and provide what information you can based on the available context.

Answer: '''

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that provides accurate answers based on the given context."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.1
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"Error generating answer: {str(e)}"
    
    async def get_stats(self) -> Dict[str, Any]:
        return self.vector_store.get_collection_stats()
    
    async def clear_database(self) -> Dict[str, str]:
        await self.vector_store.delete_collection()
        await self.retriever.initialize_bm25()
        return {"message": "Database cleared successfully"}
