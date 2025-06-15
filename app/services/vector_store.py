import chromadb
from typing import List, Dict, Any, Optional
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import Document

from app.config import settings

class VectorStore:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=settings.OPENAI_API_KEY,
            model="text-embedding-ada-002"
        )
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIRECTORY)
        
        # Initialize Langchain Chroma wrapper
        self.vectorstore = Chroma(
            client=self.client,
            collection_name=settings.CHROMA_COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=settings.CHROMA_PERSIST_DIRECTORY
        )
    
    async def add_documents(self, documents: List[Document]) -> None:
        if not documents:
            return
            
        # Add documents to ChromaDB via Langchain wrapper
        self.vectorstore.add_documents(documents)
        
        # Persist the changes
        self.vectorstore.persist()
    
    async def similarity_search(
        self, 
        query: str, 
        k: int = 5, 
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        return self.vectorstore.similarity_search(
            query=query,
            k=k,
            filter=filter
        )
    
    async def similarity_search_with_scores(
        self, 
        query: str, 
        k: int = 5, 
        filter: Optional[Dict[str, Any]] = None
    ) -> List[tuple]:
        return self.vectorstore.similarity_search_with_score(
            query=query,
            k=k,
            filter=filter
        )
    
    async def get_all_documents(self) -> List[Document]:
        collection = self.client.get_collection(settings.CHROMA_COLLECTION_NAME)
        results = collection.get()
        
        documents = []
        for i, doc_id in enumerate(results['ids']):
            doc = Document(
                page_content=results['documents'][i],
                metadata=results['metadatas'][i] if results['metadatas'] else {}
            )
            documents.append(doc)
        
        return documents
    
    async def delete_collection(self) -> None:
        try:
            self.client.delete_collection(settings.CHROMA_COLLECTION_NAME)
        except Exception as e:
            print(f"Error deleting collection: {e}")
    
    def get_collection_stats(self) -> Dict[str, Any]:
        try:
            collection = self.client.get_collection(settings.CHROMA_COLLECTION_NAME)
            return {
                "document_count": collection.count(),
                "collection_name": collection.name
            }
        except Exception as e:
            return {"document_count": 0, "error": str(e)}