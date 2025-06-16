from app.services.chat_service import ChatService
from app.config import settings
import chromadb
from chromadb.config import Settings
from openai import OpenAI
import logging
from typing import Dict, Any, Optional, List
import PyPDF2
import io
from langchain.text_splitter import RecursiveCharacterTextSplitter
from fastapi import HTTPException, UploadFile
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.schema import Document
import os
import tempfile
import traceback
from app.db.database import get_db
from app.services.vector_store import VectorStore
from app.services.retriever import HybridRetriever
from app.models.chat import ChatSession

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGService:
    def __init__(self):
        self.chat_service = ChatService(next(get_db()))
        self.vector_store = VectorStore()
        self.retriever = HybridRetriever(self.vector_store)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.MAX_CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        self.documents: List[Document] = []
        
        # Initialize ChromaDB client
        try:
            self.client = chromadb.Client(Settings(
                persist_directory=settings.CHROMA_PERSIST_DIRECTORY
            ))
            self.collection = self.client.get_or_create_collection(
                name=settings.CHROMA_COLLECTION_NAME,
                metadata={"hnsw:space": settings.CHROMA_SPACE}
            )
        except Exception as e:
            logger.error(f"Error initializing ChromaDB: {str(e)}")
            raise
        
        self.openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)

    async def process_document(self, file_path: str) -> List[Document]:
        """Process a document and return its chunks"""
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext not in settings.ALLOWED_FILE_TYPES:
                raise ValueError(f"Unsupported file type: {file_ext}")
            
            if file_ext == '.pdf':
                loader = PyPDFLoader(file_path)
            elif file_ext == '.docx':
                loader = Docx2txtLoader(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_path}")

            documents = loader.load()
            chunks = self.text_splitter.split_documents(documents)
            
            for chunk in chunks:
                chunk.metadata.update({
                    "source": file_path,
                    "chunk_size": settings.MAX_CHUNK_SIZE,
                    "chunk_overlap": settings.CHUNK_OVERLAP
                })
            
            return chunks

        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    async def process_files(self, files: List[UploadFile]) -> dict:
        """Process multiple files and store their chunks"""
        try:
            total_chunks = 0
            
            for file in files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
                    content = await file.read()
                    temp_file.write(content)
                    temp_file.flush()
                    
                    chunks = await self.process_document(temp_file.name)
                    self.documents.extend(chunks)
                    total_chunks += len(chunks)
                    
                    os.unlink(temp_file.name)
            
            return {
                "status": "success",
                "message": f"Processed {total_chunks} chunks from {len(files)} files",
                "chunks": total_chunks
            }
            
        except Exception as e:
            logger.error(f"Error processing files: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    async def is_chat_history_relevant(self, query: str, session: ChatSession) -> bool:
        """Use OpenAI to determine if chat history is relevant to the current query"""
        try:
            if not session or not session.messages:
                return False

            chat_context = "\n".join([
                f"{'User' if msg.role == 'user' else 'Assistant'}: {msg.content}"
                for msg in session.messages[-3:]
            ])

            prompt = f"""Analyze if the following chat history is relevant to the user's current question.
            Consider:
            1. If the current question refers to or builds upon previous conversation
            2. If the context from previous messages would help answer the current question
            3. If the question is a follow-up or clarification of previous topics
            NOTE: Answer Yes only if chat history is very relevant

            Chat History:
            {chat_context}

            Current Question: {query}

            Respond with only 'yes' if the chat history is relevant, or 'no' if it's not relevant."""

            response = self.openai_client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that determines if chat history is relevant to a current question."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=10
            )

            relevance_response = response.choices[0].message.content.strip().lower()
            return relevance_response.startswith('yes')

        except Exception as e:
            logger.error(f"Error checking chat history relevance: {str(e)}")
            return False

    async def query(self, query: str, session_id: str, user_id: str) -> str:
        """Query the RAG system"""
        try:
            session = self.chat_service.get_session(session_id, user_id)
            if not session:
                raise ValueError("Session not found or does not belong to user")
            
            is_relevant = await self.is_chat_history_relevant(query, session)
            chat_history = session.messages[-5:] if is_relevant else []
            
            chat_context = ""
            if is_relevant and chat_history:
                chat_context = "\n".join([
                    f"{'User' if msg.role == 'user' else 'Assistant'}: {msg.content}"
                    for msg in chat_history[-3:]
                ])
            
            enhanced_query = f"{query}\nContext from previous conversation:\n{chat_context}" if is_relevant else query
            results = await self.retriever.search(enhanced_query)
            
            if not results:
                return "I don't have enough information to answer your question. Please upload some documents first."
            
            context = "\n\n".join([doc.page_content for doc, _ in results])
            
            chat_history_text = ""
            if is_relevant and chat_history:
                chat_history_text = "\n".join([
                    f"{'User' if msg.role == 'user' else 'Assistant'}: {msg.content}"
                    for msg in chat_history
                ])
            
            prompt = f"""Use the following pieces of context to answer the question at the end.
            If you don't know the answer, just say that you don't know, don't try to make up an answer.
            
            Context:
            {context}
            
            {f'Previous conversation: {chat_history_text} ' if is_relevant else ''}
            
            Question: {query}
            
            Answer:"""
            
            response = self.openai_client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            
            return response.choices[0].message.content, results
            
        except Exception as e:
            logger.error(f"Error in RAG query: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    def get_document_count(self) -> int:
        """Get the number of loaded documents"""
        return len(self.documents)

    def get_chunk_count(self) -> int:
        """Get the total number of chunks"""
        return len(self.documents)

    async def process_documents(self, content: bytes, filename: str) -> Dict[str, Any]:
        """Process uploaded documents"""
        try:
            logger.info(f"Processing document: {filename}")
            # For now, just return a success message
            return {
                "message": "Document processing is not implemented yet",
                "chunks_created": 0
            }
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    def _split_document(self, content: bytes, filename: str) -> list:
        """Split document into chunks based on file type"""
        if filename.lower().endswith('.pdf'):
            return self._process_pdf(content)
        elif filename.lower().endswith('.txt'):
            return self._process_text(content)
        else:
            raise ValueError(f"Unsupported file type: {filename}")

    def _process_pdf(self, content: bytes) -> list:
        """Process PDF file and return chunks"""
        try:
            # Create a PDF reader object
            pdf_file = io.BytesIO(content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            # Extract text from each page
            texts = []
            for i, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                if text.strip():  # Only add non-empty pages
                    texts.append({
                        "content": text,
                        "metadata": {"page": i + 1}
                    })
            
            # Combine all text and split into chunks
            combined_text = "\n\n".join([t["content"] for t in texts])
            chunks = self.text_splitter.create_documents(
                [combined_text],
                metadatas=[{"page": t["metadata"]["page"]} for t in texts]
            )
            
            return chunks
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            raise

    def _process_text(self, content: bytes) -> list:
        """Process text file and return chunks"""
        try:
            text = content.decode('utf-8')
            # Split into chunks using LangChain's text splitter
            chunks = self.text_splitter.create_documents([text])
            return chunks
        except Exception as e:
            logger.error(f"Error processing text file: {str(e)}")
            raise

    async def process_file(self, file: UploadFile) -> dict:
        """Process a single file"""
        temp_file = None
        try:
            # Create a temporary file to store the upload
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1])
            # Write the uploaded file to the temporary file
            content = await file.read()
            temp_file.write(content)
            temp_file.flush()
            temp_file.close()
            
            # Load document based on file type
            if file.filename.endswith('.pdf'):
                loader = PyPDFLoader(temp_file.name)
            elif file.filename.endswith('.docx'):
                loader = Docx2txtLoader(temp_file.name)
            else:
                raise ValueError(f"Unsupported file type: {file.filename}")
            
            # Load and split the document
            documents = loader.load()
            chunks = self.text_splitter.split_documents(documents)
            
            # Add metadata to chunks
            for chunk in chunks:
                chunk.metadata.update({
                    "source": file.filename,
                    "chunk_size": settings.MAX_CHUNK_SIZE,
                    "chunk_overlap": settings.CHUNK_OVERLAP
                })
            
            # Store chunks in vector store
            await self.vector_store.add_documents(chunks)
            
            return {
                "status": "success",
                "filename": file.filename,
                "message": f"Processed {len(chunks)} chunks from document",
                "chunks": len(chunks)
            }
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            raise
        finally:
            # Clean up temporary file
            if temp_file and os.path.exists(temp_file.name):
                try:
                    os.unlink(temp_file.name)
                except Exception as e:
                    logger.error(f"Error cleaning up temporary file: {str(e)}")
