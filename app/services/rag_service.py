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

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
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
                name="documents",
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("ChromaDB initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing ChromaDB: {str(e)}")
            logger.error(traceback.format_exc())
            # Continue without ChromaDB for now
        
        self.openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
        
        logger.info(f"RAGService initialized with text splitter (chunk_size={settings.MAX_CHUNK_SIZE}, chunk_overlap={settings.CHUNK_OVERLAP})")
        
    async def process_document(self, file_path: str) -> List[Document]:
        """Process a document and return its chunks"""
        try:
            logger.debug(f"Processing document: {file_path}")
            
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Determine file type and load document
            if file_path.endswith('.pdf'):
                logger.debug("Using PyPDFLoader for PDF file")
                try:
                    loader = PyPDFLoader(file_path)
                except Exception as e:
                    logger.error(f"Error creating PyPDFLoader: {str(e)}")
                    raise
            elif file_path.endswith('.docx'):
                logger.debug("Using Docx2txtLoader for DOCX file")
                try:
                    loader = Docx2txtLoader(file_path)
                except Exception as e:
                    logger.error(f"Error creating Docx2txtLoader: {str(e)}")
                    raise
            else:
                raise ValueError(f"Unsupported file type: {file_path}")

            # Load and split document
            logger.debug("Loading document with LangChain loader")
            try:
                documents = loader.load()
                logger.info(f"Loaded document with {len(documents)} pages")
            except Exception as e:
                logger.error(f"Error loading document: {str(e)}")
                raise
            
            # Split documents into chunks
            logger.debug("Splitting document into chunks")
            try:
                chunks = self.text_splitter.split_documents(documents)
                logger.info(f"Split document into {len(chunks)} chunks")
            except Exception as e:
                logger.error(f"Error splitting document: {str(e)}")
                raise
            
            # Add metadata to chunks
            logger.debug("Adding metadata to chunks")
            for chunk in chunks:
                chunk.metadata.update({
                    "source": file_path,
                    "chunk_size": len(chunk.page_content)
                })
            
            return chunks

        except Exception as e:
            logger.error(f"Error processing document {file_path}: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

    async def process_files(self, files: List[UploadFile]) -> dict:
        """Process multiple files and store their chunks"""
        try:
            logger.debug(f"Processing {len(files)} files")
            total_chunks = 0
            
            for file in files:
                logger.debug(f"Processing file: {file.filename}")
                
                # Create a temporary file to store the uploaded content
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
                    logger.debug(f"Created temporary file: {temp_file.name}")
                    
                    try:
                        content = await file.read()
                        logger.debug(f"Read {len(content)} bytes from uploaded file")
                        
                        temp_file.write(content)
                        temp_file.flush()
                        logger.debug("Wrote content to temporary file")
                        
                        # Process the temporary file
                        chunks = await self.process_document(temp_file.name)
                        self.documents.extend(chunks)
                        total_chunks += len(chunks)
                        logger.info(f"Processed {file.filename}: {len(chunks)} chunks")
                    except Exception as e:
                        logger.error(f"Error processing file {file.filename}: {str(e)}")
                        logger.error(traceback.format_exc())
                        raise HTTPException(
                            status_code=500,
                            detail=f"Error processing file {file.filename}: {str(e)}"
                        )
                    finally:
                        # Clean up the temporary file
                        try:
                            os.unlink(temp_file.name)
                            logger.debug("Cleaned up temporary file")
                        except Exception as e:
                            logger.warning(f"Error cleaning up temporary file: {str(e)}")

            logger.info(f"Total chunks created: {total_chunks}")
            return {
                "message": f"Successfully processed {len(files)} file(s)",
                "document_count": len(files),
                "chunks_created": total_chunks
            }

        except Exception as e:
            logger.error(f"Error processing files: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Error processing files: {str(e)}")

    async def query(self, query: str, session_id: str, user_id: str) -> str:
        """Query the RAG system"""
        try:
            # Get recent chat history (last 5 messages)
            chat_history = self.chat_service.get_session_messages(session_id, user_id, limit=5)
            
            # Get collection stats
            stats = self.vector_store.get_collection_stats()
            if stats["document_count"] == 0:
                return "I am a RAG system. Please upload some documents first so I can answer your questions."
            
            # Perform hybrid search
            results = await self.retriever.search(
                query=query,
                top_k=5,
                use_hybrid=True
            )
            
            if not results:
                return "I couldn't find any relevant information in the documents to answer your question."
            
            # Format the retrieved context
            context = "\n\n".join([doc.page_content for doc, _ in results])
            
            # Format chat history
            chat_history_text = ""
            if chat_history:
                chat_history_text = "\n".join([
                    f"{'User' if msg.role == 'user' else 'Assistant'}: {msg.content}"
                    for msg in chat_history
                ])
            
            # Prepare the prompt for OpenAI
            prompt = f"""You are a helpful AI assistant. Use the following context and chat history to answer the user's question.
            If the context doesn't contain relevant information, say so.

            Context from documents:
            {context}

            Recent chat history:
            {chat_history_text}

            User's question: {query}

            Please provide a clear and concise answer based on the context and chat history:"""

            # Call OpenAI API
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant that answers questions based on provided context and chat history."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            # Extract and return the response
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            logger.error(traceback.format_exc())
            raise

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
