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
                name=settings.CHROMA_COLLECTION_NAME,
                metadata={"hnsw:space": settings.CHROMA_SPACE}
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
            
            # Check file extension
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext not in settings.ALLOWED_FILE_TYPES:
                raise ValueError(f"Unsupported file type: {file_ext}. Allowed types: {settings.ALLOWED_FILE_TYPES}")
            
            # Determine file type and load document
            if file_ext == '.pdf':
                logger.debug("Using PyPDFLoader for PDF file")
                try:
                    loader = PyPDFLoader(file_path)
                except Exception as e:
                    logger.error(f"Error creating PyPDFLoader: {str(e)}")
                    raise
            elif file_ext == '.docx':
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
                    "chunk_size": settings.MAX_CHUNK_SIZE,
                    "chunk_overlap": settings.CHUNK_OVERLAP
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

    async def is_chat_history_relevant(self, query: str, session: ChatSession) -> bool:
        """Use OpenAI to determine if chat history is relevant to the current query"""
        try:
            if not session or not session.messages:
                return False

            # Format chat history for analysis
            chat_context = "\n".join([
                f"{'User' if msg.role == 'user' else 'Assistant'}: {msg.content}"
                for msg in session.messages[-3:]  # Last 3 messages
            ])

            # Create prompt for relevance check
            prompt = f"""Analyze if the following chat history is relevant to the user's current question.
            Consider:
            1. If the current question refers to or builds upon previous conversation
            2. If the context from previous messages would help answer the current question
            3. If the question is a follow-up or clarification of previous topics

            Chat History:
            {chat_context}

            Current Question: {query}

            Respond with only 'yes' if the chat history is relevant, or 'no' if it's not relevant."""

            # Call OpenAI API for relevance check
            response = self.openai_client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that determines if chat history is relevant to a current question."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Low temperature for more consistent responses
                max_tokens=10  # We only need a yes/no response
            )

            # Get and clean the response
            relevance_response = response.choices[0].message.content.strip().lower()
            logger.info(f"Chat history relevance check response: {relevance_response}")
            
            return relevance_response.startswith('yes')

        except Exception as e:
            logger.error(f"Error checking chat history relevance: {str(e)}")
            logger.error(traceback.format_exc())
            return False  # Default to not using chat history if there's an error

    async def query(self, query: str, session_id: str, user_id: str) -> str:
        """Query the RAG system"""
        try:
            # Get chat history
            session = self.chat_service.get_session(session_id, user_id)
            if not session:
                raise ValueError("Session not found or does not belong to user")
            
            logger.info(f"Retrieved chat session: {session.id}")
            
            # Check if chat history is relevant
            is_relevant = await self.is_chat_history_relevant(query, session)
            logger.info(f"Chat history relevance check result: {is_relevant}")
            
            # Get chat history for context
            chat_history = session.messages[-5:] if is_relevant else []  # Last 5 messages if relevant
            logger.info(f"Retrieved {len(chat_history)} messages from chat history")
            
            # Format last 3 messages for context
            chat_context = ""
            if is_relevant and chat_history:
                chat_context = "\n".join([
                    f"{'User' if msg.role == 'user' else 'Assistant'}: {msg.content}"
                    for msg in chat_history[-3:]  # Last 3 messages
                ])
                logger.info(f"Formatted chat context: {chat_context}")
            
            # Combine query with chat context if relevant
            enhanced_query = f"{query}\nContext from previous conversation:\n{chat_context}" if is_relevant else query
            logger.info(f"Enhanced query: {enhanced_query}")
            
            # Perform hybrid search
            results = await self.retriever.search(enhanced_query)
            logger.info(f"Retrieved {len(results)} documents from hybrid search")
            
            if not results:
                return "I don't have enough information to answer your question. Please upload some documents first."
            
            # Format context from retrieved documents (each result is a tuple of (Document, score))
            context = "\n\n".join([doc.page_content for doc, _ in results])
            logger.info(f"Formatted context from {len(results)} documents")
            
            # Format chat history for the prompt
            chat_history_text = ""
            if is_relevant and chat_history:
                chat_history_text = "\n".join([
                    f"{'User' if msg.role == 'user' else 'Assistant'}: {msg.content}"
                    for msg in chat_history
                ])
                logger.info(f"Formatted chat history for prompt: {chat_history_text}")
            
            # Create prompt with chat history if relevant
            prompt = f"""Use the following pieces of context to answer the question at the end.
            If you don't know the answer, just say that you don't know, don't try to make up an answer.
            
            Context:
            {context}
            
            {f'Previous conversation: {chat_history_text} ' if is_relevant else ''}
            
            Question: {query}
            
            Answer:"""
            
            logger.info("Sending prompt to OpenAI")
            
            # Get response from OpenAI
            response = self.openai_client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            
            answer = response.choices[0].message.content
            logger.info(f"Received response from OpenAI: {answer}")
            
            return answer
            
        except Exception as e:
            logger.error(f"Error in RAG query: {str(e)}")
            logger.error(traceback.format_exc())
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
