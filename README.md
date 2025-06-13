# Hybrid RAG System

A robust Retrieval-Augmented Generation (RAG) system that combines vector search with BM25 for improved document retrieval and response generation. The system uses ChromaDB for vector storage, OpenAI for embeddings and LLM, and includes chat history integration for context-aware responses.

## Features

- **Document Processing**
  - Support for PDF and DOCX files
  - Configurable chunk size and overlap
  - Automatic metadata extraction
  - LangChain document processing pipeline

- **Hybrid Retrieval**
  - Vector search using OpenAI embeddings
  - BM25 keyword-based search
  - Intelligent combination of both methods
  - Configurable weights for hybrid scoring

- **Chat System**
  - Session-based chat management
  - Chat history integration with RAG
  - Context-aware responses
  - Intelligent chat history relevance detection

- **Vector Storage**
  - ChromaDB integration
  - Persistent storage
  - Efficient similarity search
  - Document metadata management

## Prerequisites

- Python 3.10+
- OpenAI API key
- SQLite (default) or PostgreSQL database

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd hybrid-rag-system
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

## Configuration

Create a `.env` file with the following variables:

```env
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-3.5-turbo
CHROMA_PERSIST_DIRECTORY=./data/chroma
DATABASE_URL=sqlite:///./data/app.db
```

## Running the Application

### Development

```bash
uvicorn app.main:app --reload
```

## API Endpoints

### Document Management
- `POST /upload` - Upload documents for processing
- `GET /documents` - Get document statistics

### Chat and Query
- `POST /sessions` - Create a new chat session
- `GET /sessions/{user_id}` - Get user's chat sessions
- `GET /sessions/{session_id}/messages` - Get session messages
- `POST /chat/{session_id}` - Send a message in a session
- `POST /query` - Query the RAG system

## Example Usage

1. Create a new chat session:
```bash
curl -X POST http://localhost:8000/sessions \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user123"}'
```

2. Upload a document:
```bash
curl -X POST http://localhost:8000/upload \
  -F "files=@document.pdf" \
  -F "user_id=user123"
```

3. Send a message:
```bash
curl -X POST http://localhost:8000/chat/{session_id} \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is RAG?",
    "user_id": "user123"
  }'
```

## Project Structure

```
app/
├── config.py           # Configuration settings
├── main.py            # FastAPI application
├── models/            # Database models
├── services/          # Core services
│   ├── rag_service.py    # RAG implementation
│   ├── chat_service.py   # Chat management
│   ├── vector_store.py   # Vector storage
│   └── retriever.py      # Hybrid retrieval
└── db/                # Database setup
```

## Features in Detail

### Hybrid Retrieval
The system combines vector search and BM25 to provide better search results:
- Vector search captures semantic similarity
- BM25 handles keyword matching
- Results are combined and re-ranked based on configurable weights

### Chat History Integration
- Maintains conversation context
- Uses LLM to determine chat history relevance
- Enhances query understanding with previous context
- Provides more coherent and context-aware responses

### Document Processing
- Automatic text extraction from PDF and DOCX
- Configurable chunking with overlap
- Metadata preservation
- Efficient storage and retrieval

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 