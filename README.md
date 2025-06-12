# Hybrid RAG System

A robust Retrieval-Augmented Generation (RAG) system that combines vector search with BM25 for improved document retrieval and response generation.

## Features

- Document ingestion and processing
- Hybrid retrieval (Vector + BM25)
- User authentication and session management
- Secure document handling
- Content moderation and safety checks
- Persistent storage with PostgreSQL
- Docker deployment support

## Prerequisites

- Python 3.10+
- Docker and Docker Compose
- OpenAI API key
- PostgreSQL (if running without Docker)

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
SECRET_KEY=your_secret_key
DATABASE_URL=postgresql://user:password@localhost:5432/dbname
DB_USER=postgres
DB_PASSWORD=your_password
DB_NAME=rag_db
```

## Running the Application

### Development

```bash
uvicorn app.main:app --reload
```

### Production (Docker)

```bash
docker-compose up -d
```

## API Documentation

Once the application is running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Security Features

- JWT-based authentication
- Input sanitization
- Content moderation
- Rate limiting
- CORS configuration
- Secure password hashing

## Deployment

1. Build the Docker image:
```bash
docker-compose build
```

2. Start the services:
```bash
docker-compose up -d
```

3. Monitor logs:
```bash
docker-compose logs -f
```

## Monitoring and Maintenance

- Application logs are available in Docker logs
- Database backups should be configured
- Regular security updates should be applied
- Monitor system resources and performance

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 