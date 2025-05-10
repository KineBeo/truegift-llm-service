# TrueGift LLM Service

A microservice for handling image recognition, vector embeddings, and semantic search for Vietnamese food images using Chroma DB, YOLO models, and sentence transformers.

## Features

- Image classification and recognition using YOLO models
- Vector embeddings for images using SentenceTransformers
- Semantic search capabilities with ChromaDB
- FastAPI backend for efficient API endpoints

## Setup

1. Clone the repository
```bash
git clone <repository-url>
cd truegift-llm-service
```

2. Set up environment variables
```bash
cp env.sample .env
# Edit .env with your configuration
```

3. Run the setup script
```bash
chmod +x start.sh
./start.sh
```

The script will:
- Create a Python virtual environment
- Activate the virtual environment
- Install required dependencies

## Running the Application

### Using FastAPI CLI (Recommended)
```bash
# Activate the virtual environment
source venv/bin/activate

# Install fastapi command line tools if not already installed
pip install fastapi-cli

# Run the application on port 9000
fastapi dev app/main.py --port 9000
```

This method provides automatic reloading when code changes are detected, which is ideal for development.

### Starting Application in Production
```bash
# Start with increased worker count (use CPU count)
cd truegift-llm-service
source venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 9000 --workers 4
```

### API Documentation
Once the server is running, you can access the API documentation at:
- Swagger UI: http://localhost:9000/docs
- ReDoc: http://localhost:9000/redoc

## Project Structure

- `/app` - Application source code
- `/weights` - Pre-trained model weights
- `/chroma_db` - Persistent vector database storage

## Dependencies

- FastAPI - Web framework
- Uvicorn - ASGI server
- YOLO - Object detection and classification models
- SentenceTransformers - For creating embeddings
- ChromaDB - Vector database for semantic search

## Development

To contribute to this project, please read the [CONTRIBUTING.md](CONTRIBUTING.md) file for guidelines.

## License

This project is licensed under the terms specified in the [LICENSE](LICENSE) file. 