# Groq Integration for TrueGift LLM Service

This document explains how to set up and use the Groq LLM integration in the TrueGift application.

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Set your Groq API key in the config file or as an environment variable.

3. Start the FastAPI server:
```bash
uvicorn app.main:app --reload
```

## Configuration

The default model used by Groq can be configured in the config file:

```python
# Example config.py
config = {
    "GROQ_API_KEY": "your-api-key-here",
    "CLOUD_MODEL": "llama-3.3-70b-versatile"  # Default model
}
```

## Available Endpoints

### Check Groq Status
```
GET /check-groq-status
```
Checks if the Groq API is accessible and returns available models.

### Ask Groq (Non-streaming)
```
POST /ask-groq
```
Request body:
```json
{
  "prompt": "Your prompt here",
  "model": "llama-3.3-70b-versatile",  // Optional - uses default if not provided
  "temperature": 0.7,
  "max_tokens": 1024
}
```
All parameters except `prompt` are optional.

### Chat API (Unified Endpoint)
```
POST /api/chat
```
Request body:
```json
{
  "prompt": "Your prompt here",
  "model": "llama-3.3-70b-versatile",  // Optional - uses default if not provided
  "temperature": 0.7,
  "max_tokens": 1024,
  "stream": true,
  "provider": "groq"
}
```
All parameters except `prompt` are optional. **Groq is now the default provider**. Set `provider` to "ollama" to use local Ollama models instead.

## Available Models

The Groq integration defaults to the model configured in your config settings, which is typically one of:

- llama-3.3-70b-versatile
- llama-3.1-8b-versatile
- mixtral-8x7b-32768
- gemma-7b-it

## Example Usage

### Python Client Example
```python
import requests

# Non-streaming request using Groq (default)
response = requests.post(
    "http://localhost:8000/api/chat",
    json={
        "prompt": "What is artificial intelligence?",
        "stream": False
    }
)
print(response.json()["response"])

# Non-streaming request with specific model
response = requests.post(
    "http://localhost:8000/api/chat",
    json={
        "prompt": "What is artificial intelligence?",
        "model": "llama-3.3-70b-versatile",
        "stream": False
    }
)
print(response.json()["response"])

# Non-streaming request using Ollama
response = requests.post(
    "http://localhost:8000/api/chat",
    json={
        "prompt": "What is artificial intelligence?",
        "provider": "ollama",
        "stream": False
    }
)
print(response.json()["response"])

# Streaming request using Groq (default)
response = requests.post(
    "http://localhost:8000/api/chat",
    json={
        "prompt": "Explain quantum computing",
        "stream": True
    },
    stream=True
)

for chunk in response.iter_content(chunk_size=None):
    if chunk:
        print(chunk.decode('utf-8'), end='')
```

### JavaScript/TypeScript Client Example
```typescript
// Non-streaming request using Groq (default)
const response = await fetch('http://localhost:8000/api/chat', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    prompt: 'What is artificial intelligence?',
    stream: false
  }),
});
const data = await response.json();
console.log(data.response);

// Streaming request using Groq (default)
const streamResponse = await fetch('http://localhost:8000/api/chat', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    prompt: 'Explain quantum computing',
    stream: true
  }),
});

const reader = streamResponse.body.getReader();
const decoder = new TextDecoder();
let buffer = '';

while (true) {
  const { value, done } = await reader.read();
  if (done) break;
  
  buffer += decoder.decode(value, { stream: true });
  console.log(buffer);
}
``` 