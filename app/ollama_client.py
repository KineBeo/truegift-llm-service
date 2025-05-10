import httpx
import asyncio
import json
from .config import OLLAMA_BASE_URL, OLLAMA_MODEL

async def check_ollama_status() -> dict:
    try:
        async with httpx.AsyncClient() as client:
            # Use the /api/tags endpoint to list models
            resp = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            resp.raise_for_status()
            
            # Return the full response as a dictionary
            return {
                "status": "ready" if resp.status_code == 200 else "error",
                "models": resp.json().get("models", []),
                "message": "Ollama server is running"
            }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to connect to Ollama server: {str(e)}"
        }

async def ask_ollama(prompt: str, temperature: float = 0.7) -> str:
    """
    Ask Ollama a question and get a response.
    Handles both streaming and non-streaming responses.
    """
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": True,  # Using streaming mode
        "temperature": temperature
    }
    
    full_response = ""
    
    try:
        async with httpx.AsyncClient() as client:
            async with client.stream("POST", f"{OLLAMA_BASE_URL}/api/generate", 
                                    json=payload, timeout=60.0) as response:
                response.raise_for_status()
                
                # Process the streaming response line by line
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                        
                    try:
                        chunk = json.loads(line)
                        if "response" in chunk:
                            full_response += chunk["response"]
                            
                    except json.JSONDecodeError:
                        # Skip invalid JSON lines
                        continue
        
        return full_response
        
    except Exception as e:
        # If streaming fails, fall back to non-streaming
        print(f"Error with streaming: {str(e)}, falling back to non-streaming")
        payload["stream"] = False
        
        async with httpx.AsyncClient() as client:
            resp = await client.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["response"]

async def stream_ollama(prompt: str, temperature: float = 0.7):
    """
    Stream Ollama responses to the client character by character to match frontend expectations.
    This function returns a generator that yields text incrementally.
    """
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": True,
        "temperature": temperature
    }
    
    try:
        async with httpx.AsyncClient() as client:
            async with client.stream("POST", f"{OLLAMA_BASE_URL}/api/generate", 
                                    json=payload, timeout=60.0) as response:
                response.raise_for_status()
                
                # Process the streaming response line by line
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                        
                    try:
                        chunk = json.loads(line)
                        if "response" in chunk:
                            # Send each character separately to enable the streaming effect in the frontend
                            yield chunk["response"]
                            
                    except json.JSONDecodeError:
                        # Skip invalid JSON lines
                        continue
                
    except Exception as e:
        # If streaming fails, fall back to non-streaming
        print(f"Error with streaming in generator: {str(e)}")
        # Send error notification to the stream
        yield f"Error connecting to AI model: {str(e)}"
