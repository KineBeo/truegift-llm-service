import os
from typing import AsyncGenerator
import httpx
from groq import Groq
from .config import config
# Initialize Groq client with API key from environment
groq_api_key = config["GROQ_API_KEY"]
groq_client = Groq(api_key=groq_api_key) if groq_api_key else None
default_model = config["CLOUD_MODEL"]

async def check_groq_status():
    """
    Check if Groq API is accessible and return status information
    """
    if not groq_api_key:
        return {"status": "error", "detail": "GROQ_API_KEY environment variable not set"}
    
    try:
        # List available models
        available_models = [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-versatile",
            "mixtral-8x7b-32768",
            "gemma-7b-it"
        ]
        
        return {
            "status": "running", 
            "models": available_models
        }
    except Exception as e:
        return {"status": "error", "detail": str(e)}

async def ask_groq(prompt: str, model: str = None, temperature: float = 0.7, max_tokens: int = 1024):
    """
    Ask a question to the Groq model and get a complete response
    """
    if not groq_api_key:
        raise Exception("GROQ_API_KEY environment variable not set")
    
    # Always use default_model if model is None
    if model is None:
        model = default_model
        
    try:
        response = groq_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_completion_tokens=max_tokens
        )
        
        return response.choices[0].message.content
    except Exception as e:
        raise Exception(f"Error querying Groq: {str(e)}")

async def stream_groq(
    prompt: str, 
    model: str = None,
    temperature: float = 0.7, 
    max_tokens: int = 1024
) -> AsyncGenerator[str, None]:
    """
    Stream responses from Groq API
    """
    if not groq_api_key:
        yield "Error: GROQ_API_KEY environment variable not set"
        return
    
    # Always use default_model if model is None
    if model is None:
        model = default_model
    
    try:
        stream = groq_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_completion_tokens=max_tokens,
            stream=True
        )
        
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception as e:
        yield f"Error: {str(e)}" 