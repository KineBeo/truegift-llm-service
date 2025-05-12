from typing import Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
import uvicorn
from .backend_client import BackendClient
from .ollama_client import check_ollama_status, ask_ollama, stream_ollama
from .groq_client import check_groq_status, ask_groq, stream_groq
from .rag_indexer import process_and_index_photos, collection
from .suggestion_service import generate_suggestion_by_prompt, get_available_prompts
from pydantic import BaseModel

app = FastAPI(title="TrueGift RAG Indexer", debug=False)
client = BackendClient()

class OllamaRequest(BaseModel):
    prompt: str

class GroqRequest(BaseModel):
    prompt: str
    model: str
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1024

class ChatRequest(BaseModel):
    prompt: str
    model: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = True
    provider: Optional[str] = "groq"  # Change default from "ollama" to "groq"

@app.get("/index-rag")
async def index_photos_for_current_user(
    auth_token: Optional[str] = None
):

    token = auth_token
    try:
       result = await process_and_index_photos(auth_token=token, max_photos=50)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return result

@app.get("/query-food-photos")
def query_food_photos(user_id: Optional[str] = Query(None), limit: int = 10):
    """Truy vấn các ảnh món ăn đã được index trong ChromaDB"""

    # Xây dựng filter
    where_filter = {"is_food": True}
    if user_id:
        where_filter["user_id"] = user_id

    try:
        results = collection.query(
            query_texts=["món ăn"],
            n_results=limit,
            where=where_filter,
            include=["metadatas", "documents"]
        )

        response = []
        for meta, caption in zip(results["metadatas"][0], results["documents"][0]):
            response.append({
                "photo_id": meta["photo_id"],
                "user_id": meta["user_id"],
                "user_name": meta["user_name"],
                "food_class": meta["food_class"],
                "created_at": meta["created_at"],
                "caption": caption
            })

        return {"status": "ok", "results": response}

    except Exception as e:
        return {"status": "error", "detail": str(e)}

@app.get("/check-ollama-status")
async def check_status():
    """
    Check if Ollama server is running and return available models
    """
    result = await check_ollama_status()
    return result

@app.get("/check-groq-status")
async def check_groq():
    """
    Check if Groq API is accessible and return available models
    """
    result = await check_groq_status()
    return result

@app.post("/ask-ollama")
async def query_ollama(request: OllamaRequest):
    """
    Ask a question to the Ollama model
    """
    try:
        # Get the full response from Ollama
        response = await ask_ollama(request.prompt)
        
        # Return the complete response
        return {"status": "success", "response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying Ollama: {str(e)}")

@app.post("/ask-groq")
async def query_groq(request: GroqRequest):
    """
    Ask a question to the Groq model
    """
    try:
        # Get the full response from Groq
        response = await ask_groq(
            prompt=request.prompt, 
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        
        # Return the complete response
        return {"status": "success", "response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying Groq: {str(e)}")

@app.post("/api/chat")
async def chat_with_llm(request: ChatRequest):
    """
    Endpoint for streaming chat with LLM models that matches frontend expectations
    """
    try:
        # Choose provider based on request
        if request.provider == "ollama":  # Change logic to default to groq unless "ollama" is explicitly specified
            if request.stream:
                # Create a generator for streaming the response from Ollama
                return StreamingResponse(
                    stream_ollama(request.prompt, temperature=request.temperature),
                    media_type="text/plain"
                )
            else:
                # For non-streaming requests, use the existing ollama function
                response = await ask_ollama(request.prompt)
                return {"response": response}
        else:  # Default to groq
            # Set default values for parameters if not provided
            max_tokens = request.max_tokens if request.max_tokens is not None else 1024
            
            if request.stream:
                # Create a generator for streaming the response from Groq
                return StreamingResponse(
                    stream_groq(
                        prompt=request.prompt, 
                        model=request.model,  # model will be handled in stream_groq
                        temperature=request.temperature,
                        max_tokens=max_tokens
                    ),
                    media_type="text/plain"
                )
            else:
                # For non-streaming requests, use the groq function
                response = await ask_groq(
                    prompt=request.prompt,
                    temperature=request.temperature,
                    max_tokens=max_tokens
                )
                return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in chat: {str(e)}")

@app.get("/suggest/prompts")
async def list_prompt_options():
    """
    Trả về danh sách các gợi ý (prompt_key) khả dụng để hiển thị trên UI.
    """
    return {"available_prompts": get_available_prompts()}

@app.get("/suggest/{user_id}/{prompt_key}")
async def suggest_with_prompt(user_id: str, prompt_key: str):
    result = await generate_suggestion_by_prompt(user_id, prompt_key)
    return {"suggestion": result}
