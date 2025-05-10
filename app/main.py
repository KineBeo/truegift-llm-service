from typing import Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
import uvicorn
from .backend_client import BackendClient
from .ollama_client import check_ollama_status, ask_ollama, stream_ollama
from .rag_indexer import process_and_index_photos, collection
from .suggestion_service import generate_suggestion_by_prompt, get_available_prompts
from pydantic import BaseModel

app = FastAPI(title="TrueGift RAG Indexer", debug=False)
client = BackendClient()

class OllamaRequest(BaseModel):
    prompt: str

class ChatRequest(BaseModel):
    prompt: str
    model: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = True

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

@app.post("/api/chat")
async def chat_with_ollama(request: ChatRequest):
    """
    New endpoint for streaming chat with Ollama that matches frontend expectations
    """
    try:
        if request.stream:
            # Create a generator for streaming the response
            return StreamingResponse(
                stream_ollama(request.prompt, temperature=request.temperature),
                media_type="text/plain"
            )
        else:
            # For non-streaming requests, use the existing function
            response = await ask_ollama(request.prompt)
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

