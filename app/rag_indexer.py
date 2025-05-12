import os
from fastapi import HTTPException
import httpx
import asyncio
import tempfile
from typing import List, Dict, Any, Optional
from datetime import datetime

from ultralytics import YOLO
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from .backend_client import BackendClient

from .config import (
    YOLO_MODEL_PATH,
    YOLO_GENERAL_CLS_MODEL_PATH,
)

# Load YOLO and Embedding model once
yolo_model = YOLO(YOLO_MODEL_PATH)
yolo_general_cls_model = YOLO(YOLO_GENERAL_CLS_MODEL_PATH)
embedding_model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# Init ChromaDB
chroma_client = chromadb.PersistentClient(
    path="./chroma_db", settings=Settings(allow_reset=True)
)
collection = chroma_client.get_or_create_collection(name="vietnamese_food_images")
client = BackendClient()

async def download_image(url: str) -> str:
    """Download image from IPFS to a temp file"""
    async with httpx.AsyncClient() as client:
        response = await client.get(url, timeout=10)
        response.raise_for_status()
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        temp.write(response.content)
        temp.close()
        return temp.name

def predict_with_model(image_path: str, model, label: str):
        result = model(image_path)[0]
        if result.probs:
            top_idx = result.probs.top1
            top_score = result.probs.top1conf.item()
            class_name = result.names[top_idx]
            print(f"[{label}] Predicted: {class_name} ({top_score:.2f})")
            return class_name, top_score
        else:
            print(f"[{label}] No probs in result.")
            return None, 0.0
        
def predict_food_or_general(image_path: str) -> tuple[str | None, bool]:
    """Try food classifier first, fallback to general classifier if confidence too low"""

    # Step 1: Try fine-tuned food classifier
    food_class, food_score = predict_with_model(image_path, yolo_model, "Food Classifier")
    if food_score >= 0.6:
        return food_class, True

    # Step 2: Fallback to general classifier
    general_class, general_score = predict_with_model(image_path, yolo_general_cls_model, "General Classifier")
    if general_class:
        return general_class, False

    # Step 3: Nothing found
    return None, False

def is_indexed(photo_id: str) -> bool:
    """Check if photo_id already in Chroma"""
    try:
        result = collection.get(ids=[f"photo:{photo_id}"])
        return bool(result.get("ids"))
    except:
        return False

def index_photo(photo: Dict[str, Any], food_class: str, is_food: bool):
    """Embed and index photo with metadata"""
    # Check if this is the user's own photo or a friend's photo
    is_own = photo.get("isOwnPhoto", True)
    is_friend = photo.get("isFriendPhoto", False)
    
    # Get user ID to determine relationship context
    user_id = str(photo.get("userId", ""))
    
    # Determine relationship description and context
    relationship_context = ""
    if is_own:
        user_type = "bạn"
    elif is_friend:
        user_type = "bạn bè"
        # Add explicit relationship context based on user ID
        if user_id == "1":
            relationship_context = " (Super Admin là bạn của Hoa Thanh)"
        elif user_id == "3":
            relationship_context = " (Hoa Thanh là bạn của Super Admin)"
    else:
        user_type = "người dùng khác"
    
    # Enhanced caption with more details about the food and relationship
    caption = (
        f"{photo['userName']} ({user_type}){relationship_context} đăng ảnh món {food_class} vào ngày {photo['createdAt'][:10]}. "
        f"Món ăn này thuộc loại {'thức ăn' if is_food else 'đồ uống/khác'}."
    )
    
    vector = embedding_model.encode([caption])[0]

    collection.add(
        ids=[f"photo:{photo['id']}"],
        documents=[caption],
        embeddings=[vector],
        metadatas=[
            {
                "photo_id": photo["id"],
                "user_id": user_id,
                "food_class": food_class,
                "user_name": photo["userName"],
                "created_at": photo["createdAt"],
                "is_own_photo": is_own,
                "is_friend_photo": is_friend,
                "is_food": is_food,
                "indexed_at": datetime.utcnow().isoformat(),
            }
        ],
    )


async def process_photo(photo: Dict[str, Any]) -> Dict[str, Any]:
    """Handle a single photo: check, download, detect, embed, index"""
    photo_id = photo["id"]
    # logger.debug(f"Processing photo {photo_id}")

    if is_indexed(photo_id):
        # logger.debug(f"Photo {photo_id} already indexed, skipping")
        return {"photo_id": photo_id, "status": "skipped"}

    try:
        # logger.debug(f"Downloading image from {photo['url']}")
        img_path = await download_image(photo["url"])
        
        # logger.debug(f"Predicting food class for {photo_id}")
        food_class, is_food = predict_food_or_general(img_path)
        
        # logger.debug(f"Removing temporary image file {img_path}")
        os.remove(img_path)

        if not food_class:
            # logger.debug(f"No food detected in photo {photo_id}")
            return {"photo_id": photo_id, "status": "no_food_detected"}

        # logger.debug(f"Indexing photo {photo_id} with food class {food_class}")
        index_photo(photo, food_class, is_food)
        
        # logger.debug(f"Successfully processed photo {photo_id}")
        return {"photo_id": photo_id, "status": "indexed", "food_class": food_class, "is_food": is_food}

    except Exception as e:
        # logger.error(f"Error processing photo {photo_id}: {str(e)}")
        return {"photo_id": photo_id, "status": "error", "error": str(e)}

async def process_and_index_photos(auth_token: Optional[str] = None, max_photos: int = 50) -> Dict[str, Any]:
    token = auth_token
    try:
        data = await client.fetch_user_photos(auth_token=token, max_photos=50)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    # Get user and friend photos from new API structure
    user_photos = data.get("userPhotos", [])
    friend_photos = data.get("friendPhotos", [])
    
    # Mark photos with ownership information
    for photo in user_photos:
        photo["isOwnPhoto"] = True
    
    for photo in friend_photos:
        photo["isOwnPhoto"] = False
        photo["isFriendPhoto"] = True
    
    # Combine all photos for processing
    all_photos = user_photos + friend_photos
    
    # Process all photos
    results = await asyncio.gather(*(process_photo(p) for p in all_photos))

    return {
        "status": "done",
        "total_photos": len(all_photos),
        "user_photos_count": len(user_photos),
        "friend_photos_count": len(friend_photos),
        "indexed": len([r for r in results if r["status"] == "indexed"]),
        "skipped": len([r for r in results if r["status"] == "skipped"]),
        "errors": [r for r in results if r["status"] == "error"],
        "details": results
    }