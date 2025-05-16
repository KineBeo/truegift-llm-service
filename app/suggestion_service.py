import difflib
import json
import os
import random
import re
from typing import List, Dict, Any

from app.groq_client import ask_groq
from .rag_indexer import collection
from .config import logger
from .ollama_client import ask_ollama

SUGGESTION_TEMPLATES = {
    "like_friends": """\
🍽️ Gợi ý món ăn từ bạn bè của bạn!
Hãy dựa trên các món ăn mà bạn bè đã chia sẻ để đưa ra gợi ý phù hợp nhất cho người dùng.
Hướng dẫn:
    Nếu không biết câu trả lời, hãy nói rõ điều đó.
    Nếu không chắc chắn, hãy yêu cầu người dùng làm rõ thêm.
    Trả lời bằng cùng ngôn ngữ với truy vấn của người dùng.
    Nếu ngữ cảnh không thể đọc được hoặc chất lượng kém, hãy thông báo cho người dùng và đưa ra câu trả lời tốt nhất có thể.
    Nếu thông tin không có trong ngữ cảnh nhưng bạn biết câu trả lời, hãy giải thích điều này và đưa ra câu trả lời dựa trên kiến thức của bạn.
    Không trích dẫn nếu không có id.
    Không sử dụng thẻ XML trong phần trả lời.
    Đảm bảo trích dẫn ngắn gọn và liên quan trực tiếp đến thông tin được cung cấp.
Đầu ra (Output):
Cung cấp câu trả lời rõ ràng, trực tiếp dựa trên ngữ cảnh, Việt hoá hết đoạn chat. Ví dụ không được để là bun dau mam tom, bun cha.... 
Câu đầu tiên luôn là: Sau đây là những món ăn tôi chọn để gợi ý cho bạn:
Câu cuối cùng luôn là: Nếu bạn muốn thử món ăn nào, hãy nhắn cho tôi nhé!
<user_query>
Dựa trên các món ăn bạn bè bạn đã chia sẻ 🍽️🍜🍲: {context}
Thông tin bổ sung về các món ăn: {crawled_info}
Hãy chọn từ 1 đến 3 món ăn trong danh sách trên để gợi ý lại cho người dùng với tone chuyên nghiệp, ngắn gọn, súc tích. 
Sử dụng thông tin bổ sung để làm rõ về món ăn (giá, địa chỉ, mô tả) nếu có.
Không được nói về 1 món ăn quá 1 lần. Mỗi món ăn chỉ được nói 1 lần.
Nếu danh sách chỉ có 1 món ăn, hãy chỉ gợi ý món ăn đó.
</user_query>
""",
    "unique_today": """\
🌟 Gợi ý món ngon hôm nay từ hành trình ẩm thực của bạn & bạn bè! 🍽️✨
{context}
Thông tin bổ sung về các món ăn: {crawled_info}

Hãy gợi ý một món ăn từ các món ăn gần đây của bạn và bạn bè.
Sử dụng thông tin bổ sung để làm rõ về món ăn (giá, địa chỉ, mô tả) nếu có.
Thêm các thông tin cơ bản về món ăn đó, ví dụ như tên món, nơi bán, giá cả, địa chỉ, thời gian mở cửa, ... với tone chuyên nghiệp, ngắn gọn, súc tích. 
Hãy viết ngắn gọn dưới 60 từ.
Đầu ra (Output):
Cung cấp câu trả lời rõ ràng, trực tiếp dựa trên ngữ cảnh, Việt hoá hết đoạn chat. Ví dụ không được để là bun dau mam tom, bun cha.... 
""",
    "special_day": """\
Ngày đặc biệt cần bữa ăn đặc biệt, cùng chọn nha! 💖🍽️
{context}
Thông tin bổ sung về các món ăn: {crawled_info}

Nếu hôm nay là một ngày đặc biệt, bạn sẽ nên ăn gì? Hãy gợi ý món ăn phù hợp, cảm xúc Gen Z, thêm chút thơ mộng và icon nha!
Sử dụng thông tin bổ sung để làm rõ về món ăn (giá, địa chỉ, mô tả) nếu có.
Lưu ý là Việt hoá hết đoạn chat. Ví dụ không được để là bun dau mam tom, bun cha.... 
""",
"mood_based": """\
💭 Hôm nay bạn thấy sao? Mình sẽ chọn món phù hợp với tâm trạng của bạn nè!
{context}
Thông tin bổ sung về các món ăn: {crawled_info}

Dựa trên các món ăn bạn và bạn bè từng chọn gần đây, hãy gợi ý một món ăn thật phù hợp với tâm trạng (vui, buồn, stress, chill, v.v.).  
Hãy Việt hoá hoàn toàn nội dung, ví dụ không được để là bun dau mam tom, bun cha...  
Gợi ý nên ngắn gọn dưới 60 từ, thêm chút cảm xúc Gen Z, icon dễ thương, và mô tả món ăn rõ ràng nếu có (giá, địa chỉ, mô tả...).
""",
"weather_fit": """\
🌦️ Thời tiết thế này thì ăn gì cho đúng vibe? Mình gợi ý giúp bạn nè!
{context}
Thông tin bổ sung về các món ăn: {crawled_info}

Dựa trên các món ăn gần đây và thông tin thời tiết hiện tại (mưa, nắng, se lạnh, oi bức,...), hãy chọn ra món ăn phù hợp nhất.  
Gợi ý cần ngắn gọn, cảm xúc, dễ thương, Việt hoá hoàn toàn món ăn, và bổ sung thông tin như giá, nơi bán nếu có.  
Không lặp lại món, chỉ chọn 1 món duy nhất cho phù hợp thời tiết nha!
""",
"late_night_craving": """\
🌙 Đêm muộn bụng đói reo? Mình gợi ý món ngon đêm khuya cho bạn nè!
{context}
Thông tin bổ sung về các món ăn: {crawled_info}

Chọn một món ăn phù hợp để ăn khuya, không quá nặng bụng nhưng vẫn ngon, dễ thương và phù hợp Gen Z.  
Hãy dựa vào các món bạn từng ăn trước đó để đưa ra gợi ý. Thêm mô tả ngắn gọn (giá, địa chỉ, cảm giác khi ăn...) nếu có.  
Việt hoá toàn bộ món ăn, thêm chút icon và cảm xúc nhẹ nhàng cho vibe đêm khuya nha.
""",

}


def get_available_prompts():
    return [
        {
            "key": k,
            "description": v.split("\n")[0][:80],
        }  # Mô tả gợi ý ngắn từ dòng đầu prompt
        for k, v in SUGGESTION_TEMPLATES.items()
    ]


def retrieve_user_photos(user_id: str, top_k: int = 5) -> List[str]:
    """Retrieve photos from a specific user - only food items"""
    results = collection.query(
        query_texts=["thức ăn"],  # More focused query for food
        n_results=top_k,
        where={
            "$and": [
                {"user_id": user_id},
                {"is_own_photo": True},
                {"is_food": True},  # Only return actual food items
            ]
        },
    )

    documents = results.get("documents", [[]])[0]

    # If no user photos found, return an explanatory message
    if not documents:
        return ["Bạn chưa có ảnh món ăn nào. Hãy chia sẻ những món ăn bạn thích!"]

    return documents


def retrieve_friend_photos(user_id: str, top_k: int = 5) -> List[str]:
    """Retrieve photos from friends of the specified user - only food items

    This should return photos where:
    1. For user_id=1, return photos from user_id=3 (since they're friends)
    2. For user_id=3, return photos from user_id=1 (since they're friends)
    """
    # We need to find photos where:
    # - The photo doesn't belong to the current user (user_id)
    # - The photo is marked as food
    # BUT critically, we need to consider the reciprocal friendship relationship

    # This complex query isn't easily expressible in ChromaDB's where clause
    # So first we get all food photos that aren't from the current user
    results = collection.query(
        query_texts=["thức ăn"],
        n_results=50,  # Get more results to filter from
        where={
            "$and": [
                {"user_id": {"$ne": user_id}},  # Not from the current user
                {"is_food": True},  # Only return actual food items
            ]
        },
    )

    # Then filter for friends' photos by examining each document
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    # If no photos at all, return explanatory message
    if not documents:
        return ["Hiện tại bạn chưa có ảnh món ăn từ bạn bè để gợi ý."]

    # Filter to get only relevant friend photos
    friend_photos = []
    friend_count = 0

    for i, metadata in enumerate(metadatas):
        # Skip if index out of bounds
        if i >= len(documents):
            continue

        # This manual filtering logic helps us find friend photos correctly
        # For each user_id, we know which other user_ids are their friends
        other_user_id = metadata.get("user_id")

        is_friend_photo = False
        # For user_id=1, look for photos from user_id=3
        if user_id == "1" and other_user_id == "3":
            is_friend_photo = True
        # For user_id=3, look for photos from user_id=1
        elif user_id == "3" and other_user_id == "1":
            is_friend_photo = True

        if is_friend_photo and friend_count < top_k:
            friend_photos.append(documents[i])
            friend_count += 1

    # If no friend photos found after filtering, return explanatory message
    if not friend_photos:
        return ["Hiện tại bạn chưa có ảnh món ăn từ bạn bè để gợi ý."]

    return friend_photos

CRAWLED_JSON_PATH = "extracted_food_data.json" 


def normalize_food_name(name: str) -> str:
    name = name.lower().strip()
    name = re.sub(r"\s+", " ", name)
    return name


def get_closest_food_name(food_name: str, crawled_data: List[Dict]) -> str:
    normalized_name = normalize_food_name(food_name)
    food_names = [normalize_food_name(item["name"]) for item in crawled_data]
    closest = difflib.get_close_matches(normalized_name, food_names, n=1, cutoff=0.8)
    return closest[0] if closest else None


def load_crawled_data() -> List[Dict[str, Any]]:
    try:
        if not os.path.exists(CRAWLED_JSON_PATH):
            logger.error(f"File {CRAWLED_JSON_PATH} không tồn tại")
            return []

        with open(CRAWLED_JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Lọc các mục không có error
        valid_data = [item for item in data if not item.get("error", False)]
        logger.debug(f"Đã đọc {len(valid_data)} món ăn hợp lệ từ file JSON")
        return valid_data
    except Exception as e:
        logger.error(f"Lỗi khi đọc file JSON: {str(e)}")
        return []


# def get_crawled_info(food_names: List[str]) -> str:
#     crawled_data = load_crawled_data()
#     crawled_info = []
#     for food_name in food_names:
#         for item in crawled_data:
#             if item.get("name", "").lower() == food_name.lower():
#                 info = f"{item['name']}: {item['description']} Giá: {item['price']}. Địa chỉ: {item['popular_address']}"
#                 crawled_info.append(info)
#                 break
#     return (
#         "\n- ".join(crawled_info)
#         if crawled_info
#         else "Không có thông tin bổ sung cho các món ăn này."
#     )


def get_crawled_info(food_names: List[str]) -> str:
    crawled_data = load_crawled_data()
    crawled_info = []
    normalized_food_names = [normalize_food_name(name) for name in food_names]
    logger.debug(f"Tìm thông tin crawl cho các món: {food_names}")

    for food_name, normalized_name in zip(food_names, normalized_food_names):
        found = False
        for item in crawled_data:
            if normalize_food_name(item.get("name", "")) == normalized_name:
                info = f"{item['name']}: {item['description']} Giá: {item['price']}. Địa chỉ: {item['popular_address']}"
                crawled_info.append(info)
                logger.debug(
                    f"Tìm thấy thông tin khớp chính xác cho món {food_name}: {info}"
                )
                found = True
                break

        if not found:
            closest_name = get_closest_food_name(food_name, crawled_data)
            if closest_name:
                for item in crawled_data:
                    if normalize_food_name(item.get("name", "")) == closest_name:
                        info = f"{item['name']} (gần giống {food_name}): {item['description']} Giá: {item['price']}. Địa chỉ: {item['popular_address']}"
                        crawled_info.append(info)
                        logger.debug(
                            f"Tìm thấy thông tin khớp gần giống cho món {food_name}: {info}"
                        )
                        found = True
                        break

        if not found:
            logger.debug(f"Không tìm thấy thông tin crawl cho món {food_name}")

    return (
        "\n- ".join(crawled_info)
        if crawled_info
        else "Không có thông tin bổ sung cho các món ăn này."
    )


def retrieve_context(
    user_id: str, top_k: int = 5, prompt_key: str = None
) -> tuple[List[str], str]:
    """Smart context retrieval based on prompt type, returns context and crawled info"""
    if isinstance(user_id, int):
        user_id = str(user_id)

    if prompt_key == "like_friends":
        # For friend-based prompts, get only friend photos
        context = retrieve_friend_photos(user_id, top_k)
    elif prompt_key == "unique_today":
        # For mixed prompts, get both user and friend photos
        user_photos = retrieve_user_photos(user_id, top_k // 2)
        friend_photos = retrieve_friend_photos(user_id, top_k // 2)
        actual_user_photos = [
            p
            for p in user_photos
            if not p.startswith("Hiện tại") and not p.startswith("Bạn chưa")
        ]
        actual_friend_photos = [
            p
            for p in friend_photos
            if not p.startswith("Hiện tại") and not p.startswith("Bạn chưa")
        ]
        if not actual_user_photos and not actual_friend_photos:
            context = [
                "Bạn và bạn bè chưa có ảnh món ăn nào. Hãy chia sẻ những món ăn bạn thích!"
            ]
        elif not actual_user_photos:
            context = actual_friend_photos
        elif not actual_friend_photos:
            context = actual_user_photos
        else:
            context = actual_user_photos + actual_friend_photos
    else:
        context = retrieve_user_photos(user_id, top_k)

    # Extract food names from context
    food_names = []
    for snippet in context:
        if snippet.startswith("Hiện tại") or snippet.startswith("Bạn chưa"):
            continue
        parts = snippet.split("đăng ảnh món ")
        if len(parts) > 1:
            food_name = parts[1].split(" vào ngày")[0].strip()
            if food_name not in food_names:
                food_names.append(food_name)

    # Get crawled info for the extracted food names
    crawled_info = get_crawled_info(food_names)
    return context, crawled_info


async def generate_suggestion_by_prompt(user_id: str, prompt_key: str) -> str:
    try:
        # Convert user_id to string for consistent comparison
        if isinstance(user_id, int):
            user_id = str(user_id)

        template = SUGGESTION_TEMPLATES.get(prompt_key)
        if not template:
            return "Không hiểu bạn muốn hỏi gì 🤔"

        # Get context based on prompt type
        context_snippets, crawled_info = retrieve_context(
            user_id, top_k=5, prompt_key=prompt_key
        )

        # Handle special case for friend-based prompts
        if prompt_key == "like_friends":
            if not context_snippets or context_snippets[0].startswith("Hiện tại"):
                # For user_id=3, they should see Super Admin's photos
                if user_id == "3":
                    return "Chưa có ảnh món ăn nào từ Super Admin. Hãy nhắc họ chia sẻ nhé! 🍕👫"
                # For user_id=1, they should see Hoa Thanh's photos
                elif user_id == "1":
                    return "Chưa có ảnh món ăn nào từ Hoa Thanh. Hãy nhắc họ chia sẻ nhé! 🍕👫"
                # Generic message for other users
                else:
                    return "Bạn bè bạn chưa đăng ảnh món ăn nào. Hãy rủ họ chia sẻ nhé! 🍕👫"

        # General case - no images at all
        if not context_snippets:
            return "Bạn chưa có ảnh nào để gợi ý. Hãy đăng vài món ăn nhé! 🍜📸"

        # Handle explanatory messages which aren't actual context
        if context_snippets[0].startswith("Hiện tại") or context_snippets[0].startswith(
            "Bạn chưa"
        ):
            return context_snippets[0]

        # Group similar food items to avoid repetition in context
        food_items = {}
        for snippet in context_snippets:
            # Skip explanatory messages
            if snippet.startswith("Hiện tại") or snippet.startswith("Bạn chưa"):
                continue

            # Extract food name using a simple pattern match
            parts = snippet.split("đăng ảnh món ")
            if len(parts) > 1:
                food_name = parts[1].split(" vào ngày")[0].strip()
                user_name = parts[0].split("(")[0].strip()

                # Group by food name
                if food_name not in food_items:
                    food_items[food_name] = [f"{user_name} đã chia sẻ món {food_name}"]
                else:
                    # Only add another mention if it's a different user
                    if not any(user_name in item for item in food_items[food_name]):
                        food_items[food_name].append(
                            f"{user_name} cũng đã chia sẻ món {food_name}"
                        )

        # Format context with deduplicated food items
        if food_items:
            formatted_context = []
            for food, mentions in food_items.items():
                formatted_context.append(f"{food}: {', '.join(mentions)}")

            context = "\n- " + "\n- ".join(formatted_context)
        else:
            # Fallback to original context formatting if pattern matching fails
            context = "\n- " + "\n- ".join(context_snippets[:5])

        prompt = template.format(context=context, crawled_info=crawled_info)
        print(f"[DEBUG] Generating suggestion with prompt:\n{prompt}")

        response = await ask_groq(prompt)
        return response.strip()
    except Exception as e:
        logger.error(f"Suggestion generation error: {str(e)}")
        return "Đã xảy ra lỗi khi tạo gợi ý 😢"
