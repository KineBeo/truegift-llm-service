from typing import List
from .rag_indexer import collection
from .config import logger
from .ollama_client import ask_ollama  
SUGGESTION_TEMPLATES = {
    "like_friends": """\
Dựa trên các món ăn bạn bè bạn đã chia sẻ:
{context}

Hãy gợi ý một món ăn giống phong cách bạn bè bạn nhưng phù hợp với khẩu vị Gen Z.
Hãy viết ngắn gọn, thêm biểu cảm dễ thương, icon và khuyến khích dùng thử. Dưới 60 từ.
""",

    "unique_today": """\
Từ các món ăn gần đây của bạn và bạn bè:
{context}

Hãy gợi ý một món ăn từ các món ăn gần đây của bạn và bạn bè, mang vibe Gen Z.
Thêm các thông tin cơ bản về món ăn đó, ví dụ như tên món, nơi bán, giá cả, địa chỉ, thời gian mở cửa, ...
Hãy viết ngắn gọn dưới 60 từ.
""",

    "special_day": """\
Dựa trên các món ăn trước đây:
{context}

Nếu hôm nay là một ngày đặc biệt, bạn sẽ nên ăn gì? Hãy gợi ý món ăn phù hợp, cảm xúc Gen Z, thêm chút thơ mộng và icon nha!
"""
}

def get_available_prompts():
    return [
        {"key": k, "description": v.split("\n")[0][:80]}  # Mô tả gợi ý ngắn từ dòng đầu prompt
        for k, v in SUGGESTION_TEMPLATES.items()
    ]

def retrieve_context(user_id: str, top_k: int = 5) -> List[str]:
    results = collection.query(
        query_texts=[f"user_id:{user_id}"],
        n_results=top_k,
        where={"user_id": user_id},
    )

    documents = results.get("documents", [[]])[0]
    
    return documents

async def generate_suggestion_by_prompt(user_id: str, prompt_key: str) -> str:
    try:
        context_snippets = retrieve_context(user_id)
        if not context_snippets:
            return "Bạn chưa có ảnh nào để gợi ý. Hãy đăng vài món ăn nhé! 🍜📸"

        template = SUGGESTION_TEMPLATES.get(prompt_key)
        if not template:
            return "Không hiểu bạn muốn hỏi gì 🤔"

        context = "\n- " + "\n- ".join(context_snippets[:5])
        prompt = template.format(context=context)
        print(f"[DEBUG] Generating suggestion with prompt:\n{prompt}")

        response = await ask_ollama(prompt)
        return response.strip()
    except Exception as e:
        logger.error(f"Suggestion generation error: {str(e)}")
        return "Đã xảy ra lỗi khi tạo gợi ý 😢"


