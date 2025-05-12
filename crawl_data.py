import os
import asyncio
import json
from pydantic import BaseModel, Field
from typing import List
from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CrawlerRunConfig,
    CacheMode,
    LLMConfig,
)
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from app.config import config


class Food(BaseModel):
    name: str = Field(description="The name of the food")
    price: str = Field(description="The price of the food")
    description: str = Field(description="The description of the food")
    popular_address: str = Field(description="The popular address of the food")

async def main():
    # 1. Define the LLM extraction strategy
    llm_strategy = LLMExtractionStrategy(
        llm_config=LLMConfig(
            provider="groq/llama-3.3-70b-versatile", api_token=config["GROQ_API_KEY"]
        ),
        schema=Food.model_json_schema(),  # Or use model_json_schema()
        extraction_type="schema",
        instruction="From the crawled content, extract all mentioned food names along with their "
        "price and description and popular address. Make sure not to miss anything in the entire content. "
        "One extracted food JSON format should look like this: "
        '{ "name": "Food Name", "price": "Price (VND)", "description": "Description", "popular_address": "Popular Address" }',
        chunk_token_threshold=800,
        overlap_rate=0.0,
        apply_chunking=True,
        input_format="markdown",  # or "html", "fit_markdown"
        extra_args={"temperature": 0.0, "max_tokens": 800},
    )

    # 2. Build the crawler config
    crawl_config = CrawlerRunConfig(
        extraction_strategy=llm_strategy, cache_mode=CacheMode.BYPASS
    )

    # 3. Create a browser config if needed
    browser_cfg = BrowserConfig(headless=True)

    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        # 4. Let's say we want to crawl a single page
        result = await crawler.arun(
            url="https://www.traveloka.com/vi-vn/explore/culinary/am-thuc-viet-nam/441612",
            config=crawl_config,
        )

        if result.success:
            # 5. The extracted content is presumably JSON
            data = json.loads(result.extracted_content)
            print("Extracted items:", data)

            # Save data to JSON file
            with open("extracted_food_data.json", "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            print("Data saved to extracted_food_data.json")

            # 6. Show usage stats
            llm_strategy.show_usage()  # prints token usage
        else:
            print("Error:", result.error_message)


if __name__ == "__main__":
    asyncio.run(main())
