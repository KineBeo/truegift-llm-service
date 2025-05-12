# File: async_webcrawler_multiple_urls_example.py
import os, sys
from datetime import datetime

# append 2 parent directories to sys.path to import crawl4ai
parent_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(parent_dir)

import asyncio
from crawl4ai import AsyncWebCrawler


async def main():
    # Initialize the AsyncWebCrawler
    async with AsyncWebCrawler(verbose=True) as crawler:
        # List of URLs to crawl
        urls = [
            "https://example.com",
            "https://python.org",
            "https://github.com",
            "https://stackoverflow.com",
            "https://news.ycombinator.com",
        ]

        # Set up crawling parameters
        word_count_threshold = 100

        # Run the crawling process for multiple URLs
        results = await crawler.arun_many(
            urls=urls,
            word_count_threshold=word_count_threshold,
            bypass_cache=True,
            verbose=True,
        )

        # Tạo thư mục nếu chưa tồn tại
        output_dir = "crawl_results"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Tạo tên file với timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        combined_file = os.path.join(output_dir, f"crawl_results_{timestamp}.md")

        # Tạo header cho file markdown
        with open(combined_file, "w", encoding="utf-8") as f:
            f.write(f"# Crawl Results - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Total URLs: {len(results)}\n\n")

        # Xử lý từng kết quả
        for result in results:
            if result.success:
                print(f"Successfully crawled: {result.url}")
                print(f"Title: {result.metadata.get('title', 'N/A')}")
                
                # Thêm kết quả vào file tổng hợp
                with open(combined_file, "a", encoding="utf-8") as f:
                    f.write(f"## {result.metadata.get('title', 'No Title')}\n\n")
                    f.write(f"URL: {result.url}\n\n")
                    f.write(f"### Content\n\n")
                    f.write(result.markdown)
                    f.write("\n\n---\n\n")
                
                print(f"Added to combined markdown file")
                print(f"Number of images: {len(result.media.get('images', []))}")
                print("---")
            else:
                print(f"Failed to crawl: {result.url}")
                print(f"Error: {result.error_message}")
                
                # Thêm lỗi vào file tổng hợp
                with open(combined_file, "a", encoding="utf-8") as f:
                    f.write(f"## Failed: {result.url}\n\n")
                    f.write(f"Error: {result.error_message}\n\n")
                    f.write("---\n\n")
                
                print("Error logged")
                print("---")

        print(f"Combined results saved to: {combined_file}")


if __name__ == "__main__":
    asyncio.run(main())