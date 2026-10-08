import asyncio
import os
import ssl
from typing import Any, Dict, List

import certifi
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap

from logger import Colors, log_info, log_success, log_warning, log_error, log_header

load_dotenv()

# Configure SSL context to use certifi certificates
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"]=certifi.where()
os.environ["REQUESTS_CA_BUNDLE"]=certifi.where()

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small", show_progress_bar=False, chunk_size=50, retry_min_seconds=10
    )

vectorstore = PineconeVectorStore(index_name= os.environ["INDEX_NAME"], embedding=embeddings)
tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=1000)
tavily_crawl = TavilyCrawl()


async def main():
    """Main async function to orchestrate the entire process"""
    log_header()
    log_info(
        "TavilyCrawl: Starting to Crawl documentation from https://python.langchain.com/",
        Colors.PURPLE
        )

    # Crawl the documentation site
    res = tavily_crawl.invoke(
        {
            "url": "https://python.langchain.com/",
            "max_depth":5,
            "extract_depth":"advanced",
            "instructions":"content on ai agents"
        }
    )
    all_docs=res["results"]
    log_success(f"TavilyCrawl: Successfully crawled {len(all_docs)} URLs from documentation site")


if __name__ == "__main__":
    asyncio.run(main())