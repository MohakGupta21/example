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


async def index_documents_async(documents:List[Document], batch_size:int=50):
    log_header()
    log_info(
        f"Vector Store Indexing: Preparing to add {len(documents)} documents to vector store",
        Colors.DARKCYAN
    )
    #Create Batches
    batches = [
        documents[i:i + batch_size] for i in range(0, len(documents), batch_size)
    ]

    log_info(
        f"VectorStore Indexing: Split into {len(batches)} batches of {batch_size} documents each"
    )

    # Process all batches concurrently
    async def add_batch(batch:List[Document], batch_num:int):
        try:
            await vectorstore.aadd_documents(batch)
            log_success(
                f"Vector Store Indexing: Successfully added batch {batch_num}/{len(batches)} ({len(batch)}) documents"
            )
        except Exception as e:
            log_error(
                f"Vector Store Indexing: Failed to add batch {batch_num}/{len(batches)} ({len(batch)}) documents. Error: {str(e)}"
            )
            return False
        return True

    # Create a list of tasks for all batches
    # Keep one shared Pinecone async session open for all concurrent batches;
    # otherwise each aadd_documents call closes the shared session when it finishes
    async with vectorstore:
        tasks = [add_batch(batch, i+1) for i, batch in enumerate(batches)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    #Count succcessful batches
    successful = sum(1 for result in results if result is True)

    if successful == len(batches):
        log_success(
            f"Vector Store Indexing: Successfully added all {len(documents)} documents to vector store"
        )
    else:
        log_warning(
            f"Vector Store Indexing: Added {successful}/{len(batches)} batches successfully. Some documents may not have been indexed."
        )

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
    documents = [Document(page_content=doc["raw_content"], metadata={"url": doc["url"]}) for doc in all_docs]
    log_success(f"TavilyCrawl: Successfully crawled {len(all_docs)} URLs from documentation site")

    log_header()
    log_info(" Document Chunking Phase")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    splitted_docs = text_splitter.split_documents(documents)
    log_success(f"Text Splitter: Created {len(splitted_docs)} chunks from {len(documents)} documents")

    #Process documents asynchronously
    await index_documents_async(splitted_docs, batch_size=5)

    log_header()
    log_success("Documentation ingestion pipeline finished successfully!")
    log_info("Summary:", Colors.BOLD)
    log_info(f"   + URLs mapped:{len(res['results'])}")
    log_info(f"   + Documents extracted: {len(documents)}")
    log_info(f"   + Chunks created: {len(splitted_docs)}")
if __name__ == "__main__":
    asyncio.run(main())