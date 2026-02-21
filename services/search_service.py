import logging
from ddgs import DDGS
import asyncio
from typing import List, Dict

logger = logging.getLogger(__name__)

def perform_search(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """
    Performs a synchronous web search using DuckDuckGo.
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            return results
    except Exception as e:
        logger.error(f"Search error for query '{query}': {e}")
        return []

async def get_search_context(query: str) -> str:
    """
    Asynchronous wrapper for search. Returns a formatted string of results.
    """
    try:
        # Run synchronous search in a thread to avoid blocking the event loop
        results = await asyncio.to_thread(perform_search, query)
        
        if not results:
            return ""

        context_parts = ["Information from the internet:\n"]
        for i, res in enumerate(results, 1):
            title = res.get('title', 'No Title')
            body = res.get('body', 'No Content')
            href = res.get('href', '#')
            context_parts.append(f"{i}. **{title}**\n   {body}\n   Source: {href}\n")
        
        return "\n".join(context_parts)
    except Exception as e:
        logger.error(f"Error getting search context: {e}")
        return ""
