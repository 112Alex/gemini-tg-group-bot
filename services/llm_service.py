import time
import asyncio
import logging
import os
import re
import json
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from google import genai
from google.genai import types
from openai import AsyncOpenAI
from config import config
from services.search_service import get_search_context

logger = logging.getLogger(__name__)

# Define the web search tool schema for OpenAI-compatible APIs
WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the internet for real-time information, news, or facts not in your knowledge base. Use this when the user asks about current events or specific data.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to send to the search engine."
                }
            },
            "required": ["query"]
        }
    }
}

class RateLimitExceeded(Exception):
    """Exception raised when the rate limit for a chat is exceeded."""
    pass


class BaseLLM(ABC):
    """
    Abstract Base Class for LLM providers.
    Handles system prompt loading and rate limiting.
    """

    def __init__(self) -> None:
        self._last_request_time: Dict[int, float] = {}
        self._rate_limit_seconds = 3.0
        self._system_prompt_file = "system_prompt.md"
        self._system_prompt_example_file = "system_prompt.example.md"

    def _check_rate_limit(self, chat_id: int) -> None:
        """
        Ensures that requests from the same chat are not made too frequently.
        """
        current_time = time.time()
        last_time = self._last_request_time.get(chat_id, 0)
        
        if current_time - last_time < self._rate_limit_seconds:
            raise RateLimitExceeded(
                f"Please wait {self._rate_limit_seconds - (current_time - last_time):.1f} seconds."
            )
        
        self._last_request_time[chat_id] = current_time

    def _load_system_prompt(self) -> str:
        """
        Loads the system prompt from file. Fallbacks to example file if main file is missing or empty.
        """
        prompt = ""
        try:
            if os.path.exists(self._system_prompt_file):
                with open(self._system_prompt_file, "r", encoding="utf-8") as f:
                    prompt = f.read().strip()
            
            if not prompt and os.path.exists(self._system_prompt_example_file):
                with open(self._system_prompt_example_file, "r", encoding="utf-8") as f:
                    prompt = f.read().strip()
            
            if not prompt:
                logger.warning("System prompt files are missing or empty. Using default.")
                return "You are a helpful assistant."
                
            return prompt
        except Exception as e:
            logger.error(f"Error loading system prompt: {e}")
            return "You are a helpful assistant."

    @abstractmethod
    async def generate_response(self, chat_id: int, user_name: str, prompt: str, history: List[Dict[str, str]]) -> str:
        """
        Generates a response from the LLM using conversation history.
        """
        pass


class GeminiLLM(BaseLLM):
    """
    Gemini Provider with Google Search Grounding.
    """
    def __init__(self, api_key: str, model_name: str) -> None:
        super().__init__()
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    async def generate_response(self, chat_id: int, user_name: str, prompt: str, history: List[Dict[str, str]]) -> str:
        self._check_rate_limit(chat_id)
        
        system_instruction = await asyncio.to_thread(self._load_system_prompt)
        
        # Convert history format to SDK compatible Content objects
        contents = []
        for msg in history:
            role = "user" if msg['role'] == "user" else "model"
            contents.append(types.Content(role=role, parts=[types.Part(text=msg['content'])]))
        
        # Add current user prompt
        contents.append(types.Content(role="user", parts=[types.Part(text=f"{user_name}: {prompt}")]))

        tools = None
        if config.search_enabled:
            # Use dynamic retrieval so Gemini decides when to search
            tools = [
                types.Tool(
                    google_search=types.GoogleSearchRetrieval(
                        dynamic_retrieval_config=types.DynamicRetrievalConfig(
                            mode=types.DynamicRetrievalConfigMode.MODE_DYNAMIC,
                            dynamic_threshold=0.7, # Adjust threshold as needed
                        )
                    )
                )
            ]

        try:
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    tools=tools,
                    response_mime_type="text/plain",
                    temperature=config.model_temperature,
                    top_p=config.model_top_p,
                    top_k=config.model_top_k,
                )
            )
            
            if response.text:
                return response.text
            return "Извините, я не смог сформулировать ответ."

        except Exception as e:
            logger.error(f"Gemini API Error for chat {chat_id}: {e}")
            
            # Fallback to Groq (openai/gpt-oss-120b) if configured
            if config.groq_api_key:
                logger.warning(f"Falling back to Groq (openai/gpt-oss-120b) for chat {chat_id}")
                try:
                    fallback_client = GroqLLM(
                        api_key=config.groq_api_key.get_secret_value(),
                        model_name="openai/gpt-oss-120b"
                    )
                    return await fallback_client.generate_response(chat_id, user_name, prompt, history)
                except Exception as fallback_error:
                    logger.error(f"Fallback to Groq failed: {fallback_error}")
                    raise e # Raise original error if fallback fails
            
            raise e


class GroqLLM(BaseLLM):
    """
    Groq Provider (DeepSeek, Llama, etc.) with manual Tool Calling loop.
    """
    def __init__(self, api_key: str, model_name: str) -> None:
        super().__init__()
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1"
        )
        self.model_name = model_name

    async def generate_response(self, chat_id: int, user_name: str, prompt: str, history: List[Dict[str, str]]) -> str:
        self._check_rate_limit(chat_id)
        
        system_instruction = await asyncio.to_thread(self._load_system_prompt)
        
        # Construct messages array with history
        messages = [{"role": "system", "content": system_instruction}]
        for msg in history:
            role = "user" if msg['role'] == "user" else "assistant"
            messages.append({"role": role, "content": msg['content']})
        messages.append({"role": "user", "content": f"{user_name}: {prompt}"})

        tools = [WEB_SEARCH_TOOL] if config.search_enabled else None

        try:
            # First API Call
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=config.model_temperature,
                top_p=config.model_top_p,
                tools=tools,
                tool_choice="auto" if tools else None
            )
            
            response_message = response.choices[0].message
            tool_calls = response_message.tool_calls

            # Check if model wants to call a tool
            if tool_calls:
                # Add the model's request to call a tool to the conversation history
                messages.append(response_message)

                for tool_call in tool_calls:
                    if tool_call.function.name == "web_search":
                        function_args = json.loads(tool_call.function.arguments)
                        query = function_args.get("query")
                        logger.info(f"Groq executing search tool for: {query}")
                        
                        # Execute search
                        search_result = await get_search_context(query)
                        if not search_result:
                            search_result = "No results found."

                        # Add tool result to conversation history
                        messages.append({
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": "web_search",
                            "content": search_result,
                        })

                # Second API Call (to generate final answer with tool info)
                second_response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=config.model_temperature,
                    top_p=config.model_top_p
                )
                final_content = second_response.choices[0].message.content
            else:
                final_content = response_message.content

            if final_content:
                clean_content = re.sub(r'<think>.*?</think>', '', final_content, flags=re.DOTALL).strip()
                return clean_content if clean_content else final_content
            return "Извините, я не смог сформулировать ответ."

        except Exception as e:
            logger.error(f"Groq API Error for chat {chat_id}: {e}")
            raise e


class DeepSeekOfficialLLM(BaseLLM):
    """
    Official DeepSeek Provider with manual Tool Calling loop.
    """
    def __init__(self, api_key: str, model_name: str) -> None:
        super().__init__()
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )
        self.model_name = model_name

    async def generate_response(self, chat_id: int, user_name: str, prompt: str, history: List[Dict[str, str]]) -> str:
        self._check_rate_limit(chat_id)
        
        system_instruction = await asyncio.to_thread(self._load_system_prompt)
        
        messages = [{"role": "system", "content": system_instruction}]
        for msg in history:
            role = "user" if msg['role'] == "user" else "assistant"
            messages.append({"role": role, "content": msg['content']})
        messages.append({"role": "user", "content": f"{user_name}: {prompt}"})

        # DeepSeek V3 supports function calling
        tools = [WEB_SEARCH_TOOL] if config.search_enabled else None

        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=config.model_temperature,
                top_p=config.model_top_p,
                stream=False,
                tools=tools,
                tool_choice="auto" if tools else None
            )
            
            response_message = response.choices[0].message
            tool_calls = response_message.tool_calls

            if tool_calls:
                messages.append(response_message)
                for tool_call in tool_calls:
                    if tool_call.function.name == "web_search":
                        function_args = json.loads(tool_call.function.arguments)
                        query = function_args.get("query")
                        logger.info(f"DeepSeek executing search tool for: {query}")
                        
                        search_result = await get_search_context(query)
                        messages.append({
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": "web_search",
                            "content": search_result or "No results found.",
                        })

                second_response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=config.model_temperature,
                    top_p=config.model_top_p
                )
                final_content = second_response.choices[0].message.content
            else:
                final_content = response_message.content

            if final_content:
                clean_content = re.sub(r'<think>.*?</think>', '', final_content, flags=re.DOTALL).strip()
                return clean_content if clean_content else final_content
            return "Извините, я не смог сформулировать ответ."

        except Exception as e:
            error_msg = str(e)
            if "402" in error_msg or "Insufficient Balance" in error_msg:
                 logger.error(f"DeepSeek Insufficient Balance: {e}")
                 return (
                     "⚠️ **Ошибка доступа к DeepSeek API**\n\n"
                     "Закончился баланс (Error 402). Официальный DeepSeek API не является полностью бесплатным.\n"
                     "Пожалуйста, пополните баланс на platform.deepseek.com или переключитесь на Groq в настройках."
                 )
            
            logger.error(f"DeepSeek Official API Error for chat {chat_id}: {e}")
            raise e


def get_llm_client() -> BaseLLM:
    """
    Factory function to return the configured LLM provider.
    """
    provider = config.llm_provider.lower()
    
    if provider in ["groq", "deepseek"]:
        if not config.groq_api_key:
            raise ValueError("GROQ_API_KEY is required for Groq provider.")
        return GroqLLM(
            api_key=config.groq_api_key.get_secret_value(), 
            model_name=config.groq_model
        )
    elif provider == "deepseek_official":
        if not config.deepseek_api_key:
            raise ValueError("DEEPSEEK_API_KEY is required for official DeepSeek provider.")
        return DeepSeekOfficialLLM(
            api_key=config.deepseek_api_key.get_secret_value(),
            model_name=config.deepseek_api_model
        )
    else:
        # Default to Gemini
        return GeminiLLM(
            api_key=config.gemini_api_key.get_secret_value(),
            model_name=config.gemini_model
        )

# Global instance
llm_client = get_llm_client()
