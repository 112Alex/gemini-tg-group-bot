import logging
import asyncio
from aiogram import Router, F, types
from aiogram.filters import Command
from aiogram.utils.chat_action import ChatActionSender
from aiogram.enums import ChatAction

from services.llm_service import llm_client, RateLimitExceeded
from services.memory import chat_memory

logger = logging.getLogger(__name__)
router = Router()

def split_text(text: str, max_length: int = 4096) -> list[str]:
    """
    Splits text into chunks of max_length, respecting word boundaries where possible.
    """
    if len(text) <= max_length:
        return [text]
    
    chunks = []
    while text:
        if len(text) <= max_length:
            chunks.append(text)
            break
            
        # Find the last space within the limit
        split_index = text.rfind(' ', 0, max_length)
        
        # If no space found, force split at max_length
        if split_index == -1:
            split_index = max_length
            
        chunks.append(text[:split_index])
        text = text[split_index:].lstrip()
        
    return chunks

@router.message(Command("ask"))
@router.message(F.text)
async def handle_ask(message: types.Message):
    """
    Handles /ask command, mentions, or replies to the bot.
    """
    if not message.text:
        return

    bot_info = await message.bot.get_me()
    
    # Logic to determine if the bot should respond
    is_command = message.text.startswith("/ask")
    is_mentioned = f"@{bot_info.username}" in message.text
    is_reply_to_bot = (
        message.reply_to_message 
        and message.reply_to_message.from_user.id == bot_info.id
    )

    if not (is_command or is_mentioned or is_reply_to_bot):
        return

    # Clean the prompt
    prompt = message.text
    if is_command:
        prompt = prompt.replace("/ask", "", 1).strip()
    if is_mentioned:
        prompt = prompt.replace(f"@{bot_info.username}", "", 1).strip()
    
    # If it's a mention/command but without text, and NOT a reply, ask for input
    if not prompt and not is_reply_to_bot:
        await message.reply("Пожалуйста, введите ваш вопрос.")
        return

    # If it's a reply to bot without new text, we can use a default prompt
    if not prompt and is_reply_to_bot:
        prompt = "Продолжай"

    chat_id = message.chat.id
    user_name = message.from_user.full_name

    async with ChatActionSender.typing(bot=message.bot, chat_id=chat_id):
        try:
            # Get conversation history
            history = chat_memory.get_history(chat_id)

            # Generate response using the configured LLM provider
            response_text = await llm_client.generate_response(
                chat_id=chat_id,
                user_name=user_name,
                prompt=prompt,
                history=history
            )
            
            # Save interaction to memory (full response)
            chat_memory.add_message(chat_id, "user", f"{user_name}: {prompt}")
            chat_memory.add_message(chat_id, "assistant", response_text)
            
            # Split and send response
            chunks = split_text(response_text)
            for chunk in chunks:
                try:
                    await message.reply(chunk, parse_mode="HTML")
                except Exception as e:
                    # Fallback to plain text if HTML parsing fails (common with complex LLM output)
                    await message.reply(chunk, parse_mode=None)
                # Small delay to ensure order
                await asyncio.sleep(0.1)

        except RateLimitExceeded:
            await message.reply("Подождите пару секунд... (Rate limit)")
        except Exception as e:
            logger.error(f"Error handling message in chat {chat_id}: {e}")
            await message.reply("Произошла ошибка при обращении к AI. Попробуйте позже.")
