import os
import openai

DEFAULT_MODEL = os.getenv("MODEL_NAME", "gpt-4.1-mini-2025-04-14")
SYSTEM_PROMPT = "You are a helpful assistant."

def chat_completion(user_prompt: str,
                    system_prompt: str = SYSTEM_PROMPT,
                    model: str = None,
                    **kwargs) -> str:
    """
    Send a chat completion request and return the raw assistant reply.
    
    :param user_prompt: The content of the user message.
    :param system_prompt: The content of the system message.
    :param model:      Which model to use (defaults to $MODEL_NAME).
    :param kwargs:     Any extra args (like temperature, max_tokens, etc).
    :return:           The assistant’s reply as a string.
    """
    mdl = model or DEFAULT_MODEL
    resp = openai.chat.completions.create(
        model=mdl,
        messages=[
            {"role": "system",  "content": system_prompt},
            {"role": "user",    "content": user_prompt}
        ],
        **kwargs
    )
    return resp
