""" Large Language Model (LLM) module

This module provides a base class for large language models (LLMs) and preset LLM classes.

Example:
    import a preset LLM class

    >>> from sunfounder_voice_assistant.llm import Deepseek as LLM
    >>> from sunfounder_voice_assistant.llm import Grok as LLM
    >>> from sunfounder_voice_assistant.llm import Doubao as LLM
    >>> from sunfounder_voice_assistant.llm import Qwen as LLM
    >>> from sunfounder_voice_assistant.llm import OpenAI as LLM

    initialize the LLM instance

    >>> API_KEY = "your_api_key"
    >>> MODEL = "your_model"
    >>> llm = LLM(api_key=API_KEY, model=MODEL)
    
    For Ollama, you don't need api_key, but you might need to set ip.

    >>> from sunfounder_voice_assistant.llm import Ollama as LLM
    >>> llm = LLM(ip="localhost", model="deepseek-r1:1.5b")

    You can also import a basic LLM class.

    >>> from sunfounder_voice_assistant.llm import LLM as LLM

    You will need to set the base url which compatible with OpenAI completion API.

    >>> llm = LLM(
            base_url="https://api.deepseek.com",
            model=MODEL,
            api_key=API_KEY,
        )

    Or set the whole url if it's not ends with "/v1/chat/completions"

    >>> llm = LLM(
            url="https://api.deepseek.com/v1/chat/completions",
            model=MODEL,
            api_key=API_KEY,
        )

    Set instructions

    >>> llm.set_instructions("You are a helpful assistant.")

    Set welcome message

    >>> llm.set_welcome("Hello, I am a helpful assistant. How can I help you?")

    Prompt the LLM with input text

    >>> input_text = "Hello"
    >>> response = llm.prompt(input_text, stream=True)
    >>> for next_word in response:
    >>>     if next_word:
    >>>         print(next_word, end="", flush=True)
    >>> print("")

    Prompt with image

    >>> input_text = "Hello"
    >>> image = "image.jpg"
    >>> response = llm.prompt(input_text, image=image, stream=True)
    >>> for next_word in response:
    >>>     if next_word:
    >>>         print(next_word, end="", flush=True)
    >>> print("")
"""

from .llm import LLM

__all__ = [
    "LLM",
    "Deepseek",
    "Grok",
    "Doubao",
    "Gemini",
    "Qwen",
    "OpenAI",
    "Ollama",
]

class Deepseek(LLM):
    """ Deepseek preset LLM class
        
        Args:
            *args: Passed to :class:`sunfounder_voice_assistant.llm.llm.LLM`
            **kwargs: Passed to :class:`sunfounder_voice_assistant.llm.llm.LLM`
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args,
            base_url="https://api.deepseek.com",
            **kwargs)

class Grok(LLM):
    """ Grok preset LLM class
        
        Args:
            *args: Passed to :class:`sunfounder_voice_assistant.llm.llm.LLM`
            **kwargs: Passed to :class:`sunfounder_voice_assistant.llm.llm.LLM`
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, 
            base_url="https://api.x.ai/v1",
            **kwargs)

class Doubao(LLM):
    """ Doubao preset LLM class
        
        Args:
            *args: Passed to :class:`sunfounder_voice_assistant.llm.llm.LLM`
            **kwargs: Passed to :class:`sunfounder_voice_assistant.llm.llm.LLM`
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, 
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            **kwargs)

class Qwen(LLM):
    """ Qwen preset LLM class
        
        Args:
            *args: Passed to :class:`sunfounder_voice_assistant.llm.llm.LLM`
            **kwargs: Passed to :class:`sunfounder_voice_assistant.llm.llm.LLM`
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, 
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            **kwargs)

class OpenAI(LLM):
    """ OpenAI preset LLM class
        
        Args:
            *args: Passed to :class:`sunfounder_voice_assistant.llm.llm.LLM`
            **kwargs: Passed to :class:`sunfounder_voice_assistant.llm.llm.LLM`
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, 
            base_url="https://api.openai.com/v1",
            **kwargs)

class Ollama(LLM):
    """ Ollama preset LLM class
        
        Args:
            ip (str, optional): IP address of Ollama server. Defaults to "localhost".
            *args: Passed to :class:`sunfounder_voice_assistant.llm.llm.LLM`
            api_key (str, optional): API key of Ollama server. Defaults to "ollama".
            **kwargs: Passed to :class:`sunfounder_voice_assistant.llm.llm.LLM`
    """
    def __init__(self, ip: str="localhost", *args, api_key: str="ollama", **kwargs):
        super().__init__(*args, 
            url=f"http://{ip}:11434/api/chat",
            # base_url=f"http://{ip}:11434/v1",
            api_key=api_key,
            **kwargs)
    
    def add_message(self, role: str, content: str, image_path: str=None) -> None:
        """ Add message to messages list

        Args:
            role (str): Role of message, e.g. "user", "assistant"
            content (str): Content of message
            image_path (str, optional): Image path, default is None

        Raises:
            ValueError: Role must be 'user' or 'assistant'
        """
        if role not in ["user", "assistant", "system"]:
            raise ValueError("Role must be 'user' or 'assistant'")

        data = {"role": role, "content": content}
        if image_path is not None:
            # get base64 from image
            base64 = self.get_base64_from_image(image_path)
            # add to content
            data["images"] = [base64]

        self.messages.append(data)

    def decode_stream_response(self, line: str) -> str:
        """ Decode stream response line

        Args:
            line (str): Stream response line

        Returns:
            str: Decoded content, None if error
        """
        import json
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            return None
        if "message" in data and "content" in data["message"]:
            return data["message"]["content"]
        if "error" in data:
            raise Exception(data["error"])
        return None

class Gemini(LLM):
    """ Gemini preset LLM class
        
        Args:
            *args: Passed to :class:`sunfounder_voice_assistant.llm.llm.LLM`
            **kwargs: Passed to :class:`sunfounder_voice_assistant.llm.llm.LLM`
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, 
            base_url="https://generativelanguage.googleapis.com/v1beta/openai",
            **kwargs)
