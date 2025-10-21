from fusion_hat.llm import LLM
from secret import API_KEY

# Register OpenAI API
# openai.com

# Export your openai api key with :LLM_API_KEY
# export LLM_API_KEY=sk-xxxxxxxxxxxxxxxxx


INSTRUCTIONS = "You are a helpful assistant."
WELCOME = "Hello, I am a helpful assistant. How can I help you?"

llm = LLM(
    base_url = f"",
    api_key=API_KEY,
    model="",
)

# Set how many messages to keep
llm.set_max_messages(20)
# Set instructions
llm.set_instructions(INSTRUCTIONS)
# Set welcome message
llm.set_welcome(WELCOME)

print(WELCOME)

while True:
    input_text = input(">>> ")

    # Response without stream
    # response = llm.prompt(input_text)
    # print(f"response: {response}")

    # Response with stream
    response = llm.prompt(input_text, stream=True)
    for next_word in response:
        if next_word:
            print(next_word, end="", flush=True)
    print("")
