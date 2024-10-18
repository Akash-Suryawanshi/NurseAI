from src.config import Config

from langchain_community.chat_models import ChatOpenAI

def get_gpt4_llm():
    """
    Returns an instance of OpenAI GPT-4 model using Langchain.
    """
    chat_openai_llm = ChatOpenAI(
        openai_api_key=Config.OPENAI_API_KEY,
        model="gpt-4",  # Specify GPT-4
        temperature=0,  # Adjust the temperature for creative responses
        max_tokens=10  # Adjust this based on your use case
    )
    return chat_openai_llm
