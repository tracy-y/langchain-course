# This code demonstrates how to use the langchain-anthropic library to interact with the Anthropic Large Language Model ("Claude-3 Sonnet").

# 1. Import required modules:
from dotenv import load_dotenv  # Loads environment variables from a .env file.
from langchain_core.prompts import PromptTemplate
from langchain_anthropic import ChatAnthropic  
# from langchain_ollama import ChatOllama # Imports the Ollama chat model wrapper from LangChain.

# 2. Load environment variables from .env file.
load_dotenv()

# 3. Define the main function that runs the logic.
def main():
    # Print a friendly message.
    print("Hello from langchain-course with Anthropic!")
    information = """ 
    Elon Reeve Musk (/ˈiːlɒn/ EE-lon; born June 28, 1971) is a businessman and entrepreneur known for his leadership of Tesla, SpaceX, Twitter, and xAI. Musk has been the wealthiest person in the world since 2021; as of October 2025, Forbes estimates his net worth to be US$500 billion.
    """
    summary_template = """
    Summarize the following information {information} about person I want you to create:
    1. A short summary
    2. two interesting facts about them
    """
    summary_prompt_template = PromptTemplate(
        template=summary_template,
        input_variables=["information"],
    )

    # llm = ChatOllama(
    #     model="gemma3:270m",
    #     temperature=0       
    # )
  
    llm = ChatAnthropic(
        model="claude-3-5-sonnet-20241022",  # Updated Claude model name
        temperature=0.7,                     # Controls creativity of responses (higher = more creative).
        max_tokens=1000                      # Maximum number of tokens to generate in the response.
    )

    chain = summary_prompt_template | llm
    response = chain.invoke({"information": information})
    print(response.content)


if __name__ == "__main__":
    main()
