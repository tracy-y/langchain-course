
from dotenv import load_dotenv  

load_dotenv()
from langchain import hub;
from langchain.agents import AgentExecutor
from langchain.agents.react.agent import create_react_agent
from langchain_anthropic import ChatAnthropic  
from langchain_tavily import TavilySearch

tools = [TavilySearch()]


def main():
    # Print a friendly message.
    print("Hello from langchain-course!")
    
    # Initialize the LLM
    llm = ChatAnthropic(
        model="claude-3-5-sonnet-20241022",  # Updated Claude model name
        temperature=0.7,                     # Controls creativity of responses (higher = more creative).
        max_tokens=1000                      # Maximum number of tokens to generate in the response.
    )
    
    # Create the ReAct agent
    react_prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(
        llm=llm, 
        tools=tools, 
        prompt=react_prompt,
    )
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    # Run the agent
    result = agent_executor.invoke({
        "input": "search for 3 job postings for an AI engineer using langchain in the bay area on linkedin and list their details"
    })
    print(result)


if __name__ == "__main__":
    main()
