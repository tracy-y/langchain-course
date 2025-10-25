
from dotenv import load_dotenv

load_dotenv()
from langchain import hub
from langchain.agents import AgentExecutor
from langchain.agents.react.agent import create_react_agent
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_tavily import TavilySearch

from prompt import REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS
from schemas import AgentResponse

tools = [TavilySearch()]

llm = ChatAnthropic(
        model="claude-3-5-sonnet-20241022",  # Updated Claude model name
        temperature=0,                     # Controls creativity of responses (higher = more creative).
        max_tokens=1000                      # Maximum number of tokens to generate in the response.
    )

structured_llm = llm.with_structured_output(AgentResponse)
# Create the ReAct agent
react_prompt = hub.pull("hwchase17/react")
react_prompt_with_format_instructions = PromptTemplate(
        template=REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS,
        input_variables=["input", "agent_scratchpad", "tool_names"],
    ).partial(format_instructions="")

agent = create_react_agent(
    llm=llm, 
    tools=tools, 
    prompt=react_prompt_with_format_instructions,
)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
extract_output = RunnableLambda(lambda x: x["output"])
chain = agent_executor | extract_output | structured_llm


# Run the agent
def main():
    result = chain.invoke(
        input={
        "input": "search for 1 job postings for an AI engineer using langchain in Sdyney on linkedin and list their details"
        }
    )
    print(result)


if __name__ == "__main__":
    main()
