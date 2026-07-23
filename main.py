from dotenv import load_dotenv
import openai


load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.messages import  HumanMessage
from langchain_openai import ChatOpenAI
#from tavily import TavilyClient

from langchain_tavily import TavilySearch

#tavily = TavilyClient()
newTools = [TavilySearch()]

#Langchain Tool is a function which agent can execute. 
'''
@tool
def search(query: str) -> str:
    \'''
    Tool that searches over internet
    Args:
        query (str): The search query to be executed.
    Returns:
        str: The search results.
    \'''

    print(f"Searching for: {query}")
    #return f"Tokyo is very sunny"
    return tavily.search(query=query, num_results=3)
'''    


llm = ChatOpenAI(model="gpt-5")

# ollama does not support tools
#llm = ChatOllama(temperature=0, model="gemma3:270m", max_tokens=1000)

# List of tools
#tools = [search]
agent = create_agent(model=llm, tools=newTools)


def main():
    print("Hello, World!")
    #result = agent.invoke({"messages":[HumanMessage(content="What is the weather in Tokyo?")]})
    result = agent.invoke({"messages":[HumanMessage(content="Search for 3 job postings for an AI engineer using langchain in Sydney area and list their details?")]})
    print(f"Agent Result: {result}")


if __name__ == "__main__":
    main()