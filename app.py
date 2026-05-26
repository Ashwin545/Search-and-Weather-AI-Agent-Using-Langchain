import os 
import certifi 
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain import hub
from langchain.tools import tool
import requests


from langchain.agents import create_react_agent, AgentExecutor

# ==========================================
# LOAD ENV VARIABLES
# ==========================================
os.environ["SSL_CERT_FILE"] = certifi.where()
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

search_tool = TavilySearchResults(max_results=2)

search_tool.invoke("What is the capital of France?")

# ==========================================
# LLM
# ==========================================

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
    api_key=OPENAI_API_KEY
)

response = llm.invoke("Tell me a joke about AI")
response

prompt = hub.pull("hwchase17/react")

tools = [search_tool]

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)

response = agent_executor.invoke({
    "input": (
        "Find the capital of India"
        "and then find its current weather."
    )
})

print(response["output"])