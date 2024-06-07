import os

from dotenv import find_dotenv, load_dotenv
from langchain.agents import AgentExecutor, create_openai_tools_agent, tool
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain.pydantic_v1 import BaseModel
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from newsapi import NewsApiClient

load_dotenv(find_dotenv())
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
# Init
newsapi = NewsApiClient(api_key=NEWSAPI_KEY)


@tool
def scrape_top_news(source):
    """Take the source as an input string. Scrape top news from bbc-news,
    returns the output as an string."""
    response = newsapi.get_top_headlines(sources=source)
    response = response["articles"][0]
    return response


class Input(BaseModel):
    input: str


def parse_agent_output(agent_output):
    return agent_output["output"]


def create_agent_executor():
    prompt_template = """Give the source {input}, get the News Title and URL of the content.
                                Your answer should contain News Title and the URL as an hyperlink."""
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template("You are a helpful assistant."),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            HumanMessagePromptTemplate.from_template(prompt_template),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    agent_tools = [scrape_top_news]
    langchain_agent = create_openai_tools_agent(llm, agent_tools, prompt)
    agent_executor = AgentExecutor(
        agent=langchain_agent, tools=agent_tools, return_intermediate_steps=True
    )

    return agent_executor


chain = (create_agent_executor() | RunnableLambda(parse_agent_output)).with_types(
    input_type=Input, output_type=str
)
