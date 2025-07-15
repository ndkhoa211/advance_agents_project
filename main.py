from dotenv import load_dotenv

load_dotenv()

from rich import print
from rich.markdown import Markdown
from rich.console import Console


from langchain import hub
from langchain_openai import ChatOpenAI
from langchain.agents import (
    create_react_agent,
    AgentExecutor,
    create_tool_calling_agent,
)
from langchain_experimental.tools import PythonAstREPLTool
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain.tools import Tool, tool
from typing import Dict, Any
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate

# try one between these 2:
from langchain_community.tools.tavily_search import (
    TavilySearchResults,
)  # deprecated in LangChain 0.3.25, will be removed in 1.0
from langchain_tavily import TavilySearch  # output annoying warnings


@tool
def multiply(x: float, y: float) -> float:
    """Multiply two numbers."""
    return x * y


def main():
    print("Start...")

    instructions = """You are an agent designed to write and execute python code to answer questions.
    You have access to a python REPL, which you can use to execute python code.
    If you get an error, debug your code and try again.
    Only use the output of your code to answer the question. 
    You might know the answer without running any code, but you should still run the code to get the answer.
    If it does not seem like you can write code to answer the question, just return "I don't know" as the answer.
    """

    base_prompt = hub.pull("langchain-ai/react-agent-template")

    prompt = base_prompt.partial(instructions=instructions)

    tools = [PythonAstREPLTool()]

    python_agent = create_react_agent(
        prompt=prompt,
        llm=ChatOpenAI(model="gpt-4.1-mini", temperature=0.0),
        tools=tools,
    )

    python_agent_executor = AgentExecutor(
        agent=python_agent,
        tools=tools,
        verbose=True,
    )

    # EXAMPLE 1
    # python_agent_executor.invoke(
    #     input={
    #         "input": """generate and save in current working directory QRcodes
    #         that point to https://github.com/ndkhoa211, you have qrcode package installed already"""
    #     }
    # )

    csv_agent_executor: AgentExecutor = create_csv_agent(
        llm=ChatOpenAI(model="gpt-4.1-mini", temperature=0.0),
        path="episode_info.csv",
        verbose=True,
        allow_dangerous_code=True,
    )

    # EXAMPLE 1
    # csv_agent.invoke(
    #     input={"input": "how many columns are there in file episode_info.csv"},
    # ) # Answer:8

    # EXAMPLE 2
    # csv_agent.invoke(
    #     input={
    #         "input": "print the seasons by ascending order of the number of episodes they have"
    #     },
    # )

    # EXAMPLE 3
    # csv_agent.invoke(
    #     input={
    #         "input": "in episode_info.csv, who wrote the most episode? How many episodes did he write?",
    #     }, # only part of the .csv is accessed so the answer always be 49 (correct is 58)
    # )

    ##### ROUTER GRAND AGENT #####

    def python_agent_executor_wrapper(original_prompt: str) -> Dict[str, Any]:
        return python_agent_executor.invoke({"input": original_prompt})

    tools = [
        Tool(
            name="Python Agent",
            # func=lambda x: python_agent_executor.invoke({"input": x}),
            func=python_agent_executor_wrapper,
            description="""useful when you need to transform natural language to python and execute the python code,
                        returning the results in the code execution
                        DOES NOT ACCEPT CODE AS INPUT""",
        ),
        Tool(
            name="CSV Agent",
            func=csv_agent_executor.invoke,
            description="""useful when you need to answer question over episode_info.csv file,
                        takes an input the entire question and return the answer after running pandas calculations""",
        ),
    ]

    prompt = base_prompt.partial(instructions="")
    grand_agent = create_react_agent(
        prompt=prompt,
        llm=ChatOpenAI(model="gpt-4.1-mini", temperature=0.0),
        tools=tools,
    )
    grand_agent_executor = AgentExecutor(
        agent=grand_agent,
        tools=tools,
        verbose=True,
    )

    # EXAMPLE 1
    # print(
    #     grand_agent_executor.invoke(
    #         {
    #             "input": "which season has the most episode?",
    #         }
    #     )
    # )

    # EXAMPLE 2
    # print(
    #     grand_agent_executor.invoke(
    #         {
    #             "input": "generate and save in current working directory a QR code that point to https://github.com/ndkhoa211",
    #         }
    #     )
    # )

    ########## TOOL CALLING ##########
    print(":::Hello Tool Calling:::")

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "you're a helpful assistant"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    tools = [TavilySearchResults(), multiply]
    # tools = [TavilySearch(), multiply]

    # llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.0)
    # llm = ChatAnthropic(model_name="claude-3-sonnet-20240229", temperature=0.0)
    llm = ChatAnthropic(model_name="claude-3-haiku-20240307", temperature=0.0)

    agent = create_tool_calling_agent(
        llm=llm,
        tools=tools,
        prompt=prompt,
    )

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
    )

    res = agent_executor.invoke(
        {
            "input": "what is the weather in Taipei right now? compare it with Kaohsiung, output should be in Celcius.",
        }
    )

    # ✅ Extract just the markdown-compatible string
    text_output = res["output"][0]["text"]

    # ✅ Print as markdown using Rich (beautiful formatting in terminal)
    console = Console()
    console.print(Markdown(text_output))

    print(
        "::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::"
    )
    print(res)


if __name__ == "__main__":
    main()
