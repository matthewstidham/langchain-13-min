from langgraph.graph import START, StateGraph, END
from typing import TypedDict
from langgraph.prebuilt import create_react_agent
from financial_tools import stock_data_tool, portfolio_risk_tool
from weather import weather_tool
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")


class AgentState(TypedDict):
    user_query: str
    answer: str


# --- Define Node ---
def search_agent(state: AgentState) -> dict:
    """
    Executes a ReAct-style agent that processes a user query.

    This function takes the current state (which includes the user's question),
    creates an agent, then runs the agent to get a response.
    The final answer is returned as updated state.

    Args:
        state (AgentState): A dictionary with the user's query.

    Returns:
        dict: Updated state with the generated answer.
    """
    agent = create_react_agent(llm, [stock_data_tool, portfolio_risk_tool, weather_tool])
    result = agent.invoke({"messages": state["user_query"]})
    return {"answer": result["messages"][-1].content}


def main():
    # --- Define Graph ---
    workflow = StateGraph(AgentState)

    workflow.add_node("search_agent", search_agent)

    workflow.add_edge(START, "search_agent")
    workflow.add_edge("search_agent", END)

    app = workflow.compile()

    return app


if __name__ == "__main__":
    main()
