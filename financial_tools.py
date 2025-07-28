import yfinance as yf
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")


class FinancialTools:
    def __init__(self):
        self.risk_free_rate = 0.045  # Current 10-year Treasury rate

    @staticmethod
    def get_stock_data(symbol: str, period: str = "1mo") -> str:
        """Get stock price data and basic metrics."""
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
            current_price = data['Close'].iloc[-1]
            change = ((current_price - data['Close'].iloc[0]) / data['Close'].iloc[0]) * 100

            return f"Symbol: {symbol}\nCurrent Price: ${current_price:.2f}\nPeriod Change: {change:.2f}%"
        except Exception as e:
            return f"Error retrieving data for {symbol}: {e}"

    @staticmethod
    def calculate_portfolio_risk(input_str: str) -> str:
        print(input_str)
        inputs = input_str.split(',')
        symbols = [x.split(':')[0] for x in inputs]
        weights = [x.split(':')[1] for x in inputs]
        """Calculate portfolio risk metrics."""
        try:
            # Fetch historical data
            data = yf.download(symbols, period="1y")['Adj Close']
            returns = data.pct_change().dropna()

            # Calculate portfolio return
            portfolio_return = (returns * weights).sum(axis=1)
            volatility = portfolio_return.std() * (252 ** 0.5)  # Annualized volatility

            s = f"Portfolio Volatility: {volatility:.2%}\nRisk Assessment: " \
                f"{'High' if volatility > 0.2 else 'Moderate' if volatility > 0.1 else 'Low'}"
            return s
        except Exception as e:
            return f"Error calculating portfolio risk: {e}"


# Create financial tools
financial_tools = FinancialTools()

stock_data_tool = Tool(
    name="get_stock_data",
    description="Get current stock price and performance data",
    func=financial_tools.get_stock_data
)

portfolio_risk_tool = Tool(
    name="calculate_portfolio_risk",
    description="Calculate portfolio risk metrics given symbols and weights",
    func=financial_tools.calculate_portfolio_risk
)


# Create financial agent with specialized prompt
financial_agent = initialize_agent(
    tools=[stock_data_tool, portfolio_risk_tool],
    llm=llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True),
    verbose=True,
    agent_kwargs={
        "system_message": """You are a financial analysis assistant. You help with stock analysis, 
        portfolio risk assessment, and market data interpretation. Always include disclaimers 
        about investment risks and recommend consulting with financial advisors. Use precise 
        financial terminology and provide data-driven insights."""
    }
)


def main():
    # Test financial agent
    response = financial_agent.run(
        "Analyze AAPL stock performance and assess risk for a portfolio with 60% AAPL and 40% MSFT")
    print(response)


if __name__ == "__main__":
    main()
