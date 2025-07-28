from langchain_tavily import TavilySearch
from langchain.tools import Tool


class Weather:
    @staticmethod
    def search_tavily(input_str):
        search = TavilySearch(max_results=2)
        return search.invoke(input_str)


weather = Weather()

weather_tool = Tool(
    name="search_tavily",
    description="Get the current weather in different cities",
    func=weather.search_tavily
)
