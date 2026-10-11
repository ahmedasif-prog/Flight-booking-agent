import os
from typing import TypedDict,Annotated,List
import operator
from langgraph.graph import StateGraph,START,END
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_core.messages import AnyMessage,HumanMessage,AIMessage,SystemMessage
from langchain_groq import ChatGroq
from tools.tavily_tool import tavily_search
from tools.flight_tools import search_flights
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model = "llama-3.3-70b-versatile"
)

class TravelState(TypedDict):
    messages:Annotated[List[AnyMessage],operator.add]
    userquery:str
    flight_results:str
    hotel_results:str
    itinerrary:str
    llm_calls:int
