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
import psycopg
from psycopg.rows import dict_row

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
llm = ChatGroq(
    model = "openai/gpt-oss-20b"
)

class TravelState(TypedDict):
    messages:Annotated[List[AnyMessage],operator.add]
    userquery:str
    flight_results:str
    hotel_results:str
    itinerary:str
    llm_calls:int


def flight_agent(state:TravelState):
    query = state["userquery"]
    flight_data = search_flights(query)
    return{
        "flight_results":flight_data,
        "messages":[AIMessage(content=f"Flight results fetched")],
        # "llm calls":state.get("llm_calls",0)+1

    }

def hotel_agent(state:TravelState):
     query = f"Best hotels for {state['userquery']}"
     hotel_data = tavily_search(query)
     return{
            "hotel_results":hotel_data,
            "messages":[AIMessage(content=f"Hotel Information fetched")],
            # "llm calls":state.get("llm_calls",0)+1
        }
def itinerary_agent(state:TravelState):
    prompt = f"""
    Create a travel itineray
    User query: {state['userquery']}
    Flight Results:{state['flight_results']}
    Hotel Results:{state['hotel_results']}
"""
    response = llm.invoke([
        SystemMessage(content = "You are an expert travel planner"),HumanMessage(content  = prompt)]
    )
    return{
        "itinerary":response.content,
        "messages":[response],
        "llm_calls":state.get('llm_calls',0)+1
    }

def final_agent(state:TravelState):
    final_prompt = f"""
    Generate final travel response
    User query: {state['userquery']}
    Flight Results:{state['flight_results']}
    Hotel Results:{state['hotel_results']}
"""
    response = llm.invoke(final_prompt)
    return {
        "messages":[response],
        "llm_calls":state.get('llm_calls',0)+1
    }

graph = StateGraph(TravelState)

graph.add_node("flight_agent",flight_agent)
graph.add_node("hotel_agent",hotel_agent)
graph.add_node("itinerary_agent",itinerary_agent)
graph.add_node("final_agent",final_agent)

graph.add_edge(START,"flight_agent")
graph.add_edge("flight_agent","hotel_agent")
graph.add_edge("hotel_agent","itinerary_agent")
graph.add_edge("itinerary_agent","final_agent")
graph.add_edge("final_agent",END)

_conn = psycopg.connect(DATABASE_URL,autocommit = True,row_factory=dict_row)
checkpointer = PostgresSaver(_conn)
checkpointer.setup()


app = graph.compile(checkpointer=checkpointer)

if __name__ == "__main__":
    config = {"configurable":{"thread_id":"user_ahmed"}}
    user_input = input("Enter travel request:")
    result = app.invoke(
        {
            "messages":[HumanMessage(content=user_input)],
            "userquery":user_input,
            "flight_results":"",
            "hotel_results":"",
            "itinerary":"",
            "llm_calls":0
        },

        config=config
    )
    print("Final Response:")
    for message in result["messages"]:
        print(message.content)