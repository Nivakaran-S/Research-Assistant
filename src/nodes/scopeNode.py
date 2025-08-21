
from datetime import datetime 
from typing_extensions import Literal

from src.llms.groqllm import GroqLLM
from langchain_core.messages import HumanMessage, AIMessage, get_buffer_string
from langgraph.types import Command 
from langgraph.graph import END, START

from src.utils.prompts import clarification_with_user_instrucitons, transform_messages_into_research_topic_prompt
from src.states.scopeState import AgentState, ClarifyWithUser, ResearchQuestion, AgentInputState

from src.utils.utils import get_today_str

model = GroqLLM().get_llm()

# Defining the nodes 
def clarify_with_user(state:AgentState) -> Command[Literal["write_research_brief", "__end__"]]:
    """
    Determine if the user's request contains sufficient information to proceed wiht research.

    Uses structured output to make deterministic decisions and avoid hallucination. 
    Routes to either research brief generation or ends with a clarification question.
    """
    structured_output_model = model.with_structured_output()

    response = structured_output_model.invoke([
        HumanMessage(
            content=clarification_with_user_instrucitons.format(
                messages=get_buffer_string(messages=state["messages"]),
                date=get_today_str()
            )
        )
    ])

    if response.need_clarification:
        return Command(
            goto=END,
            update={"messages", [AIMessage(content=response.question)]}
        )
    else:
        return Command(
            goto="write_research_brief",
            update={"messages": [AIMessage(content=response.verification)]}
        )

def write_research_brief(state: AgentState):
    """
    Transform the conversation history into a comprehensive research brief.

    Uses structured output to ensure the brief follows the required format
    and contains all necessary details for effective research.
    """
    structured_output_model = model.with_structured_output(ResearchQuestion)

    response = structured_output_model.invoke([
        HumanMessage(content=transform_messages_into_research_topic_prompt.format(
            messages=get_buffer_string(state.get("messages", [])),
            date=get_today_str()
        ))
    ])

    return {
        "research_brief": response.research_brief,
        "supervisor_message": [HumanMessage(content=f"{response.research_brief}")]
    }