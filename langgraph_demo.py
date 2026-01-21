"""
LangGraph Demo: Parallel guardrail execution with LLM cancellation.

This pattern solves the customer's problem: run guardrails and LLM generation
in parallel, then cancel the LLM if the guardrail detects a violation.

This is the RECOMMENDED approach for parallel process termination.
"""
import asyncio
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()


class State(TypedDict):
    messages: list
    guardrail_passed: bool
    error: str | None


async def check_guardrail(user_message: str) -> bool:
    """Fast guardrail check (1 second) - checks for policy violations"""
    print("Guardrail: Checking content policy (1s)...")
    await asyncio.sleep(1)
    
    # Simulate violation detection
    violation_detected = True  # Change to False to test passing scenario
    
    if violation_detected:
        print("Guardrail: VIOLATION DETECTED!")
        return False
    else:
        print("Guardrail: PASS - content is safe")
        return True


async def generate_llm_response(messages: list) -> str:
    """Expensive LLM generation (30 seconds)"""
    print("LLM: Starting expensive generation (30s)...")
    try:
        # Simulate expensive LLM call
        await asyncio.sleep(30)
        print("LLM: Generation complete!")
        return "AI Response: Here is a detailed answer..."
    except asyncio.CancelledError:
        print("LLM: Cancelled - tokens/time saved!")
        raise


async def parallel_guardrail_node(state: State) -> State:
    """
    Race guardrail against LLM generation.
    
    Pattern:
    1. Start both tasks simultaneously
    2. Await guardrail first (it's faster)
    3. If guardrail fails, cancel LLM immediately
    4. If guardrail passes, await LLM result
    """
    user_message = state["messages"][-1].content if state["messages"] else ""
    print(f"\nProcessing message: '{user_message}'")
    print("Starting parallel execution...\n")
    
    # Start both tasks in parallel
    guardrail_task = asyncio.create_task(check_guardrail(user_message))
    llm_task = asyncio.create_task(generate_llm_response(state["messages"]))
    
    # Check guardrail result first (it finishes faster)
    guardrail_passed = await guardrail_task
    
    if not guardrail_passed:
        # Cancel the LLM call immediately
        print("\nCancelling LLM task due to guardrail failure...")
        llm_task.cancel()
        try:
            await llm_task
        except asyncio.CancelledError:
            print("LLM task successfully cancelled\n")
        
        return {
            "messages": state["messages"],
            "guardrail_passed": False,
            "error": "Request blocked due to content policy violation"
        }
    
    # Guardrail passed, get LLM result
    print("\nGuardrail passed, waiting for LLM to complete...")
    llm_response = await llm_task
    
    return {
        "messages": state["messages"] + [HumanMessage(content=llm_response)],
        "guardrail_passed": True,
        "error": None
    }


# Build the graph
workflow = StateGraph(State)
workflow.add_node("parallel_guardrail", parallel_guardrail_node)
workflow.add_edge(START, "parallel_guardrail")
workflow.add_edge("parallel_guardrail", END)

app = workflow.compile()


if __name__ == "__main__":
    print("=" * 60)
    print("LangGraph Demo: Parallel Guardrail with Cancellation")
    print("=" * 60)
    print("\nThis demonstrates the recommended pattern for parallel")
    print("guardrail execution with the ability to cancel expensive")
    print("LLM calls mid-execution.\n")
    
    result = asyncio.run(app.ainvoke({
        "messages": [HumanMessage(content="Generate something that violates policy")],
        "guardrail_passed": False,
        "error": None
    }))
    
    print(f"{'=' * 60}")
    print(f"Guardrail passed: {result['guardrail_passed']}")
    if result['error']:
        print(f"Error: {result['error']}")
    else:
        print(f"Response: {result['messages'][-1].content}")
    print(f"{'=' * 60}\n")
