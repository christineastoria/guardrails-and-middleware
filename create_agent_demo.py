"""
create_agent Demo: Sequential guardrails using middleware.

This shows the LangChain recommended pattern for guardrails: using middleware
to check content BEFORE the LLM call (input guardrails) or AFTER (output guardrails).

Note: This is SEQUENTIAL, not parallel. For parallel execution with cancellation,
use the LangGraph approach.
"""
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, AgentState, ModelRequest, ModelResponse
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from typing import Any, Callable
from dotenv import load_dotenv

load_dotenv()


# Define a simple tool for the agent
@tool
def search_tool(query: str) -> str:
    """Search for information"""
    return f"Search results for: {query}"


class InputGuardrailMiddleware(AgentMiddleware):
    """
    Middleware that checks input guardrails BEFORE the model is called.
    
    This runs sequentially:
    1. Check user input for violations
    2. If violation: raise error (stops execution)
    3. If pass: continue to model call
    """
    
    def before_model(self, state: AgentState, runtime) -> dict[str, Any] | None:
        """Check input guardrails before model execution"""
        print("\n" + "=" * 60)
        print("MIDDLEWARE: Checking input guardrails...")
        print("=" * 60)
        
        # Get the user's message
        messages = state.get("messages", [])
        if not messages:
            return None
        
        user_message = messages[-1].content if hasattr(messages[-1], 'content') else str(messages[-1])
        
        # Simulate guardrail check
        print(f"Input: '{user_message}'")
        
        # Check for policy violations
        violation_detected = True  # Change to False to test passing scenario
        
        if violation_detected:
            print("Result: VIOLATION DETECTED")
            print("Action: Blocking model call\n")
            raise ValueError("Content policy violation: Request blocked by input guardrail")
        
        print("Result: PASS - content is safe")
        print("Action: Proceeding to model call\n")
        return None  # Don't modify state


class OutputGuardrailMiddleware(AgentMiddleware):
    """
    Middleware that checks output guardrails AFTER the model responds.
    
    This runs sequentially:
    1. Model generates response
    2. Check response for issues (hallucinations, safety, etc)
    3. If violation: raise error or modify response
    4. If pass: continue normally
    """
    
    def after_model(self, state: AgentState, runtime) -> dict[str, Any] | None:
        """Check output guardrails after model execution"""
        print("\n" + "=" * 60)
        print("MIDDLEWARE: Checking output guardrails...")
        print("=" * 60)
        
        messages = state.get("messages", [])
        if not messages:
            return None
        
        last_message = messages[-1]
        response_content = last_message.content if hasattr(last_message, 'content') else str(last_message)
        
        print(f"Output: '{response_content[:100]}...'")
        
        # Check for output issues (hallucinations, safety, etc)
        has_issues = False  # Change to True to test output blocking
        
        if has_issues:
            print("Result: OUTPUT VIOLATION DETECTED")
            print("Action: Blocking response\n")
            raise ValueError("Output guardrail violation: Response contains unsafe content")
        
        print("Result: PASS - response is safe")
        print("Action: Allowing response\n")
        return None  # Don't modify state


if __name__ == "__main__":
    print("=" * 60)
    print("create_agent Demo: Sequential Middleware Guardrails")
    print("=" * 60)
    print("\nThis demonstrates LangChain's recommended middleware pattern.")
    print("Guardrails run SEQUENTIALLY (before/after model calls).\n")
    print("For PARALLEL execution with cancellation, use LangGraph.\n")
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Create agent with guardrail middleware
    agent = create_agent(
        model=llm,
        tools=[search_tool],
        system_prompt="You are a helpful assistant.",
        middleware=[
            InputGuardrailMiddleware(),   # Checks input before model
            OutputGuardrailMiddleware(),  # Checks output after model
        ]
    )
    
    # Test 1: Input guardrail blocks request
    print("TEST 1: Input Guardrail (blocks before model call)")
    print("-" * 60)
    
    try:
        result = agent.invoke({
            "messages": [HumanMessage(content="Generate something that violates policy")]
        })
        
        print("\n" + "=" * 60)
        print("SUCCESS: Request completed")
        print("=" * 60)
        print(f"Response: {result['messages'][-1].content}\n")
        
    except ValueError as e:
        print("\n" + "=" * 60)
        print("BLOCKED: Request stopped by input guardrail")
        print("=" * 60)
        print(f"Reason: {e}\n")
    
    # Test 2: Output guardrail validates response
    print("\n" + "=" * 60)
    print("TEST 2: Output Guardrail (validates after model responds)")
    print("-" * 60)
    print("\nChanging violation_detected to False in InputGuardrailMiddleware")
    print("and has_issues to True in OutputGuardrailMiddleware to demonstrate...")
    print("\nIn a real implementation, you would check the actual response content.")
    print("-" * 60 + "\n")
    
    print("\nKEY DIFFERENCES FROM LANGGRAPH:")
    print("- Input guardrail: Checks BEFORE model (blocks bad requests)")
    print("- Output guardrail: Checks AFTER model (validates responses)")
    print("- Both run sequentially - cannot cancel mid-generation")
    print("- LangGraph parallel pattern can cancel during streaming")
    print("\nFor parallel cancellation with streaming, use LangGraph.\n")
