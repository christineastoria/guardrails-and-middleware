"""
create_deep_agent Demo: Sequential guardrails with planning.

Deep agents use planning to organize tasks. This demo shows how to use
middleware for guardrails in a planning-based agent.

Note: This is SEQUENTIAL with smart task ordering. For parallel execution
with cancellation, use the LangGraph approach.
"""
from deepagents import create_deep_agent
from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from typing import Any
from dotenv import load_dotenv

load_dotenv()


@tool
def research_tool(query: str) -> str:
    """Research a topic"""
    return f"Research findings for: {query}"


class InputGuardrailMiddleware(AgentMiddleware):
    """
    Input guardrail for deep agents - checks before planning.
    
    Deep agents create a plan with todos, then execute them. This middleware
    checks inputs before the agent creates its plan.
    """
    
    def before_model(self, state: AgentState, runtime) -> dict[str, Any] | None:
        """Check input guardrails before agent planning"""
        print("\n" + "=" * 60)
        print("INPUT MIDDLEWARE: Checking before planning...")
        print("=" * 60)
        
        messages = state.get("messages", [])
        if not messages:
            return None
        
        last_message = messages[-1]
        user_input = last_message.content if hasattr(last_message, 'content') else str(last_message)
        
        print(f"User request: '{user_input}'")
        
        # Check for policy violations before agent creates plan
        violation_detected = True  # Change to False to test passing scenario
        
        if violation_detected:
            print("Result: VIOLATION DETECTED")
            print("Action: Blocking plan creation\n")
            raise ValueError("Content policy violation: Deep agent blocked by input guardrail")
        
        print("Result: PASS - request is safe")
        print("Action: Agent will create plan and execute\n")
        return None


class OutputGuardrailMiddleware(AgentMiddleware):
    """
    Output guardrail for deep agents - validates final response.
    
    This runs after the agent completes its plan and generates a response.
    """
    
    def after_model(self, state: AgentState, runtime) -> dict[str, Any] | None:
        """Check output guardrails after agent execution"""
        print("\n" + "=" * 60)
        print("OUTPUT MIDDLEWARE: Checking agent response...")
        print("=" * 60)
        
        messages = state.get("messages", [])
        if not messages:
            return None
        
        last_message = messages[-1]
        response = last_message.content if hasattr(last_message, 'content') else str(last_message)
        
        print(f"Agent response: '{response[:100]}...'")
        
        # Check for output issues (hallucinations, safety, quality)
        has_issues = False  # Change to True to test output blocking
        
        if has_issues:
            print("Result: OUTPUT VIOLATION DETECTED")
            print("Action: Blocking response\n")
            raise ValueError("Output guardrail violation: Response contains issues")
        
        print("Result: PASS - response is safe")
        print("Action: Allowing response\n")
        return None


if __name__ == "__main__":
    print("=" * 60)
    print("create_deep_agent Demo: Sequential Guardrails")
    print("=" * 60)
    print("\nDeep agents use planning to organize tasks sequentially.")
    print("Middleware checks guardrails BEFORE plan execution.\n")
    print("For PARALLEL execution with cancellation, use LangGraph.\n")
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Create deep agent with guardrail middleware
    agent = create_deep_agent(
        model=llm,
        tools=[research_tool],
        system_prompt="""You are a helpful research assistant.
        
When given a task:
1. Create a plan with specific todos
2. Execute each todo in order
3. Provide a comprehensive answer

Always be thorough and systematic.""",
        middleware=[
            InputGuardrailMiddleware(),   # Checks input before planning
            OutputGuardrailMiddleware(),  # Checks output after execution
        ]
    )
    
    print("Testing with potentially violating input...")
    print("-" * 60)
    
    try:
        result = agent.invoke({
            "messages": [HumanMessage(content="Research something that violates policy")]
        })
        
        print("\n" + "=" * 60)
        print("SUCCESS: Request completed")
        print("=" * 60)
        
        # Extract final response
        final_message = result["messages"][-1]
        print(f"Response: {final_message.content}\n")
        
    except ValueError as e:
        print("\n" + "=" * 60)
        print("BLOCKED: Request stopped by middleware")
        print("=" * 60)
        print(f"Reason: {e}\n")
    
    print("\nDEEP AGENT WITH GUARDRAILS:")
    print("- Input guardrail: Checks BEFORE agent creates plan")
    print("- Output guardrail: Validates AFTER plan execution completes")
    print("- Both run sequentially (not parallel)")
    print("- Creates a plan with todos, executes sequentially")
    print("- Good for complex multi-step tasks with validation")
    print("- Cannot cancel mid-execution during plan execution")
    print("\nFor parallel cancellation with streaming, use LangGraph.\n")
