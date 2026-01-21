# Parallel Guardrails with Early Termination

Demonstrations of running guardrails (classifiers) in parallel with LLM generation, with the ability to terminate expensive operations when violations are detected.

## Background

After consulting with the OSS team, the recommendation is:

**For parallel execution with cancellation**: Use `asyncio` within a LangGraph node to kick off parallel tasks, then conditionally terminate one using `asyncio.Task.cancel()`.

**For sequential validation**: Use the LangChain recommended pattern with middleware - `before_model` hooks for input guardrails and `after_model` hooks for output guardrails.

This repo demonstrates both approaches.

---

## The Problem

Customer wants to run guardrails in parallel with LLM calls to reduce latency. If a guardrail detects a violation in 1 second, they need to terminate the 30-second LLM call immediately to save tokens and time.

**Challenge**: Python's threading library cannot kill running threads. Solution: Use `asyncio.Task.cancel()`.

---

## Demos

### 1. langgraph_demo.py - Parallel with Cancellation

**Why this demo**: Shows the OSS team's recommended approach for parallel execution with cancellation.

**Pattern**:
```python
async def parallel_guardrail_node(state):
    # Start both tasks in parallel
    guardrail_task = asyncio.create_task(check_guardrail(...))
    llm_task = asyncio.create_task(generate_llm(...))
    
    # Check guardrail first (it's faster)
    guardrail_passed = await guardrail_task
    
    if not guardrail_passed:
        llm_task.cancel()  # Terminate immediately
        return {"error": "blocked"}
    
    # Guardrail passed, get LLM result
    response = await llm_task
    return {"messages": [response]}
```

**Use when**: You need to cancel expensive operations mid-execution to save tokens and time.

---

### 2. create_agent_demo.py - Sequential Middleware

**Why this demo**: Shows the LangChain recommended pattern for guardrails using middleware.

**Pattern**:
```python
from langchain.agents.middleware import AgentMiddleware

class InputGuardrailMiddleware(AgentMiddleware):
    def before_model(self, state, runtime):
        # Check input before LLM runs
        if violation_detected:
            raise ValueError("Blocked")
        return None

class OutputGuardrailMiddleware(AgentMiddleware):
    def after_model(self, state, runtime):
        # Validate response after LLM completes
        if has_issues:
            raise ValueError("Blocked")
        return None

agent = create_agent(
    model=llm,
    tools=[...],
    middleware=[
        InputGuardrailMiddleware(),
        OutputGuardrailMiddleware(),
    ]
)
```

**Use when**: Sequential validation is sufficient (input checks before, output checks after). Simpler than parallel approach.

**Limitation**: Cannot cancel mid-execution. Must wait for LLM to complete before output validation runs.

---

### 3. create_deep_agent.py - Sequential with Planning

**Why this demo**: Shows how middleware works with planning-based deep agents.

**Pattern**:
```python
class InputGuardrailMiddleware(AgentMiddleware):
    def before_model(self, state, runtime):
        # Check before agent creates plan
        if violation_detected:
            raise ValueError("Blocked")
        return None

class OutputGuardrailMiddleware(AgentMiddleware):
    def after_model(self, state, runtime):
        # Validate after plan execution
        if has_issues:
            raise ValueError("Blocked")
        return None

agent = create_deep_agent(
    model=llm,
    tools=[...],
    middleware=[
        InputGuardrailMiddleware(),
        OutputGuardrailMiddleware(),
    ]
)
```

**Use when**: Complex multi-step tasks requiring planning with input/output validation.

**Limitation**: Sequential execution. Cannot cancel during plan execution.

---

## Input vs Output Guardrails

**Input Guardrails** (before LLM):
- Check user messages for toxicity, PII, policy violations
- Block bad requests before expensive processing

**Output Guardrails** (after LLM):
- Validate AI responses for hallucinations, safety, quality
- Ensure responses meet standards before returning to user

**With LangGraph parallel pattern**: Can cancel during generation if streaming output fails checks.

**With sequential middleware**: Input runs before, output runs after. Must wait for completion.

---

## Running the Demos

```bash
cd terminate-parallel-processes
uv sync
uv run langgraph_demo.py
uv run create_agent_demo.py
uv run create_deep_agent.py
```

Ensure `.env` contains:
```
OPENAI_API_KEY=your_key_here
```

---

## Summary

**Parallel cancellation**: Use LangGraph with asyncio (recommended for the customer's use case).

**Sequential validation**: Use create_agent or create_deep_agent with middleware (standard LangChain pattern, simpler but no cancellation).

Choose based on whether you need mid-execution cancellation or if sequential validation is sufficient.
