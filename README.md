# PydanticAI Reference Project

This repository serves as a comprehensive reference for using [PydanticAI](https://github.com/pydantic/pydantic-ai), a powerful framework for integrating Pydantic with Large Language Models (LLMs). The examples demonstrate how to create structured data models, build AI agents, and orchestrate complex multi-agent systems.

## Prerequisites

### API Keys

This project requires API keys for LLM providers. Create a `.env` file in the project root with the following variables:

```
OPENAI_API_KEY=your_openai_api_key
GEMINI_API_KEY=your_gemini_api_key  # Optional
ANTHROPIC_API_KEY=your_anthropic_api_key  # Optional
XAI_API_KEY=your_xai_api_key  # Optional
```

The examples primarily use OpenAI's GPT-4o model, but can be adapted to use other providers.

### Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Example Scripts

### 1. Basic PydanticAI Example (`pydantic_ai_basic.py`)

A simple introduction to PydanticAI's core concepts:
- Defining structured data models with Pydantic's `BaseModel`
- Creating an agent that connects to an LLM
- Adding tool functions to enhance agent capabilities
- Implementing a simple REPL interface for user interaction

Run with:
```bash
python pydantic_ai_basic.py
```

### 2. Advanced PydanticAI Example (`pydantic_ai_advanced.py`)

Builds on the basic example with more sophisticated features:
- Multiple agents working together (stock analyst and sentiment analyst)
- Real LLM calls within tool functions
- More complex data processing and analysis
- Enhanced user interface with detailed output

Run with:
```bash
python pydantic_ai_advanced.py
```

### 3. Clean PydanticAI Example (`pydantic_ai_clean.py`)

A minimalist version focused on clarity and readability:
- Removes verbose output and prettification
- Maintains core functionality with cleaner code
- Includes Logfire integration for observability
- Well-commented for educational purposes

Run with:
```bash
python pydantic_ai_clean.py
```

### 4. PydanticAI Orchestrator (`pydantic_ai_orchestrator.py`)

A sophisticated multi-agent system with orchestration:
- Orchestrator agent that routes tasks to specialized agents
- Multiple specialized agents with different capabilities
- Various tools that agents can use to gather data
- Complex task routing and processing
- Comprehensive Logfire integration for observability

Run with:
```bash
python pydantic_ai_orchestrator.py
```

## Logfire Observability

This project integrates [Logfire](https://github.com/pydantic/logfire), a powerful observability tool from the Pydantic team. To use Logfire:

1. Install Logfire: `pip install logfire`
2. Authenticate: `logfire auth`
3. View traces and logs at: https://logfire-us.pydantic.dev/ (or your selected region)

Logfire provides:
- Structured logging
- Distributed tracing
- Performance metrics
- Error tracking

## Key Concepts

### Agents

Agents are the core of PydanticAI. They connect to LLMs and define:
- The output type (a Pydantic model)
- The system prompt that guides the LLM's behavior
- Tools that the agent can use

### Tools

Tools are functions that agents can call to:
- Retrieve external data
- Perform calculations
- Call other agents
- Interact with APIs

### Pydantic Models

Pydantic models provide:
- Data validation
- Type checking
- Structured output from LLMs
- Clear documentation through field descriptions

## Advanced Usage

The orchestrator example demonstrates advanced concepts:
- Task routing between agents
- Complex workflows
- Error handling and fallbacks
- Hierarchical agent systems

## License

This project is provided as an educational reference. All code is available for use under the MIT License.
