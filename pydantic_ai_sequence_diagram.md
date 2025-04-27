# PydanticAI Basic Example - Sequence Diagram

This diagram illustrates the sequence of function calls in the basic PydanticAI example (`pydantic_ai_basic.py`).

```mermaid
sequenceDiagram
    participant User
    participant Main as Main Script
    participant load_dotenv
    participant StockAnalysis as StockAnalysis Model
    participant Agent as Agent Creation
    participant analyze_stock
    participant stock_analyst as stock_analyst.run_sync
    participant get_market_sentiment
    participant LLM as OpenAI GPT-4o

    User->>Main: Run script
    Main->>load_dotenv: load_dotenv()
    Note over load_dotenv: Loads API keys from .env file
    
    Main->>StockAnalysis: Define StockAnalysis model
    Note over StockAnalysis: Creates structured data model with fields:<br/>ticker, current_price, recommendation,<br/>confidence, rationale
    
    Main->>Agent: Create stock_analyst agent
    Note over Agent: Initializes agent with:<br/>- OpenAI GPT-4o model<br/>- StockAnalysis output type<br/>- System prompt<br/>- API key
    
    Main->>Agent: Register @stock_analyst.tool
    Note over Agent: Adds get_market_sentiment tool<br/>to the agent's capabilities
    
    Main->>Main: Define analyze_stock function
    
    Main->>User: Prompt for ticker input
    User->>Main: Enter ticker (e.g., "AAPL")
    
    Main->>analyze_stock: analyze_stock(ticker)
    analyze_stock->>stock_analyst: stock_analyst.run_sync(prompt)
    Note over stock_analyst: Constructs prompt with ticker
    
    stock_analyst->>LLM: Send prompt to OpenAI
    
    LLM-->>get_market_sentiment: May call tool function
    get_market_sentiment-->>LLM: Return market sentiment
    
    LLM-->>stock_analyst: Return structured response
    Note over LLM,stock_analyst: Response is validated against<br/>StockAnalysis model
    
    stock_analyst-->>analyze_stock: Return StockAnalysis object
    analyze_stock-->>Main: Return analysis result
    
    Main->>User: Display formatted analysis
    Note over Main,User: Shows ticker, price, recommendation,<br/>confidence, and rationale
    
    Main->>User: Prompt for next ticker
```

## Explanation of Key Components

1. **Environment Setup**:
   - `load_dotenv()` loads API keys from the `.env` file

2. **Model Definition**:
   - `StockAnalysis` Pydantic model defines the structure for stock analysis results

3. **Agent Creation**:
   - `stock_analyst = Agent(...)` creates an agent with:
     - LLM provider and model (`openai:gpt-4o`)
     - Output type (`StockAnalysis`)
     - System prompt (instructions for the agent)
     - API key from environment variables

4. **Tool Registration**:
   - `@stock_analyst.tool` decorator registers the `get_market_sentiment` function
   - This gives the agent the ability to retrieve market sentiment data

5. **User Interaction**:
   - Script prompts user for stock ticker input
   - User enters a ticker (e.g., "AAPL")

6. **Analysis Process**:
   - `analyze_stock(ticker)` function is called
   - This calls `stock_analyst.run_sync(prompt)` with the ticker
   - The agent sends the prompt to OpenAI's GPT-4o
   - The LLM may call the `get_market_sentiment` tool if needed
   - The LLM returns a response that's validated against the `StockAnalysis` model
   - The validated response is returned to the main script

7. **Result Display**:
   - Script displays the formatted analysis to the user
   - Process repeats for next ticker input
