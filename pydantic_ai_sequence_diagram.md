# PydanticAI Sequence Diagrams

This document contains sequence diagrams for all examples in the PydanticAI reference project, illustrating the flow of function calls and data in each implementation.

## Table of Contents

1. [Basic Example](#basic-example)
2. [Advanced Example](#advanced-example)
3. [Clean Example](#clean-example)
4. [Orchestrator Example](#orchestrator-example)

## Basic Example

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

### Key Components

1. **Environment Setup**: Loads API keys from the `.env` file
2. **Model Definition**: Defines the `StockAnalysis` Pydantic model for structured outputs
3. **Agent Creation**: Creates a stock analyst agent with GPT-4o
4. **Tool Registration**: Adds the `get_market_sentiment` tool
5. **User Interaction**: Processes user input (stock ticker)
6. **Analysis Process**: Sends prompt to LLM and validates response
7. **Result Display**: Shows formatted analysis to the user

## Advanced Example

This diagram illustrates the sequence of function calls in the advanced PydanticAI example (`pydantic_ai_advanced.py`), which adds multiple agents and more complex tools.

```mermaid
sequenceDiagram
    participant User
    participant Main as Main Script
    participant load_dotenv
    participant Logfire
    participant StockAnalysis as StockAnalysis Model
    participant MarketSentiment as MarketSentiment Model
    participant stock_analyst as Stock Analyst Agent
    participant sentiment_analyst as Sentiment Analyst Agent
    participant analyze_stock
    participant get_stock_data
    participant get_market_sentiment
    participant get_simulated_news
    participant LLM as OpenAI GPT-4o

    User->>Main: Run script
    Main->>load_dotenv: load_dotenv()
    Main->>Logfire: Configure Logfire
    Note over Logfire: Sets up observability
    
    Main->>StockAnalysis: Define StockAnalysis model
    Main->>MarketSentiment: Define MarketSentiment model
    
    Main->>stock_analyst: Create stock_analyst agent
    Note over stock_analyst: Verbose mode enabled
    
    Main->>sentiment_analyst: Create sentiment_analyst agent
    Note over sentiment_analyst: Separate agent for<br/>sentiment analysis
    
    Main->>stock_analyst: Register @stock_analyst.tool (get_stock_data)
    Main->>stock_analyst: Register @stock_analyst.tool (get_market_sentiment)
    
    Main->>User: Prompt for ticker input
    User->>Main: Enter ticker (e.g., "AAPL")
    
    Main->>Logfire: Create session span
    Main->>Logfire: Create request span
    
    Main->>analyze_stock: analyze_stock(ticker)
    analyze_stock->>Logfire: Create analysis span
    
    analyze_stock->>stock_analyst: stock_analyst.run(prompt)
    stock_analyst->>LLM: Send prompt to OpenAI
    
    LLM-->>get_stock_data: Call tool function
    get_stock_data-->>LLM: Return stock data
    
    LLM-->>get_market_sentiment: Call tool function
    get_market_sentiment->>get_simulated_news: Get news headlines
    get_market_sentiment->>sentiment_analyst: Analyze sentiment
    sentiment_analyst->>LLM: Second LLM call
    LLM-->>sentiment_analyst: Return sentiment analysis
    sentiment_analyst-->>get_market_sentiment: Return MarketSentiment
    get_market_sentiment-->>LLM: Return sentiment data
    
    LLM-->>stock_analyst: Return structured response
    stock_analyst-->>analyze_stock: Return StockAnalysis object
    
    analyze_stock->>Logfire: Log analysis results
    analyze_stock-->>Main: Return analysis result
    
    Main->>User: Display detailed analysis
    Main->>User: Prompt for next ticker
```

### Key Components

1. **Multiple Agents**: Uses separate agents for stock analysis and sentiment analysis
2. **Multiple Models**: Defines both `StockAnalysis` and `MarketSentiment` models
3. **Nested LLM Calls**: The sentiment analysis tool makes its own LLM call
4. **Observability**: Integrates Logfire for tracing and logging
5. **Verbose Output**: Provides detailed output with timing information

## Clean Example

This diagram illustrates the sequence of function calls in the clean PydanticAI example (`pydantic_ai_clean.py`), which focuses on code clarity with Logfire integration.

```mermaid
sequenceDiagram
    participant User
    participant Main as Main Script
    participant load_dotenv
    participant Logfire
    participant StockAnalysis as StockAnalysis Model
    participant MarketSentiment as MarketSentiment Model
    participant stock_analyst as Stock Analyst Agent
    participant sentiment_analyst as Sentiment Analyst Agent
    participant analyze_stock_async
    participant analyze_stock
    participant get_stock_data
    participant get_market_sentiment
    participant LLM as OpenAI GPT-4o

    User->>Main: Run script
    Main->>load_dotenv: load_dotenv()
    Main->>Logfire: Configure Logfire
    
    Main->>StockAnalysis: Define StockAnalysis model
    Main->>MarketSentiment: Define MarketSentiment model
    
    Main->>stock_analyst: Create stock_analyst agent
    Main->>sentiment_analyst: Create sentiment_analyst agent
    
    Main->>stock_analyst: Register @stock_analyst.tool (get_stock_data)
    Main->>stock_analyst: Register @stock_analyst.tool (get_market_sentiment)
    
    Main->>Logfire: Create session span
    Main->>User: Prompt for ticker input
    User->>Main: Enter ticker (e.g., "AAPL")
    
    Main->>Logfire: Create request span
    Main->>analyze_stock: analyze_stock(ticker)
    analyze_stock->>analyze_stock_async: Call async version
    
    analyze_stock_async->>Logfire: Create analysis span
    analyze_stock_async->>stock_analyst: stock_analyst.run(prompt)
    
    stock_analyst->>LLM: Send prompt to OpenAI
    
    LLM-->>get_stock_data: Call tool function
    get_stock_data->>Logfire: Log tool call
    get_stock_data-->>LLM: Return stock data
    
    LLM-->>get_market_sentiment: Call tool function
    get_market_sentiment->>sentiment_analyst: Call sentiment_analyst
    sentiment_analyst->>LLM: Second LLM call
    LLM-->>sentiment_analyst: Return sentiment analysis
    sentiment_analyst-->>get_market_sentiment: Return MarketSentiment
    get_market_sentiment-->>LLM: Return sentiment data
    
    LLM-->>stock_analyst: Return structured response
    stock_analyst-->>analyze_stock_async: Return StockAnalysis object
    
    analyze_stock_async->>Logfire: Log analysis results
    analyze_stock_async-->>analyze_stock: Return analysis
    analyze_stock-->>Main: Return analysis result
    
    Main->>User: Display minimal analysis
    Main->>User: Prompt for next ticker
```

### Key Components

1. **Clean Code Design**: Focuses on readability and maintainability
2. **Comprehensive Logging**: Uses Logfire spans for detailed tracing
3. **Async/Sync Pattern**: Uses async functions internally with sync wrappers
4. **Error Handling**: Includes proper error handling and logging

## Orchestrator Example

This diagram illustrates the sequence of function calls in the orchestrator example (`pydantic_ai_orchestrator.py`), which demonstrates a multi-agent system with task routing.

```mermaid
sequenceDiagram
    participant User
    participant Main as Main Script
    participant Logfire
    participant Models as Data Models
    participant orchestrator as Orchestrator Agent
    participant stock_analyst as Stock Analyst Agent
    participant news_analyst as News Analyst Agent
    participant technical_analyst as Technical Analyst Agent
    participant portfolio_advisor as Portfolio Advisor Agent
    participant Tools as Various Tools
    participant process_user_request
    participant analyze_user_request
    participant process_task
    participant _execute_task
    participant LLM as OpenAI GPT-4o

    User->>Main: Run script
    Main->>Logfire: Configure Logfire
    Main->>Models: Define data models
    Note over Models: TaskType, Task, TaskResult,<br/>StockAnalysis, NewsAnalysis,<br/>TechnicalAnalysis, PortfolioRecommendation
    
    Main->>orchestrator: Create orchestrator agent
    Main->>stock_analyst: Create stock_analyst agent
    Main->>news_analyst: Create news_analyst agent
    Main->>technical_analyst: Create technical_analyst agent
    Main->>portfolio_advisor: Create portfolio_advisor agent
    
    Main->>Tools: Register various tools
    Note over Tools: get_stock_data, get_recent_news,<br/>calculate_technical_indicators,<br/>get_market_data, assess_risk_profile
    
    Main->>Logfire: Create session span
    Main->>User: Prompt for request
    User->>Main: Enter request (e.g., "Analyze AAPL stock")
    
    Main->>run_orchestrator: run_orchestrator(user_request)
    run_orchestrator->>process_user_request: process_user_request(user_request)
    
    process_user_request->>Logfire: Create request span
    process_user_request->>analyze_user_request: analyze_user_request(user_request)
    
    analyze_user_request->>orchestrator: orchestrator.run(user_request)
    orchestrator->>LLM: Send request to determine task type
    LLM-->>orchestrator: Return Task object
    orchestrator-->>analyze_user_request: Return Task
    
    analyze_user_request-->>process_user_request: Return Task
    process_user_request->>process_task: process_task(task)
    
    process_task->>_execute_task: _execute_task(task)
    
    alt Stock Analysis Task
        _execute_task->>stock_analyst: stock_analyst.run(prompt)
        stock_analyst->>LLM: Send prompt to OpenAI
        LLM-->>stock_analyst: Return StockAnalysis
    else News Analysis Task
        _execute_task->>news_analyst: news_analyst.run(prompt)
        news_analyst->>LLM: Send prompt to OpenAI
        LLM-->>news_analyst: Return NewsAnalysis
    else Technical Analysis Task
        _execute_task->>technical_analyst: technical_analyst.run(prompt)
        technical_analyst->>LLM: Send prompt to OpenAI
        LLM-->>technical_analyst: Return TechnicalAnalysis
    else Portfolio Recommendation Task
        _execute_task->>portfolio_advisor: portfolio_advisor.run(prompt)
        portfolio_advisor->>LLM: Send prompt to OpenAI
        LLM-->>portfolio_advisor: Return PortfolioRecommendation
    end
    
    _execute_task-->>process_task: Return TaskResult
    process_task-->>process_user_request: Return TaskResult
    process_user_request-->>run_orchestrator: Return result
    run_orchestrator-->>Main: Return result
    
    Main->>User: Display task result
    Main->>User: Prompt for next request
```

### Key Components

1. **Multi-Agent System**: Uses multiple specialized agents for different tasks
2. **Task Routing**: Orchestrator agent determines which agent should handle each request
3. **Complex Data Models**: Defines multiple structured data models for different outputs
4. **Specialized Tools**: Each agent has access to specific tools for its domain
5. **Dynamic Task Execution**: Routes tasks based on their type to the appropriate agent
