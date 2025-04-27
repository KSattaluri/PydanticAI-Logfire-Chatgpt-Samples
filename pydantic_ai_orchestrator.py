"""
PydanticAI Orchestrator Example

This example demonstrates a multi-agent system with an orchestrator that dispatches
tasks to specialized agents. Each agent has its own tools and capabilities.
"""
import os
import time
import json
import asyncio
from enum import Enum
from typing import List, Optional, Union, Dict, Any
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

# Load environment variables from .env file
load_dotenv()

# Import and configure Logfire for observability
try:
    import logfire
    logfire.configure(
        service_name="pydantic_ai_orchestrator",
        service_version="1.0.0",
        environment="development",
        send_to_logfire=False  # Set to True if you want to send to Logfire cloud
    )
    LOGFIRE_AVAILABLE = True
    print("Logfire enabled for local logging")
except ImportError:
    print("Logfire not installed. Install with: pip install logfire")
    LOGFIRE_AVAILABLE = False

# ----- DATA MODELS -----

class TaskType(str, Enum):
    """Types of tasks that can be handled by the system"""
    STOCK_ANALYSIS = "stock_analysis"
    NEWS_ANALYSIS = "news_analysis"
    TECHNICAL_ANALYSIS = "technical_analysis"
    PORTFOLIO_RECOMMENDATION = "portfolio_recommendation"
    UNKNOWN = "unknown"

class Task(BaseModel):
    """A task to be processed by the system"""
    task_id: str = Field(description="Unique identifier for the task")
    task_type: TaskType = Field(description="Type of task to be performed")
    parameters: Dict[str, Any] = Field(description="Parameters for the task")
    priority: int = Field(description="Priority of the task (1-10)", ge=1, le=10)

class TaskResult(BaseModel):
    """Result of a processed task"""
    task_id: str = Field(description="ID of the task that was processed")
    task_type: TaskType = Field(description="Type of task that was performed")
    success: bool = Field(description="Whether the task was successful")
    result: Dict[str, Any] = Field(description="Result data")
    processing_time_ms: int = Field(description="Time taken to process the task in milliseconds")
    error: Optional[str] = Field(description="Error message if the task failed", default=None)

class StockAnalysis(BaseModel):
    """Stock analysis result"""
    ticker: str = Field(description="Stock ticker symbol")
    current_price: float = Field(description="Current stock price in USD")
    recommendation: str = Field(description="Buy, Sell, or Hold recommendation")
    confidence: int = Field(description="Confidence level in the recommendation (1-10)", ge=1, le=10)
    rationale: str = Field(description="Brief explanation for the recommendation")

class NewsAnalysis(BaseModel):
    """News analysis result"""
    ticker: str = Field(description="Stock ticker symbol")
    sentiment: str = Field(description="Overall sentiment (Positive, Neutral, Negative)")
    key_points: List[str] = Field(description="Key points from the news")
    impact_assessment: str = Field(description="Assessment of potential market impact")

class TechnicalAnalysis(BaseModel):
    """Technical analysis result"""
    ticker: str = Field(description="Stock ticker symbol")
    trend: str = Field(description="Current price trend (Uptrend, Downtrend, Sideways)")
    support_level: float = Field(description="Nearest support level")
    resistance_level: float = Field(description="Nearest resistance level")
    indicators: Dict[str, Any] = Field(description="Technical indicators and their values")
    outlook: str = Field(description="Technical outlook based on indicators")

class PortfolioRecommendation(BaseModel):
    """Portfolio recommendation result"""
    risk_profile: str = Field(description="Risk profile (Conservative, Moderate, Aggressive)")
    recommended_stocks: List[Dict[str, Any]] = Field(description="List of recommended stocks with allocation percentages")
    rationale: str = Field(description="Rationale for the recommendations")
    expected_return: float = Field(description="Expected annual return percentage")
    volatility: float = Field(description="Expected portfolio volatility")

# ----- AGENTS -----

# Orchestrator Agent - Determines which specialized agent should handle a task
orchestrator = Agent(
    'openai:gpt-4o',
    output_type=Task,
    system_prompt=(
        "You are an orchestrator that analyzes incoming financial requests and routes them to the appropriate "
        "specialized agent. Your job is to determine the type of task, extract relevant parameters, "
        "and assign a priority level. Be precise in your categorization."
    ),
    api_key=os.getenv('OPENAI_API_KEY')
)

# Stock Analyst Agent - Analyzes stocks and provides recommendations
stock_analyst = Agent(
    'openai:gpt-4o',
    output_type=StockAnalysis,
    system_prompt=(
        "You are a stock market analyst. Provide analysis for stocks based on their ticker symbol, "
        "current price, and market data. Your analysis should include a clear buy/sell/hold recommendation "
        "with confidence level and rationale."
    ),
    api_key=os.getenv('OPENAI_API_KEY')
)

# News Analyst Agent - Analyzes news sentiment and impact
news_analyst = Agent(
    'openai:gpt-4o',
    output_type=NewsAnalysis,
    system_prompt=(
        "You are a financial news analyst. Analyze news articles and headlines for specific stocks "
        "to determine sentiment, extract key points, and assess potential market impact."
    ),
    api_key=os.getenv('OPENAI_API_KEY')
)

# Technical Analyst Agent - Performs technical analysis on stock charts
technical_analyst = Agent(
    'openai:gpt-4o',
    output_type=TechnicalAnalysis,
    system_prompt=(
        "You are a technical analyst. Analyze stock price charts, patterns, and technical indicators "
        "to identify trends, support/resistance levels, and provide technical outlook."
    ),
    api_key=os.getenv('OPENAI_API_KEY')
)

# Portfolio Advisor Agent - Provides portfolio recommendations
portfolio_advisor = Agent(
    'openai:gpt-4o',
    output_type=PortfolioRecommendation,
    system_prompt=(
        "You are a portfolio advisor. Based on risk profile, market conditions, and analysis of individual stocks, "
        "provide portfolio allocation recommendations with expected returns and volatility estimates."
    ),
    api_key=os.getenv('OPENAI_API_KEY')
)

# ----- TOOLS -----

# Stock Data Tool - Get stock price and fundamental data
@stock_analyst.tool
@technical_analyst.tool
@portfolio_advisor.tool
async def get_stock_data(ctx: RunContext, ticker: str) -> dict:
    """Get current stock data including price, volume, and fundamental ratios."""
    # In a real implementation, this would call a financial API
    stock_data = {
        "AAPL": {"price": 198.45, "volume": 62500000, "pe_ratio": 32.8, "market_cap": "3.1T", "dividend_yield": 0.5},
        "MSFT": {"price": 425.22, "volume": 28300000, "pe_ratio": 37.2, "market_cap": "3.2T", "dividend_yield": 0.7},
        "GOOGL": {"price": 175.98, "volume": 31200000, "pe_ratio": 25.4, "market_cap": "2.2T", "dividend_yield": 0},
        "AMZN": {"price": 182.75, "volume": 45700000, "pe_ratio": 62.1, "market_cap": "1.9T", "dividend_yield": 0},
        "META": {"price": 491.83, "volume": 19800000, "pe_ratio": 28.7, "market_cap": "1.3T", "dividend_yield": 0.4},
    }
    
    if LOGFIRE_AVAILABLE:
        logfire.info(f"Retrieved stock data", ticker=ticker, data=stock_data.get(ticker.upper(), {}))
    
    return stock_data.get(ticker.upper(), {"price": 0.0, "volume": 0, "pe_ratio": 0.0})

# News Retrieval Tool - Get recent news for a stock
@news_analyst.tool
@stock_analyst.tool
async def get_recent_news(ctx: RunContext, ticker: str) -> List[str]:
    """Get recent news headlines for a specific stock."""
    # In a real implementation, this would call a news API
    news = {
        "AAPL": [
            "Apple's iPhone 16 sales exceed expectations in first month",
            "Apple announces expansion of AI features across product line",
            "Supply chain issues resolved for upcoming Apple Vision Pro 2"
        ],
        "MSFT": [
            "Microsoft Cloud revenue grows 28% year-over-year",
            "Microsoft faces new antitrust scrutiny in European markets",
            "Microsoft's AI integration boosts productivity suite adoption"
        ],
        "GOOGL": [
            "Google ad revenue rebounds after previous quarter's slowdown",
            "Regulatory challenges continue for Google's search dominance",
            "Google's AI models show promising results against competitors"
        ],
        "AMZN": [
            "Amazon Prime Day sets new sales record",
            "Amazon Web Services expands data center presence in Asia",
            "Amazon's logistics investments paying off with faster delivery times"
        ],
        "META": [
            "Meta's Reality Labs continues to lose billions despite VR growth",
            "Instagram user engagement reaches all-time high",
            "Meta's AI research leads to breakthrough in content moderation"
        ]
    }
    
    default_news = [
        f"No specific news found for {ticker}",
        "Market shows mixed signals across sectors",
        "Analysts remain cautious about overall economic outlook"
    ]
    
    result = news.get(ticker.upper(), default_news)
    
    if LOGFIRE_AVAILABLE:
        logfire.info(f"Retrieved news headlines", ticker=ticker, headlines=result)
    
    return result

# Technical Indicators Tool - Calculate technical indicators
@technical_analyst.tool
async def calculate_technical_indicators(ctx: RunContext, ticker: str) -> dict:
    """Calculate technical indicators for a stock."""
    # In a real implementation, this would use historical price data and calculate actual indicators
    indicators = {
        "AAPL": {
            "RSI": 62.4, "MACD": 1.2, "SMA_50": 190.25, "SMA_200": 182.30, 
            "support": 192.50, "resistance": 205.75, "trend": "Uptrend"
        },
        "MSFT": {
            "RSI": 58.7, "MACD": 0.8, "SMA_50": 415.10, "SMA_200": 390.45, 
            "support": 410.25, "resistance": 435.50, "trend": "Sideways"
        },
        "GOOGL": {
            "RSI": 45.2, "MACD": -0.5, "SMA_50": 180.30, "SMA_200": 172.15, 
            "support": 170.25, "resistance": 185.50, "trend": "Downtrend"
        },
        "AMZN": {
            "RSI": 71.3, "MACD": 2.1, "SMA_50": 175.20, "SMA_200": 165.40, 
            "support": 178.50, "resistance": 190.25, "trend": "Uptrend"
        },
        "META": {
            "RSI": 65.8, "MACD": 1.5, "SMA_50": 480.75, "SMA_200": 450.20, 
            "support": 475.25, "resistance": 505.50, "trend": "Uptrend"
        }
    }
    
    default_indicators = {
        "RSI": 50.0, "MACD": 0.0, "SMA_50": 0.0, "SMA_200": 0.0,
        "support": 0.0, "resistance": 0.0, "trend": "Unknown"
    }
    
    result = indicators.get(ticker.upper(), default_indicators)
    
    if LOGFIRE_AVAILABLE:
        logfire.info(f"Calculated technical indicators", ticker=ticker, indicators=result)
    
    return result

# Market Data Tool - Get overall market data
@portfolio_advisor.tool
@stock_analyst.tool
async def get_market_data(ctx: RunContext) -> dict:
    """Get overall market data including index values and sector performance."""
    # In a real implementation, this would call a market data API
    market_data = {
        "indices": {
            "S&P500": 5320.45, "NASDAQ": 16750.22, "DOW": 38950.75
        },
        "sector_performance": {
            "Technology": 2.3, "Healthcare": -0.5, "Financials": 1.2,
            "Consumer Discretionary": 0.8, "Energy": -1.5
        },
        "market_sentiment": "Moderately Bullish",
        "volatility_index": 18.5
    }
    
    if LOGFIRE_AVAILABLE:
        logfire.info(f"Retrieved market data", data=market_data)
    
    return market_data

# Risk Assessment Tool - Assess risk profile
@portfolio_advisor.tool
async def assess_risk_profile(ctx: RunContext, answers: List[int]) -> str:
    """Assess risk profile based on questionnaire answers."""
    # Simple risk assessment based on average score
    avg_score = sum(answers) / len(answers)
    
    if avg_score < 3:
        profile = "Conservative"
    elif avg_score < 7:
        profile = "Moderate"
    else:
        profile = "Aggressive"
    
    if LOGFIRE_AVAILABLE:
        logfire.info(f"Assessed risk profile", answers=answers, profile=profile)
    
    return profile

# ----- ORCHESTRATOR FUNCTIONS -----

async def process_task(task: Task) -> TaskResult:
    """Process a task using the appropriate agent based on task type."""
    start_time = time.time()
    
    if LOGFIRE_AVAILABLE:
        with logfire.span(f'process_task_{task.task_id}', attributes={'task_type': task.task_type}):
            logfire.info(f"Processing task", task_id=task.task_id, task_type=task.task_type)
            result = await _execute_task(task)
            processing_time = round((time.time() - start_time) * 1000)
            logfire.info(
                f"Task processed", 
                task_id=task.task_id, 
                success=result.success,
                processing_time_ms=processing_time
            )
    else:
        result = await _execute_task(task)
        processing_time = round((time.time() - start_time) * 1000)
    
    return result

async def _execute_task(task: Task) -> TaskResult:
    """Execute a task using the appropriate agent."""
    try:
        if task.task_type == TaskType.STOCK_ANALYSIS:
            ticker = task.parameters.get("ticker", "")
            result = await stock_analyst.run(f"Analyze the stock {ticker}")
            return TaskResult(
                task_id=task.task_id,
                task_type=task.task_type,
                success=True,
                result=result.output.dict(),
                processing_time_ms=0  # Will be updated by caller
            )
            
        elif task.task_type == TaskType.NEWS_ANALYSIS:
            ticker = task.parameters.get("ticker", "")
            result = await news_analyst.run(f"Analyze recent news for {ticker}")
            return TaskResult(
                task_id=task.task_id,
                task_type=task.task_type,
                success=True,
                result=result.output.dict(),
                processing_time_ms=0  # Will be updated by caller
            )
            
        elif task.task_type == TaskType.TECHNICAL_ANALYSIS:
            ticker = task.parameters.get("ticker", "")
            result = await technical_analyst.run(f"Perform technical analysis for {ticker}")
            return TaskResult(
                task_id=task.task_id,
                task_type=task.task_type,
                success=True,
                result=result.output.dict(),
                processing_time_ms=0  # Will be updated by caller
            )
            
        elif task.task_type == TaskType.PORTFOLIO_RECOMMENDATION:
            risk_profile = task.parameters.get("risk_profile", "Moderate")
            tickers = task.parameters.get("tickers", [])
            tickers_str = ", ".join(tickers) if tickers else "top stocks"
            result = await portfolio_advisor.run(
                f"Create a portfolio recommendation for a {risk_profile} investor using {tickers_str}"
            )
            return TaskResult(
                task_id=task.task_id,
                task_type=task.task_type,
                success=True,
                result=result.output.dict(),
                processing_time_ms=0  # Will be updated by caller
            )
            
        else:
            return TaskResult(
                task_id=task.task_id,
                task_type=task.task_type,
                success=False,
                result={},
                processing_time_ms=0,  # Will be updated by caller
                error=f"Unknown task type: {task.task_type}"
            )
            
    except Exception as e:
        if LOGFIRE_AVAILABLE:
            logfire.error(f"Error executing task", task_id=task.task_id, error=str(e))
        return TaskResult(
            task_id=task.task_id,
            task_type=task.task_type,
            success=False,
            result={},
            processing_time_ms=0,  # Will be updated by caller
            error=str(e)
        )

async def analyze_user_request(user_request: str) -> Task:
    """Use the orchestrator agent to analyze a user request and create a task."""
    # Handle simple ticker inputs directly
    if user_request.strip().upper() in ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]:
        # Create a default stock analysis task for simple ticker inputs
        ticker = user_request.strip().upper()
        if LOGFIRE_AVAILABLE:
            logfire.info(f"Creating default stock analysis task for ticker", ticker=ticker)
        
        return Task(
            task_id="default_task",  # Will be replaced later
            task_type=TaskType.STOCK_ANALYSIS,
            parameters={"ticker": ticker},
            priority=5
        )
    
    # For more complex requests, use the orchestrator agent
    try:
        if LOGFIRE_AVAILABLE:
            with logfire.span('analyze_user_request', attributes={'request': user_request}):
                logfire.info(f"Analyzing user request", request=user_request)
                result = await orchestrator.run(user_request)
                task = result.output
                logfire.info(
                    f"Request analyzed", 
                    task_id=task.task_id, 
                    task_type=task.task_type,
                    parameters=task.parameters
                )
        else:
            result = await orchestrator.run(user_request)
            task = result.output
        
        return task
    except Exception as e:
        # If orchestrator fails, try to extract a ticker and create a default task
        if LOGFIRE_AVAILABLE:
            logfire.warning(f"Orchestrator failed, attempting to extract ticker", error=str(e))
        
        # Very simple ticker extraction - in a real system this would be more sophisticated
        words = user_request.upper().split()
        for word in words:
            if word in ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]:
                if LOGFIRE_AVAILABLE:
                    logfire.info(f"Extracted ticker from failed request", ticker=word)
                
                return Task(
                    task_id="default_task",  # Will be replaced later
                    task_type=TaskType.STOCK_ANALYSIS,
                    parameters={"ticker": word},
                    priority=5
                )
        
        # If we can't extract a ticker, raise the original error
        raise

# ----- MAIN EXECUTION -----

async def process_user_request(user_request: str) -> Dict[str, Any]:
    """Process a user request from start to finish."""
    # Generate a unique task ID
    task_id = f"task_{int(time.time())}"
    
    if LOGFIRE_AVAILABLE:
        with logfire.span(f'user_request_{task_id}', attributes={'request': user_request}):
            # Step 1: Analyze the request and create a task
            task = await analyze_user_request(user_request)
            
            # Update the task ID to our generated one
            task.task_id = task_id
            
            # Step 2: Process the task with the appropriate agent
            result = await process_task(task)
            
            return {
                "task": task.dict(),
                "result": result.dict()
            }
    else:
        # Step 1: Analyze the request and create a task
        task = await analyze_user_request(user_request)
        
        # Update the task ID to our generated one
        task.task_id = task_id
        
        # Step 2: Process the task with the appropriate agent
        result = await process_task(task)
        
        return {
            "task": task.dict(),
            "result": result.dict()
        }

def run_orchestrator(user_request: str) -> Dict[str, Any]:
    """Synchronous wrapper for process_user_request."""
    return asyncio.run(process_user_request(user_request))

# ----- INTERACTIVE MODE -----

if __name__ == "__main__":
    print("\nPydanticAI Multi-Agent Orchestrator System")
    print("------------------------------------------")
    print("Example requests:")
    print("- Simply enter a ticker like 'AAPL' for basic stock analysis")
    print("- 'Analyze AAPL stock and give me a recommendation'")
    print("- 'What's the sentiment around MSFT based on recent news?'")
    print("- 'Perform technical analysis on GOOGL'")
    print("- 'Create a moderate risk portfolio with AAPL, MSFT, and AMZN'")
    
    # Create a session-level span for Logfire if available
    session_span = None
    if LOGFIRE_AVAILABLE:
        session_span = logfire.span("orchestrator_session")
        session_span.__enter__()
    
    try:
        while True:
            user_input = input("\nEnter your request (or 'exit' to quit): ")
            if user_input.lower() == 'exit':
                break
                
            try:
                result = run_orchestrator(user_input)
                
                print("\nRequest processed:")
                print(f"Task Type: {result['task']['task_type']}")
                print(f"Priority: {result['task']['priority']}")
                
                if result['result']['success']:
                    print("\nResult:")
                    # Pretty print the result with indentation
                    print(json.dumps(result['result']['result'], indent=2))
                    print(f"\nProcessing time: {result['result']['processing_time_ms']}ms")
                else:
                    print(f"\nError: {result['result']['error']}")
                    
            except Exception as e:
                print(f"Error processing request: {str(e)}")
    finally:
        if LOGFIRE_AVAILABLE and session_span:
            session_span.__exit__(None, None, None)
