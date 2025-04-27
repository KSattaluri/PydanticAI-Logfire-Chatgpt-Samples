"""
Clean PydanticAI example with Logfire integration.
This file demonstrates the core functionality of PydanticAI with minimal verbosity.
"""
import os
import time
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

# Load environment variables from .env file
load_dotenv()

# Import and configure Logfire for observability
try:
    import logfire
    # Configure Logfire with authentication
    logfire.configure(
        service_name="pydantic_ai_stock_analyst",
        service_version="1.0.0",
        environment="development"
    )
    LOGFIRE_AVAILABLE = True
    print("Logfire enabled")
except ImportError:
    print("Logfire not installed. Install with: pip install logfire")
    LOGFIRE_AVAILABLE = False
except Exception as e:
    print(f"Error configuring Logfire: {str(e)}")
    LOGFIRE_AVAILABLE = False

# Define a model for stock analysis
class StockAnalysis(BaseModel):
    """Structured data model for stock analysis results"""
    ticker: str = Field(description="Stock ticker symbol")
    current_price: float = Field(description="Current stock price in USD")
    recommendation: str = Field(description="Buy, Sell, or Hold recommendation")
    confidence: int = Field(description="Confidence level in the recommendation (1-10)", ge=1, le=10)
    rationale: str = Field(description="Brief explanation for the recommendation")

# Define a model for market sentiment
class MarketSentiment(BaseModel):
    """Structured data model for market sentiment analysis"""
    sentiment: str = Field(description="Overall market sentiment (Positive, Neutral, Negative)")
    reasoning: str = Field(description="Reasoning behind the sentiment analysis")

# Create an agent for stock analysis
stock_analyst = Agent(
    'openai:gpt-4o',  # Using OpenAI's GPT-4o model
    output_type=StockAnalysis,  # The agent will return a StockAnalysis object
    system_prompt=(
        "You are a stock market analyst. Provide analysis for stocks based on their ticker symbol. "
        "Your analysis should include a clear buy/sell/hold recommendation with confidence level."
    ),
    api_key=os.getenv('OPENAI_API_KEY')  # API key from environment variables
)

# Create a separate agent for sentiment analysis
sentiment_analyst = Agent(
    'openai:gpt-4o',  # Using OpenAI's GPT-4o model
    output_type=MarketSentiment,  # The agent will return a MarketSentiment object
    system_prompt=(
        "You are a financial news analyst specializing in market sentiment analysis. "
        "Analyze news headlines to determine overall market sentiment for specific stocks."
    ),
    api_key=os.getenv('OPENAI_API_KEY')  # API key from environment variables
)

# Add a tool to get stock data (simulated API call)
@stock_analyst.tool
async def get_stock_data(ctx: RunContext, ticker: str) -> dict:
    """Get current stock data for a specific ticker."""
    # In a real implementation, this would call a financial API
    # For this example, we'll simulate an API response
    stock_data = {
        "AAPL": {"price": 198.45, "volume": 62500000, "pe_ratio": 32.8},
        "MSFT": {"price": 425.22, "volume": 28300000, "pe_ratio": 37.2},
        "GOOGL": {"price": 175.98, "volume": 31200000, "pe_ratio": 25.4},
        "AMZN": {"price": 182.75, "volume": 45700000, "pe_ratio": 62.1},
        "META": {"price": 491.83, "volume": 19800000, "pe_ratio": 28.7},
    }
    
    # Log the tool call with Logfire if available
    if LOGFIRE_AVAILABLE:
        logfire.info(f"Retrieved stock data for {ticker}", ticker=ticker, data=stock_data.get(ticker.upper(), {}))
    
    return stock_data.get(ticker.upper(), {"price": 0.0, "volume": 0, "pe_ratio": 0.0})

# Add a tool that uses another LLM to analyze market sentiment
@stock_analyst.tool
async def get_market_sentiment(ctx: RunContext, ticker: str) -> str:
    """Get the overall market sentiment for a specific stock ticker using LLM analysis."""
    # First, we'll simulate getting recent news about the stock
    recent_news = get_simulated_news(ticker)
    
    # Use Logfire to track the sentiment analysis if available
    start_time = time.time()
    
    if LOGFIRE_AVAILABLE:
        with logfire.span(f'sentiment_analysis_{ticker}', attributes={'ticker': ticker}):
            # Log the news headlines being analyzed
            logfire.info(f'Analyzing sentiment for {ticker}', ticker=ticker, headlines=recent_news)
            
            # Call the sentiment analyst agent to analyze the sentiment
            prompt = f"Analyze the market sentiment for {ticker} based on these recent headlines:\n\n{recent_news}"
            result = await sentiment_analyst.run(prompt)
            
            sentiment_analysis = result.output
            # Log the sentiment analysis results
            logfire.info(
                f'Sentiment analysis completed',
                ticker=ticker,
                sentiment=sentiment_analysis.sentiment,
                duration_ms=round((time.time() - start_time) * 1000)
            )
    else:
        # Call the sentiment analyst agent without Logfire
        prompt = f"Analyze the market sentiment for {ticker} based on these recent headlines:\n\n{recent_news}"
        result = await sentiment_analyst.run(prompt)
        sentiment_analysis = result.output
    
    return f"{sentiment_analysis.sentiment}: {sentiment_analysis.reasoning}"

def get_simulated_news(ticker):
    """Simulate getting recent news headlines for a stock."""
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
    
    # Default news for unknown tickers
    default_news = [
        f"No specific news found for {ticker}",
        "Market shows mixed signals across sectors",
        "Analysts remain cautious about overall economic outlook"
    ]
    
    return "\n".join(news.get(ticker.upper(), default_news))

# Function to analyze a stock
async def analyze_stock_async(ticker):
    """Analyze a stock using the PydanticAI agent"""
    start_time = time.time()
    
    # Use Logfire span if available
    if LOGFIRE_AVAILABLE:
        with logfire.span(f'analyze_stock_{ticker}', attributes={'ticker': ticker}):
            result = await stock_analyst.run(f"Analyze the stock {ticker}")
            analysis = result.output
            # Log the analysis results
            logfire.info(
                f'Stock analysis completed',
                ticker=ticker,
                recommendation=analysis.recommendation,
                confidence=analysis.confidence,
                price=analysis.current_price,
                duration_ms=round((time.time() - start_time) * 1000)
            )
    else:
        result = await stock_analyst.run(f"Analyze the stock {ticker}")
        analysis = result.output
    
    return result.output

# Synchronous wrapper for the async function
def analyze_stock(ticker):
    """Synchronous wrapper for analyze_stock_async"""
    import asyncio
    return asyncio.run(analyze_stock_async(ticker))

if __name__ == "__main__":
    print("PydanticAI Stock Analyst with Logfire")
    
    # Create a session-level span variable for Logfire if available
    session_span = None
    if LOGFIRE_AVAILABLE:
        # Create a session span that will contain all operations
        session_span = logfire.span("stock_analysis_session")
        session_span.__enter__()
    
    while True:
        ticker = input("\nEnter a stock ticker (or 'exit' to quit): ")
        if ticker.lower() == 'exit':
            if LOGFIRE_AVAILABLE and session_span:
                # Log session end and exit the span context
                logfire.info("Stock analysis session ended")
                session_span.__exit__(None, None, None)
            break
            
        try:
            if LOGFIRE_AVAILABLE:
                with logfire.span(f"user_request_{ticker}", attributes={"ticker": ticker}):
                    analysis = analyze_stock(ticker)
                    logfire.info(f"Successfully analyzed stock", ticker=ticker)
            else:
                analysis = analyze_stock(ticker)
                
            # Display minimal analysis results
            print(f"\nAnalysis for {analysis.ticker}:")
            print(f"Current Price: ${analysis.current_price}")
            print(f"Recommendation: {analysis.recommendation}")
            print(f"Confidence: {analysis.confidence}/10")
            print(f"Rationale: {analysis.rationale}")
        except Exception as e:
            error_msg = str(e)
            print(f"Error analyzing stock: {error_msg}")
            if LOGFIRE_AVAILABLE:
                logfire.error(f"Error analyzing stock", ticker=ticker, error=error_msg)
