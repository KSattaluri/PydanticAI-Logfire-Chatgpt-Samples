import os
import json
import time
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

# Import and configure Logfire for observability
try:
    import logfire
    # Configure Logfire with authentication (using credentials from logfire auth)
    logfire.configure(
        service_name="pydantic_ai_stock_analyst",
        service_version="1.0.0",
        environment="development"
    )
    LOGFIRE_AVAILABLE = True
    print("Logfire configured with authentication - data will be sent to Logfire cloud")
except ImportError:
    print("Logfire not installed. Install with: pip install logfire")
    print("Continuing without Logfire observability...")
    LOGFIRE_AVAILABLE = False
except Exception as e:
    print(f"Error configuring Logfire: {str(e)}")
    print("Continuing without Logfire observability...")
    LOGFIRE_AVAILABLE = False
import httpx

# Load environment variables from .env file
load_dotenv()

# Define a simple data model for stock analysis
class StockAnalysis(BaseModel):
    ticker: str = Field(description="Stock ticker symbol")
    current_price: float = Field(description="Current stock price in USD")
    recommendation: str = Field(description="Buy, Sell, or Hold recommendation")
    confidence: int = Field(description="Confidence level in the recommendation (1-10)", ge=1, le=10)
    rationale: str = Field(description="Brief explanation for the recommendation")

# Define a model for market sentiment
class MarketSentiment(BaseModel):
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
    api_key=os.getenv('OPENAI_API_KEY'),  # API key from environment variables
    verbose=True  # Enable verbose logging to see prompts and responses
)

# Create a separate agent for sentiment analysis
sentiment_analyst = Agent(
    'openai:gpt-4o',  # Using OpenAI's GPT-4o model
    output_type=MarketSentiment,  # The agent will return a MarketSentiment object
    system_prompt=(
        "You are a financial news analyst specializing in market sentiment analysis. "
        "Analyze news headlines to determine overall market sentiment for specific stocks."
    ),
    api_key=os.getenv('OPENAI_API_KEY'),  # API key from environment variables
    verbose=True  # Enable verbose logging to see prompts and responses
)

# Add a tool to get real stock data (simulated API call)
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
    
    return stock_data.get(ticker.upper(), {"price": 0.0, "volume": 0, "pe_ratio": 0.0})

# Add a tool that uses another LLM to analyze market sentiment
@stock_analyst.tool
async def get_market_sentiment(ctx: RunContext, ticker: str) -> str:
    """Get the overall market sentiment for a specific stock ticker using LLM analysis."""
    # First, we'll simulate getting recent news about the stock
    # In a real implementation, this would call a news API
    recent_news = get_simulated_news(ticker)
    
    print(f"\n{'*'*80}\n📰 SENTIMENT ANALYSIS FOR: {ticker}\n{'*'*80}")
    print(f"Headlines being analyzed:\n{recent_news}\n{'*'*80}")
    
    start_time = time.time()
    
    # Use Logfire for observability if available
    if LOGFIRE_AVAILABLE:
        with logfire.span(f'sentiment_analysis_{ticker}', attributes={'ticker': ticker}):
            # Log the news headlines being analyzed
            logfire.info(f'Analyzing sentiment for {ticker} based on headlines', 
                        ticker=ticker, 
                        headlines=recent_news)
            
            # Now we'll use our sentiment analyst agent to analyze the sentiment
            # This is a real LLM call within the tool function
            prompt = f"Analyze the market sentiment for {ticker} based on these recent headlines:\n\n{recent_news}"
            result = await sentiment_analyst.run(prompt)
            
            sentiment_analysis = result.output
            # Log the sentiment analysis results
            logfire.info(
                f'Sentiment analysis completed for {ticker}',
                ticker=ticker,
                sentiment=sentiment_analysis.sentiment,
                duration_ms=round((time.time() - start_time) * 1000)
            )
    else:
        # Now we'll use our sentiment analyst agent to analyze the sentiment
        # This is a real LLM call within the tool function
        prompt = f"Analyze the market sentiment for {ticker} based on these recent headlines:\n\n{recent_news}"
        result = await sentiment_analyst.run(prompt)
    
    # Print detailed information about the result object
    print("\n🔍 RESULT OBJECT STRUCTURE:")
    print("Attributes:", ', '.join(dir(result)))
    print("\nVariables:")
    for key, value in vars(result).items():
        print(f"  {key}: {value}")
    
    sentiment_analysis = result.output
    print(f"\n📊 SENTIMENT ANALYSIS RESULT:")
    print(json.dumps(sentiment_analysis.dict(), indent=4))
    print(f"Analysis took {round((time.time() - start_time) * 1000)}ms")
    print(f"{'*'*80}")
    
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
            "Meta's ad targeting improvements offset by privacy regulations"
        ]
    }
    
    default_news = [
        f"No specific recent news for {ticker}",
        "Market shows mixed signals across tech sector",
        "Analysts divided on growth prospects for smaller tech companies"
    ]
    
    return "\n".join(news.get(ticker.upper(), default_news))

# Function to analyze a stock
async def analyze_stock_async(ticker):
    """Analyze a stock using the PydanticAI agent"""
    print(f"\n{'='*80}\n📊 ANALYZING STOCK: {ticker}\n{'='*80}")
    
    start_time = time.time()
    
    # Use Logfire span if available
    if LOGFIRE_AVAILABLE:
        with logfire.span(f'analyze_stock_{ticker}', attributes={'ticker': ticker}):
            result = await stock_analyst.run(f"Analyze the stock {ticker}")
            analysis = result.output
            # Log the analysis results
            logfire.info(
                f'Stock analysis completed for {ticker}',
                ticker=ticker,
                recommendation=analysis.recommendation,
                confidence=analysis.confidence,
                price=analysis.current_price,
                duration_ms=round((time.time() - start_time) * 1000)
            )
    else:
        result = await stock_analyst.run(f"Analyze the stock {ticker}")
    
    # Prettify the output
    print(f"\n{'='*80}\n🔍 FINAL ANALYSIS RESULT:\n")
    analysis = result.output
    print(json.dumps(analysis.dict(), indent=4))
    print(f"Analysis took {round((time.time() - start_time) * 1000)}ms")
    print(f"{'='*80}")
    
    return result.output

# Synchronous wrapper for the async function
def analyze_stock(ticker):
    """Synchronous wrapper for analyze_stock_async"""
    import asyncio
    return asyncio.run(analyze_stock_async(ticker))

if __name__ == "__main__":
    print("\n" + "*"*80)
    print("*" + " "*30 + "PYDANTICAI STOCK ANALYST" + " "*30 + "*")
    print("*" + " "*22 + "WITH LLM-POWERED SENTIMENT ANALYSIS" + " "*22 + "*")
    if LOGFIRE_AVAILABLE:
        print("*" + " "*25 + "LOGFIRE OBSERVABILITY ENABLED" + " "*25 + "*")
    print("*"*80)
    
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
                    logfire.info(f"Successfully analyzed {ticker}", ticker=ticker)
            else:
                analysis = analyze_stock(ticker)
                
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