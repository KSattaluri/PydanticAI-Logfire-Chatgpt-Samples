import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

# Load environment variables from .env file
load_dotenv()

# Define a simple data model for stock analysis
class StockAnalysis(BaseModel):
    ticker: str = Field(description="Stock ticker symbol")
    current_price: float = Field(description="Current stock price in USD")
    recommendation: str = Field(description="Buy, Sell, or Hold recommendation")
    confidence: int = Field(description="Confidence level in the recommendation (1-10)", ge=1, le=10)
    rationale: str = Field(description="Brief explanation for the recommendation")

# Create an agent that performs stock analysis
stock_analyst = Agent(
    'openai:gpt-4o',  # Using OpenAI's GPT-4o model
    output_type=StockAnalysis,  # The agent will return a StockAnalysis object
    system_prompt=(
        "You are a stock market analyst. Provide analysis for stocks based on their ticker symbol. "
        "Your analysis should include a clear buy/sell/hold recommendation with confidence level."
    ),
    api_key=os.getenv('OPENAI_API_KEY')  # API key from environment variables
)

# Add a tool to get additional market data (simulated)
@stock_analyst.tool
async def get_market_sentiment(ctx: RunContext, ticker: str) -> str:
    """Get the overall market sentiment for a specific stock ticker."""
    # In a real implementation, this would call an API or database
    # For this example, we'll use a simple dictionary
    sentiments = {
        "AAPL": "Positive - Strong product lineup and services growth",
        "MSFT": "Positive - Cloud business continues to expand",
        "GOOGL": "Neutral - Facing regulatory challenges despite strong ad revenue",
        "AMZN": "Positive - E-commerce and AWS showing strong performance",
        "META": "Neutral - Concerns about ad targeting limitations balanced by metaverse investments",
    }
    
    return sentiments.get(ticker.upper(), "Unknown - No sentiment data available")

# Simple REPL for stock analysis
def analyze_stock(ticker):
    """Analyze a stock using the PydanticAI agent"""
    result = stock_analyst.run_sync(f"Analyze the stock {ticker}")
    return result.output

if __name__ == "__main__":
    print("PydanticAI Stock Analyst")
    print("------------------------")
    
    while True:
        ticker = input("\nEnter a stock ticker (or 'exit' to quit): ")
        if ticker.lower() == 'exit':
            break
            
        try:
            analysis = analyze_stock(ticker)
            print(f"\nAnalysis for {analysis.ticker}:")
            print(f"Current Price: ${analysis.current_price}")
            print(f"Recommendation: {analysis.recommendation}")
            print(f"Confidence: {analysis.confidence}/10")
            print(f"Rationale: {analysis.rationale}")
        except Exception as e:
            print(f"Error analyzing stock: {str(e)}")
