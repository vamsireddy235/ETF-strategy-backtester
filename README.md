# ETF-strategy-backtester
"Interactive ETF backtesting platform allowing you to analyze and compare 'Buy the Dip', SIP, Momentum, and Hybrid strategies with custom parameters on historical data. Visualize results, download summaries, and optimize your investment approach easily."






This project provides an interactive web app and Python script to backtest, analyze, and visualize diverse ETF investment strategies using historical data. Easily compare how "Buy the Dip", "SIP", "Momentum", and "Hybrid" strategies perform with your custom parameters.

## Features

- **Global ETF lookup:** Search any ETF by symbol and auto-suggest using Yahoo Finance APIs.
- **Simulate multiple strategies:** Buy the Dip, SIP (Systematic Investment Plan), Momentum, and Hybrid backtests.
- **Custom timeframes:** Test for custom year-ranges from 2000 onward, or use all available data.
- **User-friendly UI:** Run interactive simulations in Streamlit, see visual breakdowns and results.
- **Yearly analysis:** Downloadable summary CSVs for investments, profits/losses, and units bought per year.
- **Comprehensive charts:** Auto-plots buy signals, prices, and annual investments.
- **Supports INR formatting & analytics:** Ready for Indian investors but modifiable for other markets.
- **Compare strategies:** Directly compare strategies side-by-side including ROI and portfolio growth over time.


## Getting Started

1. **Clone the repository:**
    
    git clone https://github.com/vamsireddy235/etf-strategy-backtester.git
    cd etf-strategy-backtester
    

2. **Install dependencies:**
    
    pip install -r requirements.txt
    

3. **Run the app:**
    
    streamlit run ETF2.py
    
   Or use the command-line version for custom batch backtesting:
  
    python etf.py
    
## Usage

- Open [http://localhost:8501](http://localhost:8500) in your browser after launching the Streamlit app.
- Search for an ETF symbol (e.g., `SPY`, `GOLDBEES.NS`), adjust investment parameters, select a strategy/time period, and press "Fetch".
- Analyze the output tables and plots. Download CSV summaries to explore results in detail.

---

## File Structure

- `ETF2.py`: Streamlit app for interactive strategy simulation and visualization.
- `etf.py`: Standalone script for fast, command-line ETF backtesting and analytics.
- `requirements.txt`: List of Python package dependencies.

## Authors

   VAMSI REDDY

## Acknowledgements

- Data via [Yahoo Finance](https://finance.yahoo.com/)
- Built with [Streamlit](https://streamlit.io/), [yfinance](https://github.com/ranaroussi/yfinance), [Pandas](https://pandas.pydata.org/), and [Matplotlib](https://matplotlib.org/)

