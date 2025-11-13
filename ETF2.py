import streamlit as st
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

st.set_page_config(page_title="Global ETF Backtest Simulator", layout="wide")
st.title("🌍 Global ETF Backtest Simulator")
st.write("Type at least 2 letters of an ETF symbol (global search).")

@st.cache_data(ttl=3600)
def fetch_yahoo_suggestions(query: str):
    if not query:
        return []
    url = "https://query2.finance.yahoo.com/v1/finance/search"
    params = {"q": query, "lang": "en-US", "region": "US"}
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(url, params=params, headers=headers, timeout=5)
        r.raise_for_status()
        quotes = r.json().get("quotes", [])
        suggestions = [
            f"{q['symbol']} | {q.get('shortname', '')} | {q.get('quoteType','')}"
            for q in quotes if q.get("symbol")
        ]
        return suggestions
    except Exception:
        return []

@st.cache_data(ttl=3600)
def download_history(symbol: str, start="2000-01-01"):
    try:
        data = yf.download(symbol, start=start, progress=False, threads=False)
        return data
    except Exception:
        return pd.DataFrame()

def parse_symbol(display: str):
    if not display:
        return None
    return display.split("|")[0].strip()

query = st.text_input("🔎 Search ETF symbol (2+ chars):", "")
suggestions = fetch_yahoo_suggestions(query) if len(query) >= 2 else []

selected_display = None
if suggestions:
    selected_display = st.selectbox("Select an ETF from suggestions:", suggestions)

if st.button("🚀 Fetch ETF Data"):
    if selected_display:
        st.session_state["chosen_etf"] = parse_symbol(selected_display)
    elif suggestions:
        st.session_state["chosen_etf"] = parse_symbol(suggestions[0])
        st.info(f"No selection made — auto-selecting first suggestion: {st.session_state['chosen_etf']}")
    else:
        st.error("No valid ETF symbol selected.")

if "chosen_etf" in st.session_state:
    etf_symbol = st.session_state["chosen_etf"]
    st.markdown(f"**Selected ETF Symbol:** `{etf_symbol}`")

    try:
        ticker = yf.Ticker(etf_symbol)
        info = ticker.info or {}
        etf_info_data = {
            "Symbol": etf_symbol,
            "ETF Name": info.get("shortName") or info.get("longName") or "N/A",
            "Market Price": info.get("previousClose", "N/A"),
            "Currency": info.get("currency", "N/A"),
            "Exchange": info.get("exchange", "N/A"),
            "Sector": [info.get("sector", "N/A") if "sector" in info else "N/A"],
            "Industry": [info.get("industry", "N/A") if "industry" in info else "N/A"],
            "Quote Type": [info.get("quoteType", "N/A")]
        }
        st.subheader("📊 ETF Information")
        st.table(pd.DataFrame([etf_info_data]))
    except Exception as e:
        st.warning("⚠️ Could not fetch ETF info.")
        st.write(str(e))

    st.subheader("⚙️ Investment Parameters")
    buy_amount_initial = st.number_input("Initial Buy Amount (INR)", value=10000, min_value=1)
    buy_amount_subsequent = st.number_input("Subsequent Buy Amount (INR)", value=5000, min_value=0)
    fall_threshold = st.number_input("Fall Threshold (%) (e.g. -1.0)", value=-1.0, step=0.1)

    full_data = download_history(etf_symbol, start="2000-01-01")
    if full_data.empty:
        st.error("❌ No historical data found for this symbol.")
        st.stop()

    start_full_date = full_data.index[0].date()
    latest_full_date = full_data.index[-1].date()
    total_years_available = (latest_full_date - start_full_date).days // 365

    years_options = [str(i) for i in range(1, min(total_years_available, 5) + 1)] + ["Max"]
    selected_year = st.selectbox("Select Backtesting Period (Years):", options=years_options, index=len(years_options) -1)

    if selected_year != "Max":
        years_int = int(selected_year)
        filtered_start_date = latest_full_date - timedelta(days=years_int * 365)
    else:
        filtered_start_date = start_full_date

    filtered_data = full_data[full_data.index.date >= filtered_start_date]

    if filtered_data.empty:
        st.error("No data available for the selected period. Please select a shorter period.")
        st.stop()

    if isinstance(filtered_data.columns, pd.MultiIndex):
        filtered_data.columns = ['_'.join(col).strip() for col in filtered_data.columns.values]
    close_cols_filtered = [c for c in filtered_data.columns if 'Close' in c]
    if not close_cols_filtered:
        st.error("No Close price column found in filtered data.")
        st.stop()
    close_prices = filtered_data[close_cols_filtered[0]].dropna()

    df = pd.DataFrame({'Date': close_prices.index, 'Close': close_prices.values})
    df['Prev Close'] = df['Close'].shift(1)
    df['Change (%)'] = (df['Close'] - df['Prev Close']) / df['Prev Close'] * 100

    total_units = 0.0
    average_price = 0.0
    buy_dates, buy_prices, buy_amounts = [], [], []

    first_price = float(close_prices.iloc[0])
    units_bought = buy_amount_initial / first_price
    total_units += units_bought
    average_price = first_price
    buy_dates.append(close_prices.index[0])
    buy_prices.append(first_price)
    buy_amounts.append(buy_amount_initial)

    for i in range(1, len(df)):
        today = df.iloc[i]
        change = today['Change (%)']
        price = float(today['Close'])
        if pd.isna(change):
            continue
        if change <= fall_threshold:
            units = buy_amount_subsequent / price if price > 0 else 0
            prev_total = total_units
            total_units += units
            if total_units > 0:
                average_price = (average_price * prev_total + price * units) / total_units
            buy_dates.append(today['Date'])
            buy_prices.append(price)
            buy_amounts.append(buy_amount_subsequent)

    final_price = float(close_prices.iloc[-1])
    total_invested = sum(buy_amounts)
    portfolio_value = total_units * final_price
    roi_percent = ((portfolio_value - total_invested) / total_invested * 100) if total_invested else 0
    gain_per_unit = final_price - average_price if total_units else 0
    price_ratio = final_price / average_price if average_price else np.nan
    years_held = (datetime.today().date() - close_prices.index[0].date()).days / 365.25
    cagr = (portfolio_value / total_invested) ** (1/years_held) - 1 if total_invested > 0 else 0

    def format_inr_short(x):
        if abs(x) >= 1e7:
            return f"{x/1e7:.2f} Cr"
        elif abs(x) >= 1e5:
            return f"{x/1e5:.2f} Lakh"
        else:
            return f"{x:,.0f}"

    total_invested_fmt = format_inr_short(total_invested)
    portfolio_value_fmt = format_inr_short(portfolio_value)
    profit_loss = portfolio_value - total_invested
    profit_fmt = format_inr_short(profit_loss)

    is_profit = portfolio_value >= total_invested
    roi_positive = roi_percent >= 0

    portfolio_data = pd.DataFrame({
        "ETF Symbol": [etf_symbol],
        "Total Units": [round(total_units, 6)],
        "Average Buy Price": [round(average_price, 4)],
        "Final Price": [round(final_price, 4)],
        "Total Invested (₹)": [total_invested_fmt],
        "Portfolio Value (₹)": [portfolio_value_fmt],
        "Profit/Loss (₹)": [profit_fmt],
        "ROI (%)": [f"{roi_percent:.2f}%"],
        "Gain per Unit (₹)": [round(gain_per_unit, 4)],
        "Price / Avg Buy": [round(price_ratio, 4) if not np.isnan(price_ratio) else "N/A"],
        "CAGR (%)": [round(cagr*100, 2)],
        "Total Buys": [len(buy_dates)]
    })

    def colorize(val, positive=True):
        color = "green" if positive else "red"
        return f"<span style='color:{color}; font-weight:600;'>{val}</span>"

    portfolio_data_html = portfolio_data.copy()
    portfolio_data_html["Portfolio Value (₹)"] = colorize(portfolio_value_fmt, is_profit)
    portfolio_data_html["Profit/Loss (₹)"] = colorize(profit_fmt, is_profit)
    portfolio_data_html["ROI (%)"] = colorize(f"{roi_percent:.2f}%", roi_positive)

    st.subheader("💼 Portfolio Summary")
    st.markdown(
        portfolio_data_html.to_html(escape=False, index=False),
        unsafe_allow_html=True
    )

    st.subheader("📊 Yearly Summary: Investment, Profit/Loss, Buys & Units Bought")
    buy_df = pd.DataFrame({'Date': buy_dates, 'Price': buy_prices, 'Amount': buy_amounts})
    buy_df['Units'] = buy_df['Amount'] / buy_df['Price']
    buy_df['Value_at_End'] = buy_df['Units'] * close_prices.iloc[-1]
    buy_df['Profit/Loss'] = buy_df['Value_at_End'] - buy_df['Amount']

    # Make sure 'Year' column exists before grouping
    if 'Year' not in buy_df.columns:
        buy_df['Year'] = pd.DatetimeIndex(buy_df['Date']).year

    yearly_summary = buy_df.groupby('Year').agg(
        Total_Invested=('Amount', 'sum'),
        Profit_Loss=('Profit/Loss', 'sum'),
        Total_Buys=('Date', 'count'),
        Units_Bought=('Units', 'sum')
    ).reset_index()
    
    # Add Download Button (below yearly_summary creation)
    csv_data = yearly_summary.to_csv(index=False).encode('utf-8')
    
    st.download_button(
        label="⬇️ Download Yearly Summary (CSV)",
        data=csv_data,
        file_name=f"{etf_symbol}_yearly_summary.csv",
        mime="text/csv",
        help="Download the yearly investment summary as a CSV file"
    )


    yearly_summary['Total Invested (INR)'] = yearly_summary['Total_Invested'].map(lambda x: f"₹{x:,.2f}")
    yearly_summary['Total Invested (Lakh)'] = yearly_summary['Total_Invested'].map(lambda x: f"{round(x/1e5, 2)} Lakh")
    yearly_summary['Profit/Loss (INR)'] = yearly_summary['Profit_Loss'].map(lambda x: f"₹{x:,.2f}")
    yearly_summary['Profit/Loss (Lakh)'] = yearly_summary['Profit_Loss'].map(lambda x: f"{round(x/1e5, 2)} Lakh")
    yearly_summary['Units Bought'] = yearly_summary['Units_Bought'].map(lambda x: round(x, 4))

    def colorize(val):
        try:
            num = float(str(val).replace('₹','').replace(',',''))
            color = 'green' if num >= 0 else 'red'
        except:
            color = 'black'
        return f'<span style="color:{color}; font-weight:600">{val}</span>'

    html_table = '<table style="width:100%; border-collapse: collapse; text-align:center;">'
    html_table += '<tr style="background-color:#1f77b4; color:white; font-weight:bold;">'
    html_table += '<th>Year</th><th>Total Invested (INR)</th><th>Total Invested (Lakh)</th>'
    html_table += '<th>Profit/Loss (INR)</th><th>Profit/Loss (Lakh)</th>'
    html_table += '<th>Total Buys</th><th>Units Bought</th></tr>'

    for i, row in yearly_summary.iterrows():
        html_table += f"<tr>"
        html_table += f"<td>{row['Year']}</td>"
        html_table += f"<td>{row['Total Invested (INR)']}</td>"
        html_table += f"<td>{row['Total Invested (Lakh)']}</td>"
        html_table += f"<td>{colorize(row['Profit/Loss (INR)'])}</td>"
        html_table += f"<td>{row['Profit/Loss (Lakh)']}</td>"
        html_table += f"<td>{row['Total_Buys']}</td>"
        html_table += f"<td>{row['Units Bought']}</td>"
        html_table += "</tr>"
    html_table += "</table>"

    st.markdown(html_table, unsafe_allow_html=True)

    st.subheader("📈 Price Chart with Buy Points")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df['Date'], df['Close'], label='Close Price')
    if buy_dates:
        ax.scatter(buy_dates, buy_prices, color='red', marker='^', s=80, label='Buy Points')
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.set_title(f"{etf_symbol} Price & Buy Points")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("💰 Annual Investment (in Lakhs)")
    buy_summary_per_year = buy_df.groupby('Year').agg(
        Total_Invested=('Amount', 'sum')
    ).reset_index()
    buy_summary_per_year['Invest_Lakhs'] = buy_summary_per_year['Total_Invested'] / 1e5
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.bar(buy_summary_per_year['Year'].astype(str), buy_summary_per_year['Invest_Lakhs'], color='#3AC0DA')
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Investment (Lakhs)")
    ax2.set_title("Annual Investment Amount")
    ax2.grid(axis='y')
    plt.tight_layout()
    st.pyplot(fig2)
    
    

    




                



        
    import streamlit as st
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    
    # ==========================
    # Load your ETF data here
    # ==========================
    # df should have at least ['Date', 'Close']
    # Example:
    # df = pd.read_csv("your_etf_data.csv")
    # df['Date'] = pd.to_datetime(df['Date'])
    
    # --------------------------
    # User Inputs
    # --------------------------
    st.markdown("---")
    st.header("🧠 Strategy Comparison & Visualization")
    st.markdown("""Compare investment strategies: **Buy the Dip**, **SIP**, **Momentum**, **Hybrid (Dip + Momentum)**
    The available strategies are:  
    - **Buy the Dip**: Invest a fixed amount whenever the price falls by a certain percentage.  
    - **SIP (Monthly)**: Invest a fixed amount periodically at a chosen interval (Systematic Investment Plan).  
    - **Momentum**: Invest when the 50 EMA short-term moving average is above the 200 EMA long-term moving average.  
    - **Hybrid (Dip + Momentum)**: Invest only when a dip occurs during a positive momentum trend.
    """)
    
    strategy = st.selectbox(
        "Choose Backtest Strategy:",
        ["Buy on Dip", "SIP (Monthly)", "Momentum", "Hybrid (Dip + Momentum)"]
    )
    
    investment_amount = st.number_input("💰 Investment per Buy (₹)", 1000, 100000, 5000, step=1000)
    fall_threshold = st.number_input("📉 Dip Threshold (%)", 0.1, 10.0, 1.0, step=0.1)
    
    # --------------------------
    # Year selection dropdown (NEW)
    # --------------------------
    max_years = df['Date'].dt.year.max() - df['Date'].dt.year.min() + 1
    year_options = list(range(1, max_years + 1))
    selected_years = st.selectbox("📅 Analyze Last n Years:", year_options, index=min(4, max_years-1))
    
    # Filter dataframe according to selected years
    latest_year = df['Date'].dt.year.max()
    start_year = latest_year - selected_years + 1
    df = df[df['Date'].dt.year >= start_year].reset_index(drop=True)
    
    st.markdown("---")
    
    # ==========================
    # Prepare Data
    # ==========================
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    
    # Helper to format INR
    def format_inr(x):
        if abs(x) >= 1e7:
            return f"{x/1e7:.2f} Cr"
        elif abs(x) >= 1e5:
            return f"{x/1e5:.2f} Lakh"
        else:
            return f"₹{x:,.0f}"
    
    # ==========================
    # Backtest Function
    # ==========================
    def backtest_strategy(df, strategy_name):
        units = 0
        total_invested = 0
    
        if strategy_name == "Buy on Dip":
            for i in range(1, len(df)):
                change = (df['Close'][i] - df['Close'][i-1]) / df['Close'][i-1] * 100
                if change <= -fall_threshold:
                    units += investment_amount / df['Close'][i]
                    total_invested += investment_amount
    
        elif strategy_name == "SIP (Monthly)":
            df['Month'] = df['Date'].dt.to_period('M')
            sip_dates = df.groupby('Month').first()['Date'][::1]
            for date in sip_dates:
                price = df.loc[df['Date'] == date, 'Close'].values[0]
                units += investment_amount / price
                total_invested += investment_amount
    
        elif strategy_name == "Momentum":
            for i in range(200, len(df)):
                if df['MA50'][i] > df['MA200'][i]:
                    units += investment_amount / df['Close'][i]
                    total_invested += investment_amount
    
        elif strategy_name == "Hybrid (Dip + Momentum)":
            for i in range(200, len(df)):
                change = (df['Close'][i] - df['Close'][i-1]) / df['Close'][i-1] * 100
                if (df['MA50'][i] > df['MA200'][i]) and (change <= -fall_threshold):
                    units += investment_amount / df['Close'][i]
                    total_invested += investment_amount
    
        final_value = units * df['Close'].iloc[-1]
        profit = final_value - total_invested
        roi = profit / total_invested * 100 if total_invested != 0 else 0
    
        return {
            "Strategy": strategy_name,
            "Total Invested": total_invested,
            "Final Value": final_value,
            "Profit / Loss": profit,
            "ROI (%)": roi,
            "Units Purchased": units,
            "Total Buys": int(total_invested / investment_amount)
        }
    
    # ==========================
    # Single Strategy Summary
    # ==========================
    result = backtest_strategy(df, strategy)
    
    st.subheader("📈 Strategy Performance Summary")
    summary_df = pd.DataFrame({
        "Metric": ["Total Invested", "Final Portfolio Value", "Profit / Loss", "ROI (%)", "Units Purchased", "Total Buys"],
        "Value": [
            format_inr(result["Total Invested"]),
            format_inr(result["Final Value"]),
            format_inr(result["Profit / Loss"]),
            f"{result['ROI (%)']:.2f}%",
            f"{result['Units Purchased']:.2f}",
            result["Total Buys"]
        ]
    })
    styled_table = summary_df.style.set_table_styles([
        {"selector": "thead", "props": [("background-color", "#0783d9"), ("color", "white"), ("font-weight", "bold")]},
        {"selector": "td", "props": [("padding", "8px")]}
    ]).hide(axis="index")
    st.dataframe(styled_table, use_container_width=True)
    
    # ==========================
    # Compare All Strategies
    # ==========================
    if st.button("📊 Compare All Strategies"):
        strategies = ["Buy on Dip", "SIP (Monthly)", "Momentum", "Hybrid (Dip + Momentum)"]
        summary_list = [backtest_strategy(df, strat) for strat in strategies]
    
        comparison_df = pd.DataFrame(summary_list)
    
        # Format INR columns
        for col in ["Total Invested", "Final Value", "Profit / Loss"]:
            comparison_df[col] = comparison_df[col].apply(format_inr)
        comparison_df["ROI (%)"] = comparison_df["ROI (%)"].apply(lambda x: f"{x:.2f}%")
        comparison_df["Units Purchased"] = comparison_df["Units Purchased"].apply(lambda x: f"{x:.2f}")
    
        # Conditional coloring for Profit / Loss
        def color_profit_loss(val):
            try:
                if 'Cr' in val:
                    num = float(val.replace('Cr', '').replace('₹', '').replace(',', '').strip()) * 1e7
                elif 'Lakh' in val:
                    num = float(val.replace('Lakh', '').replace('₹', '').replace(',', '').strip()) * 1e5
                else:
                    num = float(val.replace('₹', '').replace(',', '').strip())
            except:
                num = 0
            return 'color: green;' if num >= 0 else 'color: red;'
    
        styled_table = comparison_df.style.set_table_styles([
            {"selector": "thead", "props": [("background-color", "#0783d9"),
                                            ("color", "white"),
                                            ("font-weight", "bold")]},
            {"selector": "td", "props": [("padding", "8px")]}
        ]).applymap(color_profit_loss, subset=['Profit / Loss'])
    
        st.subheader("📈 Strategy Performance Comparison")
        st.dataframe(styled_table, use_container_width=True)
    
        # ==========================
        # Best Strategy Insight
        # ==========================
        best_strategy_index = np.argmax([res['Profit / Loss'] for res in summary_list])
        best_strategy = summary_list[best_strategy_index]
    
        best_strategy_name = best_strategy['Strategy']
        reason = f"This strategy gave the highest Profit/Loss of {format_inr(best_strategy['Profit / Loss'])} " \
                 f"with an ROI of {best_strategy['ROI (%)']:.2f}% over the last {selected_years} years."
    
        st.markdown(f"**💡 Best Strategy:** <span style='color:blue'>{best_strategy_name}</span>", unsafe_allow_html=True)
        st.markdown(f"**📌 Reason:** {reason}")



        # ==========================
        # Line Chart Visualization of All Strategies
        # ==========================
        st.subheader("📊 Portfolio Value Over Time for All Strategies")
        
        # Function to compute daily portfolio value for a strategy
        def portfolio_over_time(df, strategy_name):
            units = 0
            total_invested = 0
            values = []
        
            for i in range(len(df)):
                if strategy_name == "Buy on Dip" and i > 0:
                    change = (df['Close'][i] - df['Close'][i-1]) / df['Close'][i-1] * 100
                    if change <= -fall_threshold:
                        units += investment_amount / df['Close'][i]
                        total_invested += investment_amount
        
                elif strategy_name == "SIP (Monthly)":
                    if i == 0 or df['Date'][i].month != df['Date'][i-1].month:
                        units += investment_amount / df['Close'][i]
                        total_invested += investment_amount
        
                elif strategy_name == "Momentum" and i >= 200:
                    if df['MA50'][i] > df['MA200'][i]:
                        units += investment_amount / df['Close'][i]
                        total_invested += investment_amount
        
                elif strategy_name == "Hybrid (Dip + Momentum)" and i >= 200:
                    change = (df['Close'][i] - df['Close'][i-1]) / df['Close'][i-1] * 100
                    if (df['MA50'][i] > df['MA200'][i]) and (change <= -fall_threshold):
                        units += investment_amount / df['Close'][i]
                        total_invested += investment_amount
        
                # Append current portfolio value
                current_value = units * df['Close'][i]
                values.append(current_value)
        
            return values
        
        # Compute portfolio values for all strategies
        strategies = ["Buy on Dip", "SIP (Monthly)", "Momentum", "Hybrid (Dip + Momentum)"]
        portfolio_dict = {strat: portfolio_over_time(df, strat) for strat in strategies}
        portfolio_df = pd.DataFrame(portfolio_dict, index=df['Date'])
        
        # Plot line chart
        st.line_chart(portfolio_df)