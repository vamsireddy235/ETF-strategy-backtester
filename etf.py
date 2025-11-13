import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

# === User Input ===
etf_name = input("Enter ETF symbol (e.g., goldbees.NS): ").strip()
buy_amount_initial = 10000  # initial buy
buy_amount_subsequent = 5000
fall_threshold = -1.0       # 1% fall or more

# === Download ETF data ===
data = yf.download(etf_name, start="2000-01-01")
if data.empty:
    raise ValueError("No data found for this ETF.")

# Automatically set start and end dates
start_date = data.index[0].date()
end_date = datetime.today().date()
print(f"Data from {start_date} to {end_date}")

# --- Handle MultiIndex / DataFrame issues ---
if isinstance(data.columns, pd.MultiIndex):
    data.columns = ['_'.join(col).strip() for col in data.columns.values]

possible_close_cols = [c for c in data.columns if 'Close' in c or 'close' in c]
if not possible_close_cols:
    raise ValueError(f"No valid 'Close' column found. Columns: {list(data.columns)}")

close_prices = data[possible_close_cols[0]].copy()
if isinstance(close_prices, pd.DataFrame):
    close_prices = close_prices.iloc[:, 0]
close_prices = pd.Series(close_prices.squeeze(), dtype='float64').dropna()

# === Calculate daily change ===
df = pd.DataFrame({'Date': close_prices.index, 'Close': close_prices.values})
df['Prev Close'] = df['Close'].shift(1)
df['Change (%)'] = ((df['Close'] - df['Prev Close']) / df['Prev Close']) * 100

# === Filter negative days and mark condition met ===
fall_df = df[df['Change (%)'] < 0].copy()
fall_df['Condition Met (≥1%)'] = fall_df['Change (%)'] <= fall_threshold

# === Backtesting ===
total_units = 0.0
average_price = 0.0
buy_dates = []
buy_prices = []
buy_amounts = []

# --- Initial buy ---
first_price = close_prices.iloc[0]
units_bought = buy_amount_initial / first_price
total_units += units_bought
average_price = first_price
buy_dates.append(close_prices.index[0])
buy_prices.append(first_price)
buy_amounts.append(buy_amount_initial)
print(f"Initial buy: {units_bought:.4f} units at {first_price:.2f} on {close_prices.index[0].date()}")

# --- Subsequent buys ---
for i in range(1, len(df)):
    today = df.iloc[i]
    change = today['Change (%)']
    price = today['Close']

    if change <= fall_threshold:
        units_bought = buy_amount_subsequent / price
        total_units += units_bought
        average_price = ((average_price * (total_units - units_bought)) + (price * units_bought)) / total_units
        buy_dates.append(today['Date'])
        buy_prices.append(price)
        buy_amounts.append(buy_amount_subsequent)
        print(f"Bought {units_bought:.4f} units at {price:.2f} on {today['Date'].date()} (fall {change:.2f}%)")

# Include initial buy in fall_df if not present
fall_df['Buy Executed'] = fall_df['Condition Met (≥1%)']
if close_prices.index[0] not in fall_df['Date'].values:
    fall_df = pd.concat([
        pd.DataFrame({
            'Date':[close_prices.index[0]],
            'Prev Close':[np.nan],
            'Close':[first_price],
            'Change (%)':[np.nan],
            'Condition Met (≥1%)':[True],
            'Buy Executed':[True]
        }),
        fall_df
    ], ignore_index=True)

# === Final Portfolio Calculation ===
final_price = float(close_prices.iloc[-1])
total_invested = float(sum(buy_amounts))
total_units = float(total_units)
average_price = float(average_price)
portfolio_value = total_units * final_price
profit = portfolio_value - total_invested

# === Ensure all buy prices/amounts are numeric ===
buy_prices_float = list(map(float, buy_prices))
buy_amounts_float = list(map(float, buy_amounts))

lowest_buy_price = min(buy_prices_float)
highest_buy_price = max(buy_prices_float)
max_single_buy = max(buy_amounts_float)
gain_per_unit = final_price - average_price
price_ratio = final_price / average_price
roi_percent = (portfolio_value - total_invested) / total_invested * 100

# --- Average days between buys (convert timedelta to float days) ---
buy_dates_sorted = pd.to_datetime(buy_dates)
if len(buy_dates_sorted) > 1:
    days_between_buys = np.diff(buy_dates_sorted)
    days_between_buys_days = [d / np.timedelta64(1, 'D') for d in days_between_buys]  # convert to float
    avg_days_between_buys = float(np.mean(days_between_buys_days))
else:
    avg_days_between_buys = np.nan

# --- CAGR ---
years_held = (end_date - start_date).days / 365.25
cagr = (portfolio_value / total_invested)**(1/years_held) - 1

# === Buy Summary Per Year ===
buy_df = pd.DataFrame({'Date': buy_dates, 'Price': buy_prices, 'Amount': buy_amounts})
buy_df['Year'] = buy_df['Date'].dt.year
buy_df['Units'] = buy_df['Amount'] / buy_df['Price']

buy_summary_per_year = buy_df.groupby('Year').agg(
    Buys=('Date','count'),
    Total_Invested=('Amount','sum'),
    Units_Bought=('Units','sum')
).reset_index()

# === Backtest Summary ===
print("\n=== Backtest Summary ===")
print(f"ETF: {etf_name}")
print(f"Total Units Bought: {total_units:.4f}")
print(f"Average Buy Price: {average_price:.2f}")
print(f"Final ETF Price: {final_price:.2f}")
print(f"Total Invested: {total_invested:.2f}")
print(f"Portfolio Value: {portfolio_value:.2f}")
print(f"Profit/Loss: {profit:.2f}")
print(f"ROI: {roi_percent:.2f}%")
print(f"Unrealized Gain per Unit: {gain_per_unit:.2f}")
print(f"Lowest Buy Price: {lowest_buy_price:.2f}")
print(f"Highest Buy Price: {highest_buy_price:.2f}")
print(f"Price / Avg Buy Price Ratio: {price_ratio:.2f}")
print(f"Max Single Buy: {max_single_buy:.2f}")
print(f"Average Days Between Buys: {avg_days_between_buys:.1f}")
print(f"CAGR: {cagr*100:.2f}%")
print(f"Total Buy Transactions: {len(buy_dates)}")

# === Red Days Table (first 20 rows) ===
print("\n=== Red Days Table (≥1% Buy Triggered) ===")
print(fall_df[['Date','Prev Close','Close','Change (%)','Condition Met (≥1%)','Buy Executed']].head(20))

# === Buy Summary Per Year ===
print("\n=== Buy Summary Per Year ===")
print(buy_summary_per_year)

# === Yearly Buy Tables in Red Days Table style ===
# Merge buy_df with original df to get Prev Close and Change (%)
buy_table = pd.merge(
    buy_df,
    df[['Date','Prev Close','Close','Change (%)']],
    on='Date',
    how='left'
)
buy_table['Condition Met (≥1%)'] = True
buy_table['Buy Executed'] = True

buy_table = buy_table.sort_values('Date').reset_index(drop=True)

for year, group in buy_table.groupby(buy_table['Date'].dt.year):
    print(f"\n=== Buys in {year} ===")
    print(group[['Date','Prev Close','Close','Change (%)']])



# --- Calculate yearly Profit/Loss ---
buy_table['Value at End'] = buy_table['Units'] * final_price
buy_table['Profit/Loss'] = buy_table['Value at End'] - buy_table['Amount']

yearly_profit = buy_table.groupby(buy_table['Date'].dt.year)['Profit/Loss'].sum().reset_index()
yearly_profit.rename(columns={'Date':'Year'}, inplace=True)  # optional, nicer column name

# --- Function to format in Indian style ---
def format_inr(x):
    x = round(x, 2)
    s, *d = f"{x:.2f}".split(".")
    n = len(s)
    if n > 3:
        parts = [s[-3:]] + [s[max(i-2,0):i] for i in range(n-3, 0, -2)][::-1]
        s = ",".join(parts)
    return s + "." + d[0]

# --- Function to approximate in lakhs ---
def approx_lakh(x):
    lakh = x / 1_00_000
    if lakh >= 1:
        return f"{round(lakh)} Lakh"
    else:
        return f"{round(lakh, 2)} Lakh"

# Apply formatting
yearly_profit['Profit/Loss (INR)'] = yearly_profit['Profit/Loss'].apply(format_inr)
yearly_profit['Approx (Lakh)'] = yearly_profit['Profit/Loss'].apply(approx_lakh)

# Display
print("\n=== Yearly Profit/Loss (INR format + Approx in Lakh) ===")
print(yearly_profit[['Year', 'Profit/Loss', 'Approx (Lakh)']])


# === Visualization ===
plt.figure(figsize=(14,6))
plt.plot(df['Date'], df['Close'], label='ETF Close Price', color='blue')
plt.scatter(buy_dates, buy_prices, color='red', label='Buy Points', marker='^', s=100)
plt.title(f'{etf_name} Price & Buy Points')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()


# Annual Buy Summary Bar Chart
fig4, ax4 = plt.subplots(figsize=(12, 5))
buy_summary_per_year['Total_Invested_Lakhs'] = buy_summary_per_year['Total_Invested'] / 1e5
bars = ax4.bar(buy_summary_per_year['Year'].astype(str), buy_summary_per_year['Total_Invested_Lakhs'],
               color='LIGHTBLUE', alpha=0.7)
ax4.set_title('Annual Investment Amount (in Lakhs)')
ax4.set_xlabel('Year')
ax4.set_ylabel('Investment Amount (Lakhs)')
ax4.grid(axis='y')

plt.tight_layout()
plt.show()


