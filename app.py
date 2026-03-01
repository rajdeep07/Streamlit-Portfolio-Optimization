import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from scipy.cluster.hierarchy import linkage, leaves_list

# --- 1. CONFIG & DATA ---
st.set_page_config(page_title="HRP Portfolio Allocation", layout="wide")


@st.cache_data
def get_stock_data(tickers):
    data = yf.download(tickers, period="1y", interval="1d")['Close']
    data = data.dropna(axis=1, how='all')
    data = data.ffill().dropna()
    return data


# --- 2. THE ALGORITHMS ---
def get_hrp_allocation(returns):
    corr = returns.corr().fillna(0)
    d_matrix = np.sqrt(np.clip(0.5 * (1 - corr), 0, 1))
    from scipy.spatial.distance import squareform
    link = linkage(squareform(d_matrix), 'single')
    sort_ix = leaves_list(link)
    sorted_tickers = corr.columns[sort_ix].tolist()

    variances = returns.var()
    inv_vars = 1 / variances
    weights = inv_vars / inv_vars.sum()
    return weights, sorted_tickers


def forecast_gbm(prices, weights, investment, days=252):
    """Geometric Brownian Motion for Portfolio Forecasting"""
    returns = prices.pct_change().dropna()
    # Align weights with returns columns
    w = np.array([weights[t] for t in returns.columns])

    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    port_return = np.sum(mean_returns * w) * 252
    port_vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix * 252, w)))

    t = np.linspace(0, 1, days)
    mu_path = investment * np.exp((port_return - 0.5 * port_vol ** 2) * t)
    # 95% Confidence Interval
    lower_ci = mu_path * np.exp(-1.96 * port_vol * np.sqrt(t))
    upper_ci = mu_path * np.exp(1.96 * port_vol * np.sqrt(t))

    return np.arange(days), mu_path, lower_ci, upper_ci


# --- 3. UI LAYOUT ---
st.title("Portfolio Risk Analysis & Allocation with HRP")

# Sidebar
st.sidebar.header("Investment Parameters")
investment_amt = st.sidebar.number_input("Total Investment ($)", value=100000, step=5000)

default_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'JPM', 'V', 'XOM', 'WMT', 'ADBE']
user_tickers = st.sidebar.multiselect("Select Portfolio Subset", options=default_tickers + ['TSLA', 'META', 'NFLX'],
                                      default=default_tickers)

if len(user_tickers) >= 2:
    prices = get_stock_data(user_tickers)

    if prices.empty or prices.shape[1] < 2:
        st.error("Not enough valid data found.")
    else:
        returns = prices.pct_change().dropna()
        weights, sorted_list = get_hrp_allocation(returns)

    # 1. HEATMAP
    st.subheader("1. Heatmap (HRP Sequence)")
    corr_ordered = returns.corr().loc[sorted_list, sorted_list]
    st.plotly_chart(px.imshow(corr_ordered, text_auto=".2f", color_continuous_scale='RdBu_r'), use_container_width=True)

    # 2. METADATA
    st.divider()
    st.header("2. Asset Metadata & Institutional Context")
    meta_list = []
    network_links = []

    for t in sorted_list:
        with st.expander(f"Detailed Analysis: {t}"):
            ticker_obj = yf.Ticker(t)
            info = ticker_obj.info
            mkt_cap = info.get('marketCap', 0)

            meta_list.append({
                "Ticker": t,
                "Weight": f"{weights[t] * 100:.2f}%",
                "Dollar Allocation": f"${investment_amt * weights[t]:,.2f}",
                "Sector": info.get('sector'),
                "Industry": info.get('industry'),
                "Market Cap": f"${mkt_cap:,.0f}"
            })

            col_info, col_holders = st.columns([1, 2])
            with col_info:
                st.write(f"**Exchange:** {info.get('fullExchangeName')}")
                st.write(f"**Recommendation:** {info.get('recommendationKey', 'N/A').upper()}")

            with col_holders:
                st.write("**Top 10 Institutional Holders**")
                inst = ticker_obj.institutional_holders
                if inst is not None and not inst.empty:
                    df_inst = inst.head(10)
                    st.dataframe(df_inst[['Holder', 'pctHeld', 'Shares']], hide_index=True)
                    # Collect data for Network Viz
                    for holder in df_inst['Holder'].tolist():
                        network_links.append({"Ticker": t, "Institution": holder})
                else:
                    st.info("Institutional data unavailable.")

    # 3. SUMMARY & SECTOR VIZ
    st.subheader("Final Portfolio Summary")
    st.table(pd.DataFrame(meta_list))

    st.subheader("3. Risk Distribution by Sector")
    sector_df = pd.DataFrame(meta_list)
    sector_df['WeightVal'] = [weights[t] for t in sorted_list]
    fig_sector = px.treemap(sector_df, path=['Sector', 'Ticker'], values='WeightVal',
                            title="Portfolio Tree: Hierarchical Weights")
    st.plotly_chart(fig_sector, use_container_width=True)

    # 4. FORECASTING
    st.divider()
    st.header("4. Portfolio Return Forecast (1-Year Projection)")
    st.markdown(f"Projected growth for **${investment_amt:,.0f}** based on Geometric Brownian Motion (GBM).")

    days, mu, lower, upper = forecast_gbm(prices, weights, investment_amt)

    fig_forecast = go.Figure()
    # Confidence Intervals
    fig_forecast.add_trace(go.Scatter(x=days, y=upper, mode='lines', line=dict(width=0), showlegend=False))
    fig_forecast.add_trace(go.Scatter(x=days, y=lower, mode='lines', line=dict(width=0),
                                      fill='tonexty', fillcolor='rgba(0, 255, 0, 0.1)', name='95% Confidence Interval'))
    # Mean Path
    fig_forecast.add_trace(
        go.Scatter(x=days, y=mu, mode='lines', line=dict(color='green', width=4), name='Expected Growth'))

    fig_forecast.update_layout(height=600, hovermode="x unified", template="plotly_white",
                               yaxis_title="Value ($)", xaxis_title="Days (Trading Year)")
    st.plotly_chart(fig_forecast, use_container_width=True)

    # 5. INSTITUTIONAL GRAPH VIZ (CLEANER)
    # --- 5. INSTITUTIONAL GRAPH VIZ (CLEANED & ENHANCED) ---
    st.divider()
    st.header("5. Institutional Commonality Network")
    st.markdown("""
    **How to read this chart:**
    - **Left (Yellow):** Your selected portfolio tickers.
    - **Right (Red):** Major Institutional Holders.
    - **Lines:** The color of the line matches the **Ticker** it originates from, allowing you to track ownership overlap across the portfolio.
    """)

    # Build the network dataframe with a filter to keep it clean
    network_links = []
    unique_holders = set()

    for t in sorted_list:
        ticker_obj = yf.Ticker(t)
        inst = ticker_obj.institutional_holders
        if inst is not None and not inst.empty:
            # We take top 5 to ensure the chart remains readable
            df_inst = inst.head(5)
            for _, row in df_inst.iterrows():
                network_links.append({
                    "Ticker": t,
                    "Institution": row['Holder'],
                    "Value": row['pctHeld'] * 100  # Use percentage held for line thickness
                })
                unique_holders.add(row['Holder'])

    df_net = pd.DataFrame(network_links)

    if not df_net.empty:
        # 1. Define Nodes
        tickers = sorted_list
        institutions = sorted(list(unique_holders))
        all_nodes = tickers + institutions
        node_map = {node: i for i, node in enumerate(all_nodes)}

        # 2. Define Colors
        # Create a distinct color palette for each ticker to color the lines
        ticker_colors = px.colors.qualitative.Bold
        color_map = {t: ticker_colors[i % len(ticker_colors)] for i, t in enumerate(tickers)}

        # 3. Build Sankey
        fig_net = go.Figure(data=[go.Sankey(
            node=dict(
                pad=30,  # Increased padding between nodes
                thickness=25,
                line=dict(color="black", width=1),
                label=[n.split(' ')[0] if len(n) > 20 else n for n in all_nodes],  # Shorten long names
                color=["#FFD700" if n in tickers else "#FF4B4B" for n in all_nodes],
                hoverinfo='all'
            ),
            link=dict(
                source=[node_map[t] for t in df_net['Ticker']],
                target=[node_map[i] for i in df_net['Institution']],
                value=df_net['Value'],
                # Apply ticker-specific color with transparency to the flow lines
                color=[color_map[t].replace('rgb', 'rgba').replace(')', ', 0.4)') for t in df_net['Ticker']]
            ))])

        # 4. Dynamic Height Calculation
        # More nodes = Taller chart to prevent overlapping text
        dynamic_height = max(600, len(all_nodes) * 25)

        fig_net.update_layout(
            title_text="Common Ownership: Ticker to Institution Flow",
            font_size=14,  # Larger font for readability
            height=dynamic_height,
            margin=dict(l=10, r=10, t=40, b=10)
        )

        st.plotly_chart(fig_net, use_container_width=True)
    else:
        st.warning("No institutional data found to visualize.")

else:
    st.info("Select tickers in the sidebar to start the HRP process.")
