import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px

# Import functions from efficient_frontier.py
from efficient_frontier import (
    download_data,
    calculate_returns,
    calculate_cov_matrix,
    calculate_portfolio_performance,
    calculate_efficient_frontier,
    plot_efficient_frontier,
    maximum_sharpe_portfolio,
    get_mag7_tickers,
    analyze_portfolio,
    minimum_volatility_portfolio,
)

# Set page configuration
st.set_page_config(
    page_title="Efficient Frontier Simulator",
    page_icon="📈",
    layout="wide",
)

# Add custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    h1, h2, h3 {
        color: #1E3A8A;
    }
    .stButton>button {
        background-color: #1E3A8A;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# Create title
st.title('📊 Portfolio Efficient Frontier Simulator')
st.markdown("""
This app helps you create an efficient frontier for your investment portfolio. 
You can select assets, define expected returns, and visualize optimal portfolios.
""")

# Create tabs
tab1, tab2, tab3 = st.tabs(["Asset Selection", "Efficient Frontier", "Portfolio Analysis"])

# Define default tickers
default_tickers = {
    'ACWI': 'ACWI',  # iShares MSCI ACWI ETF
    'BOND': 'BOND', # PIMCO Active Bond ETF
    'GLDM': 'GLDM', 
}

mag7_tickers = get_mag7_tickers()
mag7_dict = {ticker: ticker for ticker in mag7_tickers}

# Combine all default tickers
all_default_tickers = {**default_tickers, **mag7_dict}

# Default dates
end_date = datetime.now()
start_date = end_date - timedelta(days=3*365)  # 3 years of data

# Set session state for storing data between tabs
if 'data' not in st.session_state:
    st.session_state.data = None
if 'returns' not in st.session_state:
    st.session_state.returns = None
if 'selected_tickers' not in st.session_state:
    st.session_state.selected_tickers = []
if 'expected_returns' not in st.session_state:
    st.session_state.expected_returns = {}
if 'cov_matrix' not in st.session_state:
    st.session_state.cov_matrix = None
if 'efficient_portfolios' not in st.session_state:
    st.session_state.efficient_portfolios = None
if 'custom_tickers' not in st.session_state:
    st.session_state.custom_tickers = {}

# Asset Selection Tab
with tab1:
    st.header("Select Assets")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Predefined Assets")
        # Select predefined assets
        st.markdown("### Core Assets")
        use_acwi = st.checkbox("ACWI (iShares MSCI ACWI ETF)", value=True)
        use_bond = st.checkbox("BOND (PIMCO Active Bond ETF)", value=True)
        use_gldm = st.checkbox("GLDM (SPDR Gold Shares)", value=True)
        
        st.markdown("### MAG7 Stocks")
        mag7_selections = {}
        for ticker in mag7_tickers:
            mag7_selections[ticker] = st.checkbox(f"{ticker}", value=True)
    
    with col2:
        st.subheader("Add Custom Assets")
        # Allow users to add custom tickers
        custom_ticker = st.text_input("Enter ticker symbol (e.g., SPY, VTI)")
        add_ticker_button = st.button("Add Ticker")
        
        if add_ticker_button and custom_ticker:
            try:
                # Verify ticker exists
                test_data = yf.download(custom_ticker, period="1d")
                if not test_data.empty:
                    st.session_state.custom_tickers[custom_ticker] = custom_ticker
                    st.success(f"Added {custom_ticker} to your asset list!")
                else:
                    st.error(f"Could not find ticker {custom_ticker}. Please check the symbol.")
            except Exception as e:
                st.error(f"Error adding ticker: {str(e)}")
                
        # Display and manage custom tickers
        if st.session_state.custom_tickers:
            st.markdown("### Your Custom Assets")
            custom_selections = {}
            for ticker in st.session_state.custom_tickers:
                keep = st.checkbox(f"{ticker}", value=True, key=f"custom_{ticker}")
                if not keep:
                    # Mark for removal
                    custom_selections[ticker] = False
                else:
                    custom_selections[ticker] = True
            
            # Remove unselected custom tickers
            st.session_state.custom_tickers = {k: v for k, v in st.session_state.custom_tickers.items() 
                                           if custom_selections.get(k, True)}
    
    # Date range selection
    st.subheader("Select Historical Data Range")
    col1, col2 = st.columns(2)
    with col1:
        selected_start_date = st.date_input("Start Date", value=start_date)
    with col2:
        selected_end_date = st.date_input("End Date", value=end_date)
    
    if selected_start_date >= selected_end_date:
        st.error("Start date must be before end date")
    
    # Prepare final ticker list
    selected_tickers = []
    if use_acwi:
        selected_tickers.append(default_tickers['ACWI'])
    if use_bond:
        selected_tickers.append(default_tickers['BOND'])
    if use_gldm:
        selected_tickers.append(default_tickers['GLDM'])
    
    # Add selected MAG7 stocks
    for ticker, selected in mag7_selections.items():
        if selected:
            selected_tickers.append(ticker)
    
    # Add custom tickers
    for ticker in st.session_state.custom_tickers.values():
        selected_tickers.append(ticker)
    
    # Fetch data button
    if len(selected_tickers) >= 2:
        if st.button("Fetch Data and Proceed"):
            with st.spinner("Downloading historical data..."):
                try:
                    # Download data
                    data = download_data(selected_tickers, selected_start_date, selected_end_date)
                    
                    # Calculate returns
                    returns = calculate_returns(data)
                    
                    # Calculate covariance matrix
                    cov_matrix = calculate_cov_matrix(returns)
                    
                    # Historical annualized returns
                    hist_returns = returns.mean() * 252
                    
                    # Store in session state
                    st.session_state.data = data
                    st.session_state.returns = returns
                    st.session_state.selected_tickers = selected_tickers
                    st.session_state.cov_matrix = cov_matrix
                    
                    # Initialize expected returns with historical values
                    st.session_state.expected_returns = {ticker: float(hist_returns[ticker]) for ticker in selected_tickers}
                    
                    st.success("Data downloaded successfully! Please go to the 'Efficient Frontier' tab.")
                except Exception as e:
                    st.error(f"Error fetching data: {str(e)}")
    else:
        st.warning("Please select at least 2 assets to continue")

# Efficient Frontier Tab
with tab2:
    st.header("Efficient Frontier")
    
    if st.session_state.returns is not None and st.session_state.selected_tickers:
        # Display historical statistics
        st.subheader("Historical Statistics")
        
        annual_returns = st.session_state.returns.mean() * 252 * 100  # Convert to percentages
        annual_volatility = st.session_state.returns.std() * np.sqrt(252) * 100  # Convert to percentages
        
        stats_df = pd.DataFrame({
            'Annualized Return (%)': annual_returns,
            'Annualized Volatility (%)': annual_volatility
        })
        
        st.dataframe(stats_df.style.format('{:.2f}'))
        
        # Allow user to set custom expected returns
        st.subheader("Set Expected Returns")
        st.markdown("Set your own expected annual returns for each asset or use historical values.")
        
        # Create columns for expected returns inputs
        expected_returns_dict = {}
        col1, col2 = st.columns(2)
        
        for i, ticker in enumerate(st.session_state.selected_tickers):
            with col1 if i < len(st.session_state.selected_tickers) / 2 else col2:
                hist_return = float(annual_returns[ticker])
                expected_returns_dict[ticker] = st.number_input(
                    f"Expected Return for {ticker} (%)",
                    value=float(st.session_state.expected_returns.get(ticker, hist_return/100) * 100),
                    format="%.2f"
                ) / 100  # Convert from percentage to decimal
        
        # Update session state
        st.session_state.expected_returns = expected_returns_dict
        
        # Number of points for efficient frontier
        num_points = st.slider("Number of points on efficient frontier", 10, 100, 50)
        
        # Calculate efficient frontier
        if st.button("Calculate Efficient Frontier"):
            with st.spinner("Calculating efficient frontier..."):
                expected_returns_array = np.array([expected_returns_dict[ticker] for ticker in st.session_state.selected_tickers])
                
                efficient_portfolios = calculate_efficient_frontier(
                    st.session_state.returns,
                    st.session_state.cov_matrix,
                    expected_returns_array,
                    points=num_points
                )
                
                st.session_state.efficient_portfolios = efficient_portfolios
                
                st.success("Efficient frontier calculated! See the plot below.")
        
        # Display efficient frontier
        if st.session_state.efficient_portfolios:
            st.subheader("Efficient Frontier Plot")
            
            # Get matplotlib figure
            fig = plot_efficient_frontier(
                st.session_state.efficient_portfolios,
                st.session_state.returns,
                st.session_state.selected_tickers,
                np.array([expected_returns_dict[ticker] for ticker in st.session_state.selected_tickers])
            )
            
            # Convert to plotly for better interactivity
            returns_list = [p[0] for p in st.session_state.efficient_portfolios]
            volatility_list = [p[1] for p in st.session_state.efficient_portfolios]
            
            # Create plotly figure
            fig_plotly = go.Figure()
            
            # Add efficient frontier line
            fig_plotly.add_trace(
                go.Scatter(
                    x=volatility_list,
                    y=returns_list,
                    mode='lines',
                    name='Efficient Frontier',
                    line=dict(color='blue', width=3)
                )
            )
            
            # Add individual assets
            asset_volatility = np.sqrt(np.diag(st.session_state.cov_matrix))
            asset_returns = np.array([expected_returns_dict[ticker] for ticker in st.session_state.selected_tickers])
            
            fig_plotly.add_trace(
                go.Scatter(
                    x=asset_volatility,
                    y=asset_returns,
                    mode='markers+text',
                    name='Individual Assets',
                    marker=dict(color='red', size=12),
                    text=st.session_state.selected_tickers,
                    textposition="top center"
                )
            )
            
            # Find maximum Sharpe ratio portfolio
            expected_returns_array = np.array([expected_returns_dict[ticker] for ticker in st.session_state.selected_tickers])
            max_sharpe_weights = maximum_sharpe_portfolio(
                st.session_state.returns,
                st.session_state.cov_matrix,
                expected_returns_array
            )
            max_sharpe_return, max_sharpe_vol, _ = calculate_portfolio_performance(
                max_sharpe_weights,
                st.session_state.returns,
                st.session_state.cov_matrix,
                expected_returns_array
            )
            # หาพอร์ตที่มีความผันผวนน้อยที่สุด (Minimum Volatility Portfolio)
            min_vol_weights = minimum_volatility_portfolio(st.session_state.cov_matrix)
            min_vol_return, min_vol_vol, _ = calculate_portfolio_performance(
                min_vol_weights,
                st.session_state.returns,
                st.session_state.cov_matrix,
                expected_returns_array
            )
            # Equal weight portfolio
            equal_weights = np.ones(len(st.session_state.selected_tickers)) / len(st.session_state.selected_tickers)
            equal_return, equal_vol, _ = calculate_portfolio_performance(
                equal_weights,
                st.session_state.returns,
                st.session_state.cov_matrix,
                expected_returns_array
            )
            
            # Add maximum Sharpe ratio point
            fig_plotly.add_trace(
                go.Scatter(
                    x=[max_sharpe_vol],
                    y=[max_sharpe_return],
                    mode='markers',
                    name='Maximum Sharpe Ratio',
                    marker=dict(color='green', size=15, symbol='star')
                )
            )
            
            # Add equal weight portfolio
            fig_plotly.add_trace(
                go.Scatter(
                    x=[equal_vol],
                    y=[equal_return],
                    mode='markers',
                    name='Equal Weight Portfolio',
                    marker=dict(color='purple', size=12, symbol='diamond')
                )
            )
                        # เพิ่มจุด Minimum Volatility Portfolio ลงในกราฟ
            fig_plotly.add_trace(
                go.Scatter(
                    x=[min_vol_vol],
                    y=[min_vol_return],
                    mode='markers',
                    name='Minimum Volatility Portfolio',
                    marker=dict(color='orange', size=15, symbol='star')
                )
            )
            
            # Update layout
            fig_plotly.update_layout(
                title='Portfolio Efficient Frontier',
                xaxis_title='Volatility (Standard Deviation)',
                yaxis_title='Expected Return',
                height=600,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                )
            )
            
            # Convert axis values to percentages
            fig_plotly.update_layout(
                xaxis=dict(
                    tickformat='.1%',
                    title='Volatility (Standard Deviation)'
                ),
                yaxis=dict(
                    tickformat='.1%',
                    title='Expected Return'
                )
            )
            
            st.plotly_chart(fig_plotly, use_container_width=True)
            
            # Store optimized portfolios for the next tab
            st.session_state.max_sharpe_weights = max_sharpe_weights
            st.session_state.equal_weights = equal_weights
            st.session_state.min_vol_weights = min_vol_weights
            
            # Store portfolios for analysis
            st.session_state.max_sharpe_portfolio = {
                'weights': max_sharpe_weights,
                'return': max_sharpe_return,
                'volatility': max_sharpe_vol
            }
            
            st.session_state.min_vol__portfolio = {
                'weights': min_vol_weights,
                'return': min_vol_return,
                'volatility': min_vol_vol
            }

            st.session_state.equal_weight_portfolio = {
                'weights': equal_weights,
                'return': equal_return,
                'volatility': equal_vol
            }
            
            st.success("You can now go to the 'Portfolio Analysis' tab to see detailed portfolio allocations.")
            
    else:
        st.info("Please select assets and fetch data in the 'Asset Selection' tab first.")

# Portfolio Analysis Tab
with tab3:
    st.header("Portfolio Analysis")
    
    if st.session_state.returns is not None and st.session_state.selected_tickers:
        # Portfolio selection
        portfolio_option = st.radio(
            "Select portfolio to analyze:",
            ["Maximum Sharpe Ratio Portfolio", "Equal Weight Portfolio", "Min Vol Portfolio","Custom Weights"]
        )
        
        if portfolio_option == "Maximum Sharpe Ratio Portfolio" and hasattr(st.session_state, 'max_sharpe_weights'):
            weights = st.session_state.max_sharpe_weights
            portfolio_type = "Maximum Sharpe Ratio Portfolio"
        
        elif portfolio_option == "Equal Weight Portfolio" and hasattr(st.session_state, 'equal_weights'):
            weights = st.session_state.equal_weights
            portfolio_type = "Equal Weight Portfolio"
        
        elif portfolio_option == "Min Vol Portfolio" and hasattr(st.session_state,'min_vol_weights'):
            weights = st.session_state.min_vol_weights
            portfolio_type = "Min Vol Portfolio"

        elif portfolio_option == "Custom Weights":
            st.subheader("Set Custom Weights")
            
            # Create sliders for each asset
            custom_weights = {}
            for ticker in st.session_state.selected_tickers:
                custom_weights[ticker] = st.slider(
                    f"Weight for {ticker} (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=100.0 / len(st.session_state.selected_tickers),
                    step=1.0
                ) / 100.0  # Convert to decimal
            
            # Normalize weights to sum to 1
            total_weight = sum(custom_weights.values())
            
            if abs(total_weight - 1.0) > 0.01:  # If weights don't sum to ~1
                st.warning(f"Total weight is {total_weight:.2%}. Weights will be normalized to sum to 100%.")
                normalized_weights = {k: v/total_weight for k, v in custom_weights.items()}
                custom_weights = normalized_weights
            
            weights = np.array([custom_weights[ticker] for ticker in st.session_state.selected_tickers])
            portfolio_type = "Custom Portfolio"
        
        else:
            st.warning("Please calculate the efficient frontier in the previous tab first.")
            st.stop()
        
        # Analyze the selected portfolio
        expected_returns_array = np.array([st.session_state.expected_returns[ticker] for ticker in st.session_state.selected_tickers])
        
        portfolio_results = analyze_portfolio(
            weights,
            st.session_state.returns,
            st.session_state.cov_matrix,
            st.session_state.selected_tickers,
            expected_returns_array
        )
        
        # Display results
        st.subheader(f"{portfolio_type} Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Expected Annual Return", f"{portfolio_results['portfolio_return']:.2%}")
        with col2:
            st.metric("Annual Volatility", f"{portfolio_results['portfolio_volatility']:.2%}")
        with col3:
            st.metric("Sharpe Ratio", f"{portfolio_results['sharpe_ratio']:.2f}")
        
        # Portfolio allocation
        st.subheader("Portfolio Allocation")
        
        # Format the allocation table
        allocation_df = portfolio_results['allocation'].copy()
        allocation_df['Weight'] = allocation_df['Weight'] * 100  # Convert to percentage
        
        # Sort by weight descending
        allocation_df = allocation_df.sort_values('Weight', ascending=False)
        
        # Display the table
        st.dataframe(allocation_df.style.format({'Weight': '{:.2f}%'}))
        
        # Create a pie chart
        fig = px.pie(
            allocation_df,
            values='Weight',
            names='Asset',
            title=f'{portfolio_type} Allocation',
            hole=0.4
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Historical performance simulation
        st.subheader("Historical Performance Simulation")
        
        historical_returns = st.session_state.returns
        
        # Calculate historical portfolio performance
        portfolio_historical = (historical_returns * weights).sum(axis=1)
        benchmark_weights = {'ACWI': 0.6, 'BOND': 0.4}
        missing_assets = [asset for asset in benchmark_weights if asset not in historical_returns.columns]
        if missing_assets:
            st.warning(f"ไม่พบสินทรัพย์เบนช์มาร์ค: {', '.join(missing_assets)} ในข้อมูลย้อนหลัง")
            benchmark_historical = None
        else:
            benchmark_historical = (
                historical_returns['ACWI'] * benchmark_weights['ACWI'] +
                historical_returns['BOND'] * benchmark_weights['BOND']
            )
        benchmark_cumulative = (1 + benchmark_historical).cumprod()        
        cumulative_returns = (1 + portfolio_historical).cumprod()
        
        # Plot cumulative returns
        fig = px.line(
            cumulative_returns,
            title=f'Historical Cumulative Returns of {portfolio_type}',
            labels={'value': 'Cumulative Return', 'index': 'Date'},
        )
        
        # เพิ่ม benchmark line ถ้ามีข้อมูล
        if benchmark_cumulative is not None:
            fig.add_trace(
                go.Scatter(
                    x=benchmark_cumulative.index,
                    y=benchmark_cumulative.values,
                    mode='lines',
                    name='Benchmark (60% ACWI, 40% BOND)',
                    line=dict(color='black', dash='solid')
                )
            )
        
        # Format y-axis to show as percentages
        fig.update_layout(
            yaxis=dict(tickformat='.2f'),
            height=500,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display risk metrics
        st.subheader("Risk Metrics")
        
        # Calculate metrics
        daily_returns = portfolio_historical
        annual_return = daily_returns.mean() * 252
        annual_volatility = daily_returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / annual_volatility
        
        # Calculate drawdowns
        cum_returns = (1 + daily_returns).cumprod()
        running_max = cum_returns.cummax()
        drawdown = (cum_returns / running_max) - 1
        max_drawdown = drawdown.min()
        
        # Calculate benchmark drawdowns if available
        if benchmark_historical is not None:
            benchmark_cum_returns = (1 + benchmark_historical).cumprod()
            benchmark_running_max = benchmark_cum_returns.cummax()
            benchmark_drawdown = (benchmark_cum_returns / benchmark_running_max) - 1
            benchmark_max_drawdown = benchmark_drawdown.min()
        
        # Calculate value at risk (VaR)
        var_95 = np.percentile(daily_returns, 5)
        var_99 = np.percentile(daily_returns, 1)
        
        metrics = {
            'Annual Return': f"{annual_return:.2%}",
            'Annual Volatility': f"{annual_volatility:.2%}",
            'Sharpe Ratio': f"{sharpe_ratio:.2f}",
            'Maximum Drawdown': f"{max_drawdown:.2%}",
            'Value at Risk (95%)': f"{var_95:.2%}",
            'Value at Risk (99%)': f"{var_99:.2%}"
        }
        
        
        metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
        st.dataframe(metrics_df, use_container_width=True)
        
        # Plot drawdowns
        fig = px.line(
            drawdown,
            title='Portfolio Drawdowns',
            labels={'value': 'Drawdown', 'index': 'Date'}
        )
        
        # Add benchmark drawdowns if available
        if benchmark_historical is not None:
            fig.add_trace(
                go.Scatter(
                    x=benchmark_drawdown.index,
                    y=benchmark_drawdown.values,
                    mode='lines',
                    name='Benchmark Drawdown',
                    line=dict(color='black', dash='solid')
                )
            )
        
        fig.update_layout(
            yaxis=dict(tickformat='.1%'),
            height=400,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Please select assets and calculate the efficient frontier in the previous tabs first.")