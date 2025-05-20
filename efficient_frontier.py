import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from datetime import datetime, timedelta

def download_data(tickers, start_date, end_date):
    """
    Download historical price data for given tickers
    """
    data = yf.download(tickers, start=start_date, end=end_date)['Close']
    # If only one ticker, data will be a Series, convert to DataFrame
    if len(tickers) == 1:
        data = pd.DataFrame(data)
        data.columns = tickers
    return data

def calculate_returns(prices):
    """
    Calculate daily returns from price data
    """
    returns = prices.pct_change().dropna()
    return returns

def calculate_cov_matrix(returns):
    """
    Calculate covariance matrix of returns
    """
    return returns.cov() * 252  # Annualize by multiplying by trading days

def calculate_portfolio_performance(weights, returns, cov_matrix, expected_returns=None):
    """
    Calculate portfolio performance metrics
    
    Parameters:
    weights: Array of weights for each asset
    returns: DataFrame of historical returns
    cov_matrix: Covariance matrix
    expected_returns: Array of expected returns, if None use historical means
    
    Returns:
    portfolio_return, portfolio_volatility, sharpe_ratio
    """
    weights = np.array(weights)
    
    # Use provided expected returns or calculate from historical data
    if expected_returns is None:
        expected_returns = returns.mean() * 252  # Annualize
    else:
        expected_returns = np.array(expected_returns)
    
    portfolio_return = np.sum(weights * expected_returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    # Assuming risk-free rate of 0 for simplicity
    sharpe_ratio = portfolio_return / portfolio_volatility
    
    return portfolio_return, portfolio_volatility, sharpe_ratio

def negative_sharpe(weights, returns, cov_matrix, expected_returns=None):
    """
    Returns negative Sharpe ratio to be minimized
    """
    return -calculate_portfolio_performance(weights, returns, cov_matrix, expected_returns)[2]

def portfolio_volatility(weights, cov_matrix):
    """
    Calculate portfolio volatility (standard deviation)
    """
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

def minimize_volatility(target_return, expected_returns, cov_matrix, num_assets):
    """
    Find portfolio with minimum volatility for a given target return
    """
    # Initial guess (equal weights)
    init_weights = np.ones(num_assets) / num_assets
    
    # Constraint: weights sum to 1
    constraints = (
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
        {'type': 'eq', 'fun': lambda x: np.sum(x * expected_returns) - target_return}
    )
    
    # Bounds: no short selling
    bounds = tuple((0, 1) for _ in range(num_assets))
    
    result = minimize(
        lambda x: portfolio_volatility(x, cov_matrix),
        init_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    return result['x']

def maximum_sharpe_portfolio(returns, cov_matrix, expected_returns=None):
    """
    Find portfolio with maximum Sharpe ratio
    """
    num_assets = len(returns.columns)
    
    # Initial guess (equal weights)
    init_weights = np.ones(num_assets) / num_assets
    
    # Constraint: weights sum to 1
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    
    # Bounds: no short selling
    bounds = tuple((0, 1) for _ in range(num_assets))
    
    result = minimize(
        negative_sharpe,
        init_weights,
        args=(returns, cov_matrix, expected_returns),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    return result['x']

def minimum_volatility_portfolio(cov_matrix):
    num_assets = len(cov_matrix)
    init_weights = np.ones(num_assets) / num_assets
    bounds = tuple((0, 1) for _ in range(num_assets))
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    result = minimize(lambda x: portfolio_volatility(x, cov_matrix),
                      init_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x


def calculate_efficient_frontier(returns, cov_matrix, expected_returns=None, points=50):
    """
    Calculate the efficient frontier
    """
    # Get min and max returns from the assets
    if expected_returns is None:
        expected_returns = returns.mean() * 252  # Annualize
    
    min_return = min(expected_returns)
    max_return = max(expected_returns)
    
    # Generate target returns
    target_returns = np.linspace(min_return, max_return, points)
    efficient_portfolios = []
    
    num_assets = len(returns.columns)
    
    for target in target_returns:
        weights = minimize_volatility(target, expected_returns, cov_matrix, num_assets)
        portfolio_return, portfolio_vol, _ = calculate_portfolio_performance(
            weights, returns, cov_matrix, expected_returns
        )
        efficient_portfolios.append([portfolio_return, portfolio_vol, weights])
    
    return efficient_portfolios

def plot_efficient_frontier(efficient_portfolios, returns, asset_names, expected_returns=None, show_assets=True):
    """
    Plot the efficient frontier and assets
    """
    returns_list = [p[0] for p in efficient_portfolios]
    volatility_list = [p[1] for p in efficient_portfolios]
    
    # Create plot
    plt.figure(figsize=(12, 8))
    plt.plot(volatility_list, returns_list, 'b-', linewidth=3, label='Efficient Frontier')
    
    # If expected_returns is None, use historical
    if expected_returns is None:
        expected_returns = returns.mean() * 252
    
    # Plot individual assets
    if show_assets:
        asset_volatility = np.sqrt(np.diag(returns.cov() * 252))
        plt.scatter(
            asset_volatility, 
            expected_returns,
            s=200, 
            c='red', 
            marker='o', 
            label='Individual Assets'
        )
        
        # Add labels for individual assets
        for i, txt in enumerate(asset_names):
            plt.annotate(
                txt, 
                (asset_volatility[i], expected_returns[i]),
                xytext=(10, 0), 
                textcoords='offset points'
            )
    
    # Find maximum Sharpe ratio portfolio
    max_sharpe_weights = maximum_sharpe_portfolio(returns, returns.cov() * 252, expected_returns)
    max_sharpe_return, max_sharpe_vol, _ = calculate_portfolio_performance(
        max_sharpe_weights, returns, returns.cov() * 252, expected_returns
    )
    # หาพอร์ตที่มีความผันผวนน้อยที่สุด (Minimum Volatility Portfolio)
    min_vol_weights = minimum_volatility_portfolio(returns.cov() * 252)
    min_vol_return, min_vol_vol, _ = calculate_portfolio_performance(
        min_vol_weights, returns, returns.cov() * 252, expected_returns
    )
    
    # Equal weight portfolio
    equal_weights = np.ones(len(returns.columns)) / len(returns.columns)
    equal_return, equal_vol, _ = calculate_portfolio_performance(
        equal_weights, returns, returns.cov() * 252, expected_returns
    )
    
    # Plot maximum Sharpe ratio point
    plt.scatter(
        max_sharpe_vol, 
        max_sharpe_return, 
        s=300, 
        c='green', 
        marker='*', 
        label='Maximum Sharpe Ratio'
    )
    
    # Plot equal weight portfolio
    plt.scatter(
        equal_vol, 
        equal_return, 
        s=300, 
        c='purple', 
        marker='d', 
        label='Equal Weight Portfolio'
    )
        # แสดงจุด Minimum Volatility Portfolio บนกราฟ
    plt.scatter(
        min_vol_vol,
        min_vol_return,
        s=300,
        c='orange',
        marker='*',
        label='Minimum Volatility Portfolio'
    )
    
    plt.title('Efficient Frontier', fontsize=20)
    plt.xlabel('Volatility (Standard Deviation)', fontsize=16)
    plt.ylabel('Expected Return', fontsize=16)
    plt.legend(fontsize=14)
    plt.grid(True)
    
    return plt

def get_portfolio_allocation(weights, asset_names):
    """
    Get portfolio allocation as a DataFrame
    """
    return pd.DataFrame({
        'Asset': asset_names,
        'Weight': weights
    })

def analyze_portfolio(weights, returns, cov_matrix, asset_names, expected_returns=None):
    """
    Perform detailed analysis of a specific portfolio
    """
    portfolio_return, portfolio_vol, sharpe = calculate_portfolio_performance(
        weights, returns, cov_matrix, expected_returns
    )
    
    allocation = get_portfolio_allocation(weights, asset_names)
    
    results = {
        'portfolio_return': portfolio_return,
        'portfolio_volatility': portfolio_vol,
        'sharpe_ratio': sharpe,
        'allocation': allocation
    }
    
    return results

def get_mag7_tickers():
    """
    Return list of MAG7 tickers
    """
    return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA']

