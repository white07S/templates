import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.tsa.stattools as ts
import statsmodels.stats.diagnostic as sm_diag
from scipy import stats
from scipy.stats import norm
import os
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# Comprehensive Financial Time Series EDA
# =============================================================================

# This script performs a detailed EDA of financial data for the prediction pipeline.
# The analysis includes data quality assessment, advanced time series analysis
# and risk/return metrics. The insights will inform feature engineering and enhance 
# risk management. Financial rationale is embedded in each step to maximize alpha.


def load_and_clean_data(file_path):
    """
    Load data from a CSV file using financial-aware parsing.
    The Date column is parsed as dates. Forward fill missing values then drop remaining NAs.
    Outliers are capped using 3 * MAD based robustly.
    """
    try:
        df = pd.read_csv(file_path, parse_dates=['Date'])
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

    # Sort by Date for time series consistency
    df = df.sort_values('Date').reset_index(drop=True)

    # Forward fill missing values, then drop remaining NA
    df_ffill = df.fillna(method='ffill').dropna()

    # Identify numerical columns to cap outliers
    num_cols = df_ffill.select_dtypes(include=[np.number]).columns.tolist()
    for col in num_cols:
        try:
            # Using median and MAD for robust outlier detection
            median_val = df_ffill[col].median()
            mad = np.median(np.abs(df_ffill[col] - median_val))
            if mad == 0:
                # Avoid division by zero
                mad = df_ffill[col].std()
            robust_std = 1.4826 * mad
            # Cap using median +- 3 * robust_std
            upper_bound = median_val + 3 * robust_std
            lower_bound = median_val - 3 * robust_std
            df_ffill[col] = np.where(df_ffill[col] > upper_bound, upper_bound,
                                np.where(df_ffill[col] < lower_bound, lower_bound, df_ffill[col]))
        except Exception as e:
            print(f"Error processing column {col}: {e}")
    return df_ffill


# =============================================================================
# Data Quality Assessment
# =============================================================================

def data_quality_assessment(df, dataset_name=''):
    summary = []
    summary.append(f"Dataset: {dataset_name}")
    # Basic info
    summary.append(f"Shape: {df.shape}")
    summary.append(f"Columns: {list(df.columns)}")
    summary.append("Data Types:")
    summary.append(df.dtypes.to_string())

    # Missing value analysis
    missing = df.isnull().sum()
    summary.append("Missing values per column:")
    summary.append(missing.to_string())

    # Heatmap of missing values
    try:
        plt.figure(figsize=(10, 6))
        sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
        plt.title(f'Missing Data Heatmap: {dataset_name}')
        plt.savefig(f'{dataset_name}_missing_heatmap.png', dpi=300)
        plt.close()
    except Exception as e:
        print(f"Plotting error in missing heatmap: {e}")

    # Outlier detection summary using descriptive stats and z-scores
    desc_stats = df.describe().to_string()
    summary.append("Descriptive Statistics:")
    summary.append(desc_stats)

    for col in df.select_dtypes(include=[np.number]).columns:
        try:
            # z-score method
            z_scores = np.abs(stats.zscore(df[col]))
            outlier_count = np.sum(z_scores > 3)
            summary.append(f"Column {col} has {outlier_count} outliers based on z-score >3")
        except Exception as e:
            summary.append(f"Error in outlier detection for {col}: {e}")

    # Temporal consistency check
    if 'Date' in df.columns:
        date_diff = df['Date'].diff().dropna()
        summary.append("Temporal consistency check - Statistics of Date differences:")
        summary.append(date_diff.describe().to_string())

    return "\n".join(summary)


# =============================================================================
# Financial Time Series Analysis
# =============================================================================

def stationarity_tests(series, col_name=''):
    test_results = []
    try:
        adf_result = ts.adfuller(series.dropna())
        test_results.append(f"ADF Statistic for {col_name}: {adf_result[0]:.4f}, p-value: {adf_result[1]:.4f}")
    except Exception as e:
        test_results.append(f"ADF test error: {e}")

    try:
        kpss_result = ts.kpss(series.dropna(), regression='c')
        test_results.append(f"KPSS Statistic for {col_name}: {kpss_result[0]:.4f}, p-value: {kpss_result[1]:.4f}")
    except Exception as e:
        test_results.append(f"KPSS test error: {e}")

    try:
        pp_result = ts.phillips_perron(series.dropna())
        test_results.append(f"Phillips-Perron Statistic for {col_name}: {pp_result[0]:.4f}, p-value: {pp_result[1]:.4f}")
    except Exception as e:
        test_results.append(f"PP test error: {e}")
    return test_results


def chow_test(data, split_index, dep_var, regressors):
    """
    Implement a simple Chow test by splitting the data at split_index.
    """
    try:
        df1 = data.iloc[:split_index]
        df2 = data.iloc[split_index:]
        X1 = sm.add_constant(df1[regressors])
        X2 = sm.add_constant(df2[regressors])
        y1 = df1[dep_var]
        y2 = df2[dep_var]

        # Fit the models
        model1 = sm.OLS(y1, X1).fit()
        model2 = sm.OLS(y2, X2).fit()
        # Combined model
        X = sm.add_constant(data[regressors])
        y = data[dep_var]
        full_model = sm.OLS(y, X).fit()

        # Sum of squared residuals
        SSR_pooled = full_model.ssr
        SSR1 = model1.ssr
        SSR2 = model2.ssr
        k = X.shape[1]
        n1 = len(df1)
        n2 = len(df2)
        chow_stat = ((SSR_pooled - (SSR1 + SSR2)) / k) / (((SSR1 + SSR2)/(n1+n2-2*k)))
        p_value = 1 - stats.f.cdf(chow_stat, k, n1+n2-2*k)
        return chow_stat, p_value
    except Exception as e:
        print(f"Chow test error: {e}")
        return None, None


def regime_detection_hmm(series):
    """
    Regime identification using Markov Switching Model (HMM).
    Uses statsmodels MarkovRegression for regime switching.
    """
    try:
        mod = sm.tsa.MarkovRegression(series, k_regimes=2, trend='c', switching_variance=True)
        res = mod.fit(em_iter=50, search_reps=20)
        # Plot smoothed probabilities
        regime_prob = res.smoothed_marginal_probabilities[0]
        plt.figure(figsize=(10, 4))
        plt.plot(series.index, regime_prob, label='Regime 0 probability')
        plt.title('Regime Probability via Markov Switching Model')
        plt.legend()
        plt.savefig('regime_probability.png', dpi=300)
        plt.close()
        return res, regime_prob
    except Exception as e:
        print(f"Regime detection error: {e}")
        return None, None


def volatility_clustering(series):
    """
    Analyze volatility clustering using rolling window statistics and ARCH effect test.
    """
    try:
        rolling_window = 20
        rolling_vol = series.rolling(rolling_window).std()
        plt.figure(figsize=(10,4))
        plt.plot(series.index, series, alpha=0.5, label='Series')
        plt.plot(rolling_vol.index, rolling_vol, color='red', label=f'{rolling_window}-day Rolling Volatility')
        plt.title('Volatility Clustering Analysis')
        plt.legend()
        plt.savefig('volatility_clustering.png', dpi=300)
        plt.close()

        # ARCH effect test
        arch_test = sm_diag.het_arch(series.dropna())
        arch_pvalue = arch_test[1]
        return rolling_vol, arch_pvalue
    except Exception as e:
        print(f"Volatility clustering error: {e}")
        return None, None


def seasonal_decomposition(series):
    """
    Seasonal decomposition using statsmodels seasonal_decompose.
    Financial data may have weekly patterns. Assume frequency=5 for trading days.
    """
    try:
        decomposition = sm.tsa.seasonal_decompose(series.dropna(), model='additive', period=5)
        fig = decomposition.plot()
        fig.set_size_inches(10, 8)
        plt.suptitle('Seasonal Decomposition')
        plt.savefig('seasonal_decomposition.png', dpi=300)
        plt.close()
        return decomposition
    except Exception as e:
        print(f"Seasonal decomposition error: {e}")
        return None


def rolling_statistics(series):
    """
    Calculate rolling mean, volatility, skewness, and kurtosis.
    """
    try:
        roll_mean = series.rolling(window=20).mean()
        roll_vol = series.rolling(window=20).std()
        roll_skew = series.rolling(window=20).apply(lambda x: stats.skew(x), raw=False)
        roll_kurt = series.rolling(window=20).apply(lambda x: stats.kurtosis(x), raw=False)
        # Plot rolling statistics
        plt.figure(figsize=(12,8))
        plt.subplot(2,2,1)
        plt.plot(roll_mean, label='Rolling Mean')
        plt.title('Rolling Mean')
        plt.legend()

        plt.subplot(2,2,2)
        plt.plot(roll_vol, label='Rolling Volatility', color='orange')
        plt.title('Rolling Volatility')
        plt.legend()

        plt.subplot(2,2,3)
        plt.plot(roll_skew, label='Rolling Skewness', color='green')
        plt.title('Rolling Skewness')
        plt.legend()

        plt.subplot(2,2,4)
        plt.plot(roll_kurt, label='Rolling Kurtosis', color='red')
        plt.title('Rolling Kurtosis')
        plt.legend()

        plt.tight_layout()
        plt.savefig('rolling_statistics.png', dpi=300)
        plt.close()

        return roll_mean, roll_vol, roll_skew, roll_kurt
    except Exception as e:
        print(f"Rolling statistics error: {e}")
        return None, None, None, None


# =============================================================================
# Risk and Return Analysis
# =============================================================================

def return_distribution_analysis(returns):
    summary = []
    try:
        mean_ret = returns.mean()
        std_ret = returns.std()
        skew_ret = stats.skew(returns.dropna())
        kurt_ret = stats.kurtosis(returns.dropna())
        summary.append(f"Return Mean: {mean_ret:.6f}, Std: {std_ret:.6f}, Skew: {skew_ret:.4f}, Kurtosis: {kurt_ret:.4f}")

        # Plot histogram with density
        plt.figure(figsize=(8,5))
        sns.histplot(returns, kde=True, bins=30, color='blue')
        plt.title('Return Distribution')
        plt.savefig('return_distribution.png', dpi=300)
        plt.close()
    except Exception as e:
        summary.append(f"Return distribution error: {e}")
    return summary


def tail_risk_analysis(returns, alpha=0.05):
    """
    Calculate Value at Risk (VaR) and Conditional VaR (CVaR)
    """
    try:
        var = np.percentile(returns.dropna(), 100*alpha)
        cvar = returns[returns <= var].mean()
        return var, cvar
    except Exception as e:
        print(f"Tail risk analysis error: {e}")
        return None, None


def sharpe_ratio_evolution(returns, window=20, risk_free_rate=0.0):
    """
    Calculate rolling Sharpe ratio. Annualize if needed, but here assume daily metrics.
    """
    try:
        excess_returns = returns - risk_free_rate
        rolling_sharpe = excess_returns.rolling(window).mean() / excess_returns.rolling(window).std()
        plt.figure(figsize=(10,4))
        plt.plot(rolling_sharpe, label='Rolling Sharpe Ratio')
        plt.title('Rolling Sharpe Ratio Evolution')
        plt.legend()
        plt.savefig('rolling_sharpe_ratio.png', dpi=300)
        plt.close()
        return rolling_sharpe
    except Exception as e:
        print(f"Sharpe ratio error: {e}")
        return None


def drawdown_analysis(series):
    """
    Drawdown analysis: Compute cumulative returns, drawdown and maximum drawdown.
    """
    try:
        cumulative = (1 + series).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        plt.figure(figsize=(10,4))
        plt.plot(drawdown, label='Drawdown')
        plt.title('Drawdown Analysis')
        plt.legend()
        plt.savefig('drawdown_analysis.png', dpi=300)
        plt.close()
        return drawdown, max_drawdown
    except Exception as e:
        print(f"Drawdown analysis error: {e}")
        return None, None


def correlation_and_pca(returns_df):
    """
    Compute correlation matrix, plot hierarchical clustering heatmap and perform PCA (via SVD).
    """
    summary = []
    try:
        corr_matrix = returns_df.corr()
        plt.figure(figsize=(8,6))
        sns.clustermap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm')
        plt.title('Correlation Matrix with Hierarchical Clustering')
        plt.savefig('correlation_clustermap.png', dpi=300)
        plt.close()

        # PCA using SVD decomposition
        # Standardize the data
        X = returns_df.dropna().values
        X_standardized = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        U, s, Vt = np.linalg.svd(X_standardized, full_matrices=False)
        explained_variance = (s**2) / (X_standardized.shape[0]-1)
        total_variance = explained_variance.sum()
        ratio = explained_variance / total_variance
        summary.append('PCA Explained Variance Ratios: ' + str(ratio))
        return summary, corr_matrix, ratio
    except Exception as e:
        summary.append(f"Correlation/PCA error: {e}")
        return summary, None, None


# =============================================================================
# Advanced Visualization: Correlation, Time Series, and Distribution Plots
# =============================================================================

def advanced_visualizations(df):
    try:
        # Time series plot for Close price with regimes if available
        plt.figure(figsize=(12,6))
        plt.plot(df['Date'], df['Close'], label='Close Price')
        plt.title('Close Price Time Series')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.savefig('close_price_timeseries.png', dpi=300)
        plt.close()

        # Rolling correlation between Open and Close
        try:
            rolling_corr = df['Open'].rolling(window=20).corr(df['Close'])
            plt.figure(figsize=(10,4))
            plt.plot(df['Date'], rolling_corr, label='Rolling Correlation (Open, Close)')
            plt.title('Rolling Correlation between Open and Close')
            plt.legend()
            plt.savefig('rolling_correlation.png', dpi=300)
            plt.close()
        except Exception as e:
            print(f"Rolling correlation plot error: {e}")

        # Distribution plot with KDE for Volume
        plt.figure(figsize=(8,5))
        sns.histplot(df['Volume'], kde=True, bins=30, color='purple')
        plt.title('Volume Distribution')
        plt.savefig('volume_distribution.png', dpi=300)
        plt.close()
    except Exception as e:
        print(f"Advanced visualization error: {e}")


# =============================================================================
# Main Execution
# =============================================================================

def main():
    eda_summary_lines = []

    # Load datasets
    datasets = {}
    for split in ['train', 'val', 'test']:
        file_name = split + '.csv'
        df = load_and_clean_data(file_name)
        if df is not None:
            datasets[split] = df
            # Save cleaned data
            clean_file = split + '_clean.csv'
            df.to_csv(clean_file, index=False)
            eda_summary_lines.append(f"{split.upper()} data cleaned and saved as {clean_file}. Shape: {df.shape}")
        else:
            eda_summary_lines.append(f"Failed to load {file_name}")

    # Data Quality Assessment for train set (as representative)
    if 'train' in datasets:
        dq_summary = data_quality_assessment(datasets['train'], dataset_name='TRAIN')
        eda_summary_lines.append('Data Quality Assessment for TRAIN dataset:')
        eda_summary_lines.append(dq_summary)
    else:
        eda_summary_lines.append('TRAIN dataset not available for Data Quality Assessment.')

    # Financial Time Series Analysis on TRAIN dataset
    if 'train' in datasets and 'Date' in datasets['train'].columns:
        train_df = datasets['train']
        train_df.set_index('Date', inplace=True)

        # Stationarity Tests on Close Price
        if 'Close' in train_df.columns:
            st_results = stationarity_tests(train_df['Close'], col_name='Close')
            eda_summary_lines.append('Stationarity Test Results for Close Price:')
            eda_summary_lines.extend(st_results)

        # Structural Break Detection using Chow Test on Close Price vs Open
        if 'Close' in train_df.columns and 'Open' in train_df.columns:
            # Use mid point splitting for demonstration
            split_idx = int(len(train_df)/2)
            chow_stat, chow_p = chow_test(train_df.reset_index(), split_idx, 'Close', ['Open'])
            eda_summary_lines.append(f'Chow Test Statistic: {chow_stat}, p-value: {chow_p}')

        # Regime Detection with Hidden Markov Model on Close
        if 'Close' in train_df.columns:
            hmm_res, regime_prob = regime_detection_hmm(train_df['Close'])
            if hmm_res is not None:
                eda_summary_lines.append('Regime detection via Markov Switching Model completed.')

        # Volatility Clustering and ARCH Effects
        if 'Close' in train_df.columns:
            rolling_vol, arch_pvalue = volatility_clustering(train_df['Close'])
            eda_summary_lines.append(f'ARCH test p-value: {arch_pvalue}')

        # Seasonal Decomposition on Close Price
        if 'Close' in train_df.columns:
            decomposition = seasonal_decomposition(train_df['Close'])
            if decomposition is not None:
                eda_summary_lines.append('Seasonal decomposition completed for Close Price.')

        # Rolling Statistics
        roll_mean, roll_vol, roll_skew, roll_kurt = rolling_statistics(train_df['Close'])
        eda_summary_lines.append('Rolling statistics computed for Close Price.')

        # Risk and Return Analysis
        # Analyzing Target_return if available, else compute returns from Close prices
        if 'Target_return' in train_df.columns:
            returns = train_df['Target_return']
        else:
            returns = train_df['Close'].pct_change()
        ret_dist_summary = return_distribution_analysis(returns)
        eda_summary_lines.extend(ret_dist_summary)

        var, cvar = tail_risk_analysis(returns)
        eda_summary_lines.append(f'VaR at 5%: {var}, CVaR: {cvar}')

        rolling_sharpe = sharpe_ratio_evolution(returns)
        eda_summary_lines.append('Rolling Sharpe Ratio computed.')

        drawdown, max_drawdown = drawdown_analysis(returns)
        eda_summary_lines.append(f'Maximum Drawdown: {max_drawdown}')

        # Cross-sectional correlation analysis and PCA using available numeric features
        returns_df = train_df[['Close', 'Open', 'High', 'Low', 'Volume']].pct_change()
        pca_summary, corr_matrix, pca_ratio = correlation_and_pca(returns_df)
        eda_summary_lines.extend(pca_summary)

        # Advanced Visualizations
        train_df_reset = train_df.reset_index()
        advanced_visualizations(train_df_reset)
        eda_summary_lines.append('Advanced visualizations generated and saved as PNG files.')

    else:
        eda_summary_lines.append('TRAIN dataset or Date column missing for Time Series Analysis.')

    # Combine all summaries
    eda_summary = '\n'.join(eda_summary_lines)
    return eda_summary


if __name__ == '__main__':
    final_summary = main()
    print(final_summary)
    # For PythonTools return variable
    eda_summary = final_summary