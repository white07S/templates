import pandas as pd
import numpy as np
from scipy import stats
from scipy.fft import fft
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from sklearn.preprocessing import RobustScaler
import os

# Set display options for debugging
pd.set_option('mode.chained_assignment', None)

# -------------------------
# Utility and Technical indicator functions
# -------------------------

def compute_RSI(series, window):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=window).mean()
    RS = gain / loss.replace(to_replace=0, method='ffill').replace(0, 1e-10)
    rsi = 100 - (100 / (1 + RS))
    return rsi


def compute_MACD(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist


def compute_WilliamsR(high, low, close, window):
    highest_high = high.rolling(window=window, min_periods=window).max()
    lowest_low = low.rolling(window=window, min_periods=window).min()
    wr = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    return wr


def compute_StochasticOscillator(high, low, close, window, smooth_k=3, smooth_d=3):
    lowest_low = low.rolling(window=window, min_periods=window).min()
    highest_high = high.rolling(window=window, min_periods=window).max()
    fast_k = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
    slow_k = fast_k.rolling(window=smooth_k, min_periods=smooth_k).mean()
    slow_d = slow_k.rolling(window=smooth_d, min_periods=smooth_d).mean()
    return slow_k, slow_d


def compute_ROC(series, window):
    roc = series.diff(window) / series.shift(window)
    return roc


# -------------------------
# Volatility and Trend Indicators
# -------------------------

def compute_BollingerBands(series, window, num_std=2):
    rolling_mean = series.rolling(window=window, min_periods=window).mean()
    rolling_std = series.rolling(window=window, min_periods=window).std()
    upper_band = rolling_mean + num_std * rolling_std
    lower_band = rolling_mean - num_std * rolling_std
    band_width = upper_band - lower_band
    return rolling_mean, upper_band, lower_band, band_width


def compute_ATR(high, low, close, window):
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=window, min_periods=window).mean()
    return atr


def compute_KeltnerChannels(high, low, close, window, atr_window, multiplier=1.5):
    ema = close.ewm(span=window, adjust=False).mean()
    atr = compute_ATR(high, low, close, atr_window)
    upper_channel = ema + multiplier * atr
    lower_channel = ema - multiplier * atr
    return ema, upper_channel, lower_channel


def compute_CCI(high, low, close, window):
    typical_price = (high + low + close) / 3
    sma = typical_price.rolling(window=window, min_periods=window).mean()
    mad = typical_price.rolling(window=window, min_periods=window).apply(lambda x: np.fabs(x - x.mean()).mean(), raw=True)
    cci = (typical_price - sma) / (0.015 * mad + 1e-10)
    return cci


# -------------------------
# Volume based indicators
# -------------------------

def compute_OBV(close, volume):
    direction = np.sign(close.diff()).fillna(0)
    obv = (direction * volume).fillna(0).cumsum()
    return obv


def compute_VPT(close, volume):
    pct_change = close.pct_change().fillna(0)
    vpt = (pct_change * volume).cumsum()
    return vpt


def compute_AD(high, low, close, volume):
    # Accumulation/Distribution Line
    mfm = ((close - low) - (high - close)) / (high - low + 1e-10)
    ad = (mfm * volume).cumsum()
    return ad


def compute_MFI(high, low, close, volume, window):
    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume
    mf_positive = []
    mf_negative = []
    for i in range(len(typical_price)):
        if i == 0:
            mf_positive.append(0)
            mf_negative.append(0)
        else:
            if typical_price.iloc[i] > typical_price.iloc[i-1]:
                mf_positive.append(money_flow.iloc[i])
                mf_negative.append(0)
            else:
                mf_positive.append(0)
                mf_negative.append(money_flow.iloc[i])
    mf_positive = pd.Series(mf_positive, index=typical_price.index)
    mf_negative = pd.Series(mf_negative, index=typical_price.index)
    mfi = 100 - 100 / (1 + mf_positive.rolling(window=window, min_periods=window).sum() /
                         (mf_negative.rolling(window=window, min_periods=window).sum() + 1e-10))
    return mfi


def compute_VWMA(close, volume, window):
    vwma = (close * volume).rolling(window=window, min_periods=window).sum() / (volume.rolling(window=window, min_periods=window).sum()+1e-10)
    return vwma

# -------------------------
# Realized Volatility Estimators
# -------------------------

def garman_klass_vol(high, low, close):
    log_hl = np.log(high/low + 1e-10)
    log_co = np.log(close/close.shift() + 1e-10)
    vk = 0.5 * (log_hl**2) - (2*np.log(2)-1)*(log_co**2)
    return vk.rolling(window=10, min_periods=10).std()


def rogers_satchell_vol(high, low, close):
    log_hc = np.log(high/close + 1e-10)
    log_hl = np.log(high/low + 1e-10)
    log_lc = np.log(low/close + 1e-10)
    rs = (log_hc*(log_hc - log_hl)) + (log_lc*(log_lc - log_hl))
    return rs.rolling(window=10, min_periods=10).std()


def yang_zhang_vol(open, high, low, close):
    # Yang-Zhang volatility estimator
    log_oc = np.log(close/open + 1e-10)
    log_ho = np.log(high/open + 1e-10)
    log_lo = np.log(low/open + 1e-10)
    rs = rogers_satchell_vol(high, low, close)
    # Using a mix of overnight and RogersSatchell
    oc_var = log_oc.rolling(window=10, min_periods=10).var()
    yz = np.sqrt((oc_var + rs**2)/2 + 0.5*oc_var)
    return yz


def parkinson_vol(high, low):
    log_hl = np.log(high/low + 1e-10)
    pv = (1/(4*np.log(2)))*(log_hl**2)
    return pv.rolling(window=10, min_periods=10).std()

# -------------------------
# Alternative Volatility Surface Features
# -------------------------

def volatility_of_vol(vol_series, window=10):
    vol_of_vol = vol_series.rolling(window=window, min_periods=window).std()
    return vol_of_vol


def volatility_clustering(vol_series, window=20):
    # Simple clustering detection using rolling variance
    cluster = vol_series.rolling(window=window, min_periods=window).var()
    return cluster

# -------------------------
# Cross-Asset and Factor Signal Functions
# -------------------------

def dynamic_beta(target_returns, benchmark_returns, window=30):
    # Updated function with safe polyfit implementation
    def calc_beta(x):
        if len(x.dropna()) != window:
            return np.nan
        try:
            bench = benchmark_returns.loc[x.index]
            return np.polyfit(bench, x, 1)[0]
        except Exception:
            return np.nan
    betas = target_returns.rolling(window=window, min_periods=window).apply(calc_beta, raw=False)
    return betas


def rolling_correlation(series1, series2, window=30):
    return series1.rolling(window=window, min_periods=window).corr(series2)

# For factor exposures, create a dummy factor series based on rolling quantiles

def factor_exposure(series, factor_type='momentum', window=30):
    if factor_type == 'momentum':
        exposure = series.pct_change().rolling(window=window, min_periods=window).mean()
    elif factor_type == 'size':
        exposure = series.rolling(window=window, min_periods=window).mean()
    elif factor_type == 'value':
        exposure = 1 / (series.rolling(window=window, min_periods=window).mean()+1e-10)
    else:
        exposure = series
    return exposure

# -------------------------
# Alternative Data Features: Spectral and Information Theory
# -------------------------

def spectral_features(series):
    fft_values = fft(series.fillna(0).values)
    # take magnitude
    mag = np.abs(fft_values)
    # use first few coefficients as features
    features = pd.Series(mag[:5], index=[f'fft_coef_{i}' for i in range(5)])
    return features


def shannon_entropy(series, bins=10):
    hist, bin_edges = np.histogram(series.dropna(), bins=bins, density=True)
    hist = hist + 1e-10  # avoid log(0)
    return -np.sum(hist * np.log(hist))


def mutual_information(x, y, bins=10):
    c_xy, _, _ = np.histogram2d(x.dropna(), y.dropna(), bins=bins)
    p_xy = c_xy / np.sum(c_xy)
    p_x = np.sum(p_xy, axis=1)
    p_y = np.sum(p_xy, axis=0)
    mi = 0.0
    for i in range(len(p_x)):
        for j in range(len(p_y)):
            if p_xy[i, j] > 0:
                mi += p_xy[i, j] * np.log(p_xy[i, j] / (p_x[i]*p_y[j] + 1e-10))
    return mi

# -------------------------
# Feature Engineering Pipeline: Lag Generation and Orthogonalization
# -------------------------

def generate_lags(df, col, lags=[1,2,3]):
    for lag in lags:
        df[f'{col}_lag{lag}'] = df[col].shift(lag)
    return df


def gram_schmidt(df):
    # Orthogonalize features using Gram-Schmidt
    cols = df.columns
    Q = pd.DataFrame(index=df.index)
    for i, col in enumerate(cols):
        qi = df[col].fillna(0)
        for j in range(i):
            qj = Q[cols[j]]
            proj = (qi * qj).sum() / ((qj * qj).sum() + 1e-10)
            qi = qi - proj * qj
        Q[col] = qi
    return Q

# -------------------------
# Main processing function
# -------------------------

def process_features(df):
    # Ensure sorted by date if exists, else index
    if 'Date' in df.columns:
        df = df.sort_values('Date').reset_index(drop=True)

    # Replace fillna calls with ffill and bfill
    df = df.ffill().bfill()

    # Technical indicators suite
    # Assuming df contains columns: 'Open', 'High', 'Low', 'Close', 'Volume'
    for window in [5, 10, 20, 50]:
        df[f'RSI_{window}'] = compute_RSI(df['Close'], window)
    macd, macd_signal, macd_hist = compute_MACD(df['Close'])
    df['MACD'] = macd
    df['MACD_signal'] = macd_signal
    df['MACD_hist'] = macd_hist
    df['Williams_%R'] = compute_WilliamsR(df['High'], df['Low'], df['Close'], 14)
    slow_k, slow_d = compute_StochasticOscillator(df['High'], df['Low'], df['Close'], 14)
    df['Stoch_K'] = slow_k
    df['Stoch_D'] = slow_d
    df['ROC'] = compute_ROC(df['Close'], 12)

    # Volatility Indicators
    ma, upper_bb, lower_bb, bb_width = compute_BollingerBands(df['Close'], 20)
    df['BB_MA'] = ma
    df['BB_upper'] = upper_bb
    df['BB_lower'] = lower_bb
    df['BB_width'] = bb_width
    df['ATR'] = compute_ATR(df['High'], df['Low'], df['Close'], 14)
    ema, kc_upper, kc_lower = compute_KeltnerChannels(df['High'], df['Low'], df['Close'], 20, 14)
    df['KC_MA'] = ema
    df['KC_upper'] = kc_upper
    df['KC_lower'] = kc_lower
    df['CCI'] = compute_CCI(df['High'], df['Low'], df['Close'], 20)

    # Volume Indicators
    df['OBV'] = compute_OBV(df['Close'], df['Volume'])
    df['VPT'] = compute_VPT(df['Close'], df['Volume'])
    df['AD_Line'] = compute_AD(df['High'], df['Low'], df['Close'], df['Volume'])
    df['MFI'] = compute_MFI(df['High'], df['Low'], df['Close'], df['Volume'], 14)
    df['VWMA'] = compute_VWMA(df['Close'], df['Volume'], 20)

    # Advanced Volatility Modeling
    df['GarmanKlass'] = garman_klass_vol(df['High'], df['Low'], df['Close'])
    df['RogersSatchell'] = rogers_satchell_vol(df['High'], df['Low'], df['Close'])
    df['YangZhang'] = yang_zhang_vol(df['Open'], df['High'], df['Low'], df['Close'])
    df['Parkinson'] = parkinson_vol(df['High'], df['Low'])

    # Volatility surface features
    df['Vol_of_Vol'] = volatility_of_vol(df['ATR'], window=10)
    df['Vol_Cluster'] = volatility_clustering(df['ATR'], window=20)

    # Cross-Asset and Factor Signals
    # Simulate a benchmark return series using a rolling median of Close
    df['Benchmark'] = df['Close'].rolling(window=30, min_periods=30).median()
    df['Target_Return'] = df['Close'].pct_change()
    df['Benchmark_Return'] = df['Benchmark'].pct_change()
    df['Dynamic_Beta'] = dynamic_beta(df['Target_Return'], df['Benchmark_Return'], window=30)
    df['Rolling_Corr'] = rolling_correlation(df['Target_Return'], df['Benchmark_Return'], window=30)

    # Factor exposures using dummy calculations
    df['Momentum_Exp'] = factor_exposure(df['Close'], 'momentum', 30)
    df['Size_Exp'] = factor_exposure(df['Volume'], 'size', 30)
    df['Value_Exp'] = factor_exposure(df['Close'], 'value', 30)
    df['Market_Exp'] = factor_exposure(df['Close'], 'market', 30)

    # Alternative Data Features
    # Spectral analysis: Compute FFT features on Close price
    fft_feat = spectral_features(df['Close'])
    for name, value in fft_feat.items():
        df[name] = value  # assigning constant feature for simplicity

    # Information theory features
    df['Shannon_Entropy'] = shannon_entropy(df['Close'])
    # Mutual information between Close and Volume (using rolling windows)
    mi_list = []
    roll_window = 30
    for i in range(len(df)):
        if i < roll_window:
            mi_list.append(np.nan)
        else:
            x = df['Close'].iloc[i-roll_window:i]
            y = df['Volume'].iloc[i-roll_window:i]
            mi_list.append(mutual_information(x, y, bins=10))
    df['Mutual_Info'] = mi_list

    # Feature Engineering Pipeline: Generate lags for selected features
    features_to_lag = ['RSI_5', 'RSI_10', 'RSI_20', 'RSI_50', 'MACD', 'Williams_%R', 'Stoch_K', 'ROC',
                        'BB_width', 'ATR', 'CCI', 'OBV', 'VPT', 'MFI', 'VWMA', 'GarmanKlass',
                        'RogersSatchell', 'YangZhang', 'Parkinson', 'Vol_of_Vol', 'Vol_Cluster',
                        'Dynamic_Beta', 'Rolling_Corr', 'Momentum_Exp', 'Size_Exp', 'Value_Exp',
                        'Market_Exp', 'Shannon_Entropy', 'Mutual_Info']
    for feat in features_to_lag:
        if feat in df.columns:
            df = generate_lags(df, feat, lags=[1,2,3])

    # Fill any remaining NA values
    df = df.ffill().bfill()

    # Remove features with near zero variance (< 0.1)
    variances = df.var(numeric_only=True)
    low_var_features = variances[variances < 0.1].index.tolist()
    df.drop(columns=low_var_features, inplace=True, errors='ignore')

    # Remove highly correlated pairs (|r| > 0.95)
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr().abs()
    # Using np.triu to consider upper triangle
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    df.drop(columns=to_drop, inplace=True, errors='ignore')

    # Exclude date columns from scaling and final feature set
    feature_df = df.copy()
    for col in ['Date', 'Benchmark', 'Target_Return', 'Benchmark_Return']:
        if col in feature_df.columns:
            feature_df = feature_df.drop(columns=[col])

    # Feature Neutralization: Orthogonalize features using Gram-Schmidt
    feature_df = gram_schmidt(feature_df)

    # Robust Scaling
    scaler = RobustScaler()
    scaled_features = scaler.fit_transform(feature_df)
    feature_df = pd.DataFrame(scaled_features, columns=feature_df.columns, index=feature_df.index)

    # Return the processed dataframe and summary statistics in a dictionary
    summary = {
        'num_features': feature_df.shape[1],
        'num_samples': feature_df.shape[0],
        'removed_low_var': low_var_features,
        'removed_high_corr': to_drop,
        'scaled_feature_means': feature_df.mean().to_dict(),
        'scaled_feature_std': feature_df.std().to_dict()
    }
    return feature_df, summary

# -------------------------
# Read Input Files and Process
# -------------------------

# List of files
files = ['train_clean.csv', 'val_clean.csv', 'test_clean.csv']
feature_dfs = {}
summaries = {}

for file in files:
    if os.path.exists(file):
        df = pd.read_csv(file)
        processed_df, summary = process_features(df)
        feature_dfs[file] = processed_df
        summaries[file] = summary
    else:
        print(f"File {file} not found. Please ensure it exists.")
        processed_df = pd.DataFrame()
        summaries[file] = {}

# Save feature dataframes to CSV files
if 'train_clean.csv' in feature_dfs:
    feature_dfs['train_clean.csv'].to_csv('train_features.csv', index=False)
if 'val_clean.csv' in feature_dfs:
    feature_dfs['val_clean.csv'].to_csv('val_features.csv', index=False)
if 'test_clean.csv' in feature_dfs:
    feature_dfs['test_clean.csv'].to_csv('test_features.csv', index=False)

# Create a summary variable
feature_summary = {
    'train_summary': summaries.get('train_clean.csv', {}),
    'val_summary': summaries.get('val_clean.csv', {}),
    'test_summary': summaries.get('test_clean.csv', {})
}

# Print a brief summary (can be removed in production)
print('Feature engineering completed. Summary:', feature_summary)