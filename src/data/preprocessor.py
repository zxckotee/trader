"""
Data preprocessing module for multi-timeframe cryptocurrency data.
Handles feature engineering, normalization, and sequence preparation for MoE model.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import ta
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pathlib import Path
import pickle


class CryptoDataPreprocessor:
    """
    Preprocessor for cryptocurrency market data across multiple timeframes.
    Creates features and sequences suitable for MoE architecture.
    """
    
    def __init__(self, 
                 sequence_length: int = 100,
                 prediction_horizons: Dict[str, int] = None,
                 scaler_type: str = 'standard'):
        """
        Initialize the preprocessor.
        
        Args:
            sequence_length: Length of input sequences
            prediction_horizons: Dict mapping timeframes to prediction steps ahead
            scaler_type: Type of scaler ('standard' or 'minmax')
        """
        self.sequence_length = sequence_length
        self.prediction_horizons = prediction_horizons or {
            '5m': 12,    # 1 hour ahead (12 * 5min)
            '30m': 8,    # 4 hours ahead (8 * 30min)
            '1h': 6,     # 6 hours ahead
            '1d': 7,     # 1 week ahead
            '1w': 4      # 1 month ahead
        }
        
        self.scaler_type = scaler_type
        self.scalers = {}
        self.feature_columns = []
        
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for the given DataFrame.
        Now uses percentage changes instead of raw prices for better generalization.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with additional technical indicators
        """
        df = df.copy()
        
        # Calculate percentage changes first (this is the key change!)
        df['price_change'] = df['close'].pct_change()
        df['volume_change'] = df['volume'].pct_change()
        df['high_change'] = df['high'].pct_change()
        df['low_change'] = df['low'].pct_change()
        df['open_change'] = df['open'].pct_change()
        
        # Price range changes
        df['hl_change'] = (df['high'] - df['low']) / (df['close'].shift(1) + 1e-8)
        df['co_change'] = (df['close'] - df['open']) / (df['open'] + 1e-8)
        
        try:
            # Price-based indicators using percentage changes
            df['sma_20_change'] = df['price_change'].rolling(window=20).mean()
            df['sma_50_change'] = df['price_change'].rolling(window=50).mean()
            df['ema_12_change'] = df['price_change'].ewm(span=12).mean()
            df['ema_26_change'] = df['price_change'].ewm(span=26).mean()
            
            # Volatility indicators using percentage changes
            df['volatility_20'] = df['price_change'].rolling(window=20).std()
            df['volatility_50'] = df['price_change'].rolling(window=50).std()
        except Exception as e:
            print(f"Warning: Error calculating moving averages: {e}")
            # Fallback calculations
            df['sma_20_change'] = df['price_change'].rolling(window=20).mean()
            df['sma_50_change'] = df['price_change'].rolling(window=50).mean()
            df['ema_12_change'] = df['price_change'].ewm(span=12).mean()
            df['ema_26_change'] = df['price_change'].ewm(span=26).mean()
            df['volatility_20'] = df['price_change'].rolling(window=20).std()
            df['volatility_50'] = df['price_change'].rolling(window=50).std()
        
        try:
            # Bollinger Bands using percentage changes
            bb = ta.volatility.BollingerBands(df['price_change'])
            df['bb_upper_change'] = bb.bollinger_hband()
            df['bb_middle_change'] = bb.bollinger_mavg()
            df['bb_lower_change'] = bb.bollinger_lband()
            df['bb_width_change'] = (df['bb_upper_change'] - df['bb_lower_change']) / (df['bb_middle_change'].abs() + 1e-8)
            df['bb_position_change'] = (df['price_change'] - df['bb_lower_change']) / (df['bb_upper_change'] - df['bb_lower_change'] + 1e-8)
        except Exception as e:
            print(f"Warning: Error calculating Bollinger Bands: {e}")
            # Fallback Bollinger Bands using percentage changes
            sma_20_change = df['price_change'].rolling(window=20).mean()
            std_20_change = df['price_change'].rolling(window=20).std()
            df['bb_upper_change'] = sma_20_change + (std_20_change * 2)
            df['bb_middle_change'] = sma_20_change
            df['bb_lower_change'] = sma_20_change - (std_20_change * 2)
            df['bb_width_change'] = (df['bb_upper_change'] - df['bb_lower_change']) / (df['bb_middle_change'].abs() + 1e-8)
            df['bb_position_change'] = (df['price_change'] - df['bb_lower_change']) / (df['bb_upper_change'] - df['bb_lower_change'] + 1e-8)
        
        try:
            # RSI using percentage changes
            df['rsi_change'] = ta.momentum.rsi(df['price_change'], window=14)
        except Exception as e:
            print(f"Warning: Error calculating RSI: {e}")
            # Fallback RSI calculation using percentage changes
            delta = df['price_change'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / (loss + 1e-8)  # Avoid division by zero
            df['rsi_change'] = 100 - (100 / (1 + rs))
        
        try:
            # MACD using percentage changes
            macd = ta.trend.MACD(df['price_change'])
            df['macd_change'] = macd.macd()
            df['macd_signal_change'] = macd.macd_signal()
            df['macd_histogram_change'] = macd.macd_diff()
        except Exception as e:
            print(f"Warning: Error calculating MACD: {e}")
            # Fallback MACD calculation using percentage changes
            ema_12_change = df['price_change'].ewm(span=12).mean()
            ema_26_change = df['price_change'].ewm(span=26).mean()
            df['macd_change'] = ema_12_change - ema_26_change
            df['macd_signal_change'] = df['macd_change'].ewm(span=9).mean()
            df['macd_histogram_change'] = df['macd_change'] - df['macd_signal_change']
        
        try:
            # Stochastic Oscillator using percentage changes
            stoch = ta.momentum.StochasticOscillator(df['high_change'], df['low_change'], df['price_change'])
            df['stoch_k_change'] = stoch.stoch()
            df['stoch_d_change'] = stoch.stoch_signal()
        except Exception as e:
            print(f"Warning: Error calculating Stochastic: {e}")
            # Fallback Stochastic calculation using percentage changes
            low_min_change = df['low_change'].rolling(window=14).min()
            high_max_change = df['high_change'].rolling(window=14).max()
            df['stoch_k_change'] = 100 * (df['price_change'] - low_min_change) / (high_max_change - low_min_change + 1e-8)
            df['stoch_d_change'] = df['stoch_k_change'].rolling(window=3).mean()
        
        # Volume indicators using percentage changes
        # === VOLUME ANALYSIS (КРИТИЧНО ДЛЯ ОБНАРУЖЕНИЯ КИТОВ!) ===
        
        # Basic volume indicators using percentage changes
        df['volume_sma_change'] = df['volume_change'].rolling(window=20).mean()
        df['volume_ema_change'] = df['volume_change'].ewm(span=20).mean()
        
        # MAVOL - Moving Average of Volume (percentage changes)
        df['mavol_5'] = df['volume_change'].rolling(window=5).mean()
        df['mavol_10'] = df['volume_change'].rolling(window=10).mean()
        df['mavol_20'] = df['volume_change'].rolling(window=20).mean()
        df['mavol_50'] = df['volume_change'].rolling(window=50).mean()
        
        # Volume-weighted price changes
        try:
            # VWAP using percentage changes
            df['vwap_change'] = ta.volume.volume_weighted_average_price(df['high_change'], df['low_change'], df['price_change'], df['volume'])
        except:
            # Fallback VWAP calculation using percentage changes
            df['vwap_change'] = (df['price_change'] * df['volume']).rolling(window=20).sum() / df['volume'].rolling(window=20).sum()
        
        # OBV - On Balance Volume using percentage changes
        try:
            df['obv_change'] = ta.volume.on_balance_volume(df['price_change'], df['volume_change'])
        except:
            # Fallback OBV using percentage changes
            obv_change = [0]
            for i in range(1, len(df)):
                if df['price_change'].iloc[i] > 0:
                    obv_change.append(obv_change[-1] + df['volume_change'].iloc[i])
                elif df['price_change'].iloc[i] < 0:
                    obv_change.append(obv_change[-1] - df['volume_change'].iloc[i])
                else:
                    obv_change.append(obv_change[-1])
            df['obv_change'] = obv_change
        
        # OBV trend using percentage changes
        df['obv_ema_change'] = df['obv_change'].ewm(span=20).mean()
        df['obv_signal_change'] = (df['obv_change'] - df['obv_ema_change']) / (df['obv_ema_change'].abs() + 1e-8)
        
        # CMF - Chaikin Money Flow using percentage changes
        try:
            df['cmf_change'] = ta.volume.chaikin_money_flow(df['high_change'], df['low_change'], df['price_change'], df['volume_change'], window=20)
        except:
            # Fallback CMF using percentage changes
            mfv_change = ((df['price_change'] - df['low_change']) - (df['high_change'] - df['price_change'])) / (df['high_change'] - df['low_change'] + 1e-8) * df['volume_change']
            df['cmf_change'] = mfv_change.rolling(window=20).sum() / df['volume_change'].rolling(window=20).sum()
        
        # MFI - Money Flow Index using percentage changes
        try:
            df['mfi_change'] = ta.volume.money_flow_index(df['high_change'], df['low_change'], df['price_change'], df['volume_change'], window=14)
        except:
            # Fallback MFI using percentage changes
            typical_price_change = (df['high_change'] + df['low_change'] + df['price_change']) / 3
            money_flow_change = typical_price_change * df['volume_change']
            
            # Positive and negative money flow using percentage changes
            positive_flow_change = []
            negative_flow_change = []
            for i in range(1, len(df)):
                if typical_price_change.iloc[i] > typical_price_change.iloc[i-1]:
                    positive_flow_change.append(money_flow_change.iloc[i])
                    negative_flow_change.append(0)
                else:
                    positive_flow_change.append(0)
                    negative_flow_change.append(money_flow_change.iloc[i])
            
            positive_flow_change = [0] + positive_flow_change
            negative_flow_change = [0] + negative_flow_change
            
            df['positive_mf_change'] = positive_flow_change
            df['negative_mf_change'] = negative_flow_change
            
            positive_mf_sum_change = df['positive_mf_change'].rolling(window=14).sum()
            negative_mf_sum_change = df['negative_mf_change'].rolling(window=14).sum()
            
            mfi_change = 100 - (100 / (1 + positive_mf_sum_change / (negative_mf_sum_change + 1e-8)))
            df['mfi_change'] = mfi_change
            df.drop(['positive_mf_change', 'negative_mf_change'], axis=1, inplace=True)
        
        # Volume spikes using percentage changes (обнаружение китов!)
        df['volume_ratio_change'] = df['volume_change'] / (df['volume_sma_change'].abs() + 1e-8)
        df['volume_spike_change'] = (df['volume_ratio_change'].abs() > 2.0).astype(int)  # Объем > 2x среднего
        
        # Force Index using percentage changes (сила движения с учетом объема)
        df['force_index_change'] = df['price_change'].diff() * df['volume_change']
        df['force_index_ema_change'] = df['force_index_change'].ewm(span=13).mean()
        
        # Ease of Movement using percentage changes (легкость движения цены относительно объема)
        distance_change = ((df['high_change'] + df['low_change']) / 2).diff()
        box_ratio_change = df['volume_change'] / (df['high_change'] - df['low_change'] + 1e-8)
        df['ease_of_movement_change'] = distance_change / (box_ratio_change + 1e-8)
        df['ease_of_movement_ema_change'] = df['ease_of_movement_change'].ewm(span=14).mean()
        
        # Volume-Price Trend using percentage changes (VPT)
        df['vpt_change'] = (df['price_change'] * df['volume_change']).cumsum()
        df['vpt_signal_change'] = df['vpt_change'] - df['vpt_change'].rolling(window=20).mean()
        
        # Accumulation/Distribution Line using percentage changes (A/D)
        try:
            df['ad_change'] = ta.volume.acc_dist_index(df['high_change'], df['low_change'], df['price_change'], df['volume_change'])
        except:
            clv_change = ((df['price_change'] - df['low_change']) - (df['high_change'] - df['price_change'])) / (df['high_change'] - df['low_change'] + 1e-8)
            df['ad_change'] = (clv_change * df['volume_change']).cumsum()
        
        df['ad_ema_change'] = df['ad_change'].ewm(span=20).mean()
        df['ad_signal_change'] = (df['ad_change'] - df['ad_ema_change']) / (df['ad_ema_change'].abs() + 1e-8)
        
        try:
            # Volatility indicators using percentage changes
            df['atr_change'] = ta.volatility.average_true_range(df['high_change'], df['low_change'], df['price_change'])
        except Exception as e:
            print(f"Warning: Error calculating ATR: {e}")
            # Fallback ATR calculation using percentage changes
            high_low_change = df['high_change'] - df['low_change']
            high_close_change = np.abs(df['high_change'] - df['price_change'].shift())
            low_close_change = np.abs(df['low_change'] - df['price_change'].shift())
            true_range_change = np.maximum(high_low_change, np.maximum(high_close_change, low_close_change))
            df['atr_change'] = true_range_change.rolling(window=14).mean()
        
        # Additional price change ratios
        df['high_low_ratio_change'] = df['high_change'] / (df['low_change'] + 1e-8)
        df['close_open_ratio_change'] = df['price_change'] / (df['open_change'] + 1e-8)
        
        # Rolling statistics using percentage changes
        for window in [5, 10, 20]:
            df[f'return_std_{window}'] = df['price_change'].rolling(window).std()
            df[f'volume_std_{window}'] = df['volume_change'].rolling(window).std()
            df[f'price_max_{window}'] = df['price_change'].rolling(window).max()
            df[f'price_min_{window}'] = df['price_change'].rolling(window).min()
            df[f'volume_max_{window}'] = df['volume_change'].rolling(window).max()
            df[f'volume_min_{window}'] = df['volume_change'].rolling(window).min()
        
        return df
    
    def create_price_targets(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Create price prediction targets for the given timeframe.
        Now works with percentage changes for better generalization.
        
        Args:
            df: DataFrame with price data
            timeframe: Timeframe identifier
            
        Returns:
            DataFrame with target columns added
        """
        df = df.copy()
        horizon = self.prediction_horizons.get(timeframe, 1)
        
        # Future price change (percentage)
        df['future_price_change'] = df['price_change'].shift(-horizon)
        
        # Cumulative price change over horizon
        df['target_price_change'] = df['price_change'].rolling(horizon).sum().shift(-horizon)
        
        # Price direction (classification target)
        df['target_direction'] = (df['target_price_change'] > 0).astype(int)
        
        # Volatility target (for risk assessment) - using percentage changes
        df['target_volatility'] = df['price_change'].rolling(horizon).std().shift(-horizon)
        
        # Additional targets for probability distribution
        # Price change magnitude (absolute value)
        df['target_price_magnitude'] = df['target_price_change'].abs()
        
        # Price change percentile (for distribution modeling)
        df['target_price_percentile'] = df['target_price_change'].rolling(window=100, min_periods=50).rank(pct=True).shift(-horizon)
        
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare feature matrix from processed DataFrame.
        
        Args:
            df: DataFrame with all indicators
            
        Returns:
            DataFrame with selected features
        """
        # Select feature columns (exclude timestamp, targets, and raw OHLCV)
        exclude_cols = [
            'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover',
            'future_price_change', 'target_price_change', 'target_direction', 'target_volatility',
            'target_price_magnitude', 'target_price_percentile'
        ]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Store feature columns for later use
        if not self.feature_columns:
            self.feature_columns = feature_cols
        
        return df[feature_cols]
    
    def fit_scaler(self, data: np.ndarray, timeframe: str) -> None:
        """
        Fit scaler on training data.
        
        Args:
            data: Training data array
            timeframe: Timeframe identifier
        """
        if self.scaler_type == 'standard':
            scaler = StandardScaler()
        elif self.scaler_type == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaler type: {self.scaler_type}")
        
        scaler.fit(data)
        self.scalers[timeframe] = scaler
    
    def transform_data(self, data: np.ndarray, timeframe: str) -> np.ndarray:
        """
        Transform data using fitted scaler.
        
        Args:
            data: Data to transform
            timeframe: Timeframe identifier
            
        Returns:
            Scaled data
        """
        if timeframe not in self.scalers:
            raise ValueError(f"Scaler for {timeframe} not fitted")
        
        return self.scalers[timeframe].transform(data)
    
    def create_sequences(self, 
                        features: np.ndarray, 
                        targets: np.ndarray,
                        sequence_length: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series prediction.
        
        Args:
            features: Feature array
            targets: Target array
            sequence_length: Length of sequences (uses self.sequence_length if None)
            
        Returns:
            Tuple of (X_sequences, y_sequences)
        """
        seq_len = sequence_length or self.sequence_length
        
        X_sequences = []
        y_sequences = []
        
        for i in range(seq_len, len(features)):
            X_sequences.append(features[i-seq_len:i])
            y_sequences.append(targets[i])
        
        return np.array(X_sequences), np.array(y_sequences)
    
    def process_timeframe_data(self, 
                             df: pd.DataFrame, 
                             timeframe: str,
                             fit_scaler: bool = False) -> Dict[str, np.ndarray]:
        """
        Complete preprocessing pipeline for a single timeframe.
        
        Args:
            df: Raw OHLCV DataFrame
            timeframe: Timeframe identifier
            fit_scaler: Whether to fit scaler (True for training data)
            
        Returns:
            Dictionary with processed arrays
        """
        print(f"Processing {timeframe} data...")
        
        # Calculate technical indicators
        df_processed = self.calculate_technical_indicators(df)
        
        # Create targets
        df_processed = self.create_price_targets(df_processed, timeframe)
        
        # Prepare features
        features_df = self.prepare_features(df_processed)
        
        # Handle missing values
        features_df = features_df.ffill().bfill()
        df_processed = df_processed.ffill().bfill()
        
        # Handle infinite values and extreme outliers
        features_df = features_df.replace([np.inf, -np.inf], np.nan)
        df_processed = df_processed.replace([np.inf, -np.inf], np.nan)
        
        # Fill remaining NaN values with median
        for col in features_df.columns:
            if features_df[col].isna().any():
                median_val = features_df[col].median()
                if pd.isna(median_val):  # If median is also NaN, use 0
                    median_val = 0
                features_df[col] = features_df[col].fillna(median_val)
        
        for col in df_processed.select_dtypes(include=[np.number]).columns:
            if df_processed[col].isna().any():
                median_val = df_processed[col].median()
                if pd.isna(median_val):
                    median_val = 0
                df_processed[col] = df_processed[col].fillna(median_val)
        
        # Convert to numpy arrays
        features = features_df.values
        targets = df_processed[['target_price_change', 'target_direction', 'target_volatility', 
                              'target_price_magnitude', 'target_price_percentile']].values
        
        # Remove rows with NaN targets (at the end due to future shift)
        valid_mask = ~np.isnan(targets).any(axis=1)
        features = features[valid_mask]
        targets = targets[valid_mask]
        
        # Additional check for infinite values in features
        finite_mask = np.isfinite(features).all(axis=1)
        features = features[finite_mask]
        targets = targets[finite_mask]
        
        # Check for remaining invalid values
        if features.size == 0:
            print(f"Warning: No valid features remaining for {timeframe}")
            return {
                'X': np.array([]),
                'y': np.array([]),
                'features_raw': np.array([]),
                'targets_raw': np.array([]),
                'timestamps': np.array([])
            }
        
        # Clip extreme values to reasonable ranges
        features = np.clip(features, -1e6, 1e6)
        
        # Fit or transform features
        if fit_scaler:
            self.fit_scaler(features, timeframe)
        
        features_scaled = self.transform_data(features, timeframe)
        
        # Create sequences
        X_seq, y_seq = self.create_sequences(features_scaled, targets)
        
        print(f"Created {len(X_seq)} sequences for {timeframe}")
        
        return {
            'X': X_seq,
            'y': y_seq,
            'features_raw': features,
            'targets_raw': targets,
            'timestamps': df_processed['timestamp'].values[valid_mask]
        }
    
    def process_multi_timeframe_data(self, 
                                   data_dict: Dict[str, pd.DataFrame],
                                   fit_scalers: bool = False) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Process data for all timeframes.
        
        Args:
            data_dict: Dictionary mapping timeframes to DataFrames
            fit_scalers: Whether to fit scalers (True for training data)
            
        Returns:
            Dictionary mapping timeframes to processed data
        """
        processed_data = {}
        
        for timeframe, df in data_dict.items():
            processed_data[timeframe] = self.process_timeframe_data(
                df, timeframe, fit_scaler=fit_scalers
            )
        
        return processed_data
    
    def save_scalers(self, filepath: str) -> None:
        """Save fitted scalers to file."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'scalers': self.scalers,
                'feature_columns': self.feature_columns,
                'sequence_length': self.sequence_length,
                'prediction_horizons': self.prediction_horizons
            }, f)
        print(f"Scalers saved to {filepath}")
    
    def load_scalers(self, filepath: str) -> None:
        """Load scalers from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.scalers = data['scalers']
            self.feature_columns = data['feature_columns']
            self.sequence_length = data['sequence_length']
            self.prediction_horizons = data['prediction_horizons']
        print(f"Scalers loaded from {filepath}")


def main():
    """Example usage of CryptoDataPreprocessor."""
    from bybit_parser import BybitParser
    
    # Load data
    parser = BybitParser(data_dir="./data")
    symbol = "BTCUSDT"
    timeframes = ['5m', '30m', '1h', '1d', '1w']
    
    data_dict = {}
    for tf in timeframes:
        df = parser.load_data(symbol, tf)
        if not df.empty:
            data_dict[tf] = df
    
    if not data_dict:
        print("No data found. Please run bybit_parser.py first.")
        return
    
    # Initialize preprocessor
    preprocessor = CryptoDataPreprocessor(sequence_length=100)
    
    # Process data
    processed_data = preprocessor.process_multi_timeframe_data(
        data_dict, fit_scalers=True
    )
    
    # Save scalers
    preprocessor.save_scalers("./data/scalers.pkl")
    
    # Display summary
    print("\n=== Preprocessing Summary ===")
    for tf, data in processed_data.items():
        print(f"{tf}: {data['X'].shape[0]} sequences, "
              f"feature dim: {data['X'].shape[2]}, "
              f"sequence length: {data['X'].shape[1]}")


if __name__ == "__main__":
    main()
