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
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with additional technical indicators
        """
        df = df.copy()
        
        try:
            # Price-based indicators
            df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
            df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
            df['ema_12'] = ta.trend.ema_indicator(df['close'], window=12)
            df['ema_26'] = ta.trend.ema_indicator(df['close'], window=26)
        except Exception as e:
            print(f"Warning: Error calculating moving averages: {e}")
            # Fallback calculations
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()
            df['ema_12'] = df['close'].ewm(span=12).mean()
            df['ema_26'] = df['close'].ewm(span=26).mean()
        
        try:
            # Bollinger Bands
            bb = ta.volatility.BollingerBands(df['close'])
            df['bb_upper'] = bb.bollinger_hband()
            df['bb_middle'] = bb.bollinger_mavg()
            df['bb_lower'] = bb.bollinger_lband()
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / (df['bb_middle'] + 1e-8)
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-8)
        except Exception as e:
            print(f"Warning: Error calculating Bollinger Bands: {e}")
            # Fallback Bollinger Bands
            sma_20 = df['close'].rolling(window=20).mean()
            std_20 = df['close'].rolling(window=20).std()
            df['bb_upper'] = sma_20 + (std_20 * 2)
            df['bb_middle'] = sma_20
            df['bb_lower'] = sma_20 - (std_20 * 2)
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / (df['bb_middle'] + 1e-8)
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-8)
        
        try:
            # RSI
            df['rsi'] = ta.momentum.rsi(df['close'], window=14)
        except Exception as e:
            print(f"Warning: Error calculating RSI: {e}")
            # Fallback RSI calculation
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / (loss + 1e-8)  # Avoid division by zero
            df['rsi'] = 100 - (100 / (1 + rs))
        
        try:
            # MACD
            macd = ta.trend.MACD(df['close'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_histogram'] = macd.macd_diff()
        except Exception as e:
            print(f"Warning: Error calculating MACD: {e}")
            # Fallback MACD calculation
            ema_12 = df['close'].ewm(span=12).mean()
            ema_26 = df['close'].ewm(span=26).mean()
            df['macd'] = ema_12 - ema_26
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        try:
            # Stochastic Oscillator
            stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
            df['stoch_k'] = stoch.stoch()
            df['stoch_d'] = stoch.stoch_signal()
        except Exception as e:
            print(f"Warning: Error calculating Stochastic: {e}")
            # Fallback Stochastic calculation
            low_min = df['low'].rolling(window=14).min()
            high_max = df['high'].rolling(window=14).max()
            df['stoch_k'] = 100 * (df['close'] - low_min) / (high_max - low_min + 1e-8)
            df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        try:
            df['vwap'] = ta.volume.volume_weighted_average_price(df['high'], df['low'], df['close'], df['volume'])
        except:
            # Fallback VWAP calculation
            df['vwap'] = (df['close'] * df['volume']).rolling(window=20).sum() / df['volume'].rolling(window=20).sum()
        
        try:
            # Volatility indicators
            df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
        except Exception as e:
            print(f"Warning: Error calculating ATR: {e}")
            # Fallback ATR calculation
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            df['atr'] = true_range.rolling(window=14).mean()
        
        # Price changes and returns
        df['price_change'] = df['close'].pct_change()
        df['high_low_ratio'] = df['high'] / (df['low'] + 1e-8)
        df['close_open_ratio'] = df['close'] / (df['open'] + 1e-8)
        
        # Rolling statistics
        for window in [5, 10, 20]:
            df[f'return_std_{window}'] = df['price_change'].rolling(window).std()
            df[f'volume_std_{window}'] = df['volume'].rolling(window).std()
            df[f'price_max_{window}'] = df['close'].rolling(window).max()
            df[f'price_min_{window}'] = df['close'].rolling(window).min()
        
        return df
    
    def create_price_targets(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Create price prediction targets for the given timeframe.
        
        Args:
            df: DataFrame with price data
            timeframe: Timeframe identifier
            
        Returns:
            DataFrame with target columns added
        """
        df = df.copy()
        horizon = self.prediction_horizons.get(timeframe, 1)
        
        # Future price
        df['future_close'] = df['close'].shift(-horizon)
        
        # Price change (percentage)
        df['target_price_change'] = (df['future_close'] - df['close']) / df['close']
        
        # Price direction (classification target)
        df['target_direction'] = (df['target_price_change'] > 0).astype(int)
        
        # Volatility target (for risk assessment)
        df['target_volatility'] = df['price_change'].rolling(horizon).std().shift(-horizon)
        
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
            'future_close', 'target_price_change', 'target_direction', 'target_volatility'
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
        targets = df_processed[['target_price_change', 'target_direction', 'target_volatility']].values
        
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
