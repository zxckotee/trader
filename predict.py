#!/usr/bin/env python3
"""
Main prediction script for MoE cryptocurrency prediction model.
"""

import argparse
import sys
from pathlib import Path
import json

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.inference.predictor import CryptoPredictionService, RealTimePredictionService
from src.utils.config import load_config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Make cryptocurrency predictions using MoE model')
    
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--model-path', type=str, default='./models/best_model.pt',
                       help='Path to trained model')
    parser.add_argument('--preprocessor-path', type=str, default='./data/scalers.pkl',
                       help='Path to fitted preprocessor')
    parser.add_argument('--symbols', nargs='+', default=['BTCUSDT'],
                       help='Trading symbols to predict')
    parser.add_argument('--timeframes', nargs='+', default=['5m', '30m', '1h', '1d', '1w'],
                       help='Timeframes to use for prediction')
    parser.add_argument('--output', type=str, help='Output file for predictions')
    parser.add_argument('--real-time', action='store_true',
                       help='Run continuous real-time predictions')
    parser.add_argument('--interval', type=int, default=300,
                       help='Prediction interval in seconds (for real-time mode)')
    parser.add_argument('--duration', type=int,
                       help='Duration to run real-time predictions (hours)')
    parser.add_argument('--expert-outputs', action='store_true',
                       help='Include individual expert predictions')
    parser.add_argument('--device', type=str, choices=['auto', 'cpu', 'cuda', 'mps'],
                       default='auto', help='Device to use for inference')
    parser.add_argument('--force-cpu', action='store_true',
                       help='Force CPU usage even if GPU is available')
    
    return parser.parse_args()


def single_prediction(predictor, symbols, timeframes, expert_outputs, output_file):
    """Make single predictions for given symbols."""
    print("=== Single Prediction Mode ===")
    
    if len(symbols) == 1:
        # Single symbol prediction
        symbol = symbols[0]
        print(f"Making prediction for {symbol}...")
        
        try:
            prediction = predictor.predict(
                symbol=symbol,
                timeframes=timeframes,
                return_expert_outputs=expert_outputs
            )
            
            print(f"\nPrediction for {symbol}:")
            print(json.dumps(prediction, indent=2, default=str))
            
            if output_file:
                predictor.save_predictions(prediction, output_file)
            
        except Exception as e:
            print(f"Error making prediction for {symbol}: {e}")
    
    else:
        # Multiple symbols prediction
        print(f"Making predictions for {len(symbols)} symbols...")
        
        try:
            predictions = predictor.predict_multiple_symbols(
                symbols=symbols,
                timeframes=timeframes
            )
            
            print("\nPrediction Summary:")
            print("-" * 60)
            
            for symbol, pred in predictions.items():
                if 'error' not in pred:
                    agg = pred['aggregated_predictions']
                    print(f"{symbol:10} | {agg['predicted_direction']:4} | "
                          f"{agg['price_change_pct']:+7.2f}% | "
                          f"Conf: {agg['confidence']:.3f}")
                else:
                    print(f"{symbol:10} | ERROR: {pred['error']}")
            
            print("-" * 60)
            
            if output_file:
                predictor.save_predictions(predictions, output_file)
            
        except Exception as e:
            print(f"Error making predictions: {e}")


def real_time_prediction(predictor, symbols, interval, duration):
    """Run continuous real-time predictions."""
    print("=== Real-Time Prediction Mode ===")
    
    rt_service = RealTimePredictionService(
        predictor=predictor,
        symbols=symbols,
        prediction_interval=interval
    )
    
    try:
        rt_service.run_continuous_predictions(duration_hours=duration)
    except KeyboardInterrupt:
        print("\nReal-time predictions stopped by user")


def main():
    """Main prediction function."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    print("=== MoE Cryptocurrency Prediction ===")
    print(f"Model: {args.model_path}")
    print(f"Preprocessor: {args.preprocessor_path}")
    print(f"Symbols: {args.symbols}")
    print(f"Timeframes: {args.timeframes}")
    
    # Initialize prediction service
    try:
        predictor = CryptoPredictionService(
            model_path=args.model_path,
            preprocessor_path=args.preprocessor_path,
            device=args.device,
            force_cpu=args.force_cpu
        )
        print("Prediction service initialized successfully")
        
    except Exception as e:
        print(f"Error initializing prediction service: {e}")
        print("Make sure you have:")
        print("1. Trained a model (run train.py)")
        print("2. Valid model and preprocessor files")
        return
    
    # Run predictions
    if args.real_time:
        real_time_prediction(
            predictor=predictor,
            symbols=args.symbols,
            interval=args.interval,
            duration=args.duration
        )
    else:
        single_prediction(
            predictor=predictor,
            symbols=args.symbols,
            timeframes=args.timeframes,
            expert_outputs=args.expert_outputs,
            output_file=args.output
        )


if __name__ == "__main__":
    main()
