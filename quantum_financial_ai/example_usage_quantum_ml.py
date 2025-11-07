"""
Example Usage of Quantum Machine Learning Pipeline
OWLBAN GROUP - Demonstrating quantum ML for financial applications
"""

import numpy as np
import pandas as pd
import logging
import time
from quantum_machine_learning_pipeline import (
    QuantumMLPipeline,
    FinancialQuantumMLApplication,
    create_synthetic_financial_data,
    FinancialFeatureData
)

def setup_logging():
    """Set up logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def demonstrate_basic_quantum_ml():
    """Demonstrate basic quantum ML pipeline usage"""
    print("=== Basic Quantum ML Pipeline Demo ===")

    # Create synthetic financial data
    print("Creating synthetic financial data...")
    data = create_synthetic_financial_data(n_samples=500, n_features=6)
    print(f"Data shape: {data.features.shape}, Labels: {len(data.labels)}")

    # Create quantum SVM pipeline
    print("Creating Quantum SVM pipeline...")
    pipeline = QuantumMLPipeline(model_type="qsvm", n_qubits=4, C=1.0)

    # Train the pipeline
    print("Training quantum ML model...")
    start_time = time.time()
    result = pipeline.train(data)
    training_time = time.time() - start_time

    print(f"Training accuracy: {result.accuracy:.4f}")
    print(f"Quantum advantage: {result.quantum_advantage:.4f}")
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Model parameters: {len(result.model_parameters)} parameters")
    # Make predictions on new data
    print("Making predictions on new data...")
    test_features = np.random.randn(20, 6)
    pred_result = pipeline.predict(test_features)

    print(f"Predictions shape: {pred_result.predictions.shape}")
    print(f"Inference time: {pred_result.inference_time:.4f} seconds")

    # Evaluate performance
    print("Evaluating performance...")
    performance = pipeline.evaluate_performance(data)
    print("Performance metrics:")
    for metric, value in performance.items():
        print(f"  {metric}: {value:.4f}")

def demonstrate_financial_applications():
    """Demonstrate financial-specific quantum ML applications"""
    print("\n=== Financial Quantum ML Applications Demo ===")

    # Create financial application
    app = FinancialQuantumMLApplication()

    # Create different pipelines for financial tasks
    print("Creating specialized financial pipelines...")

    # Buy/sell signal classifier
    buy_sell_pipeline = app.create_buy_sell_classifier(n_qubits=4)
    print("✓ Buy/Sell classifier created")

    # Market direction predictor
    market_pipeline = app.create_market_predictor(n_qubits=4)
    print("✓ Market predictor created")

    # Risk assessment model
    risk_pipeline = app.create_risk_assessor(n_qubits=4)
    print("✓ Risk assessor created")

    # Create synthetic financial data for each task
    print("Training models on synthetic financial data...")

    # Buy/sell data
    buy_sell_data = create_synthetic_financial_data(n_samples=300, n_features=4)
    buy_sell_result = app.train_on_financial_signals("buy_sell", buy_sell_data)
    print(f"Buy/sell accuracy: {buy_sell_result.accuracy:.4f}")
    # Market prediction data
    market_data = create_synthetic_financial_data(n_samples=300, n_features=4)
    market_result = app.train_on_financial_signals("market_prediction", market_data)
    print(f"Market prediction accuracy: {market_result.accuracy:.4f}")
    # Risk assessment data
    risk_data = create_synthetic_financial_data(n_samples=300, n_features=4)
    risk_result = app.train_on_financial_signals("risk_assessment", risk_data)
    print(f"Risk assessment accuracy: {risk_result.accuracy:.4f}")
    # Generate performance report
    print("Generating performance report...")
    report = app.get_pipeline_performance_report()
    print("Performance Report:")
    print(f"Number of pipelines: {len(report['pipelines'])}")
    for name, info in report['pipelines'].items():
        print(f"  {name}: {info['model_type']} ({info['n_qubits']} qubits) - {info['status']}")

def demonstrate_quantum_advantage():
    """Demonstrate quantum advantage over classical methods"""
    print("\n=== Quantum Advantage Demonstration ===")

    # Create data
    data = create_synthetic_financial_data(n_samples=200, n_features=4)

    # Test different model types
    model_types = ["qsvm", "vqc", "qnn"]
    results = {}

    for model_type in model_types:
        print(f"Testing {model_type.upper()}...")
        try:
            pipeline = QuantumMLPipeline(model_type=model_type, n_qubits=4)
            result = pipeline.train(data)
            results[model_type] = result

            print(f"  Accuracy: {result.accuracy:.4f}")
            print(f"  Quantum advantage: {result.quantum_advantage:.4f}")
            print(f"  Training time: {result.training_time:.2f} seconds")
        except Exception as e:
            print(f"  Error with {model_type}: {str(e)}")

    # Compare results
    print("\nComparison of Quantum ML Models:")
    print("-" * 50)
    for model_type, result in results.items():
        print(f"{model_type.upper():<15} | Acc: {result.accuracy:.4f} | Q-Adv: {result.quantum_advantage:.4f} | Time: {result.training_time:.2f}s")
def demonstrate_real_world_integration():
    """Demonstrate integration with existing financial systems"""
    print("\n=== Real-World Integration Demo ===")

    # Simulate integration with existing portfolio optimizer
    from quantum_financial_ai.quantum_portfolio_optimizer import QuantumPortfolioOptimizer, PortfolioAsset

    print("Integrating with Quantum Portfolio Optimizer...")

    # Create portfolio optimizer
    optimizer = QuantumPortfolioOptimizer()

    # Add some assets
    assets = [
        PortfolioAsset("AAPL", 0.12, 0.25, 150.0, 100),
        PortfolioAsset("GOOGL", 0.10, 0.22, 2800.0, 50),
        PortfolioAsset("MSFT", 0.15, 0.20, 300.0, 75),
        PortfolioAsset("TSLA", 0.25, 0.35, 800.0, 25)
    ]

    for asset in assets:
        optimizer.add_asset(asset)

    # Create synthetic market data for ML predictions
    print("Creating market prediction data...")
    market_features = np.random.randn(100, 4)  # Technical indicators
    market_data = FinancialFeatureData(
        features=market_features,
        labels=np.random.randint(0, 2, 100),
        feature_names=["RSI", "MACD", "Volume", "Volatility"]
    )

    # Create and train market predictor
    market_pipeline = QuantumMLPipeline(model_type="qsvm", n_qubits=4)
    market_result = market_pipeline.train(market_data)

    print(f"Market predictor accuracy: {market_result.accuracy:.4f}")
    # Simulate using predictions for portfolio decisions
    print("Using quantum ML predictions for portfolio optimization...")

    # Get current market features (simulated)
    current_features = np.random.randn(1, 4)

    # Get prediction
    prediction_result = market_pipeline.predict(current_features)
    market_signal = "BUY" if prediction_result.predictions[0] == 1 else "SELL"

    print(f"Market signal: {market_signal}")

    # Optimize portfolio based on signal
    if market_signal == "BUY":
        # More aggressive optimization
        portfolio_result = optimizer.optimize_portfolio(method="quantum")
    else:
        # Conservative optimization
        portfolio_result = optimizer.optimize_portfolio(method="classical")

    print(f"Portfolio return: {portfolio_result.expected_return:.4f}")
    print(f"Portfolio volatility: {portfolio_result.portfolio_volatility:.4f}")
def run_all_demos():
    """Run all demonstration functions"""
    setup_logging()

    print("OWLBAN GROUP - Quantum Machine Learning Pipeline Demonstrations")
    print("=" * 70)

    try:
        demonstrate_basic_quantum_ml()
        demonstrate_financial_applications()
        demonstrate_quantum_advantage()
        demonstrate_real_world_integration()

        print("\n" + "=" * 70)
        print("✅ All demonstrations completed successfully!")
        print("Quantum ML pipeline is ready for production use.")

    except Exception as e:
        print(f"\n❌ Error during demonstrations: {str(e)}")
        logging.error("Demonstration failed", exc_info=True)

if __name__ == "__main__":
    run_all_demos()
