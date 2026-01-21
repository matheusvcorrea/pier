import argparse
import classify_model

def main():
    """
    Main function to train the lottery accumulation prediction model.
    
    This function sets up the command-line interface, parses arguments,
    and orchestrates the model training pipeline.
    """
    parser = argparse.ArgumentParser(
        description='Train a LightGBM classification model to predict lottery accumulation.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Train with default dataset (dataset.json)
  python train.py
  
  # Train with custom dataset file
  python train.py --dataset my_data.json
  
  # Train with custom MLflow tracking
  python train.py --mlflow-tracking-uri http://localhost:5000 --experiment-name lottery_v1
  
  # Save model to custom location
  python train.py --model-path custom_model.pkl --features-path custom_features.txt
  
  # Show help
  python train.py --help
        '''
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        default='dataset.json',
        help='Path to the JSON dataset file (default: dataset.json)'
    )
    
    parser.add_argument(
        '--model-path',
        type=str,
        default='lottery_model.pkl',
        help='Path to save the trained model file (default: lottery_model.pkl)'
    )
    
    parser.add_argument(
        '--features-path',
        type=str,
        default='features.txt',
        help='Path to save the feature names file (default: features.txt)'
    )
    
    parser.add_argument(
        '--mlflow-tracking-uri',
        type=str,
        default=None,
        help='MLflow tracking URI (default: local file system)'
    )
    
    parser.add_argument(
        '--experiment-name',
        type=str,
        default='lottery_prediction',
        help='MLflow experiment name (default: lottery_prediction)'
    )
    
    args = parser.parse_args()
    
    print("Training LightGBM to predict if the next lottery draw will accumulate")
    print(f"Dataset: {args.dataset}")
    print(f"Model output: {args.model_path}")
    print(f"Features output: {args.features_path}")
    
    # Set up MLflow tracking
    if args.mlflow_tracking_uri:
        import mlflow
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
        print(f"MLflow tracking URI: {args.mlflow_tracking_uri}")
    print(f"MLflow experiment: {args.experiment_name}")
    
    # Load and preprocess data
    X, y = classify_model.load_and_preprocess_data(args.dataset)
    
    # Train model with MLflow tracking
    model, feature_names, run_id = classify_model.train_model(X, y, args.experiment_name)

    # Save artifacts
    classify_model.save_model_artifacts(model, feature_names, args.model_path, args.features_path)

    print(f"\nModel training and saving completed successfully!")
    print(f"MLflow Run ID: {run_id}")

if __name__ == "__main__":
    main()