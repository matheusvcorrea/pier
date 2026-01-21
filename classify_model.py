# Lottery Accumulation Prediction Model
# This script implements a classification model to predict whether the next lottery draw will accumulate
# Based on the README.md requirements

import json
import pandas as pd
import joblib
import warnings
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from evaluate_report import evaluate_model_report, calculate_metrics
import mlflow
import mlflow.lightgbm


# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

target = 'acumulou'

def load_data(filepath: str='dataset.json') -> pd.DataFrame:
    """ Load lottery dataset from JSON filepath variable"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    if type(data) is list:
        print(f'Dataset contains {len(data)} samples.')
    df = pd.DataFrame(data)
    return df


def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Engineer features from lottery data
    
    Returns:
    --------
    X : pd.DataFrame
        Feature dataframe
    y : pd.Series
        Target variable series
    """
    # Drop rows with missing values on target variable
    missing_target_indices = df[df[target].isnull()].index
    if not missing_target_indices.empty:
        df = df.drop(index=missing_target_indices)
    print(f'Dropped {len(missing_target_indices)} rows with missing target values.')

    y = df[target].astype(int)
    X = df.drop(columns=[target])

    object_columns = X.select_dtypes(include=['object']).columns
    X = X.drop(columns=object_columns)

    # Check for duplicated rows
    duplicated_rows = X.duplicated().sum()
    print(f'Number of duplicated rows: {duplicated_rows}')

    # Drop duplicated rows if any from X and y
    if duplicated_rows > 0:
        X = X.drop_duplicates()
        y = y.loc[X.index]
        print(f'Duplicated rows dropped. New shape: {X.shape}, {y.shape}')
    
    return X, y


def load_and_preprocess_data(filepath: str='dataset.json') -> tuple[pd.DataFrame, pd.Series]:
    """
    Load and preprocess the lottery dataset from a JSON file.
    
    Parameters:
    -----------
    filepath : str
        Path to the JSON dataset file
    
    Returns:
    --------
    X : pd.DataFrame
        Preprocessed feature dataframe
    y : pd.Series
        Target variable series
    """
    print("Loading and preprocessing data...")
    
    # Load and clean data
    df = load_data(filepath)
    print(f"Original dataset size: {len(df)}")
    
    # Engineer features
    X, y = engineer_features(df)
    
    print(f"Clean dataset size: {len(X)}")
    print(f"Target distribution:\n{y.value_counts()}")
    
    return X, y

def split_data(
        X: pd.DataFrame, 
        y: pd.Series
) -> tuple[
    pd.DataFrame, 
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.Series,
    list[str]
]:
    """
    Split data into train, validation, and test sets

    Parameters:
    -----------
    X : pd.DataFrame
        Feature dataframe
    y : pd.Series
        Target variable series

    Returns:
    --------
    X_train : pd.DataFrame
        Training feature set
    X_val : pd.DataFrame
        Validation feature set
    X_test : pd.DataFrame
        Test feature set
    y_train : pd.Series
        Training target set
    y_val : pd.Series
        Validation target set
    y_test : pd.Series
        Test target set
    columns : list[str]
        List of feature column names
    """
    selected_features = ['valorAcumuladoProximoConcurso', 
                         'valorEstimadoProximoConcurso', 
                         'valorArrecadado', 
                         'valorAcumuladoConcurso_0_5',
                         'valorAcumuladoConcursoEspecial']
    
    # Random split: separate test set (20%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Second split: separate validation set from remaining data
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )
    
    X_train = X_train[selected_features]
    X_val = X_val[selected_features]
    X_test = X_test[selected_features]

    print(f"Training: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test, selected_features


def train_model(X: pd.DataFrame, y: pd.Series, experiment_name: str = "lottery_prediction") -> tuple[Pipeline, list[str], str]:
    """
    Train LightGBM model with MLflow tracking and evaluation

    Parameters:
    -----------
    X : pd.DataFrame
        Feature dataframe
    y : pd.Series
        Target variable series
    experiment_name : str
        MLflow experiment name

    Returns:
    --------
    model : sklearn.pipeline.Pipeline
        Trained model pipeline
    feature_names : list[str]
        List of feature names used in the model
    run_id : str
        MLflow run ID for version tracking
    """
    print("\nTraining classification model...")
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test, selected_features = split_data(X, y)
    
    # Create preprocessing pipeline
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X_train.select_dtypes(include=['object']).columns
    
    # Set up MLflow experiment
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        
        # Log parameters
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 5)
        mlflow.log_param("learning_rate", 0.05)
        mlflow.log_param("num_leaves", 31)
        mlflow.log_param("min_child_samples", 20)
        mlflow.log_param("subsample", 0.8)
        mlflow.log_param("colsample_bytree", 0.8)
        mlflow.log_param("reg_alpha", 0.1)
        mlflow.log_param("reg_lambda", 0.1)
        mlflow.log_param("random_state", 42)
        
        model = Pipeline(steps=[
            ('classifier', LGBMClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.05,
                num_leaves=31,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42,
                verbosity=-1
            ))
        ])

        model.fit(X_train, y_train)

        # Calculate metrics using common function
        train_metrics = calculate_metrics(model, X_train, y_train)
        val_metrics = calculate_metrics(model, X_val, y_val)
        test_metrics = calculate_metrics(model, X_test, y_test)
        
        # Log metrics to MLflow
        mlflow.log_metric("train_accuracy", train_metrics['accuracy'])
        mlflow.log_metric("train_roc_auc", train_metrics['roc_auc'])
        mlflow.log_metric("val_accuracy", val_metrics['accuracy'])
        mlflow.log_metric("val_roc_auc", val_metrics['roc_auc'])
        mlflow.log_metric("test_accuracy", test_metrics['accuracy'])
        mlflow.log_metric("test_roc_auc", test_metrics['roc_auc'])
        
        # Evaluate and print report
        evaluate_model_report(model, X_train, y_train, X_val, y_val, X_test, y_test, selected_features)
        
        # Log model with MLflow
        mlflow.lightgbm.log_model(model.named_steps['classifier'], "model")
        
        print(f"\nMLflow run ID: {run_id}")
        print(f"MLflow experiment: {experiment_name}")
    
    return model, selected_features, run_id

def save_model_artifacts(
        model, 
        selected_features, 
        model_path='lottery_model.pkl', 
        features_path='features.txt'
):
    """
    Save the trained model and feature information

    Parameters:
    -----------
    model : sklearn.pipeline.Pipeline
        The trained model pipeline
    selected_features : list
        List of feature names used in the model
    model_path : str
        Path to save the trained model
    features_path : str
        Path to save the feature names
    """
    print(f"\nSaving model to {model_path}...")
    joblib.dump(model, model_path)
    
    print(f"Features saved to: {features_path}")
    with open(features_path, 'w') as f:
        for feature in selected_features:
            f.write(f"{feature}\n")
        
    print(f"Model and features saved successfully!")

