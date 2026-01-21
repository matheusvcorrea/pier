# Evaluate model
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix
import pandas as pd
from typing import Dict, Any


def calculate_metrics(
        model,
        X: pd.DataFrame,
        y: pd.Series
) -> Dict[str, Any]:
    """
    Calculate classification metrics for a given dataset.

    This function computes accuracy and ROC AUC metrics for a trained model
    on a given dataset.

    Parameters:
    -----------
        model:
            A trained scikit-learn pipeline or classifier with predict and
            predict_proba methods.
        X: pd.DataFrame
            Feature matrix.
        y: pd.Series
            Target labels.

    Returns:
    --------
        Dict[str, float]
            Dictionary containing:
            - 'accuracy': Accuracy score
            - 'roc_auc': ROC AUC score

    Notes:
        - Assumes binary classification (uses probability for class 1).
    """
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]
    
    return {
        'accuracy': accuracy_score(y, y_pred),
        'roc_auc': roc_auc_score(y, y_pred_proba)
    }


def evaluate_model_report(
        model, 
        X_train: pd.DataFrame,
        y_train: pd.Series, 
        X_val: pd.DataFrame,
        y_val: pd.Series, 
        X_test: pd.DataFrame,
        y_test: pd.Series,
        numeric_features: list
):
    """
    Evaluate a classification model and generate a comprehensive performance report.

    This function performs a thorough evaluation of a trained classification model
    across training, validation, and test sets. It calculates various performance
    metrics, detects overfitting, performs cross-validation, and extracts feature
    importance information.

    Parameters:
    -----------
        model:
            A trained scikit-learn pipeline or classifier with predict and
            predict_proba methods.
        X_train: pd.DataFrame
            Training feature matrix.
        y_train: pd.Series
            Training target labels.
        X_val: pd.DataFrame
            Validation feature matrix .
        y_val: pd.Series
            Validation target labels .
        X_test: pd.DataFr
            Test feature matrix ame.
        y_test: pd.Series
            Test target labels .
        numeric_features: list[str] 
            List of numeric feature names for feature importance extraction.

    Returns:
    --------
        None: This function prints all results to stdout.

    Raises:
        Exception: If feature importance extraction fails (caught and printed).

    Notes:
        - Assumes binary classification (uses probability for class 1).
        - Uses StratifiedKFold with 5 splits for cross-validation.
        - Overfitting is flagged if train-test ROC AUC gap > 0.1.
        - Minimal overfitting is indicated if gap < 0.05.
        - Feature importance requires model to have 'classifier' step with
          feature_importances_ attribute.
    """
    print("\n=== Model Performance Comparison ===")

    # Calculate metrics using common function
    train_metrics = calculate_metrics(model, X_train, y_train)
    val_metrics = calculate_metrics(model, X_val, y_val)
    test_metrics = calculate_metrics(model, X_test, y_test)
    
    print(f"Training Accuracy: {train_metrics['accuracy']:.4f}")
    print(f"Training ROC AUC:  {train_metrics['roc_auc']:.4f} \n")

    print(f"Validation Accuracy: {val_metrics['accuracy']:.4f}")
    print(f"Validation ROC AUC:  {val_metrics['roc_auc']:.4f} \n")

    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test ROC AUC:  {test_metrics['roc_auc']:.4f} \n")

    # Overfitting gap
    overfitting_gap = train_metrics['roc_auc'] - test_metrics['roc_auc']
    print(f"Overfitting Gap: {overfitting_gap:.4f}")
    if overfitting_gap > 0.1:
        print("⚠️  Warning: Significant overfitting detected")
    elif overfitting_gap < 0.05:
        print("✓ Good: Minimal overfitting")

    # Cross-validation
    print("\n=== Cross-Validation Results ===")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
    print(f"5-Fold CV ROC AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    # Test evaluation
    print("\n=== Test Set Evaluation ===")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test ROC AUC: {test_metrics['roc_auc']:.4f}")

    y_test_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_test_pred)
    print(f"\nConfusion Matrix:\nTN:{cm[0][0]} FP:{cm[0][1]} FN:{cm[1][0]} TP:{cm[1][1]}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_test_pred)}")

    # Feature importance
    try:
        feature_importances = model.named_steps['classifier'].feature_importances_
        if len(numeric_features) == len(feature_importances):
            feature_importance_df = pd.DataFrame({
                'feature': numeric_features,
                'importance': feature_importances
            }).sort_values('importance', ascending=False)
            print(f"\nTop Features:\n{feature_importance_df.head(10)}")
    except Exception as e:
        print(f"Could not extract feature importances: {e}")
    