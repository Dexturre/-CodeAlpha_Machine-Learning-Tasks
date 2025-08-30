import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix, 
                           classification_report, roc_curve)
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

def load_and_preprocess_data(file_path):
    """Load and preprocess the credit scoring data"""
    # Load data
    df = pd.read_csv(file_path)
    
    print("Dataset Info:")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Missing values:\n{df.isnull().sum()}")
    print(f"Target distribution:\n{df['creditworthy'].value_counts()}")
    
    # Feature engineering
    df = feature_engineering(df)
    
    # Separate features and target
    X = df.drop('creditworthy', axis=1)
    y = df['creditworthy']
    
    return X, y, df

def feature_engineering(df):
    """Perform feature engineering on the dataset"""
    # Create new features
    df['debt_to_income_ratio'] = df['debt'] / df['income']
    df['loan_to_income_ratio'] = df['loan_amount'] / df['income']
    df['payment_consistency'] = df['payment_history'] * df['employment_length']
    df['credit_utilization_score'] = (1 - df['credit_utilization']) * df['credit_score']
    
    # Drop original columns that were used to create new features
    df = df.drop(['debt', 'loan_amount', 'payment_history', 'credit_utilization'], axis=1)
    
    print("After feature engineering:")
    print(f"New columns: {list(df.columns)}")
    
    return df

def train_and_evaluate_models(X, y):
    """Train and evaluate different classification models"""
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f"Training {name}")
        print(f"{'='*50}")
        
        # Train model
        if name == 'Logistic Regression':
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Store results
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'model': model,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        # Print results
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"Confusion Matrix:\n{cm}")
        
        # Classification report
        print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
    
    return results, X_test, y_test

def plot_roc_curves(results, y_test):
    """Plot ROC curves for all models"""
    plt.figure(figsize=(10, 8))
    
    for name, result in results.items():
        fpr, tpr, _ = roc_curve(y_test, result['y_pred_proba'])
        plt.plot(fpr, tpr, label=f'{name} (AUC = {result["roc_auc"]:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Credit Scoring Models')
    plt.legend()
    plt.grid(True)
    plt.savefig('roc_curves.png')
    plt.show()

def plot_feature_importance(model, feature_names, model_name):
    """Plot feature importance for tree-based models"""
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        indices = np.argsort(importance)[::-1]
        
        plt.figure(figsize=(12, 8))
        plt.title(f'Feature Importance - {model_name}')
        plt.bar(range(len(importance)), importance[indices])
        plt.xticks(range(len(importance)), [feature_names[i] for i in indices], rotation=45)
        plt.tight_layout()
        plt.savefig(f'feature_importance_{model_name.lower().replace(" ", "_")}.png')
        plt.show()

def main():
    """Main function to run the credit scoring model"""
    print("Credit Scoring Model - Predicting Creditworthiness")
    print("="*60)
    
    # Load and preprocess data
    X, y, df = load_and_preprocess_data('data.csv')
    
    # Train and evaluate models
    results, X_test, y_test = train_and_evaluate_models(X, y)
    
    # Plot ROC curves
    plot_roc_curves(results, y_test)
    
    # Plot feature importance for tree-based models
    feature_names = X.columns.tolist()
    for name, result in results.items():
        if name in ['Decision Tree', 'Random Forest']:
            plot_feature_importance(result['model'], feature_names, name)
    
    # Compare model performance
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    
    comparison_df = pd.DataFrame({
        'Model': list(results.keys()),
        'Accuracy': [results[m]['accuracy'] for m in results],
        'Precision': [results[m]['precision'] for m in results],
        'Recall': [results[m]['recall'] for m in results],
        'F1-Score': [results[m]['f1'] for m in results],
        'ROC-AUC': [results[m]['roc_auc'] for m in results]
    })
    
    print(comparison_df.round(4))
    
    # Save results to CSV
    comparison_df.to_csv('model_comparison.csv', index=False)
    print("\nModel comparison saved to 'model_comparison.csv'")

if __name__ == "__main__":
    main()
