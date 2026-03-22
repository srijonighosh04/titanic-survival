import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def download_data():
    os.makedirs('data', exist_ok=True)
    if not os.path.exists('data/titanic.csv'):
        print("Downloading Titanic dataset...")
        url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
        response = requests.get(url)
        with open('data/titanic.csv', 'wb') as f:
            f.write(response.content)
        print("Download complete.")
    else:
        print("Dataset already exists.")

def preprocess_data(df):
    print("Preprocessing data...")
    # Handle missing values
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    
    # Drop features that are redundant or have too many missing values (Cabin)
    df = df.drop(['Cabin', 'Ticket', 'Name', 'PassengerId'], axis=1)
    
    # Encode categorical variables
    df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)
    
    return df

def train_and_evaluate(df):
    print("Training Logistic Regression model...")
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train
    model = LogisticRegression(random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Predict
    y_pred = model.predict(X_test_scaled)
    
    # Evaluate
    acc = accuracy_score(y_test, y_pred)
    print(f"\nModel Evaluation:\nAccuracy: {acc:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    return model, X_test_scaled, y_test, y_pred, X.columns

def plot_results(model, X_test, y_test, y_pred, feature_names):
    print("Generating visualizations...")
    os.makedirs('output', exist_ok=True)
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Died', 'Survived'], 
                yticklabels=['Died', 'Survived'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig('output/confusion_matrix.png')
    plt.close()
    
    # 2. Feature Importance (Coefficients)
    coefs = pd.Series(model.coef_[0], index=feature_names).sort_values()
    plt.figure(figsize=(8, 6))
    coefs.plot(kind='barh')
    plt.title('Logistic Regression Coefficients (Feature Importance)')
    plt.xlabel('Log-Odds Coefficient')
    plt.tight_layout()
    plt.savefig('output/feature_importance.png')
    plt.close()
    
    print("Visualizations saved to 'output/' folder.")

def main():
    download_data()
    df = pd.read_csv('data/titanic.csv')
    df_clean = preprocess_data(df)
    model, X_test, y_test, y_pred, feature_names = train_and_evaluate(df_clean)
    plot_results(model, X_test, y_test, y_pred, feature_names)
    print("\nPipeline execution completed successfully.")

if __name__ == "__main__":
    main()
