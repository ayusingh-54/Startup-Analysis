import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_extraction.text import CountVectorizer
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data(filepath='startup_data_processed.csv'):
    """
    Load and preprocess the data for model training
    """
    print("Loading data...")
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"File {filepath} not found. Please run data_cleaning.py first.")
        return None
    
    # Convert date column to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Extract features from date
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    
    # Clean and prepare features
    print("Preparing features...")
    
    # Handle missing values
    df['city'] = df['city'].fillna('Unknown')
    df['vertical'] = df['vertical'].fillna('Unknown')
    df['round'] = df['round'].fillna('Unknown')
    df['investors'] = df['investors'].fillna('Unknown')
    
    # Create investor count feature
    df['investor_count'] = df['investors'].apply(lambda x: len(str(x).split(',')) if x != 'Unknown' else 0)
    
    # Filter out outliers for better model performance
    upper_bound = df['amount'].quantile(0.99)
    df_filtered = df[df['amount'] <= upper_bound]
    
    print(f"Data loaded and processed. Shape: {df_filtered.shape}")
    return df_filtered

def create_features_target(df):
    """
    Create features and target variables for modeling
    """
    # Define features and target
    X = df[['city', 'vertical', 'round', 'investor_count', 'year', 'month', 'quarter']]
    y = df['amount']
    
    return X, y

def build_and_train_model(X, y):
    """
    Build and train the prediction model
    """
    print("Building and training model...")
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define preprocessing for categorical features
    categorical_features = ['city', 'vertical', 'round']
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Define preprocessing for numerical features
    numerical_features = ['investor_count', 'year', 'month', 'quarter']
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_features),
            ('num', numerical_transformer, numerical_features)
        ])
    
    # Create the model pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', GradientBoostingRegressor(
            n_estimators=100, 
            learning_rate=0.1, 
            max_depth=5, 
            random_state=42,
            loss='huber'
        ))
    ])
    
    # Train the model
    print("Training model...")
    pipeline.fit(X_train, y_train)
    
    # Evaluate the model
    print("Evaluating model...")
    y_pred = pipeline.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model Performance:")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"Root Mean Squared Error: {rmse:.2f}")
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"R² Score: {r2:.2f}")
    
    # Feature importance
    if hasattr(pipeline.named_steps['regressor'], 'feature_importances_'):
        # Get feature names from the preprocessor
        cat_features = pipeline.named_steps['preprocessor'].transformers_[0][1]['onehot'].get_feature_names_out(categorical_features)
        features = list(cat_features) + numerical_features
        
        # Get feature importance
        feature_importance = pipeline.named_steps['regressor'].feature_importances_
        
        # Create feature importance plot
        plt.figure(figsize=(10, 6))
        
        # Handle the case where feature_importance might have a different length than features
        if len(feature_importance) <= len(features):
            sorted_idx = np.argsort(feature_importance)[::-1][:10]  # Top 10 features
            top_features = [features[i] if i < len(features) else f"Feature_{i}" for i in sorted_idx]
            plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx])
            plt.yticks(range(len(sorted_idx)), [top_features[i] for i in range(len(sorted_idx))])
        else:
            # If feature_importance has more elements than features, just show top N importances
            sorted_idx = np.argsort(feature_importance)[::-1][:10]
            plt.barh(range(10), feature_importance[sorted_idx])
            plt.yticks(range(10), [f"Feature_{i}" for i in sorted_idx])
            
        plt.xlabel('Feature Importance')
        plt.title('Top Features by Importance')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()
    
    return pipeline, (X_test, y_test, y_pred)

def save_model(model, filename='funding_prediction_model.pkl'):
    """
    Save the trained model to a file
    """
    try:
        with open(filename, 'wb') as file:
            pickle.dump(model, file)
        print(f"Model saved to {filename}")
        return True
    except Exception as e:
        print(f"Error saving model: {str(e)}")
        return False

def generate_evaluation_plots(eval_data, save_path='model_evaluation.png'):
    """
    Generate plots to evaluate model performance
    """
    X_test, y_test, y_pred = eval_data
    
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Actual vs Predicted plot
    axes[0, 0].scatter(y_test, y_pred, alpha=0.5)
    axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    axes[0, 0].set_xlabel('Actual')
    axes[0, 0].set_ylabel('Predicted')
    axes[0, 0].set_title('Actual vs Predicted Funding')
    
    # Residuals plot
    residuals = y_test - y_pred
    axes[0, 1].scatter(y_pred, residuals, alpha=0.5)
    axes[0, 1].axhline(y=0, color='r', linestyle='--')
    axes[0, 1].set_xlabel('Predicted')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].set_title('Residuals Plot')
    
    # Distribution of predicted values
    axes[1, 0].hist(y_pred, bins=30, alpha=0.7, color='skyblue')
    axes[1, 0].set_xlabel('Predicted Funding')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution of Predicted Funding')
    
    # Distribution of residuals
    axes[1, 1].hist(residuals, bins=30, alpha=0.7, color='salmon')
    axes[1, 1].set_xlabel('Residuals')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Distribution of Residuals')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Evaluation plots saved to {save_path}")

def train_and_save_model(data_path='startup_data_processed.csv'):
    """
    Complete model training pipeline
    """
    start_time = datetime.now()
    print(f"Starting model training at {start_time}")
    
    # Load and preprocess data
    df = load_and_preprocess_data(data_path)
    if df is None:
        return False
    
    # Create features and target
    X, y = create_features_target(df)
    
    # Build and train model
    model, eval_data = build_and_train_model(X, y)
    
    # Generate evaluation plots
    generate_evaluation_plots(eval_data)
    
    # Save model
    save_success = save_model(model)
    
    end_time = datetime.now()
    print(f"Model training completed at {end_time}")
    print(f"Total training time: {end_time - start_time}")
    
    return save_success

if __name__ == "__main__":
    print("Starting model training process...")
    train_and_save_model()
    print("Model training process completed!")
    print("Model training process completed!")
