import pandas as pd
import numpy as np
import pickle
import joblib
from datetime import datetime
import os

def load_prediction_model(model_path='funding_prediction_model.pkl'):
    """
    Load the trained prediction model
    """
    try:
        if os.path.exists(model_path):
            with open(model_path, 'rb') as file:
                model = pickle.load(file)
            return model
        else:
            print(f"Model file {model_path} not found.")
            return None
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

def prepare_prediction_input(input_data):
    """
    Prepare input data for prediction
    
    Args:
        input_data (dict): Dictionary with input features
            - city: City of the startup
            - vertical: Sector/vertical of the startup
            - round: Funding round
            - investor_count: Number of investors
            - date: Date of funding (optional, defaults to current date)
    
    Returns:
        pd.DataFrame: Prepared input for the model
    """
    # Create a DataFrame from input data
    input_df = pd.DataFrame([input_data])
    
    # Process date if provided, otherwise use current date
    if 'date' in input_data and input_data['date']:
        try:
            input_date = pd.to_datetime(input_data['date'])
        except:
            input_date = datetime.now()
    else:
        input_date = datetime.now()
    
    # Extract features from date
    input_df['year'] = input_date.year
    input_df['month'] = input_date.month
    input_df['quarter'] = (input_date.month - 1) // 3 + 1
    
    # Ensure all required columns are present
    required_columns = ['city', 'vertical', 'round', 'investor_count', 'year', 'month', 'quarter']
    for col in required_columns:
        if col not in input_df.columns:
            if col == 'investor_count':
                input_df[col] = 1  # Default to 1 investor
            else:
                input_df[col] = 'Unknown'  # Default value for categorical variables
    
    # Keep only the required columns
    input_df = input_df[required_columns]
    
    return input_df

def predict_funding(input_data):
    """
    Predict funding amount based on input data
    
    Args:
        input_data (dict): Dictionary with input features
    
    Returns:
        dict: Prediction results
    """
    # Load the model
    model = load_prediction_model()
    if model is None:
        return {
            'success': False,
            'error': 'Failed to load prediction model. Please train the model first.'
        }
    
    try:
        # Prepare input data
        input_df = prepare_prediction_input(input_data)
        
        # Make prediction
        predicted_amount = model.predict(input_df)[0]
        
        # Round to 2 decimal places
        predicted_amount = round(predicted_amount, 2)
        
        # Prepare results
        result = {
            'success': True,
            'predicted_amount': predicted_amount,
            'input_data': input_data
        }
        
        return result
    
    except Exception as e:
        return {
            'success': False,
            'error': f'Prediction error: {str(e)}'
        }

def get_funding_range_description(amount):
    """
    Get a descriptive range for the predicted funding amount
    """
    if amount <= 1:
        return "Seed stage funding (≤ ₹1 Cr)"
    elif amount <= 10:
        return "Early stage funding (₹1-10 Cr)"
    elif amount <= 50:
        return "Series A range (₹10-50 Cr)"
    elif amount <= 100:
        return "Series B range (₹50-100 Cr)"
    elif amount <= 500:
        return "Series C/D range (₹100-500 Cr)"
    else:
        return "Late stage funding (> ₹500 Cr)"

def get_startup_recommendations(input_data, predicted_amount):
    """
    Generate recommendations based on startup details and predicted funding
    """
    recommendations = []
    
    # Recommendation based on funding range
    if predicted_amount <= 5:
        recommendations.append("Focus on bootstrapping and developing a minimum viable product.")
        recommendations.append("Consider approaching angel investors and participating in startup incubators.")
    elif predicted_amount <= 20:
        recommendations.append("Develop a strong pitch deck with clear revenue projections.")
        recommendations.append("Approach seed funding investors and early-stage VCs.")
    elif predicted_amount <= 100:
        recommendations.append("Prepare detailed growth and expansion plans.")
        recommendations.append("Target established venture capital firms with expertise in your sector.")
    else:
        recommendations.append("Develop comprehensive market expansion and product diversification strategies.")
        recommendations.append("Consider approaching multiple investment sources including major VCs and private equity firms.")
    
    # Sector-specific recommendations
    vertical = input_data.get('vertical', '').lower()
    if 'tech' in vertical or 'software' in vertical:
        recommendations.append("Focus on demonstrating user growth and engagement metrics.")
    elif 'ecommerce' in vertical:
        recommendations.append("Highlight customer acquisition cost and lifetime value metrics.")
    elif 'health' in vertical or 'healthcare' in vertical:
        recommendations.append("Emphasize regulatory compliance and clinical validation if applicable.")
    elif 'fintech' in vertical:
        recommendations.append("Showcase your user security measures and regulatory compliance framework.")
    
    # Location-based recommendations
    city = input_data.get('city', '').lower()
    if city in ['bangalore', 'bengaluru']:
        recommendations.append("Leverage Bangalore's tech ecosystem by connecting with established startups.")
    elif city in ['mumbai']:
        recommendations.append("Tap into Mumbai's financial networks for potential investors.")
    elif city in ['delhi', 'new delhi', 'gurugram']:
        recommendations.append("Connect with the NCR startup community for mentorship and networking opportunities.")
    
    return recommendations

if __name__ == "__main__":
    # Test prediction with sample input
    sample_input = {
        'city': 'Bengaluru',
        'vertical': 'Tech',
        'round': 'Series A',
        'investor_count': 3,
        'date': '2023-01-15'
    }
    
    result = predict_funding(sample_input)
    
    if result['success']:
        print(f"Predicted funding amount: ₹{result['predicted_amount']} Cr")
        print(f"Funding range: {get_funding_range_description(result['predicted_amount'])}")
        print("\nRecommendations:")
        for rec in get_startup_recommendations(sample_input, result['predicted_amount']):
            print(f"- {rec}")
    else:
        print(f"Prediction failed: {result['error']}")
