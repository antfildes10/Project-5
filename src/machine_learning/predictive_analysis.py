"""
Predictive Analysis Module
==========================
Functions for making predictions with the trained model.
"""

import pandas as pd


def predict_conversion(pipeline, input_data):
    """
    Predict lead conversion using the trained pipeline.

    Args:
        pipeline: Trained sklearn pipeline
        input_data: DataFrame with lead features

    Returns:
        tuple: (prediction, probability)
            - prediction: 0 or 1
            - probability: float between 0 and 1
    """
    # Make prediction
    prediction = pipeline.predict(input_data)[0]

    # Get probability
    probability = pipeline.predict_proba(input_data)[0][1]

    return prediction, probability


def batch_predict(pipeline, data):
    """
    Make predictions for multiple leads.

    Args:
        pipeline: Trained sklearn pipeline
        data: DataFrame with multiple leads

    Returns:
        DataFrame: Original data with predictions and probabilities
    """
    predictions = pipeline.predict(data)
    probabilities = pipeline.predict_proba(data)[:, 1]

    result = data.copy()
    result['Prediction'] = predictions
    result['Probability'] = probabilities
    result['Priority'] = pd.cut(
        probabilities,
        bins=[0, 0.4, 0.7, 1.0],
        labels=['Low', 'Medium', 'High']
    )

    return result


def get_priority_label(probability):
    """
    Get priority label based on probability.

    Args:
        probability: Conversion probability (0-1)

    Returns:
        str: Priority label ('High', 'Medium', 'Low')
    """
    if probability >= 0.7:
        return 'High'
    elif probability >= 0.4:
        return 'Medium'
    else:
        return 'Low'


def get_recommendation(probability):
    """
    Get action recommendation based on probability.

    Args:
        probability: Conversion probability (0-1)

    Returns:
        dict: Recommendation with priority, action, and description
    """
    if probability >= 0.7:
        return {
            'priority': 'HIGH',
            'action': 'Immediate Contact',
            'description': (
                'This lead shows strong conversion signals. '
                'Contact within 24 hours with personalised outreach.'
            )
        }
    elif probability >= 0.4:
        return {
            'priority': 'MEDIUM',
            'action': 'Nurture Campaign',
            'description': (
                'This lead has moderate potential. '
                'Add to email nurture sequence and follow up in 3-5 days.'
            )
        }
    else:
        return {
            'priority': 'LOW',
            'action': 'Long-term Nurture',
            'description': (
                'This lead has lower probability. '
                'Add to long-term drip campaign; do not prioritise.'
            )
        }
