"""
Data Management Module
======================
Functions for loading data and model artifacts.
"""

import streamlit as st
import pandas as pd
import joblib
import json
import os


@st.cache_data
def load_data(filepath='outputs/datasets/cleaned/leads_cleaned.csv'):
    """
    Load and cache the cleaned dataset.

    Args:
        filepath: Path to the CSV file

    Returns:
        pd.DataFrame: Loaded dataset
    """
    return pd.read_csv(filepath)


@st.cache_resource
def load_pipeline(version='v1'):
    """
    Load and cache the ML pipeline.

    Args:
        version: Model version to load (default: 'v1')

    Returns:
        sklearn Pipeline: Fitted classification pipeline
    """
    path = f'outputs/ml_pipeline/{version}/clf_pipeline.pkl'
    if os.path.exists(path):
        return joblib.load(path)
    return None


@st.cache_data
def load_feature_importance(version='v1'):
    """
    Load feature importance data.

    Args:
        version: Model version (default: 'v1')

    Returns:
        pd.DataFrame: Feature importance rankings
    """
    path = f'outputs/ml_pipeline/{version}/feature_importance.csv'
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


@st.cache_data
def load_evaluation_metrics(version='v1'):
    """
    Load model evaluation metrics.

    Args:
        version: Model version (default: 'v1')

    Returns:
        dict: Evaluation metrics
    """
    path = f'outputs/ml_pipeline/{version}/evaluation_report.json'
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None


@st.cache_data
def load_feature_names(version='v1'):
    """
    Load feature names used by the model.

    Args:
        version: Model version (default: 'v1')

    Returns:
        list: Feature names
    """
    path = f'outputs/ml_pipeline/{version}/feature_names.json'
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None


def load_hypothesis_summary():
    """
    Load hypothesis testing summary.

    Returns:
        pd.DataFrame: Hypothesis results summary
    """
    path = 'outputs/hypothesis_summary.csv'
    if os.path.exists(path):
        return pd.read_csv(path)
    return None
