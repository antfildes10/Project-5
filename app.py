"""
Lead Conversion Predictor - Streamlit Dashboard
================================================
Main application entry point.

This dashboard provides:
- Lead conversion analysis and visualizations
- Hypothesis validation results
- Interactive lead scoring predictor
- Model performance metrics

Author: [Your Name]
Date: December 2024
"""

import streamlit as st
from app_pages.multipage import MultiPage

# Import page functions
from app_pages.page_summary import page_summary_body
from app_pages.page_lead_study import page_lead_study_body
from app_pages.page_hypothesis import page_hypothesis_body
from app_pages.page_predictor import page_predictor_body
from app_pages.page_model_performance import page_model_performance_body
from app_pages.page_technical import page_technical_body

# Create app instance
app = MultiPage(app_name="Lead Conversion Predictor")

# Add pages
app.add_page("Project Summary", page_summary_body)
app.add_page("Lead Conversion Study", page_lead_study_body)
app.add_page("Hypothesis Validation", page_hypothesis_body)
app.add_page("Lead Predictor", page_predictor_body)
app.add_page("Model Performance", page_model_performance_body)
app.add_page("Technical Details", page_technical_body)

# Run the app
app.run()
