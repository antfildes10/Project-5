"""
Project Summary Page
====================
Displays project overview, business requirements, and dataset information.
"""

import streamlit as st


def page_summary_body():
    """
    Render the Project Summary page.
    """
    st.title("Lead Conversion Predictor")

    st.markdown("---")

    # Project Overview
    st.header("Project Overview")
    st.info(
        """
        This dashboard uses **Machine Learning** to predict which sales leads
        are most likely to convert into paying customers. By identifying
        high-probability leads, sales teams can prioritise their efforts
        and improve conversion rates.

        **Business Context:** Velocity Software Solutions is a B2B software company
        selling to SMEs. The sales team currently pursues all leads equally,
        resulting in a 30% conversion rate and significant wasted effort.
        """
    )

    # Business Requirements
    st.header("Business Requirements")

    st.markdown(
        """
        The project addresses three key business requirements:

        **BR1 - Lead Characteristic Analysis**
        * Understand which lead attributes correlate with conversion
        * Identify the most important predictors of conversion success
        * Visualise relationships between features and conversion rates

        **BR2 - Conversion Prediction**
        * Build a classification model to predict whether a lead will convert
        * Achieve minimum 75% recall (capture most potential converters)
        * Achieve minimum 80% precision (accuracy of positive predictions)

        **BR3 - Interactive Dashboard**
        * Provide an interface for sales staff to score new leads
        * Display conversion probability and key influencing factors
        * Enable data-driven prioritisation of sales outreach
        """
    )

    # Dataset Information
    st.header("Dataset Information")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Records", "9,240")
    with col2:
        st.metric("Features", "37")
    with col3:
        st.metric("Target Variable", "Converted")
    with col4:
        st.metric("Conversion Rate", "~30%")

    st.markdown(
        """
        **Data Source:** [Kaggle Leads Dataset](https://www.kaggle.com/datasets/ashydv/leads-dataset)

        The dataset contains historical lead data including:
        * **Behavioural features:** Website visits, time spent, page views
        * **Acquisition features:** Lead source, lead origin
        * **Demographic features:** Country, occupation
        * **Engagement features:** Last activity, communication preferences
        """
    )

    # Hypotheses
    st.header("Project Hypotheses")

    st.markdown(
        """
        We investigate four hypotheses about lead conversion:

        | # | Hypothesis | Test Method |
        |---|------------|-------------|
        | H1 | Leads spending more time on website convert at higher rates | t-test |
        | H2 | Lead source significantly impacts conversion probability | Chi-square |
        | H3 | Recent high-engagement activities indicate higher conversion | Chi-square |
        | H4 | Optimal visit frequency exists (3-10 visits) | Chi-square |

        *See the Hypothesis Validation page for detailed results.*
        """
    )

    # Navigation Guide
    st.header("Dashboard Navigation")

    st.markdown(
        """
        Use the sidebar to navigate between pages:

        | Page | Description | Business Requirement |
        |------|-------------|---------------------|
        | **Lead Conversion Study** | EDA visualisations and correlations | BR1 |
        | **Hypothesis Validation** | Statistical test results | BR1 |
        | **Lead Predictor** | Interactive prediction tool | BR2, BR3 |
        | **Model Performance** | Metrics and success assessment | BR2 |
        | **Technical Details** | Pipeline and methodology | - |
        """
    )

    # Link to README
    st.markdown("---")
    st.markdown(
        """
        For full documentation, see the
        [Project README](https://github.com/yourusername/lead-conversion-predictor)
        """
    )
