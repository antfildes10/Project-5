"""
Lead Predictor Page
===================
Interactive form for predicting lead conversion probability.
Addresses Business Requirements BR2 and BR3.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os


def page_predictor_body():
    """
    Render the Lead Predictor page with interactive form.
    """
    st.title("Lead Conversion Predictor")

    st.info(
        """
        **Business Requirements 2 & 3**

        Enter the lead's characteristics below to predict their likelihood
        of conversion. The model will provide a probability score and
        classification to help prioritise sales outreach.
        """
    )

    st.markdown("---")

    # Load model
    @st.cache_resource
    def load_model():
        model_path = 'outputs/ml_pipeline/v1/clf_pipeline.pkl'
        if os.path.exists(model_path):
            return joblib.load(model_path)
        return None

    pipeline = load_model()

    if pipeline is None:
        st.warning(
            """
            Model not found. Please run the modelling notebook first
            to generate the trained pipeline.

            Expected path: `outputs/ml_pipeline/v1/clf_pipeline.pkl`
            """
        )
        return

    # Create input form
    st.header("Enter Lead Details")

    col1, col2 = st.columns(2)

    with col1:
        lead_origin = st.selectbox(
            "Lead Origin",
            options=['API', 'Landing Page Submission', 'Lead Add Form',
                     'Lead Import', 'Quick Add Form'],
            help="How did this lead enter the system?"
        )

        lead_source = st.selectbox(
            "Lead Source",
            options=['Google', 'Organic Search', 'Direct Traffic',
                     'Referral Sites', 'Facebook', 'Reference', 'Other'],
            help="Where did this lead come from?"
        )

        time_spent = st.slider(
            "Total Time on Website (seconds)",
            min_value=0,
            max_value=3000,
            value=300,
            step=10,
            help="Total time the lead has spent on the website"
        )

        total_visits = st.slider(
            "Total Website Visits",
            min_value=0,
            max_value=50,
            value=5,
            help="Number of times the lead has visited the website"
        )

    with col2:
        page_views = st.slider(
            "Page Views Per Visit",
            min_value=0.0,
            max_value=20.0,
            value=3.0,
            step=0.5,
            help="Average number of pages viewed per visit"
        )

        last_activity = st.selectbox(
            "Last Activity",
            options=['Email Opened', 'Page Visited on Website',
                     'Olark Chat Conversation', 'Form Submitted on Website',
                     'Email Link Clicked', 'SMS Sent', 'Other'],
            help="The lead's most recent interaction"
        )

        occupation = st.selectbox(
            "Current Occupation",
            options=['Working Professional', 'Unemployed', 'Student',
                     'Businessman', 'Other'],
            help="Lead's current employment status"
        )

        st.markdown("**Contact Preferences:**")
        do_not_email = st.checkbox("Do Not Email", value=False)
        do_not_call = st.checkbox("Do Not Call", value=False)

    st.markdown("---")

    # Load feature names
    @st.cache_data
    def load_feature_names():
        import json
        feature_path = 'outputs/ml_pipeline/v1/feature_names.json'
        if os.path.exists(feature_path):
            with open(feature_path, 'r') as f:
                return json.load(f)
        return None

    feature_names = load_feature_names()

    # Predict button
    if st.button("Predict Conversion", type="primary", use_container_width=True):
        # Show spinner while predicting
        with st.spinner("Analysing lead..."):
            try:
                # Create input DataFrame with all required features
                input_data = {name: 0 for name in feature_names}

                # Set basic features
                input_data['Do Not Email'] = 1 if do_not_email else 0
                input_data['Do Not Call'] = 1 if do_not_call else 0
                input_data['TotalVisits'] = total_visits
                input_data['Total Time Spent on Website'] = time_spent
                input_data['Page Views Per Visit'] = page_views

                # Set engagement score (derived feature)
                input_data['Engagement_Score'] = (time_spent / 100) + (total_visits * 2) + (page_views * 3)
                input_data['High_Engagement_Time'] = 1 if time_spent > 300 else 0
                input_data['Contact_Restricted'] = 1 if (do_not_email and do_not_call) else 0

                # Set visit category
                if total_visits <= 2:
                    input_data['Visit_Category_Low (1-2)'] = 1
                elif total_visits <= 10:
                    input_data['Visit_Category_Medium (3-10)'] = 1

                # Set lead origin one-hot encoding
                origin_mapping = {
                    'Landing Page Submission': 'Lead Origin_Landing Page Submission',
                    'Lead Add Form': 'Lead Origin_Lead Add Form',
                    'Lead Import': 'Lead Origin_Lead Import'
                }
                if lead_origin in origin_mapping:
                    input_data[origin_mapping[lead_origin]] = 1

                # Set lead source one-hot encoding
                source_mapping = {
                    'Google': 'Lead Source_Google',
                    'Organic Search': 'Lead Source_Organic Search',
                    'Reference': 'Lead Source_Reference',
                    'Other': 'Lead Source_Other'
                }
                if lead_source in source_mapping:
                    input_data[source_mapping[lead_source]] = 1
                elif lead_source not in ['Direct Traffic', 'Referral Sites', 'Facebook']:
                    input_data['Lead Source_Other'] = 1

                # Set last activity one-hot encoding
                activity_mapping = {
                    'Email Opened': 'Last Activity_Email Opened',
                    'Page Visited on Website': 'Last Activity_Page Visited on Website',
                    'Olark Chat Conversation': 'Last Activity_Olark Chat Conversation',
                    'Email Link Clicked': 'Last Activity_Email Link Clicked',
                    'SMS Sent': 'Last Activity_SMS Sent',
                    'Other': 'Last Activity_Other'
                }
                if last_activity in activity_mapping:
                    input_data[activity_mapping[last_activity]] = 1
                else:
                    input_data['Last Activity_Other'] = 1

                # Set occupation one-hot encoding
                occupation_mapping = {
                    'Working Professional': 'What is your current occupation_Working Professional',
                    'Student': 'What is your current occupation_Student',
                    'Unemployed': 'What is your current occupation_Unemployed',
                    'Other': 'What is your current occupation_Other'
                }
                if occupation in occupation_mapping:
                    input_data[occupation_mapping[occupation]] = 1
                else:
                    input_data['What is your current occupation_Unknown'] = 1

                # Set unknown categories for unspecified fields
                input_data['Tags_Unknown'] = 1
                input_data['Country_Unknown'] = 1
                input_data['Specialization_Unknown'] = 1
                input_data['What matters most to you in choosing a course_Unknown'] = 1
                input_data['City_Unknown'] = 1
                input_data['Asymmetrique Activity Index_Unknown'] = 1
                input_data['Asymmetrique Profile Index_Unknown'] = 1
                input_data['Last Notable Activity_Other'] = 1

                # Create DataFrame and make prediction
                input_df = pd.DataFrame([input_data])
                input_df = input_df[feature_names]  # Ensure correct column order

                # Use actual model prediction
                probability = pipeline.predict_proba(input_df)[0][1]
                prediction = 1 if probability >= 0.5 else 0

                # Display results
                st.markdown("---")
                st.header("Prediction Result")

                col1, col2, col3 = st.columns(3)

                with col1:
                    if prediction == 1:
                        st.success("### HOT LEAD")
                        st.markdown("This lead is **likely to convert**.")
                    else:
                        st.warning("### COLD LEAD")
                        st.markdown("This lead is **unlikely to convert**.")

                with col2:
                    st.metric(
                        "Conversion Probability",
                        f"{probability:.0%}",
                        delta=f"{(probability - 0.3):.0%} vs avg"
                    )

                with col3:
                    if probability >= 0.7:
                        priority = "HIGH"
                        priority_color = "green"
                    elif probability >= 0.4:
                        priority = "MEDIUM"
                        priority_color = "orange"
                    else:
                        priority = "LOW"
                        priority_color = "red"

                    st.markdown(f"### Priority: :{priority_color}[{priority}]")

                # Probability bar
                st.progress(probability)

                # Recommendation
                st.markdown("---")
                st.subheader("Recommendation")

                if probability >= 0.7:
                    st.success(
                        """
                        **Priority: HIGH - Immediate Action Recommended**

                        This lead shows strong conversion signals:
                        - High engagement metrics
                        - Quality lead source
                        - Recent meaningful activity

                        **Suggested Actions:**
                        1. Contact within 24 hours
                        2. Personalised outreach
                        3. Schedule product demo
                        """
                    )
                elif probability >= 0.4:
                    st.warning(
                        """
                        **Priority: MEDIUM - Nurture Recommended**

                        This lead has moderate conversion potential:
                        - Some positive signals present
                        - May need additional engagement

                        **Suggested Actions:**
                        1. Add to nurture email sequence
                        2. Send relevant content
                        3. Follow up in 3-5 days
                        """
                    )
                else:
                    st.info(
                        """
                        **Priority: LOW - Long-term Nurture**

                        This lead has lower conversion probability:
                        - Limited engagement signals
                        - Early-stage awareness

                        **Suggested Actions:**
                        1. Add to long-term drip campaign
                        2. Do not prioritise for direct outreach
                        3. Monitor for engagement increases
                        """
                    )

                # Key Factors
                st.markdown("---")
                st.subheader("Key Factors Influencing This Prediction")

                factors = []
                if time_spent > 500:
                    factors.append(("High engagement time", "+", "green"))
                elif time_spent < 100:
                    factors.append(("Low engagement time", "-", "red"))

                if total_visits >= 3 and total_visits <= 10:
                    factors.append(("Optimal visit frequency", "+", "green"))
                elif total_visits < 3:
                    factors.append(("Low visit count", "-", "red"))

                if lead_source in ['Referral Sites', 'Reference']:
                    factors.append(("High-quality lead source", "+", "green"))

                if last_activity in ['Email Opened', 'Page Visited on Website']:
                    factors.append(("Recent high engagement", "+", "green"))

                if do_not_email and do_not_call:
                    factors.append(("Contact restrictions", "-", "red"))

                if factors:
                    for factor, sign, color in factors:
                        st.markdown(f":{color}[{sign}] {factor}")
                else:
                    st.markdown("No standout factors identified.")

            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                st.info("Please check that all fields are filled correctly.")

    # Batch Prediction
    st.markdown("---")
    st.header("Batch Prediction")

    st.markdown(
        """
        Upload a CSV file with multiple leads to score them all at once.
        The file should include columns for the features shown above.
        """
    )

    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.write(f"Uploaded {len(batch_df)} leads")
            st.dataframe(batch_df.head())

            if st.button("Score All Leads"):
                st.info("Batch prediction feature will be available after model training.")
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
