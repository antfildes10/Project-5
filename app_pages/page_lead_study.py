"""
Lead Conversion Study Page
==========================
Displays EDA visualizations, correlation analysis, and PPS results.
Addresses Business Requirement 1.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


def page_lead_study_body():
    """
    Render the Lead Conversion Study page.
    """
    st.title("Lead Conversion Study")

    st.info(
        """
        **Business Requirement 1**

        The client wants to understand which lead characteristics most strongly
        correlate with conversion. This page presents the data analysis findings.
        """
    )

    st.markdown("---")

    # Load data
    @st.cache_data
    def load_data():
        return pd.read_csv('outputs/datasets/cleaned/leads_cleaned.csv')

    try:
        df = load_data()
        data_loaded = True
    except FileNotFoundError:
        st.warning("Dataset not found. Please run the data pipeline notebooks first.")
        data_loaded = False
        df = None

    if data_loaded:
        # Target Distribution
        st.header("1. Target Variable Distribution")

        col1, col2 = st.columns([1, 2])

        with col1:
            st.metric("Total Leads", f"{len(df):,}")
            st.metric("Converted", f"{df['Converted'].sum():,}")
            st.metric("Conversion Rate", f"{df['Converted'].mean():.1%}")

        with col2:
            fig, ax = plt.subplots(figsize=(8, 4))
            colors = ['#DC3545', '#28A745']
            df['Converted'].value_counts().plot(
                kind='bar', color=colors, ax=ax, edgecolor='black'
            )
            ax.set_title('Target Variable Distribution')
            ax.set_xlabel('Converted (0 = No, 1 = Yes)')
            ax.set_ylabel('Count')
            ax.set_xticklabels(['Not Converted', 'Converted'], rotation=0)
            st.pyplot(fig)
            plt.close()

        st.markdown(
            """
            **Interpretation:**
            The dataset shows class imbalance with approximately 30% converted leads
            and 70% non-converted. This imbalance is handled during modelling using
            class weighting techniques.
            """
        )

        st.markdown("---")

        # Numerical Features Analysis
        st.header("2. Key Behavioural Features")

        feature_option = st.selectbox(
            "Select feature to analyse:",
            ['Total Time Spent on Website', 'TotalVisits', 'Page Views Per Visit']
        )

        if feature_option in df.columns:
            col1, col2 = st.columns(2)

            with col1:
                # Distribution by conversion
                fig, ax = plt.subplots(figsize=(8, 5))
                for converted, color, label in [(0, '#DC3545', 'Not Converted'),
                                                 (1, '#28A745', 'Converted')]:
                    data = df[df['Converted'] == converted][feature_option].dropna()
                    ax.hist(data, bins=30, alpha=0.6, color=color,
                            label=label, edgecolor='black')
                ax.set_xlabel(feature_option)
                ax.set_ylabel('Frequency')
                ax.set_title(f'{feature_option} Distribution by Conversion')
                ax.legend()
                st.pyplot(fig)
                plt.close()

            with col2:
                # Box plot
                fig, ax = plt.subplots(figsize=(8, 5))
                df.boxplot(column=feature_option, by='Converted', ax=ax,
                           patch_artist=True, boxprops=dict(facecolor='lightblue'))
                ax.set_title(f'{feature_option} by Conversion Status')
                ax.set_xlabel('Converted')
                plt.suptitle('')
                st.pyplot(fig)
                plt.close()

            # Statistics
            st.subheader("Statistics")
            stats_df = df.groupby('Converted')[feature_option].describe()
            st.dataframe(stats_df)

            st.markdown(
                f"""
                **Interpretation:**
                Converted leads typically show higher values for {feature_option}.
                This suggests that increased engagement is a positive signal
                for conversion likelihood.
                """
            )

        st.markdown("---")

        # Correlation Analysis
        st.header("3. Correlation Analysis")

        # Select numerical columns
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

        # Correlation matrix
        corr_matrix = df[numerical_cols].corr()

        fig, ax = plt.subplots(figsize=(12, 8))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
                    cmap='RdBu_r', center=0, ax=ax, square=True)
        ax.set_title('Feature Correlation Matrix')
        st.pyplot(fig)
        plt.close()

        # Correlation with target
        target_corr = corr_matrix['Converted'].drop('Converted').sort_values(ascending=False)

        st.subheader("Correlation with Target (Converted)")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Top Positive Correlations:**")
            st.dataframe(target_corr.head(5).to_frame('Correlation'))

        with col2:
            st.markdown("**Top Negative Correlations:**")
            st.dataframe(target_corr.tail(5).to_frame('Correlation'))

        st.markdown(
            """
            **Interpretation:**
            - **Total Time Spent on Website** shows the strongest positive correlation
              with conversion, indicating engagement duration is a key predictor.
            - **TotalVisits** also shows positive correlation - repeat visitors are
              more likely to convert.
            - Some features show weak or no correlation and may be less predictive.
            """
        )

        st.markdown("---")

        # PPS Analysis
        st.header("4. Predictive Power Score (PPS) Analysis")

        st.info(
            """
            **What is PPS?**

            Predictive Power Score measures how well one variable predicts another.
            Unlike correlation, PPS:
            - Works with categorical variables
            - Detects non-linear relationships
            - Is asymmetric (A predicting B ≠ B predicting A)
            """
        )

        # Display PPS chart if it exists
        pps_image_path = 'outputs/figures/pps_scores.png'
        if os.path.exists(pps_image_path):
            st.image(pps_image_path, caption='Predictive Power Scores for Target Variable')
        else:
            st.info("PPS chart will be generated after running the EDA notebook.")

        st.markdown(
            """
            **Key PPS Findings:**
            1. **Total Time Spent on Website** - Strongest predictor of conversion
            2. **Tags** - Sales qualification significantly impacts conversion
            3. **Lead Source** - Traffic source has moderate predictive power

            These findings inform our feature selection for the ML model.
            """
        )

        st.markdown("---")

        # Categorical Analysis
        st.header("5. Conversion by Categorical Features")

        cat_cols = ['Lead Source', 'Lead Origin', 'Last Activity']
        cat_cols = [c for c in cat_cols if c in df.columns]

        if cat_cols:
            cat_feature = st.selectbox("Select categorical feature:", cat_cols)

            # Calculate conversion rate
            conv_rate = df.groupby(cat_feature)['Converted'].agg(['mean', 'count'])
            conv_rate.columns = ['Conversion Rate', 'Count']
            conv_rate = conv_rate[conv_rate['Count'] >= 50]  # Filter low counts
            conv_rate = conv_rate.sort_values('Conversion Rate', ascending=True)

            fig, ax = plt.subplots(figsize=(10, 6))
            colors = plt.cm.RdYlGn(conv_rate['Conversion Rate'])
            bars = ax.barh(conv_rate.index, conv_rate['Conversion Rate'], color=colors)
            ax.axvline(x=df['Converted'].mean(), color='red', linestyle='--',
                       label=f'Overall Rate ({df["Converted"].mean():.1%})')
            ax.set_xlabel('Conversion Rate')
            ax.set_title(f'Conversion Rate by {cat_feature}')
            ax.set_xlim(0, 1)
            ax.legend()
            st.pyplot(fig)
            plt.close()

            st.dataframe(conv_rate.sort_values('Conversion Rate', ascending=False))

            st.markdown(
                f"""
                **Interpretation:**
                Different {cat_feature} categories show varying conversion rates.
                Categories above the red line (overall average) represent
                higher-converting segments that should be prioritised.
                """
            )

    # Summary
    st.markdown("---")
    st.header("Key Takeaways")

    st.success(
        """
        **Main Findings for Business Requirement 1:**

        1. **Engagement Time is Critical:** Leads spending more time on the website
           are significantly more likely to convert.

        2. **Source Matters:** Referral leads show highest conversion rates,
           while some paid channels underperform.

        3. **Recent Activity Signals Intent:** Leads with recent high-engagement
           activities (email opens, page visits) are more likely to convert.

        4. **Optimal Visit Range Exists:** Leads with 3-10 visits tend to convert
           better than those with very few or very many visits.
        """
    )
