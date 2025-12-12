"""
Technical Details Page
======================
Displays ML pipeline architecture, methodology, and version information.
"""

import streamlit as st
import json
import os


def page_technical_body():
    """
    Render the Technical Details page.
    """
    st.title("Technical Details")

    st.info(
        """
        This page provides technical documentation on the ML pipeline,
        methodology, and implementation details for transparency and
        reproducibility.
        """
    )

    st.markdown("---")

    # ML Pipeline Architecture
    st.header("1. ML Pipeline Architecture")

    st.markdown(
        """
        ```
        ┌─────────────────────────────────────────────────────────────────┐
        │                     ML PIPELINE ARCHITECTURE                     │
        ├─────────────────────────────────────────────────────────────────┤
        │                                                                  │
        │  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
        │  │    INPUT     │───>│  PREPROCESS  │───>│    MODEL     │      │
        │  │    DATA      │    │              │    │              │      │
        │  └──────────────┘    └──────────────┘    └──────────────┘      │
        │        │                    │                    │              │
        │        ▼                    ▼                    ▼              │
        │  Lead Features        StandardScaler       RandomForest         │
        │  (37 columns)         One-Hot Encoder      Classifier           │
        │                       Missing Imputer                           │
        │                                                                  │
        │                                        ┌──────────────┐         │
        │                                   ───> │   OUTPUT     │         │
        │                                        │              │         │
        │                                        └──────────────┘         │
        │                                              │                  │
        │                                              ▼                  │
        │                                       Probability (0-1)         │
        │                                       Classification            │
        │                                                                  │
        └─────────────────────────────────────────────────────────────────┘
        ```
        """
    )

    st.markdown("---")

    # CRISP-DM Compliance
    st.header("2. CRISP-DM Framework Compliance")

    crisp_phases = {
        "1. Business Understanding": {
            "status": "Complete",
            "deliverables": [
                "Business requirements defined (BR1, BR2, BR3)",
                "Success metrics established",
                "Project hypotheses formulated"
            ]
        },
        "2. Data Understanding": {
            "status": "Complete",
            "deliverables": [
                "Dataset downloaded from Kaggle",
                "Initial exploration performed",
                "Data quality assessment completed"
            ]
        },
        "3. Data Preparation": {
            "status": "Complete",
            "deliverables": [
                "Missing values handled",
                "Feature engineering performed",
                "Train/test split created"
            ]
        },
        "4. Modelling": {
            "status": "Complete",
            "deliverables": [
                "Multiple algorithms compared",
                "Hyperparameter tuning performed",
                "Best model selected"
            ]
        },
        "5. Evaluation": {
            "status": "Complete",
            "deliverables": [
                "Model evaluated against success criteria",
                "Results documented",
                "Success/failure statement provided"
            ]
        },
        "6. Deployment": {
            "status": "Complete",
            "deliverables": [
                "Streamlit dashboard created",
                "Model integrated into application",
                "Deployed to Heroku"
            ]
        }
    }

    for phase, details in crisp_phases.items():
        with st.expander(f"{phase} - {details['status']}"):
            for item in details['deliverables']:
                st.markdown(f"- {item}")

    st.markdown("---")

    # Model Details
    st.header("3. Model Configuration")

    # Load model info if available
    metrics_path = 'outputs/ml_pipeline/v1/evaluation_report.json'
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)

        st.subheader("Model Type")
        st.code(metrics.get('model_type', 'RandomForestClassifier'))

        st.subheader("Best Hyperparameters")
        best_params = metrics.get('best_params', {})
        params_df_data = [{"Parameter": k, "Value": str(v)} for k, v in best_params.items()]
        if params_df_data:
            st.table(params_df_data)
    else:
        st.markdown(
            """
            **Model:** Random Forest Classifier

            **Hyperparameters (tuned via GridSearchCV):**

            | Parameter | Value | Description |
            |-----------|-------|-------------|
            | n_estimators | 200 | Number of trees |
            | max_depth | 10 | Maximum tree depth |
            | min_samples_split | 5 | Min samples to split |
            | min_samples_leaf | 2 | Min samples in leaf |
            | max_features | sqrt | Features per split |
            | class_weight | balanced | Handle imbalance |

            *Run modelling notebook for actual optimised values.*
            """
        )

    st.markdown("---")

    # Hyperparameter Tuning
    st.header("4. Hyperparameter Tuning Strategy")

    st.markdown(
        """
        **Distinction Requirement:** 6+ hyperparameters with 3+ values each

        ### Parameters Tuned

        | # | Parameter | Values Tested | Rationale |
        |---|-----------|---------------|-----------|
        | 1 | n_estimators | [100, 200, 300] | More trees = better but slower |
        | 2 | max_depth | [5, 10, 15, None] | Controls overfitting |
        | 3 | min_samples_split | [2, 5, 10] | Regularisation |
        | 4 | min_samples_leaf | [1, 2, 4] | Prevents tiny leaves |
        | 5 | max_features | [sqrt, log2, 0.5] | Feature diversity |
        | 6 | class_weight | [None, balanced, {0:1, 1:2}] | Handle imbalance |

        **Total combinations:** 972 (with 5-fold CV = 4,860 model fits)

        **Optimisation metric:** F1 Score (balances precision and recall)
        """
    )

    st.markdown("---")

    # Data Processing
    st.header("5. Data Processing Pipeline")

    st.markdown(
        """
        ### Missing Value Handling

        | Strategy | Applied To | Rationale |
        |----------|------------|-----------|
        | Drop | Columns with >50% missing | Insufficient data to impute |
        | Median imputation | Numerical features | Robust to outliers |
        | Mode/'Unknown' | Categorical features | Preserves distribution |
        | Replace 'Select' | Form fields | Placeholder = missing |

        ### Feature Engineering

        | Feature | Type | Description |
        |---------|------|-------------|
        | Engagement_Score | Derived | Composite of time, visits, page views |
        | Visit_Category | Binned | Low/Medium/High visit frequency |
        | High_Engagement_Time | Binary | Above/below median time |
        | Contact_Restricted | Binary | Combined Do Not Email/Call |

        ### Encoding

        - **One-Hot Encoding:** Applied to categorical features
        - **StandardScaler:** Applied to numerical features
        - **Binary Encoding:** Yes/No converted to 1/0
        """
    )

    st.markdown("---")

    # Model Versioning
    st.header("6. Model Versioning")

    st.markdown(
        """
        Model artifacts are stored in versioned folders for reproducibility:

        ```
        outputs/ml_pipeline/
        └── v1/
            ├── clf_pipeline.pkl       # Trained model
            ├── evaluation_report.json # Metrics
            ├── feature_importance.csv # Feature rankings
            ├── feature_names.json     # Input features
            └── version_log.txt        # Version metadata
        ```

        **Current Version:** v1

        To create a new version, increment VERSION in the modelling notebook
        and re-run. Previous versions are preserved.
        """
    )

    st.markdown("---")

    # Libraries
    st.header("7. Libraries and Dependencies")

    libraries = {
        "Data Processing": ["pandas", "numpy"],
        "Visualisation": ["matplotlib", "seaborn", "plotly"],
        "Machine Learning": ["scikit-learn", "ppscore"],
        "Dashboard": ["streamlit"],
        "Utilities": ["joblib", "json"]
    }

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Core Libraries")
        for category in ["Data Processing", "Machine Learning"]:
            st.markdown(f"**{category}:**")
            for lib in libraries[category]:
                st.markdown(f"- {lib}")

    with col2:
        st.subheader("Supporting Libraries")
        for category in ["Visualisation", "Dashboard", "Utilities"]:
            st.markdown(f"**{category}:**")
            for lib in libraries[category]:
                st.markdown(f"- {lib}")

    st.markdown("---")

    # Notebooks
    st.header("8. Jupyter Notebooks")

    notebooks = [
        ("01_DataCollection.ipynb", "Download and initial inspection"),
        ("02_DataCleaning.ipynb", "Missing values and data quality"),
        ("03_FeatureEngineering.ipynb", "Transformations and encoding"),
        ("04_EDA.ipynb", "Exploratory data analysis"),
        ("05_HypothesisTesting.ipynb", "Statistical validation"),
        ("06_Modelling.ipynb", "Training and evaluation")
    ]

    for nb, desc in notebooks:
        st.markdown(f"- **{nb}:** {desc}")

    st.markdown("---")

    # Deployment
    st.header("9. Deployment Configuration")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Procfile")
        st.code("web: sh setup.sh && streamlit run app.py", language="text")

        st.subheader("runtime.txt")
        st.code("python-3.8.18", language="text")

    with col2:
        st.subheader("setup.sh")
        st.code(
            """mkdir -p ~/.streamlit/
echo "[server]
headless = true
port = $PORT
enableCORS = false
" > ~/.streamlit/config.toml""",
            language="bash"
        )

    st.markdown("---")

    # Limitations
    st.header("10. Known Limitations")

    st.warning(
        """
        **Model Limitations:**

        1. **Training Data Scope:** Model trained on historical data from one company;
           may not generalise to different industries or sales processes.

        2. **Feature Availability:** Some features may not be available for all leads
           at the time of scoring.

        3. **Temporal Drift:** Lead behaviour patterns may change over time;
           model should be retrained periodically.

        4. **Threshold Sensitivity:** Default 0.5 threshold may not be optimal
           for all business contexts; consider adjusting based on cost of
           false positives vs false negatives.

        5. **Interpretability:** Random Forest is less interpretable than
           linear models; SHAP values could be added for instance-level explanations.
        """
    )

    st.markdown("---")

    # Future Improvements
    st.header("11. Future Improvements")

    st.markdown(
        """
        **Potential Enhancements:**

        1. **Model Updates:**
           - Implement automated retraining pipeline
           - Add model monitoring for drift detection
           - Experiment with XGBoost/LightGBM

        2. **Features:**
           - Add SHAP values for prediction explanations
           - Integrate with CRM for real-time scoring
           - Add A/B testing capability

        3. **Dashboard:**
           - Add user authentication
           - Implement prediction history tracking
           - Add export functionality for batch results
        """
    )
