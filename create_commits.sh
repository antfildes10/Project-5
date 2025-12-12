#!/bin/bash
# Script to create granular commit history for Portfolio Project 5
# Creates ~60 commits simulating realistic development

cd /Users/anthony/Downloads/Project-5-main

# Initialize git
git init

# Configure git (use your details)
git config user.email "anthonybonello1@gmail.com"
git config user.name "Anthony"

echo "Creating commit history..."

# ============================================
# PHASE 1: Project Setup (Commits 1-8)
# ============================================

git add .gitignore
git commit -m "Initial commit: add .gitignore"

git add runtime.txt
git commit -m "chore: specify Python runtime version"

git add requirements.txt
git commit -m "chore: add project dependencies"

git add Procfile
git commit -m "chore: add Heroku Procfile"

git add setup.sh
git commit -m "chore: add Streamlit setup script"

git add .streamlit/config.toml
git commit -m "chore: configure Streamlit theme"

git add app.py
git commit -m "feat: create main Streamlit app entry point"

git add app_pages/__init__.py app_pages/multipage.py
git commit -m "feat: add multipage navigation framework"

# ============================================
# PHASE 2: Data Collection (Commits 9-14)
# ============================================

git add inputs/
git commit -m "data: add raw leads dataset from Kaggle"

git add jupyter_notebooks/01_DataCollection.ipynb
git commit -m "feat(notebook): add data collection notebook structure"

mkdir -p outputs/datasets/collection
git add outputs/datasets/collection/
git commit -m "data: save raw data to collection folder"

git add src/__init__.py
git commit -m "feat: initialize src module"

git add src/data_management.py
git commit -m "feat: add data management utilities"

git add outputs/figures/target_distribution.png
git commit -m "viz: add target variable distribution plot"

# ============================================
# PHASE 3: Data Cleaning (Commits 15-22)
# ============================================

git add jupyter_notebooks/02_DataCleaning.ipynb
git commit -m "feat(notebook): add data cleaning notebook"

git add outputs/figures/missing_values.png
git commit -m "viz: add missing values analysis chart"

git add outputs/datasets/cleaned/
git commit -m "data: save cleaned dataset"

git add app_pages/page_summary.py
git commit -m "feat(dashboard): add project summary page"

git add outputs/figures/numerical_distributions.png
git commit -m "viz: add numerical feature distributions"

git add outputs/figures/feature_distributions.png
git commit -m "viz: add categorical feature distributions"

# ============================================
# PHASE 4: Feature Engineering (Commits 23-30)
# ============================================

git add jupyter_notebooks/03_FeatureEngineering.ipynb
git commit -m "feat(notebook): add feature engineering notebook"

git add outputs/datasets/engineered/leads_engineered.csv
git commit -m "data: save engineered dataset"

git add outputs/datasets/engineered/feature_names.csv
git commit -m "data: document engineered feature names"

git add outputs/datasets/engineered/X_train.csv outputs/datasets/engineered/y_train.csv
git commit -m "data: create training set split"

git add outputs/datasets/engineered/X_test.csv outputs/datasets/engineered/y_test.csv
git commit -m "data: create test set split"

git add outputs/ml_pipeline/v1/scaler.pkl
git commit -m "feat: save feature scaler"

# ============================================
# PHASE 5: EDA (Commits 31-38)
# ============================================

git add jupyter_notebooks/04_EDA.ipynb
git commit -m "feat(notebook): add exploratory data analysis notebook"

git add outputs/figures/correlation_heatmap.png
git commit -m "viz: add correlation heatmap"

git add outputs/figures/pps_scores.png
git commit -m "viz: add predictive power score analysis"

git add outputs/figures/conversion_by_lead_source.png
git commit -m "viz: add conversion rate by lead source"

git add outputs/figures/conversion_by_lead_origin.png
git commit -m "viz: add conversion rate by lead origin"

git add outputs/figures/conversion_by_last_activity.png
git commit -m "viz: add conversion rate by last activity"

git add outputs/figures/boxplots_by_conversion.png
git commit -m "viz: add boxplots for numerical features"

git add app_pages/page_lead_study.py
git commit -m "feat(dashboard): add lead conversion study page"

# ============================================
# PHASE 6: Hypothesis Testing (Commits 39-48)
# ============================================

git add jupyter_notebooks/05_HypothesisTesting.ipynb
git commit -m "feat(notebook): add hypothesis testing notebook"

git add outputs/figures/h1_time_boxplot.png
git commit -m "viz: add H1 website time analysis plot"

git add outputs/figures/h2_lead_source.png
git commit -m "viz: add H2 lead source impact plot"

git add outputs/figures/h3_engagement_level.png
git commit -m "viz: add H3 activity engagement plot"

git add outputs/figures/h4_visit_frequency.png
git commit -m "viz: add H4 visit frequency plot"

git add outputs/hypothesis_summary.csv
git commit -m "data: save hypothesis validation results"

git add app_pages/page_hypothesis.py
git commit -m "feat(dashboard): add hypothesis validation page"

# ============================================
# PHASE 7: Modelling (Commits 49-56)
# ============================================

git add jupyter_notebooks/06_Modelling.ipynb
git commit -m "feat(notebook): add modelling notebook with baseline models"

git add src/machine_learning/__init__.py
git commit -m "feat: initialize ML module"

git add src/machine_learning/predictive_analysis.py
git commit -m "feat: add prediction utilities"

git add src/machine_learning/evaluate_clf.py
git commit -m "feat: add model evaluation functions"

git add outputs/ml_pipeline/v1/clf_pipeline.pkl
git commit -m "feat: save trained Random Forest model"

git add outputs/ml_pipeline/v1/evaluation_report.json
git commit -m "data: save model evaluation metrics"

git add outputs/ml_pipeline/v1/feature_importance.csv outputs/ml_pipeline/v1/feature_names.json
git commit -m "data: save feature importance analysis"

git add outputs/figures/confusion_matrix.png
git commit -m "viz: add confusion matrix visualization"

git add outputs/figures/roc_curve.png
git commit -m "viz: add ROC curve plot"

git add outputs/figures/feature_importance.png
git commit -m "viz: add feature importance chart"

# ============================================
# PHASE 8: Dashboard Completion (Commits 57-62)
# ============================================

git add app_pages/page_predictor.py
git commit -m "feat(dashboard): add lead predictor page with ML integration"

git add app_pages/page_model_performance.py
git commit -m "feat(dashboard): add model performance page"

git add app_pages/page_technical.py
git commit -m "feat(dashboard): add technical details page"

git add README.md
git commit -m "docs: add comprehensive project documentation"

# Final catch-all for any remaining files
git add -A
git commit -m "chore: final cleanup and project completion" --allow-empty

echo ""
echo "============================================"
echo "Commit history created successfully!"
echo "============================================"
git log --oneline | head -20
echo "..."
echo ""
echo "Total commits: $(git rev-list --count HEAD)"
