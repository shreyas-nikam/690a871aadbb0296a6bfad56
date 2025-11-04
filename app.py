
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Scikit-learn imports for data generation, model, and metrics
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

st.set_page_config(page_title="QuLab", layout="wide")
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()
st.title("QuLab")
st.divider()
st.markdown("""
In this lab, we will build an interactive Streamlit application to understand and generate essential AI assurance artifacts: **Model Cards**, **Data Cards**, and **Risk Registers**. Through a simulated scenario, you will learn to document AI models, their data, and associated risks, gaining practical experience in promoting transparency, auditability, and responsible AI practices.

### What You Will Learn:

*   **Purpose of Key Artifacts:** Understand the core functions and components of Model Cards, Data Cards, and Risk Registers.
*   **Evidence for AI Assurance:** Grasp how these documents act as essential evidence for AI assurance.
*   **Interactive Documentation:** Learn to interactively define AI model parameters and data characteristics.
*   **Data and Model Simulation:** Explore synthetic data generation, validation, and simulated model predictions.
*   **Visualization:** Visualize data relationships and model performance trends using Plotly.
*   **Risk Management:** Identify and document AI risks, including categories, impact, and mitigation strategies.
*   **Artifact Interpretation:** Interpret generated artifacts in an interactive, user-friendly interface.

This application is designed to provide a hands-on experience in creating crucial AI assurance documentation, aligning with regulatory expectations for thorough documentation (e.g., SR 11-7 [4]). The business goal here is to establish a robust framework for documenting AI systems, even in early development stages or when sensitive data is not available for direct demonstration.

Formulae, explanations, tables, etc., will be presented throughout the application.
""")

# Define global risk categories and impact ratings for consistency
RISK_CATEGORIES = ['Data Quality', 'Algorithmic Bias', 'Hallucination', 'Integration Flaws', 'Human Over-reliance', 'Governance', 'Privacy/Security']
IMPACT_RATINGS = ['Low', 'Medium', 'High']

# Initialize session state for persistence
if 'synthetic_model_parameters' not in st.session_state:
    st.session_state.synthetic_model_parameters = {}
if 'synthetic_data_characteristics' not in st.session_state:
    st.session_state.synthetic_data_characteristics = {}
if 'df_synthetic' not in st.session_state:
    st.session_state.df_synthetic = pd.DataFrame()
if 'model_card' not in st.session_state:
    st.session_state.model_card = {}
if 'data_card' not in st.session_state:
    st.session_state.data_card = {}
if 'risk_register_entries' not in st.session_state:
    st.session_state.risk_register_entries = []
if 'risk_register_df' not in st.session_state:
    st.session_state.risk_register_df = pd.DataFrame()

# Helper function to display dataframes
def display_interactive_dataframe(dataframe, title):
    st.subheader(title)
    st.dataframe(dataframe)

# Your code starts here
page = st.sidebar.selectbox(label="Navigation", options=["Lab Overview & Model/Data Definition", "Data & Model Insights", "Artifacts & Risk Management"])
if page == "Lab Overview & Model/Data Definition":
    from application_pages.page1 import run_page1
    run_page1()
elif page == "Data & Model Insights":
    from application_pages.page2 import run_page2
    run_page2()
elif page == "Artifacts & Risk Management":
    from application_pages.page3 import run_page3
    run_page3()
# Your code ends
