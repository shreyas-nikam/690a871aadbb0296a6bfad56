id: 690a871aadbb0296a6bfad56_documentation
summary: AI Analyst Assistant for Model Governance Documentation
feedback link: https://docs.google.com/forms/d/e/1FAIpQLSfWkOK-in_bMMoHSZfcIvAeO58PAH9wrDqcxnJABHaxiDqhSA/viewform?usp=sf_link
environments: Web
status: Published
# Building AI Assurance Artifacts with Streamlit: Model Cards, Data Cards, and Risk Registers

## Introduction and Project Overview
Duration: 05:00

Welcome to **QuLab**, an interactive Streamlit application designed to demystify and demonstrate the creation of essential AI assurance artifacts: **Model Cards**, **Data Cards**, and **Risk Registers**. In today's rapidly evolving AI landscape, regulatory bodies and stakeholders increasingly demand transparency, auditability, and responsible practices in AI development and deployment. This codelab provides a hands-on guide to understanding and generating these critical documents through a simulated scenario.

<aside class="positive">
This lab is crucial for developers, data scientists, and AI governance professionals aiming to implement **Responsible AI** practices. It directly addresses the need for thorough documentation, aligning with regulatory expectations such as those outlined in SR 11-7 [4], which calls for robust model risk management.
</aside>

### Why are these artifacts important?

*   **Transparency and Trust:** They provide clear, structured information about AI models and the data they use, fostering trust among users, stakeholders, and regulators.
*   **Auditability and Accountability:** These documents serve as evidence for internal and external audits, demonstrating due diligence and adherence to ethical guidelines.
*   **Risk Management:** They help identify, assess, and mitigate potential risks associated with AI systems, from data quality issues to algorithmic bias and deployment challenges.
*   **Effective Challenge:** As highlighted in various AI assurance principles [1], these artifacts enable stakeholders to "effectively challenge" AI models, ensuring their reliability and fairness.

### Application Architecture Overview

The QuLab application is built using Streamlit, a Python library for creating interactive web applications. It's structured into three main pages, facilitating a logical flow from defining model/data parameters to generating and visualizing artifacts. Data and model parameters are persisted across pages using Streamlit's session state.

Here's a high-level overview of the application's structure and data flow:

```
+-+
|  Streamlit Main Application |
|      (app.py)              |
| -- |
| - Initializes Session State|
| - Sidebar Navigation       |
| - Global Risk Definitions  |
+-+
       |
       |  Navigates to
       V
+--+
|  Page 1: Lab Overview & Model/Data Definition (application_pages/page1.py)             |
|  |
| - Environment Setup & Methodology                                                       |
| - Define Synthetic Model Parameters (UI Input -> Session State)                         |
| - Define Synthetic Data Characteristics (UI Input -> Session State)                     |
| - **Generates Synthetic Data** (sklearn.make_classification)                            |
| - **Simulates Model Predictions** (sklearn.linear_model.LogisticRegression)             |
|   -> Stores `df_synthetic` in Session State                                             |
+--+
       |
       |  Utilizes Session State (df_synthetic, model_params, data_params)
       V
+--+
|  Page 2: Data & Model Insights (application_pages/page2.py)                             |
|  |
| - Performs Data Validation on `df_synthetic`                                            |
| - Visualizes Feature Relationships (Plotly Scatter Plot)                                |
| - Visualizes Simulated Model Performance Trend Over Time (Plotly Line Plot)             |
+--+
       |
       |  Utilizes Session State (df_synthetic, model_params, data_params, risk_entries)
       V
+--+
|  Page 3: Artifacts & Risk Management (application_pages/page3.py)                       |
|  |
| - Calculates Model Performance Metrics (sklearn.metrics)                                |
| - **Generates Model Card Content** (from model_params & metrics -> Session State)       |
| - **Generates Data Card Content** (from data_params & df_synthetic stats -> Session State) |
| - User Input for AI Risks (UI Input -> Session State `risk_register_entries`)           |
| - **Compiles Risk Register** (from risk_register_entries -> Session State `risk_register_df`) |
| - Visualizes Aggregated AI Risks (Plotly Bar Chart)                                     |
| - Displays Interactive Tables for Model Card, Data Card, Risk Register                  |
| - Discussion, Conclusion, References                                                    |
+--+
```

### What You Will Learn:

*   **Purpose of Key Artifacts:** Understand the core functions and components of Model Cards, Data Cards, and Risk Registers.
*   **Evidence for AI Assurance:** Grasp how these documents act as essential evidence for AI assurance.
*   **Interactive Documentation:** Learn to interactively define AI model parameters and data characteristics.
*   **Data and Model Simulation:** Explore synthetic data generation, validation, and simulated model predictions.
*   **Visualization:** Visualize data relationships and model performance trends using Plotly.
*   **Risk Management:** Identify and document AI risks, including categories, impact, and mitigation strategies.
*   **Artifact Interpretation:** Interpret generated artifacts in an interactive, user-friendly interface.

Let's begin!

## Environment Setup and Core Concepts
Duration: 08:00

This step introduces the environment and fundamental concepts powering our AI assurance application.

### Environment Setup Overview

The application relies on standard Python libraries. If you were to run this locally, you'd ensure these are installed:

```python
import streamlit as st # For interactive UI
import pandas as pd     # For data manipulation
import numpy as np      # For numerical operations
import plotly.express as px # For interactive visualizations
import plotly.graph_objects as go # For advanced Plotly charts

# Scikit-learn for data generation, model, and metrics
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
```
These imports provide the capabilities for everything from user interface elements to machine learning functionalities and data visualization.

### Data/Inputs Overview

To demonstrate AI assurance artifacts without using sensitive or proprietary real-world data, this application leverages **synthetic data**. This approach allows for a controlled environment to showcase the principles of AI documentation.

Our synthetic data will simulate a typical classification task and include:
*   **Numeric features:** `feature_1`, `feature_2`, etc., representing measurable attributes.
*   **Categorical features:** `category_1`, `category_2`, etc., simulating discrete variables (e.g., product types, regions).
*   **A timestamp column:** `timestamp`, enabling the simulation of time-series trends and performance monitoring over time.
*   **A binary target variable:** `true_label`, for a classification task (e.g., 0 or 1, indicating 'not high-value' or 'high-value').

<aside class="positive">
Using synthetic data is a best practice for demonstrating AI systems in public or non-production environments. It allows for exploration of concepts like bias, data quality, and model performance without exposing sensitive information.
</aside>

### Methodology Overview

Our approach to generating AI assurance artifacts follows a structured, step-by-step process:

1.  **Define Model & Data Characteristics**: Interactively specify parameters for a synthetic AI model and its dataset.
2.  **Generate Synthetic Data**: Create a dataset based on defined characteristics, including features, a target variable, and a timestamp.
3.  **Simulate Model Predictions**: Train a simple classification model (Logistic Regression) on the synthetic data and generate predictions.
4.  **Validate & Explore Data**: Perform basic data validation and visualize relationships and trends within the synthetic data.
5.  **Generate Model Card**: Compile key model information, performance metrics, and usage details into a Model Card.
6.  **Generate Data Card**: Document the dataset's characteristics, provenance, and identified biases in a Data Card.
7.  **Identify AI Risks**: Allow users to input and manage AI-related risks in a Risk Register.
8.  **Visualize & Display Artifacts**: Present all generated artifacts and visualizations in an interactive format.

### Key Formulae and Their Business Rationale:

Understanding the underlying mathematics of AI models and their evaluation metrics is crucial for effective AI assurance.

**1. Logistic Regression Probability (for Model Simulation):**
This formula describes how Logistic Regression estimates the probability of a positive outcome.

$$ P(Y=1|X) = \frac{1}{1 + e^{-(\beta_0 + \sum_{i=1}^k \beta_i X_i)}} $$

*   Where:
    *   $P(Y=1|X)$ is the probability of the positive class (e.g., a customer being `high-value`).
    *   $X_i$ represents the input features.
    *   $\beta_0$ is the intercept term.
    *   $\beta_i$ are the coefficients (weights) assigned to each feature.

**Business Rationale:** Knowing how a model calculates probabilities is vital for interpretability and trust. Logistic Regression is transparent in its decision-making, as the coefficients ($\beta_i$) indicate the influence of each feature. This helps in explaining *why* a model made a certain prediction, which is critical for human oversight and model review.

**2. Classification Performance Metrics (for Model Cards):**
These metrics are fundamental for evaluating a classification model's effectiveness and identifying potential issues like bias or poor generalization.

*   **Accuracy**: The proportion of total predictions that were correct.
    $$ \text{Accuracy} = \frac{\text{Number of Correct Predictions}}{\text{Total Number of Predictions}} $$
*   **Precision**: The proportion of positive predictions that were actually correct (minimizes false positives).
    $$ \text{Precision} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}} $$
*   **Recall (Sensitivity)**: The proportion of actual positives that were correctly identified (minimizes false negatives).
    $$ \text{Recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}} $$
*   **F1-Score**: The harmonic mean of Precision and Recall, providing a single metric that balances both. It's especially useful for imbalanced datasets.
    $$ \text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} $$
*   **ROC AUC**: The Area Under the Receiver Operating Characteristic Curve. It measures the ability of a classifier to distinguish between classes. A higher AUC indicates a better model performance at distinguishing between positive and negative classes.

**Business Rationale:** These metrics are essential for evaluating whether an AI model meets its intended performance targets and for identifying potential issues. For instance, in a fraud detection system, high **Recall** is crucial to catch as many fraudulent transactions as possible (minimizing false negatives), while high **Precision** might be important in a medical diagnosis system to avoid unnecessary treatments (minimizing false positives). The **F1-Score** offers a balanced view, and **ROC AUC** provides an overall assessment of discriminative power.

## Defining Synthetic AI Model Parameters
Duration: 03:00

In this step, you will interactively define the characteristics of our simulated AI model, which will form the basis of our Model Card.

### Understanding Model Cards

A **Model Card** is a structured document that provides key information about an AI model. This includes its purpose, type, intended use, performance characteristics, and any known limitations or ethical considerations. It serves as a vital tool for transparency, accountability, and responsible deployment.

### Interacting with the Application

Navigate to the "Lab Overview & Model/Data Definition" page in the Streamlit sidebar. Scroll down to "Section 3: Defining Synthetic AI Model Parameters".

You will see an expandable section titled "Define Synthetic AI Model Parameters". Expand it to reveal the input fields.

The Streamlit application code snippet for this section looks like this:

```python
# application_pages/page1.py - define_model_parameters_streamlit()
def define_model_parameters_streamlit():
    with st.expander("Define Synthetic AI Model Parameters", expanded=True):
        st.markdown("A Model Card provides a structured overview of an AI model's purpose, characteristics, performance, and ethical considerations. It serves as a crucial document for transparency, accountability, and responsible deployment. Use the widgets below to specify parameters for our synthetic AI model.")

        col1, col2 = st.columns(2)
        with col1:
            model_name = st.text_input(
                "Model Name:",
                value=st.session_state.synthetic_model_parameters.get('model_name', "Synthetic AI Analyst Assistant"),
                help="A descriptive name for the AI model (e.g., Customer Churn Predictor)."
            )
            model_type = st.selectbox(
                "Model Type:",
                options=['Classification', 'Regression', 'Generative'],
                index=['Classification', 'Regression', 'Generative'].index(st.session_state.synthetic_model_parameters.get('model_type', "Classification")),
                help="Choose the type of AI model being simulated (e.g., Classification, Regression)."
            )
            performance_threshold = st.number_input(
                "Min F1-score:",
                value=st.session_state.synthetic_model_parameters.get('performance_threshold', 0.75),
                min_value=0.0, max_value=1.0, step=0.01,
                help="Minimum F1-score considered acceptable for this model (0.0 to 1.0)."
            )
        with col2:
            purpose = st.text_area(
                "Purpose:",
                value=st.session_state.synthetic_model_parameters.get('purpose', "To classify synthetic customer data for analytical insights."),
                height=100,
                help="Describe the primary objective and function of this AI model."
            )
            known_limitations = st.text_area(
                "Limitations:",
                value=st.session_state.synthetic_model_parameters.get('known_limitations', "Performance may degrade on highly imbalanced datasets."),
                height=100,
                help="Document any known limitations or failure modes of the model."
            )
            usage_notes = st.text_area(
                "Usage Notes:",
                value=st.session_state.synthetic_model_parameters.get('usage_notes', "Intended for internal analytical use by data science teams."),
                height=100,
                help="Provide guidance on the appropriate use and users of this model."
            )
        st.session_state.synthetic_model_parameters = {
            'model_name': model_name, 'purpose': purpose, 'model_type': model_type,
            'performance_threshold': performance_threshold, 'known_limitations': known_limitations,
            'usage_notes': usage_notes
        }
    st.success("Synthetic AI model parameters captured.")
```

*   **Model Name:** Provide a descriptive name (e.g., "Synthetic AI Analyst Assistant").
*   **Purpose:** Describe the model's main objective (e.g., "To classify synthetic customer data for analytical insights.").
*   **Model Type:** Select "Classification" for this lab.
*   **Min F1-score:** Set an acceptable performance threshold (e.g., 0.75). This will be used later for evaluating our model.
*   **Limitations:** Document any known shortcomings (e.g., "Performance may degrade on highly imbalanced datasets.").
*   **Usage Notes:** Specify who should use the model and under what conditions (e.g., "Intended for internal analytical use by data science teams.").

The inputs are automatically saved to Streamlit's session state, ensuring they are preserved as you navigate the application.

## Defining Synthetic Data Characteristics
Duration: 03:00

Following the model parameters, this step focuses on defining the characteristics of the synthetic dataset that will be used to train and evaluate our model. This information is crucial for building a comprehensive Data Card.

### Understanding Data Cards

A **Data Card** documents the dataset's characteristics, provenance (origin), collection methods, and any identified biases or privacy considerations. It is a vital component of data governance, ensuring fair, transparent, and responsible AI development by meticulously detailing the input data.

### Interacting with the Application

Still on the "Lab Overview & Model/Data Definition" page, scroll down to "Section 4: Defining Synthetic Data Characteristics".

Expand the "Define Synthetic Data Characteristics" section to see the input fields.

The Streamlit application code snippet for this section looks like this:

```python
# application_pages/page1.py - define_data_characteristics_streamlit()
def define_data_characteristics_streamlit():
    with st.expander("Define Synthetic Data Characteristics", expanded=True):
        st.markdown("A Data Card documents the dataset's characteristics, provenance, collection methods, and any identified biases or privacy considerations. This is vital for data governance and ensuring fair and responsible AI development. Use the widgets below to specify parameters for our synthetic dataset.")

        col1, col2 = st.columns(2)
        with col1:
            dataset_name = st.text_input(
                'Dataset Name:',
                value=st.session_state.synthetic_data_characteristics.get('dataset_name', 'Synthetic Customer Data'),
                help='A descriptive name for the synthetic dataset.'
            )
            n_samples = st.number_input(
                'Num Samples:',
                value=st.session_state.synthetic_data_characteristics.get('n_samples', 1000),
                min_value=0, step=100,
                help='Number of synthetic data samples to generate.'
            )
            n_features = st.number_input(
                'Num Features (for classification):',
                value=st.session_state.synthetic_data_characteristics.get('n_features', 5),
                min_value=0, step=1,
                help='Number of numeric features for the classification task.'
            )
            n_categorical_features = st.number_input(
                'Num Categorical Features (added later):',
                value=st.session_state.synthetic_data_characteristics.get('n_categorical_features', 2),
                min_value=0, step=1,
                help='Number of synthetic categorical features to add.'
            )
        with col2:
            data_provenance = st.text_area(
                'Data Provenance:',
                value=st.session_state.synthetic_data_characteristics.get('data_provenance', 'Generated by internal Python script for simulation.'),
                help='Where did this data come from?'
            )
            collection_method = st.text_area(
                'Collection Method:',
                value=st.session_state.synthetic_data_characteristics.get('collection_method', 'Simulated algorithmic collection based on predefined rules.'),
                help='How was this data collected or generated?'
            )
            identified_biases_description = st.text_area(
                'Identified Biases:',
                value=st.session_state.synthetic_data_characteristics.get('identified_biases_description', 'Potential class imbalance and simulated feature correlations.'),
                help='Describe any known biases present in the data.'
            )
            privacy_notes = st.text_area(
                'Privacy Notes:',
                value=st.session_state.synthetic_data_characteristics.get('privacy_notes', 'All data is synthetic and contains no personal identifiable information.'),
                help='Important privacy considerations for this dataset.'
            )
        st.session_state.synthetic_data_characteristics = {
            'dataset_name': dataset_name, 'n_samples': n_samples, 'n_features': n_features,
            'n_categorical_features': n_categorical_features, 'data_provenance': data_provenance,
            'collection_method': collection_method, 'identified_biases_description': identified_biases_description,
            'privacy_notes': privacy_notes,
        }
    st.success("Synthetic dataset characteristics captured.")
```

*   **Dataset Name:** A descriptive name for the dataset (e.g., "Synthetic Customer Data").
*   **Num Samples:** The total number of rows in the synthetic dataset (e.g., 1000).
*   **Num Features:** The number of numerical features for classification (e.g., 5).
*   **Num Categorical Features:** The number of categorical features to add (e.g., 2).
*   **Data Provenance:** Where the data came from (e.g., "Generated by internal Python script for simulation.").
*   **Collection Method:** How the data was gathered (e.g., "Simulated algorithmic collection based on predefined rules.").
*   **Identified Biases:** Any known biases (e.g., "Potential class imbalance and simulated feature correlations.").
*   **Privacy Notes:** Important privacy considerations (e.g., "All data is synthetic and contains no personal identifiable information.").

As with model parameters, these inputs are automatically saved to `st.session_state.synthetic_data_characteristics`.

## Generating Synthetic Data and Simulating Model Predictions
Duration: 05:00

This is a core operational step where the application generates a synthetic dataset based on your defined characteristics and then simulates an AI model's training and prediction process.

### Synthetic Data Generation

The `generate_synthetic_data` function uses `sklearn.datasets.make_classification` to create a dataset that simulates a binary classification problem. It then adds categorical features and a timestamp column to enrich the dataset.

```python
# application_pages/page1.py - generate_synthetic_data()
@st.cache_data
def generate_synthetic_data(n_samples, n_features, n_classes=2, random_state=42):
    if n_samples == 0:
        return pd.DataFrame()

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=min(n_features, 3), # At most 3 informative features, or fewer if n_features is small
        n_redundant=max(0, n_features - min(n_features, 3) - 1), # Ensure n_redundant is non-negative
        n_repeated=0,
        n_classes=n_classes,
        random_state=random_state
    )

    df = pd.DataFrame(X, columns=[f'feature_{i+1}' for i in range(n_features)])
    df['true_label'] = y

    # Add categorical features
    n_categorical_features = st.session_state.synthetic_data_characteristics.get('n_categorical_features', 2)
    for i in range(n_categorical_features):
        df[f'category_{i+1}'] = np.random.choice(['A', 'B', 'C', 'D'], size=n_samples)

    # Add a timestamp column
    start_date = pd.to_datetime('2023-01-01')
    end_date = pd.to_datetime('2023-12-31')
    df['timestamp'] = pd.to_datetime(np.random.uniform(start_date.timestamp(), end_date.timestamp(), n_samples).astype(int), unit='s')

    return df
```

### Model Simulation

After generating the data, the `simulate_model_predictions` function trains a simple `LogisticRegression` model on the numeric features and the `true_label`. It then generates `predicted_label` and `prediction_score` for each sample.

```python
# application_pages/page1.py - simulate_model_predictions()
@st.cache_data
def simulate_model_predictions(features, labels, model_type='Logistic Regression', random_state=42):
    if features.empty or labels.empty:
        return pd.Series([], dtype=int), pd.Series([], dtype=float)

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=random_state)

    if model_type == 'Logistic Regression':
        model = LogisticRegression(random_state=random_state, solver='liblinear')
    else:
        model = LogisticRegression(random_state=random_state, solver='liblinear')

    model.fit(X_train, y_train)
    predictions = model.predict(features)
    prediction_scores = model.predict_proba(features)[:, 1] # Probability of the positive class

    return pd.Series(predictions, index=features.index), pd.Series(prediction_scores, index=features.index)
```

### Interacting with the Application

On the "Lab Overview & Model/Data Definition" page, scroll down to "Section 5: Synthetic Data Generation and Model Simulation".

You will see a button labeled "Generate Synthetic Data & Simulate Model".

```python
# application_pages/page1.py - run_data_generation_and_simulation()
def run_data_generation_and_simulation():
    if st.button("Generate Synthetic Data & Simulate Model"):
        # ... (validation checks for n_samples and n_features) ...

        with st.spinner("Generating synthetic data and simulating model predictions..."):
            df_synthetic_local = generate_synthetic_data(
                st.session_state.synthetic_data_characteristics['n_samples'],
                st.session_state.synthetic_data_characteristics['n_features'],
                random_state=42
            )

            X = df_synthetic_local.select_dtypes(include=np.number).drop(['true_label'], axis=1, errors='ignore')
            y = df_synthetic_local['true_label']

            if not X.empty and not y.empty:
                predicted_label, prediction_score = simulate_model_predictions(
                    X, y,
                    model_type=st.session_state.synthetic_model_parameters.get('model_type', 'Classification')
                )
                df_synthetic_local['predicted_label'] = predicted_label
                df_synthetic_local['prediction_score'] = prediction_score
            else:
                df_synthetic_local['predicted_label'] = pd.Series([], dtype=int)
                df_synthetic_local['prediction_score'] = pd.Series([], dtype=float)
                st.warning("Could not simulate model predictions: no valid features or labels were found in the generated data.")

            st.session_state.df_synthetic = df_synthetic_local
            st.success("Generated synthetic data with simulated model predictions.")
            if not st.session_state.df_synthetic.empty:
                st.write("Here's a preview of the generated data:")
                st.dataframe(st.session_state.df_synthetic.head())
            else:
                st.info("No data generated due to input parameters (e.g., N_samples = 0).")
```

1.  Click the **"Generate Synthetic Data & Simulate Model"** button.
2.  Observe the spinner indicating data generation and model simulation in progress.
3.  Once complete, a success message will appear, and a preview of the generated DataFrame, including `true_label`, `predicted_label`, and `prediction_score` columns, will be displayed. This `pd.DataFrame` is stored in `st.session_state.df_synthetic` for use in subsequent pages.

## Data Validation and Exploration
Duration: 07:00

Data validation and exploration are crucial preliminary steps in any AI project. They ensure the quality, integrity, and suitability of the data for model training and help uncover insights and potential issues before they impact model performance.

### Data Validation

On the "Data & Model Insights" page, scroll to "Section 6: Data Validation and Exploration".
This section performs basic data validation checks:

*   **Data Overview:** Displays the shape (rows, columns) and first few rows of the DataFrame.
*   **Missing Values:** Identifies columns with missing values.
*   **Data Types:** Lists the data type for each column.
*   **Critical Field Check:** Verifies the presence of essential columns (`feature_X`, `true_label`, `predicted_label`, `prediction_score`, `timestamp`).

The `perform_data_validation` function encapsulates these checks:

```python
# application_pages/page2.py - perform_data_validation()
@st.cache_data
def perform_data_validation(dataframe, critical_fields):
    st.subheader("Data Overview:")
    st.write(f"Dataset contains {dataframe.shape[0]} rows and {dataframe.shape[1]} columns.")
    st.write("First 5 rows:")
    st.dataframe(dataframe.head())

    st.subheader("Missing Values:")
    missing_data = dataframe.isnull().sum()
    missing_data = missing_data[missing_data > 0]
    if not missing_data.empty:
        st.warning("Missing values detected in the following columns:")
        st.dataframe(missing_data.rename("Missing Count"))
    else:
        st.info("No missing values detected.")

    st.subheader("Data Types:")
    st.dataframe(dataframe.dtypes.rename("Data Type"))

    st.subheader("Critical Field Check:")
    missing_critical_fields = [field for field in critical_fields if field not in dataframe.columns]
    if missing_critical_fields:
        st.error(f"Critical fields missing: {missing_critical_fields}. This may impact subsequent analysis.")
    else:
        st.info("All critical fields are present.")
```
Review the output to ensure the synthetic data meets basic quality standards.

### Core Visual: Relationship Plot

Understanding the relationships between features can reveal underlying patterns and potential correlations that the AI model might exploit or be sensitive to. A scatter plot is an excellent tool for this.

On the "Data & Model Insights" page, scroll to "Section 7: Core Visual: Relationship Plot".

You can select two numeric features from your synthetic dataset to plot against each other. The points will be colored by their `true_label` to highlight class separation.

```python
# application_pages/page2.py - plot_relationship()
@st.cache_data
def plot_relationship(dataframe, feature_x, feature_y, hue_column, title, save_path=None):
    if not dataframe.empty:
        if feature_x not in dataframe.columns or feature_y not in dataframe.columns or hue_column not in dataframe.columns:
            st.error(f"Required columns missing for relationship plot: {feature_x}, {feature_y}, {hue_column}")
            return

        fig = px.scatter(dataframe, x=feature_x, y=feature_y, color=hue_column, title=title,
                         color_discrete_sequence=px.colors.qualitative.Plotly,
                         labels={feature_x: feature_x, feature_y: feature_y, hue_column: 'True Label'})

        fig.update_layout(title_font_size=16, font_size=12, legend_title_text='True Label')
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("DataFrame is empty, cannot generate relationship plot.")
```
1.  Use the dropdowns to select different `X-axis Feature` and `Y-axis Feature`.
2.  Observe how the distribution of `True Label` changes with different feature combinations. This helps in understanding data separability.

### Core Visual: Trend Plot for Simulated Performance

Monitoring model performance over time is critical for AI governance. It helps detect performance degradation, concept drift, or other issues post-deployment. This plot visualizes how a key metric, like accuracy, evolves over the simulated timeline.

On the "Data & Model Insights" page, scroll to "Section 8: Core Visual: Trend Plot for Simulated Performance".

The application calculates the daily average `simulated_accuracy` (where `true_label == predicted_label`) and plots its trend.

```python
# application_pages/page2.py - plot_performance_trend()
@st.cache_data
def plot_performance_trend(dataframe, timestamp_column, metric_column, title, save_path=None):
    if not dataframe.empty:
        if timestamp_column not in dataframe.columns or metric_column not in dataframe.columns:
            st.error(f"Required columns missing for trend plot: {timestamp_column}, {metric_column}")
            return

        fig = px.line(dataframe, x=timestamp_column, y=metric_column, title=title,
                      line_shape="linear", render_mode="svg",
                      color_discrete_sequence=['steelblue'],
                      labels={timestamp_column: "Date", metric_column: "Simulated Accuracy"})

        fig.update_layout(title_font_size=16, font_size=12)
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey', title_font_size=12)
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey', title_font_size=12, range=[0,1.05])
        fig.update_traces(mode='lines+markers', marker=dict(size=6, line=dict(width=0)))

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("DataFrame is empty, cannot generate performance trend plot.")
```
Observe the trend line to see if the simulated accuracy remains stable or shows any noticeable fluctuations over time.

## Generating and Displaying the Model Card
Duration: 04:00

Now that we have simulated data and model predictions, we can compile the Model Card. This step calculates essential performance metrics and integrates them into a comprehensive Model Card document.

### Calculating Model Metrics

On the "Artifacts & Risk Management" page, scroll to "Section 9: Generating Model Card Content".

The application first calculates the performance metrics previously discussed (Accuracy, Precision, Recall, F1-Score, ROC AUC) using `sklearn.metrics` from the `true_label`, `predicted_label`, and `prediction_score` columns in the `df_synthetic`.

```python
# application_pages/page3.py - calculate_model_metrics()
@st.cache_data
def calculate_model_metrics(true_labels, predicted_labels, prediction_scores):
    if true_labels.empty or predicted_labels.empty or len(true_labels) == 0:
        return {
            'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0,
            'f1_score': 0.0, 'roc_auc': 0.0
        }

    true_labels_arr = true_labels.values if isinstance(true_labels, pd.Series) else true_labels
    predicted_labels_arr = predicted_labels.values if isinstance(predicted_labels, pd.Series) else predicted_labels
    prediction_scores_arr = prediction_scores.values if isinstance(prediction_scores, pd.Series) else prediction_scores

    accuracy = accuracy_score(true_labels_arr, predicted_labels_arr)
    precision = precision_score(true_labels_arr, predicted_labels_arr, zero_division=0)
    recall = recall_score(true_labels_arr, predicted_labels_arr, zero_division=0)
    f1 = f1_score(true_labels_arr, predicted_labels_arr, zero_division=0)

    roc_auc = 0.0
    if prediction_scores_arr is not None and len(np.unique(true_labels_arr)) > 1:
        try:
            roc_auc = roc_auc_score(true_labels_arr, prediction_scores_arr)
        except ValueError:
            roc_auc = 0.0

    metrics = {
        'accuracy': accuracy, 'precision': precision, 'recall': recall,
        'f1_score': f1, 'roc_auc': roc_auc
    }
    return metrics
```

These metrics, along with the model parameters defined in Step 3, are then compiled into the `st.session_state.model_card`.

```python
# application_pages/page3.py - generate_model_card_content()
@st.cache_data
def generate_model_card_content(model_params, true_labels, predicted_labels, prediction_scores):
    model_card = {
        "Model Name": model_params.get('model_name', 'N/A'),
        "Purpose": model_params.get('purpose', 'N/A'),
        "Model Type": model_params.get('model_type', 'N/A'),
        "Minimum Acceptable F1-score": model_params.get('performance_threshold', 'N/A'),
        "Known Limitations": model_params.get('known_limitations', 'N/A'),
        "Usage Notes": model_params.get('usage_notes', 'N/A'),
    }

    performance_metrics = calculate_model_metrics(true_labels, predicted_labels, prediction_scores)
    model_card["Performance Metrics"] = performance_metrics

    return model_card
```
This process happens automatically when you navigate to this page, provided the data has been generated.

### Displaying the Model Card (Interactive Table)

Scroll to "Section 10: Displaying the Model Card (Interactive Table)".
The generated Model Card content is then presented in an interactive table format.

```python
# application_pages/page3.py - display_model_card()
def display_model_card():
    st.header("Section 10: Displaying the Model Card (Interactive Table)")
    st.markdown(r"""The generated Model Card content is now ready for display. This interactive table summarizes the key attributes and performance metrics of the synthetic AI model, offering a quick and comprehensive overview for stakeholders.""")
    if st.session_state.model_card:
        model_card_flat = {
            'Model Name': st.session_state.model_card.get('Model Name', 'N/A'),
            'Purpose': st.session_state.model_card.get('Purpose', 'N/A'),
            'Model Type': st.session_state.model_card.get('Model Type', 'N/A'),
            'Minimum Acceptable F1-score': st.session_state.model_card.get('Minimum Acceptable F1-score', 'N/A'),
            'Known Limitations': st.session_state.model_card.get('Known Limitations', 'N/A'),
            'Usage Notes': st.session_state.model_card.get('Usage Notes', 'N/A')
        }

        performance_metrics = st.session_state.model_card.get('Performance Metrics', {})
        for metric, value in performance_metrics.items():
            if isinstance(value, (int, float)):
                model_card_flat[f'Metric: {metric.replace("_", " ").title()}'] = f"{value:.4f}"
            else:
                model_card_flat[f'Metric: {metric.replace("_", " ").title()}'] = str(value)

        model_card_df = pd.DataFrame(list(model_card_flat.items()), columns=['Attribute', 'Value'])
        # Helper function from app.py
        def display_interactive_dataframe(dataframe, title):
            st.subheader(title)
            st.dataframe(dataframe)
        display_interactive_dataframe(model_card_df, "Synthetic AI Model Card")
        st.success("Interactive Model Card displayed.")
    else:
        st.warning("Model Card is empty. Please generate Model Card content first.")
```
<aside class="positive">
Review the generated Model Card. Pay attention to the performance metrics and compare the F1-score against the "Minimum Acceptable F1-score" you defined. This provides a direct assessment of whether the simulated model meets its performance target.
</aside>

## Generating and Displaying the Data Card
Duration: 04:00

Just as the Model Card documents the AI model, the Data Card provides comprehensive documentation for the dataset used. This step generates and displays the Data Card, detailing data characteristics and statistics.

### Generating Data Card Content

On the "Artifacts & Risk Management" page, scroll to "Section 11: Generating Data Card Content".

The application compiles information from the data characteristics defined in Step 4 and computes detailed statistics for each feature in the `st.session_state.df_synthetic`. This includes data types, missing value counts, unique values, and for numeric features, mean, std, min, max; for categorical, top values and their frequencies.

```python
# application_pages/page3.py - generate_data_card_content()
@st.cache_data
def generate_data_card_content(data_params, dataframe):
    if dataframe.empty:
        # Returns an empty data card if the dataframe is empty
        # ... (simplified for brevity) ...
        return {}

    data_card = {
        "dataset_name": data_params.get('dataset_name', 'N/A'),
        "n_samples": data_params.get('n_samples', 0),
        "n_features": data_params.get('n_features', 0),
        "n_categorical_features": data_params.get('n_categorical_features', 0),
        "data_provenance": data_params.get('data_provenance', 'N/A'),
        "collection_method": data_params.get('collection_method', 'N/A'),
        "identified_biases_description": data_params.get('identified_biases_description', 'N/A'),
        "privacy_notes": data_params.get('privacy_notes', 'N/A'),
        "feature_statistics": {}
    }

    feature_stats = {}
    for col in dataframe.columns:
        col_stats = {
            'dtype': str(dataframe[col].dtype),
            'missing_count': dataframe[col].isnull().sum(),
            'unique_count': dataframe[col].nunique(),
        }
        if pd.api.types.is_numeric_dtype(dataframe[col]):
            col_stats['mean'] = dataframe[col].mean()
            col_stats['std'] = dataframe[col].std()
            col_stats['min'] = dataframe[col].min()
            col_stats['max'] = dataframe[col].max()
        elif pd.api.types.is_categorical_dtype(dataframe[col]) or pd.api.types.is_object_dtype(dataframe[col]):
            top_value = dataframe[col].mode()[0] if not dataframe[col].mode().empty else None
            top_freq = dataframe[col].value_counts().max() if top_value is not None else 0
            col_stats['top'] = top_value
            col_stats['top_frequency'] = top_freq
        feature_stats[col] = col_stats
    data_card["feature_statistics"] = feature_stats
    return data_card
```
This process also occurs automatically upon navigating to the page, populating `st.session_state.data_card`.

### Displaying the Data Card (Interactive Table)

Scroll to "Section 12: Displaying the Data Card (Interactive Table)".
The generated Data Card is presented in two interactive tables: an overview of the dataset's general characteristics and a detailed breakdown of statistics for each feature.

```python
# application_pages/page3.py - display_data_card()
def display_data_card():
    st.header("Section 12: Displaying the Data Card (Interactive Table)")
    st.markdown(r"""The generated Data Card content is now ready for display. This section provides an interactive overview of the dataset and detailed statistics for each feature, enhancing transparency and aiding in data quality assessment.""")
    if st.session_state.data_card:
        data_card_flat = { # ... (overview attributes) ... }
        feature_stats_list = []
        for feature, stats in st.session_state.data_card.get('feature_statistics', {}).items():
            stats_summary = f"Dtype: {stats.get('dtype', 'N/A')}, Missing: {stats.get('missing_count', 'N/A')}, Unique: {stats.get('unique_count', 'N/A')}"
            if 'mean' in stats and stats['mean'] is not None:
                stats_summary += f", Mean: {stats['mean']:.2f}, Std: {stats['std']:.2f}"
            if 'top' in stats and stats['top'] is not None:
                stats_summary += f", Top: {stats['top']} (Freq: {stats['top_frequency']})"
            feature_stats_list.append({'Feature': feature, 'Summary Statistics': stats_summary})

        data_card_df_top = pd.DataFrame(list(data_card_flat.items()), columns=['Attribute', 'Value'])
        data_card_df_features = pd.DataFrame(feature_stats_list)

        # Helper function from app.py
        def display_interactive_dataframe(dataframe, title):
            st.subheader(title)
            st.dataframe(dataframe)
        display_interactive_dataframe(data_card_df_top, "Synthetic Dataset Data Card (Overview)")
        st.markdown("\nDetailed Feature Statistics:")
        display_interactive_dataframe(data_card_df_features, "Synthetic Dataset Data Card (Feature Statistics)")
        st.success("Interactive Data Card displayed.")
    else:
        st.warning("Data Card is empty. Please generate Data Card content first.")
```
<aside class="positive">
Carefully examine the "Detailed Feature Statistics" for each column. Look for high missing counts, unexpected data types, or unusual distributions (reflected in mean/std deviations) which could indicate data quality issues or potential biases.
</aside>

## Identifying and Managing AI Risks
Duration: 08:00

The Risk Register is a living document that captures, assesses, and tracks potential risks associated with an AI system. It's a proactive tool for managing compliance and ensuring responsible AI deployment.

### User Input: Identifying AI Risks for the Risk Register

On the "Artifacts & Risk Management" page, scroll to "Section 13: User Input: Identifying AI Risks for the Risk Register".

The application provides a form to add new risks and pre-fills some common simulated AI risks to kickstart the register. These risks are stored in `st.session_state.risk_register_entries`.

```python
# application_pages/page3.py - run_risk_input_form()
def run_risk_input_form():
    st.header("Section 13: User Input: Identifying AI Risks for the Risk Register")
    st.markdown(r"A Risk Register is a living document that captures, assesses, and tracks potential risks associated with an AI system. It is a critical tool for proactive risk management and compliance. Use the form below to add synthetic risks to our register.")

    if not st.session_state.risk_register_entries:
        # Pre-fill initial risks for demonstration
        initial_risks = [
            {'Category': 'Data Quality', 'Description': 'Synthetic data contains outliers affecting model robustness.', 'Impact Rating': 'Medium', 'Mitigation Strategy': 'Implement robust data cleansing and outlier detection mechanisms.', 'Status': 'Open'},
            {'Category': 'Algorithmic Bias', 'Description': 'Simulated model shows bias towards certain synthetic demographic features.', 'Impact Rating': 'High', 'Mitigation Strategy': 'Re-evaluate feature engineering; implement fairness-aware training techniques; monitor bias metrics.', 'Status': 'Open'},
            {'Category': 'Human Over-reliance', 'Description': 'Users may over-trust synthetic model predictions without understanding limitations.', 'Impact Rating': 'Low', 'Mitigation Strategy': 'Provide clear documentation on model limitations and uncertainty; conduct user training.', 'Status': 'Open'},
            {'Category': 'Privacy/Security', 'Description': 'Risk of unintended data exposure if synthetic data generation is not fully controlled.', 'Impact Rating': 'Medium', 'Mitigation Strategy': 'Ensure synthetic data generation environments are isolated and audited; review data anonymization techniques.', 'Status': 'Open'}
        ]
        for risk in initial_risks:
            risk_id = len(st.session_state.risk_register_entries) + 1
            st.session_state.risk_register_entries.append(create_risk_entry(risk_id, risk['Category'], risk['Description'], risk['Impact Rating'], risk['Mitigation Strategy']))
        st.info("A few simulated risks have been pre-filled for demonstration purposes.")

    with st.expander("Add New AI Risk", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            category = st.selectbox("Category:", options=RISK_CATEGORIES, key="risk_category_input")
            impact = st.selectbox("Impact Rating:", options=IMPACT_RATINGS, key="risk_impact_input")
        with col2:
            description = st.text_area("Description:", placeholder='Enter a detailed description of the AI risk...', height=100, key="risk_description_input")
            mitigation = st.text_area("Mitigation Strategies:", placeholder='Enter strategies to mitigate the identified risk...', height=100, key="risk_mitigation_input")

        if st.button("Add AI Risk (Simulated)"):
            if description and mitigation:
                add_risk_to_register(st.session_state.risk_register_entries, category, description, impact, mitigation)
                st.success(f"Risk '{category}' (Impact: {impact}) added to register.")
            else:
                st.warning("Please fill in both the Description and Mitigation Strategies before adding a risk.")
    # Displays current risks from session state
    # ... (dataframe display) ...
```
1.  Review the pre-filled risks.
2.  Use the form to add your own simulated risks: select a `Category` (e.g., 'Data Quality', 'Algorithmic Bias'), describe the `Description`, choose an `Impact Rating`, and suggest `Mitigation Strategies`.
3.  Click **"Add AI Risk (Simulated)"** to add it to the register.

### Generating Risk Register Content

Scroll to "Section 14: Generating Risk Register Content".
This section compiles all individual risk entries into a structured Pandas DataFrame, `st.session_state.risk_register_df`.

```python
# application_pages/page3.py - compile_risk_register()
@st.cache_data
def compile_risk_register(risk_entries_list):
    if not risk_entries_list:
        return pd.DataFrame(columns=['Risk ID', 'Category', 'Description', 'Impact Rating', 'Mitigation Strategy', 'Status'])
    df = pd.DataFrame(risk_entries_list)
    return df
```
This action is triggered automatically on page load if risks exist.

### Core Visual: Risk Register Aggregated Comparison (Bar Chart)

Visualizing the Risk Register helps in quickly identifying high-priority areas and understanding the distribution of risks.

Scroll to "Section 15: Core Visual: Risk Register Aggregated Comparison (Bar Chart)".
This bar chart aggregates the identified risks by `Category` and `Impact Rating`, providing a clear overview for risk management.

```python
# application_pages/page3.py - plot_risk_aggregation()
@st.cache_data
def aggregate_risks_by_category_and_impact(risk_register_dataframe):
    if risk_register_dataframe.empty:
        return pd.DataFrame(columns=['Category', 'Impact Rating', 'Count of Risks'])
    # ... (aggregation logic) ...
    return aggregated_df

def plot_risk_aggregation(risk_dataframe, category_column, impact_column, title, save_path=None):
    if not risk_dataframe.empty:
        aggregated_data = aggregate_risks_by_category_and_impact(risk_dataframe)

        fig = px.bar(
            aggregated_data,
            x=category_column,
            y='Count of Risks',
            color=impact_column,
            title=title,
            barmode='group',
            category_orders={category_column: RISK_CATEGORIES, impact_column: IMPACT_RATINGS},
            color_discrete_sequence=px.colors.qualitative.Vivid,
            labels={'Category': 'Risk Category', 'Count of Risks': 'Count of Risks', 'Impact Rating': 'Impact Rating'}
        )
        # ... (layout updates) ...
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Risk DataFrame is empty, cannot generate risk aggregation plot.")
```
Examine the bar chart to see which categories have the most risks, particularly those with a 'High' impact rating.

### Displaying the Risk Register (Interactive Table)

Scroll to "Section 16: Displaying the Risk Register (Interactive Table)".
Finally, the complete Risk Register is displayed as an interactive table, providing a detailed view of each identified risk.

```python
# application_pages/page3.py - display_risk_register()
def display_risk_register():
    st.header("Section 16: Displaying the Risk Register (Interactive Table)")
    st.markdown(r"""To provide a detailed view of each identified risk, we will display the complete AI System Risk Register. This interactive table allows for easy review of all risks, their descriptions, impact ratings, mitigation strategies, and current status.""")
    if not st.session_state.risk_register_df.empty:
        # Helper function from app.py
        def display_interactive_dataframe(dataframe, title):
            st.subheader(title)
            st.dataframe(dataframe)
        display_interactive_dataframe(st.session_state.risk_register_df, "AI System Risk Register")
        st.success("Interactive Risk Register table displayed.")
    else:
        st.warning("Cannot display Risk Register: DataFrame is empty. Please add and compile risks first on this page (Sections 13 & 14).")
```
This table provides a central place to review and manage all documented AI-related risks.

## Discussion, Conclusion, and References
Duration: 05:00

This final step connects the generated artifacts to the broader context of AI assurance, summarizes the key learnings, and provides references for further study.

### Discussion: Connecting Artifacts to AI Assurance

On the "Artifacts & Risk Management" page, scroll to "Section 17: Discussion: Connecting Artifacts to AI Assurance".

The Model Card, Data Card, and Risk Register, when used together, form a powerful set of "**Evidence Artifacts**" for AI assurance [3]. These documents are not merely bureaucratic exercises but fundamental tools for building and maintaining trustworthy and responsible AI systems.

*   The **Model Card** provides transparency into the model's behavior, intended use, and limitations, which is critical for an "effective challenge" [1]. It helps ensure that stakeholders understand what the model does, how it performs, and where its boundaries lie.
*   The **Data Card** ensures accountability regarding data provenance, characteristics, and potential biases, directly addressing the "data" dimension of AI risk [1]. By meticulously documenting the dataset, organizations can better manage data quality, privacy, and fairness issues.
*   The **Risk Register** explicitly maps identified hazards to mitigation strategies, directly supporting proactive risk management and compliance with frameworks like SR 11-7 and NIST AI RMF [1, 4]. It serves as a dynamic record of potential pitfalls and the actions taken to address them.

These structured documents not only improve auditability but also foster trust and responsible development of AI systems by ensuring all key aspects are thoroughly documented and reviewed. They facilitate stakeholder review and can be mapped to NIST AI RMF functions (Govern, Map, Measure, Manage) [4]. Their systematic application ensures a comprehensive approach to AI governance, moving from abstract principles to concrete, actionable evidence.

### Conclusion

On the "Artifacts & Risk Management" page, scroll to "Section 18: Conclusion".

This lab has provided a hands-on experience in generating crucial AI assurance documentation. By creating synthetic Model Cards, Data Cards, and Risk Registers, you've gained practical insight into how these artifacts:

*   Make AI models and data transparent, fostering greater understanding and trust among users and stakeholders.
*   Facilitate auditability and stakeholder review, enabling rigorous scrutiny of AI systems and their components.
*   Provide essential evidence for robust AI risk management, ensuring that potential issues are identified, assessed, and mitigated proactively.

The ability to produce and maintain such artifacts is fundamental to building trustworthy and responsible AI systems. These practices are not just about compliance, but about instilling confidence and enabling the safe and ethical deployment of AI technologies in real-world scenarios.

### References

On the "Artifacts & Risk Management" page, scroll to "Section 19: References".

[1] "Unit 1: Principles of AI Risk and Assurance," including "Model Risk Management" and "Risk Taxonomy," [Provided Document, Page 1]. These sections introduce the need for documentation and the classification of AI risks.
[2] "Unit 2: Large Language Models and Agentic Architectures," [Provided Document, Page 2]. General context for AI models.
[3] "Evidence Artifacts," [Provided Document, Page 2]. This section specifically describes model cards, data cards, and risk registers as evidence.
[4] "SR 11-7 $ \leftrightarrow $ NIST AI RMF," [Provided Document, Page 1]. Mentions SR 11-7's call for thorough documentation.
[5] Python Software Foundation. Python Language Reference, version 3.x. Available at python.org.
[6] The Pandas Development Team. Pandas: a Python Data Analysis Library. Available at pandas.pydata.org.
[7] Hunter, J. D. (2007). Matplotlib: A 2D Graphics Environment. Computing in Science & Engineering, 9(3), 90-95. (Note: This reference is for Matplotlib, but Plotly is used in this app as per requirements.)
[8] Waskom, M. L. (2021). Seaborn: statistical data visualization. Journal of Open Source Software, 6(60), 3021. (Note: This reference is for Seaborn, but Plotly is used in this app as per requirements.)
[9] Pedregosa et al., (2011). Scikit-learn: Machine Learning in Python. JMLR 12, pp. 2825-2830.
[10] The Project Jupyter Team. ipywidgets: Interactive HTML Widgets for Jupyter. Available at ipywidgets.readthedocs.io. (Note: ipywidgets are replaced by Streamlit widgets in this app.)
