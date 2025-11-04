
# Streamlit Application Requirements Specification

## 1. Application Overview

This Streamlit application will provide an interactive workbench for understanding and generating essential AI assurance artifacts: Model Cards, Data Cards, and Risk Registers. Through a simulated scenario, users will learn to document AI models, their data, and associated risks.

**Learning Goals:**
*   Understand the purpose and components of Model Cards, Data Cards, and Risk Registers as key evidence artifacts for AI assurance.
*   Grasp how these documents contribute to transparency, auditability, and responsible AI practices.
*   Learn to interactively define AI model parameters and data characteristics.
*   Explore synthetic data generation, validation, and simulated model predictions.
*   Visualize data relationships and model performance trends.
*   Identify and document AI risks, including categories, impact, and mitigation strategies.
*   Interpret generated artifacts in an interactive, user-friendly interface.

## 2. User Interface Requirements

### Layout and Navigation Structure

The application will feature a clear, step-by-step layout, likely using a sidebar for navigation between major sections or displaying high-level controls, and the main area for content, inputs, visualizations, and artifact displays. Each major section of the Jupyter Notebook will correspond to a distinct part of the Streamlit application, potentially separated by Streamlit expanders or clear headings.

### Input Widgets and Controls

The application will incorporate various interactive input widgets to allow users to define parameters for the synthetic AI model and dataset, and to log AI risks. These will directly replace the `ipywidgets` used in the Jupyter Notebook. Each input will include descriptive labels and placeholder text.

| Notebook Widget Type (ipywidgets) | Streamlit Widget | Purpose | Help Text/Tooltip Content |
| :-------------------------------- | :--------------- | :------ | :------------------------ |
| `widgets.Text` (`model_name_widget`, `dataset_name_widget`) | `st.text_input` | Define model/dataset name. | E.g., "A descriptive name for the AI model (e.g., Customer Churn Predictor)." |
| `widgets.Textarea` (`purpose_widget`, `known_limitations_widget`, `usage_notes_widget`, `data_provenance_widget`, `collection_method_widget`, `identified_biases_widget`, `privacy_notes_widget`, `description_textarea`, `mitigation_textarea`) | `st.text_area` | Provide detailed descriptions, notes, or strategies. | Context-specific; e.g., "Describe the primary objective and function of this AI model." |
| `widgets.Dropdown` (`model_type_widget`, `category_dropdown`, `impact_dropdown`) | `st.selectbox` | Select from predefined options. | Context-specific; e.g., "Choose the type of AI model being simulated (e.g., Classification, Regression)." |
| `widgets.FloatText` (`performance_threshold_widget`) | `st.number_input` | Input a floating-point numerical value within a range. | "Minimum F1-score considered acceptable for this model (0.0 to 1.0)." |
| `widgets.IntText` (`n_samples_widget`, `n_features_widget`, `n_categorical_features_widget`) | `st.number_input` | Input an integer numerical value. | Context-specific; e.g., "Number of synthetic data samples to generate." |
| `widgets.Button` (`add_risk_button`) | `st.button` | Trigger an action (e.g., add a risk to the register). | "Click to add the identified AI risk to the register." |

### Visualization Components

The application will include the following visualizations to provide insights into the synthetic data, model performance, and risk profile:

1.  **Relationship Plot (Scatter Plot)**:
    *   **Content**: Visualizes the correlation between two numeric features, colored by the true label.
    *   **Configuration**: `feature_x`, `feature_y`, `hue_column` (true_label), `title`.
    *   **Style**: Color-blind friendly palette (`palette='colorblind'`), clear titles, labeled axes, font size $\ge 12$ pt.
2.  **Trend Plot (Line Plot)**:
    *   **Content**: Displays a simulated performance metric (e.g., accuracy) over time, utilizing the `timestamp` column.
    *   **Configuration**: `timestamp_column` (date_key), `metric_column` (simulated_accuracy), `title`.
    *   **Style**: Color-blind friendly palette (`color='steelblue'`), clear titles, labeled axes, font size $\ge 12$ pt, subtle grid.
3.  **Risk Register Aggregated Comparison (Bar Chart)**:
    *   **Content**: Shows the count of identified risks aggregated by "Risk Category" and "Impact Rating".
    *   **Configuration**: `category_column` (`Category`), `impact_column` (`Impact Rating`), `title`.
    *   **Style**: Color-blind friendly palette (`palette='viridis'`), clear titles, labeled axes, font size $\ge 12$ pt, x-axis labels rotated for readability, legend placed outside.
    *   **Interactivity**: Bar labels for counts on top of bars.

The application will also display interactive tables for the generated artifacts:

1.  **Synthetic AI Model Card (Interactive Table)**:
    *   **Content**: Presents key model attributes and performance metrics in a tabular format.
    *   **Components**: `st.dataframe` to display `model_card_df`.
2.  **Synthetic Dataset Data Card (Interactive Tables)**:
    *   **Content**: Two tables: an overview of data characteristics and detailed feature statistics.
    *   **Components**: `st.dataframe` for `data_card_df_top` and `data_card_df_features`.
3.  **AI System Risk Register (Interactive Table)**:
    *   **Content**: Displays all identified risks, their descriptions, impact, and mitigation strategies.
    *   **Components**: `st.dataframe` to display `risk_register_df`.

### Interactive Elements and Feedback Mechanisms

*   **Input-driven updates**: Changes in input widgets will trigger re-execution of relevant parts of the application, dynamically updating derived calculations and visualizations.
*   **Action buttons**: Buttons (e.g., "Add AI Risk (Simulated)") will trigger specific actions and update the application state.
*   **Progress indicators**: Streamlit's built-in caching (`st.cache_data`, `st.cache_resource`) will optimize performance. Status messages (`st.info`, `st.success`, `st.warning`) will provide user feedback on operations (e.g., "Synthetic AI model parameters captured.").
*   **Error handling**: Robust error handling will be implemented for data validation and function calls, displaying user-friendly error messages (`st.error`).

## 3. Additional Requirements

*   **Annotation and Tooltip Specifications**:
    *   Each user input widget (text inputs, dropdowns, number inputs) will be accompanied by inline help text or tooltips (`help` parameter in Streamlit widgets) that clearly describe its purpose and expected input format, as demonstrated in the "Input Widgets and Controls" section.
*   **Save the states of the fields properly so that changes are not lost**:
    *   All user-defined parameters (`synthetic_model_parameters`, `synthetic_data_characteristics`), generated data (`df_synthetic`), calculated metrics, and compiled artifact content (`model_card`, `data_card`, `risk_register_entries`, `risk_register_df`) will be stored in `st.session_state`. This ensures that user inputs and application state persist across re-runs or interactions within the Streamlit session.

## 4. Notebook Content and Code Requirements

This section outlines how the Jupyter Notebook's markdown content and Python code will be integrated into the Streamlit application, maintaining logical flow and fulfilling interactive requirements.

---

### Section: Application Title and Introduction
**Notebook Content:**
```markdown
# AI Assurance Artifact Workbench: Creating Model Cards, Data Cards, and Risk Registers

This Jupyter Notebook serves as a comprehensive guide to understanding and generating essential AI assurance artifacts: **Model Cards**, **Data Cards**, and **Risk Registers**. By simulating a practical scenario involving a synthetic AI model and its associated data, users will gain hands-on experience in creating these crucial documents.

### What You Will Learn:

*   **Purpose of Key Artifacts:** Understand the core functions and components of Model Cards, Data Cards, and Risk Registers.
*   **Evidence for AI Assurance:** Grasp how these documents act as
```
**Streamlit Implementation:**
*   `st.title("AI Assurance Artifact Workbench: Creating Model Cards, Data Cards, and Risk Registers")`
*   `st.markdown("This Streamlit application serves as a comprehensive guide...")`
*   `st.header("What You Will Learn:")`
*   `st.markdown("- **Purpose of Key Artifacts:** Understand the core functions and components of Model Cards, Data Cards, and Risk Registers.")`
*   `st.markdown("- **Evidence for AI Assurance:** Grasp how these documents act as essential evidence for AI assurance.")`

---

### Section: Environment Setup and Imports
**Notebook Content:**
```markdown
## 2. Environment Setup

Before diving into the specifics of AI assurance artifacts, we need to set up our development environment by installing the necessary Python libraries and importing them. This ensures that all required functionalities for data generation, manipulation, visualization, and interactive elements are available.
```
```python
# Standard library imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn imports for data generation, model, and metrics
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Streamlit-specific imports
import streamlit as st

# Configure plotting styles for better aesthetics
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6) # Default figure size
plt.rcParams['font.size'] = 12 # Default font size
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

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

# Helper function to display dataframes (replaces ipywidgets display)
def display_interactive_dataframe(dataframe, title):
    st.subheader(title)
    st.dataframe(dataframe)

# (Other notebook specific functions will be defined as needed later)
```
**Streamlit Implementation:**
*   `st.header("2. Environment Setup")`
*   `st.markdown("Before diving into the specifics...")`
*   All imports will be placed at the top of the `app.py` file.
*   Global `RISK_CATEGORIES` and `IMPACT_RATINGS` will be defined.
*   `plt.rcParams` settings will be applied for consistent plot styling.
*   `st.session_state` will be initialized for all persistent variables as shown above.
*   The `display_interactive_dataframe` helper will be used to show dataframes.

---

### Section: Data/Inputs Overview
**Notebook Content:**
```markdown
## 3. Data/Inputs Overview

In this notebook, we will leverage **synthetic data** to simulate a realistic AI development and deployment scenario. This approach allows us to demonstrate the creation of AI assurance artifacts without relying on sensitive or proprietary real-world datasets.

Our synthetic data will feature:
*   **Numeric features:** Representing various measurable attributes.
*   **Categorical features:** Simulating discrete variables.
*   **A timestamp column:** Enabling the simulation of time-series trends and performance monitoring over time.
*   **A binary target variable:** For a classification task, which is common in many AI applications.

This simulated environment is critical for showcasing how Model Cards, Data Cards, and Risk Registers can be generated and utilized to promote transparency, auditability, and responsible AI practices, aligning with regulatory expectations for thorough documentation (e.g., SR 11-7 [4]). The business goal here is to establish a robust framework for documenting AI systems, even in early development stages or when sensitive data is not available for direct demonstration.
```
**Streamlit Implementation:**
*   `st.header("3. Data/Inputs Overview")`
*   `st.markdown("In this application, we will leverage **synthetic data**...")`

---

### Section: Methodology Overview & Key Formulae
**Notebook Content:**
```markdown
## 4. Methodology Overview

Our approach to generating AI assurance artifacts follows a structured, step-by-step process:
... (list of steps) ...

### Key Formulae and Their Business Rationale:

**1. Logistic Regression Probability (for Model Simulation):**
$$ P(Y=1|X) = \frac{1}{1 + e^{-(\beta_0 + \sum_{i=1}^k \beta_i X_i)}} $$
Where:
*   $P(Y=1|X)$ is the probability of the positive class (e.g., a customer being `high-value`).
*   $X_i$ represents the input features.
*   $\beta_0$ is the intercept term.
*   $\beta_i$ are the coefficients (weights) assigned to each feature.

**Business Rationale:** Understanding how a model arrives at its predictions...

**2. Classification Performance Metrics (for Model Cards):**

*   **Accuracy**:
    $$ \text{Accuracy} = \frac{\text{Number of Correct Predictions}}{\text{Total Number of Predictions}} $$
*   **Precision**:
    $$ \text{Precision} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}} $$
*   **Recall (Sensitivity)**:
    $$ \text{Recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}} $$
*   **F1-Score**:
    $$ \text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} $$

**Business Rationale:** These metrics are critical for evaluating whether an AI model meets its intended performance targets...
```
**Streamlit Implementation:**
*   `st.header("4. Methodology Overview")`
*   `st.markdown("Our approach to generating AI assurance artifacts follows...")`
*   `st.subheader("Key Formulae and Their Business Rationale:")`
*   `st.markdown("**1. Logistic Regression Probability (for Model Simulation):**")`
*   `st.latex("P(Y=1|X) = \\frac{1}{1 + e^{-(\\beta_0 + \\sum_{i=1}^k \\beta_i X_i)}}")`
*   `st.markdown("Where: ...")` (for explanation of variables)
*   `st.markdown("**Business Rationale:** Understanding how a model arrives at its predictions...")`
*   `st.markdown("**2. Classification Performance Metrics (for Model Cards):**")`
*   `st.markdown("*   **Accuracy**: ")`
*   `st.latex("\\text{Accuracy} = \\frac{\\text{Number of Correct Predictions}}{\\text{Total Number of Predictions}}")`
*   Similarly for Precision, Recall, and F1-Score using `st.latex()`.
*   `st.markdown("**Business Rationale:** These metrics are critical for evaluating...")`

---

### Section: Defining Synthetic AI Model Parameters
**Notebook Content:**
```markdown
### Section 3: Defining Synthetic AI Model Parameters
...
```
```python
def define_model_parameters_streamlit():
    """
    Streamlit-compatible function to capture synthetic AI model parameters.
    """
    with st.expander("Define Synthetic AI Model Parameters", expanded=True):
        st.markdown("A Model Card provides a structured overview... Use the widgets below to specify.")

        col1, col2 = st.columns(2)
        with col1:
            model_name = st.text_input(
                "Model Name:",
                value=st.session_state.synthetic_model_parameters.get('model_name', "Synthetic AI Analyst Assistant"),
                placeholder="e.g., Customer Sentiment Analyzer",
                help="A descriptive name for the AI model."
            )
            model_type = st.selectbox(
                "Model Type:",
                options=['Classification', 'Regression', 'Generative'],
                index=['Classification', 'Regression', 'Generative'].index(st.session_state.synthetic_model_parameters.get('model_type', "Classification")),
                help="The type of AI model being simulated."
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
                placeholder="e.g., Classifies customer reviews into positive, neutral, or negative.",
                height=100,
                help="Describe the primary objective and function of this AI model."
            )
            known_limitations = st.text_area(
                "Limitations:",
                value=st.session_state.synthetic_model_parameters.get('known_limitations', "Performance may degrade on highly imbalanced datasets."),
                placeholder="e.g., Struggles with sarcasm; performance degrades on noisy data.",
                height=100,
                help="Document any known limitations or failure modes of the model."
            )
            usage_notes = st.text_area(
                "Usage Notes:",
                value=st.session_state.synthetic_model_parameters.get('usage_notes', "Intended for internal analytical use by data science teams."),
                placeholder="e.g., Recommended for internal use by data scientists only.",
                height=100,
                help="Provide guidance on the appropriate use and users of this model."
            )

        st.session_state.synthetic_model_parameters = {
            'model_name': model_name,
            'purpose': purpose,
            'model_type': model_type,
            'performance_threshold': performance_threshold,
            'known_limitations': known_limitations,
            'usage_notes': usage_notes
        }
    st.success("Synthetic AI model parameters captured.")
```
**Streamlit Implementation:**
*   `st.header("Section 3: Defining Synthetic AI Model Parameters")`
*   The `define_model_parameters_streamlit` function will be called directly in the app, using `st.text_input`, `st.text_area`, `st.selectbox`, `st.number_input`.
*   Values will be read from and saved to `st.session_state.synthetic_model_parameters`.
*   Widgets will be organized using `st.columns` for better layout and wrapped in `st.expander`.
*   Help text will be added using the `help` parameter for each widget.

---

### Section: Defining Synthetic Data Characteristics
**Notebook Content:**
```markdown
### Section 4: Defining Synthetic Data Characteristics
...
```
```python
def define_data_characteristics_streamlit():
    """
    Streamlit-compatible function to capture synthetic dataset characteristics.
    """
    with st.expander("Define Synthetic Data Characteristics", expanded=True):
        st.markdown("A Data Card documents the dataset... Use the widgets below to specify.")

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
                help='Number of data samples to generate.'
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
            'dataset_name': dataset_name,
            'n_samples': n_samples,
            'n_features': n_features,
            'n_categorical_features': n_categorical_features,
            'data_provenance': data_provenance,
            'collection_method': collection_method,
            'identified_biases_description': identified_biases_description,
            'privacy_notes': privacy_notes,
        }
    st.success("Synthetic dataset characteristics captured.")
```
**Streamlit Implementation:**
*   `st.header("Section 4: Defining Synthetic Data Characteristics")`
*   The `define_data_characteristics_streamlit` function will be called, using `st.text_input`, `st.text_area`, `st.number_input`.
*   Values will be read from and saved to `st.session_state.synthetic_data_characteristics`.
*   Widgets will be organized using `st.columns` and wrapped in `st.expander`.
*   Help text will be added using the `help` parameter.

---

### Section: Synthetic Data Generation and Model Simulation
**Notebook Content:**
```markdown
### Section 5: Synthetic Data Generation and Model Simulation
...
$$ P(Y=1|X) = \frac{1}{1 + e^{-(\beta_0 + \sum_{i=1}^k \beta_i X_i)}} $$
...
```
```python
@st.cache_data
def generate_synthetic_data(n_samples, n_features, n_classes=2, random_state=42):
    # ... (exact code from notebook) ...
    return df

@st.cache_data
def simulate_model_predictions(features, labels, model_type='Logistic Regression', random_state=42):
    # ... (exact code from notebook) ...
    return predicted_label, prediction_score

def run_data_generation_and_simulation():
    if st.button("Generate Synthetic Data & Simulate Model"):
        if st.session_state.synthetic_data_characteristics['n_samples'] == 0:
            st.warning("Number of samples is 0. Dataframe will be empty.")
            st.session_state.df_synthetic = pd.DataFrame()
            return

        with st.spinner("Generating synthetic data and simulating model predictions..."):
            df_synthetic_local = generate_synthetic_data(
                st.session_state.synthetic_data_characteristics['n_samples'],
                st.session_state.synthetic_data_characteristics['n_features'],
                random_state=42
            )

            X = df_synthetic_local.select_dtypes(include=np.number).drop(['true_label'], axis=1, errors='ignore')
            y = df_synthetic_local['true_label']

            if not X.empty and not y.empty:
                predicted_label, prediction_score = simulate_model_predictions(X, y)
                df_synthetic_local['predicted_label'] = predicted_label
                df_synthetic_local['prediction_score'] = prediction_score
            else:
                df_synthetic_local['predicted_label'] = pd.Series([], dtype=int)
                df_synthetic_local['prediction_score'] = pd.Series([], dtype=float)

            st.session_state.df_synthetic = df_synthetic_local
            st.success("Generated synthetic data with simulated model predictions.")
            st.write("Here's a preview of the generated data:")
            st.dataframe(st.session_state.df_synthetic.head())

```
**Streamlit Implementation:**
*   `st.header("Section 5: Synthetic Data Generation and Model Simulation")`
*   `st.markdown("Now we will generate a synthetic dataset...")`
*   `st.latex("P(Y=1|X) = \\frac{1}{1 + e^{-(\\beta_0 + \\sum_{i=1}^k \\beta_i X_i)}}")`
*   The `generate_synthetic_data` and `simulate_model_predictions` functions will be defined. They will be decorated with `@st.cache_data` for performance.
*   A Streamlit button (`st.button`) will trigger the execution of `run_data_generation_and_simulation()`.
*   Progress feedback will be provided using `st.spinner`.
*   The generated `df_synthetic` will be stored in `st.session_state`.
*   `st.dataframe(st.session_state.df_synthetic.head())` will display a preview.

---

### Section: Data Validation and Exploration
**Notebook Content:**
```markdown
### Section 6: Data Validation and Exploration
...
```
```python
@st.cache_data
def perform_data_validation(dataframe, critical_fields):
    # ... (exact code from notebook) ...
    # Replace display(dataframe.head()) with st.dataframe(dataframe.head())
    return None # Function prints directly in Streamlit context

def run_data_validation():
    if not st.session_state.df_synthetic.empty:
        critical_fields = ['feature_1', 'feature_2', 'feature_3', 'true_label', 'predicted_label', 'prediction_score', 'timestamp']
        st.header("Section 6: Data Validation and Exploration")
        st.markdown("Before proceeding, it's crucial to validate the generated synthetic data...")
        with st.expander("Perform Data Validation", expanded=True):
            perform_data_validation(st.session_state.df_synthetic, critical_fields)
        st.success("Synthetic data validated.")
    else:
        st.warning("Cannot perform data validation: Synthetic DataFrame is empty. Please generate data first.")
```
**Streamlit Implementation:**
*   `st.header("Section 6: Data Validation and Exploration")`
*   `st.markdown("Before proceeding, it's crucial to validate...")`
*   The `perform_data_validation` function will be defined with `@st.cache_data`.
*   `st.warning`, `st.info`, `st.dataframe` will be used within `perform_data_validation` for output.
*   The `run_data_validation()` function will be called, potentially inside an `st.expander`.

---

### Section: Core Visual: Relationship Plot
**Notebook Content:**
```markdown
### Section 7: Core Visual: Relationship Plot
...
```
```python
def plot_relationship(dataframe, feature_x, feature_y, hue_column, title, save_path=None):
    # ... (exact code from notebook) ...
    # Replace plt.show() with st.pyplot(plt)
    # Ensure plt.close() is called after st.pyplot()
    if not dataframe.empty:
        if feature_x not in dataframe.columns or feature_y not in dataframe.columns or hue_column not in dataframe.columns:
            st.error(f"Required columns missing for relationship plot: {feature_x}, {feature_y}, {hue_column}")
            return
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=dataframe, x=feature_x, y=feature_y, hue=hue_column, palette='colorblind', s=80, alpha=0.7)
        plt.title(title, fontsize=16)
        plt.xlabel(feature_x, fontsize=12)
        plt.ylabel(feature_y, fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        st.pyplot(plt)
        plt.close() # Important to release memory

def run_relationship_plot():
    st.header("Section 7: Core Visual: Relationship Plot")
    st.markdown("Understanding relationships within the data is essential...")
    if not st.session_state.df_synthetic.empty:
        numeric_features = st.session_state.df_synthetic.select_dtypes(include=np.number).columns.tolist()
        if 'true_label' in numeric_features: numeric_features.remove('true_label')
        if 'predicted_label' in numeric_features: numeric_features.remove('predicted_label')
        if 'prediction_score' in numeric_features: numeric_features.remove('prediction_score')

        if len(numeric_features) >= 2:
            col_x, col_y = st.columns(2)
            with col_x:
                feature_x_plot = st.selectbox("Select X-axis Feature:", options=numeric_features, index=0, key="rel_x_feat")
            with col_y:
                feature_y_plot = st.selectbox("Select Y-axis Feature:", options=numeric_features, index=1 if len(numeric_features) > 1 else 0, key="rel_y_feat")

            plot_relationship(st.session_state.df_synthetic, feature_x_plot, feature_y_plot, 'true_label',
                              f'Relationship between {feature_x_plot} and {feature_y_plot} by True Label')
            st.success(f"Relationship plot generated for {feature_x_plot} and {feature_y_plot}.")
        else:
            st.warning("Not enough numeric features to generate a relationship plot.")
    else:
        st.warning("Cannot generate relationship plot: Synthetic DataFrame is empty. Please generate data first.")
```
**Streamlit Implementation:**
*   `st.header("Section 7: Core Visual: Relationship Plot")`
*   `st.markdown("Understanding relationships within the data is essential...")`
*   The `plot_relationship` function will be defined, replacing `plt.show()` with `st.pyplot(plt)`. `plt.close()` will be called.
*   Dynamic feature selection will be implemented using `st.selectbox` for `feature_x_plot` and `feature_y_plot`.
*   The `run_relationship_plot()` function will be called.

---

### Section: Core Visual: Trend Plot for Simulated Performance
**Notebook Content:**
```markdown
### Section 8: Core Visual: Trend Plot for Simulated Performance
...
```
```python
def plot_performance_trend(dataframe, timestamp_column, metric_column, title, save_path=None):
    # ... (exact code from notebook) ...
    # Replace plt.show() with st.pyplot(plt)
    # Ensure plt.close() is called after st.pyplot()
    if not dataframe.empty:
        if timestamp_column not in dataframe.columns or metric_column not in dataframe.columns:
            st.error(f"Required columns missing for trend plot: {timestamp_column}, {metric_column}")
            return
        plt.figure(figsize=(10, 6))
        sns.lineplot(x=timestamp_column, y=metric_column, data=dataframe.sort_values(by=timestamp_column), color='steelblue')
        plt.title(title, fontsize=16)
        plt.xlabel(timestamp_column, fontsize=12)
        plt.ylabel(metric_column, fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        st.pyplot(plt)
        plt.close()

def run_performance_trend_plot():
    st.header("Section 8: Core Visual: Trend Plot for Simulated Performance")
    st.markdown("Monitoring model performance over time is a critical aspect...")
    if not st.session_state.df_synthetic.empty and 'true_label' in st.session_state.df_synthetic.columns and 'predicted_label' in st.session_state.df_synthetic.columns and 'timestamp' in st.session_state.df_synthetic.columns:
        df_synthetic_copy = st.session_state.df_synthetic.copy()
        df_synthetic_copy['simulated_accuracy'] = (df_synthetic_copy['true_label'] == df_synthetic_copy['predicted_label']).astype(int)
        daily_performance = df_synthetic_copy.groupby(df_synthetic_copy['timestamp'].dt.date)['simulated_accuracy'].mean().reset_index()
        daily_performance.rename(columns={'timestamp': 'date_key'}, inplace=True)

        plot_performance_trend(daily_performance, 'date_key', 'simulated_accuracy', 'Simulated Model Accuracy Over Time')
        st.success("Simulated model accuracy trend plot generated.")
    else:
        st.warning("Cannot generate simulated performance trend plot: DataFrame is empty or missing 'true_label'/'predicted_label'/'timestamp'. Please generate data first.")
```
**Streamlit Implementation:**
*   `st.header("Section 8: Core Visual: Trend Plot for Simulated Performance")`
*   `st.markdown("Monitoring model performance over time is a critical aspect...")`
*   The `plot_performance_trend` function will be defined, replacing `plt.show()` with `st.pyplot(plt)`. `plt.close()` will be called.
*   The calculation for `daily_performance` will be integrated.
*   The `run_performance_trend_plot()` function will be called.

---

### Section: Generating Model Card Content
**Notebook Content:**
```markdown
### Section 9: Generating Model Card Content
...
*   **Accuracy**: $$ \text{Accuracy} = \frac{\text{Number of Correct Predictions}}{\text{Total Number of Predictions}} $$
... (Precision, Recall, F1-Score formulas) ...
```
```python
@st.cache_data
def calculate_model_metrics(true_labels, predicted_labels, prediction_scores):
    # ... (exact code from notebook) ...
    return metrics

@st.cache_data
def generate_model_card_content(model_params, true_labels, predicted_labels, prediction_scores):
    # ... (exact code from notebook) ...
    return model_card

def run_model_card_generation():
    st.header("Section 9: Generating Model Card Content")
    st.markdown("Now we will compile all the relevant information for our Model Card...")
    st.markdown("We will calculate common classification metrics:")
    st.markdown("*   **Accuracy**: ")
    st.latex("\\text{Accuracy} = \\frac{\\text{Number of Correct Predictions}}{\\text{Total Number of Predictions}}")
    st.markdown("*   **Precision**: ")
    st.latex("\\text{Precision} = \\frac{\\text{True Positives}}{\\text{True Positives} + \\text{False Positives}}")
    st.markdown("*   **Recall**: ")
    st.latex("\\text{Recall} = \\frac{\\text{True Positives}}{\\text{True Positives} + \\text{False Negatives}}")
    st.markdown("*   **F1-Score**: The harmonic mean of Precision and Recall. ")
    st.latex("\\text{F1-Score} = 2 \\times \\frac{\\text{Precision} \\times \\text{Recall}}{\\text{Precision} + \\text{Recall}}")

    if not st.session_state.df_synthetic.empty and st.session_state.synthetic_model_parameters:
        model_performance_metrics = calculate_model_metrics(
            st.session_state.df_synthetic['true_label'],
            st.session_state.df_synthetic['predicted_label'],
            st.session_state.df_synthetic['prediction_score']
        )

        st.session_state.model_card = generate_model_card_content(
            st.session_state.synthetic_model_parameters,
            st.session_state.df_synthetic['true_label'],
            st.session_state.df_synthetic['predicted_label'],
            st.session_state.df_synthetic['prediction_score']
        )
        st.success("Model Card content generated successfully.")
    else:
        st.warning("Cannot generate Model Card: Synthetic DataFrame is empty or model parameters are not defined. Please generate data and define model parameters first.")
```
**Streamlit Implementation:**
*   `st.header("Section 9: Generating Model Card Content")`
*   Markdown and LaTeX for metrics will be displayed using `st.markdown` and `st.latex`.
*   The `calculate_model_metrics` and `generate_model_card_content` functions will be defined with `@st.cache_data`.
*   The `run_model_card_generation()` function will be called, updating `st.session_state.model_card`.

---

### Section: Displaying the Model Card (Interactive Table)
**Notebook Content:**
```markdown
### Section 10: Displaying the Model Card (Interactive Table)
...
```
```python
def display_model_card():
    st.header("Section 10: Displaying the Model Card (Interactive Table)")
    st.markdown("The generated Model Card content is now ready for display...")
    if st.session_state.model_card:
        model_card_flat = {
            'Model Name': st.session_state.model_card['Model Name'],
            'Purpose': st.session_state.model_card['Purpose'],
            'Model Type': st.session_state.model_card['Model Type'],
            'Minimum Acceptable F1-score': st.session_state.model_card['Minimum Acceptable F1-score'],
            'Known Limitations': st.session_state.model_card['Known Limitations'],
            'Usage Notes': st.session_state.model_card['Usage Notes']
        }
        for metric, value in st.session_state.model_card['Performance Metrics'].items():
            model_card_flat[f'Metric: {metric.replace("_", " ").title()}'] = f"{value:.4f}"
        
        model_card_df = pd.DataFrame(list(model_card_flat.items()), columns=['Attribute', 'Value'])
        display_interactive_dataframe(model_card_df, "Synthetic AI Model Card")
        st.success("Interactive Model Card displayed.")
    else:
        st.warning("Model Card is empty. Please generate Model Card content first.")
```
**Streamlit Implementation:**
*   `st.header("Section 10: Displaying the Model Card (Interactive Table)")`
*   `st.markdown("The generated Model Card content is now ready...")`
*   The `display_model_card()` function will be called, flattening the dictionary and using `display_interactive_dataframe` (which uses `st.dataframe`).

---

### Section: Generating Data Card Content
**Notebook Content:**
```markdown
### Section 11: Generating Data Card Content
...
```
```python
@st.cache_data
def generate_data_card_content(data_params, dataframe):
    # ... (exact code from notebook) ...
    return data_card

def run_data_card_generation():
    st.header("Section 11: Generating Data Card Content")
    st.markdown("Similar to the Model Card, a Data Card provides critical documentation...")
    if not st.session_state.df_synthetic.empty and st.session_state.synthetic_data_characteristics:
        st.session_state.data_card = generate_data_card_content(st.session_state.synthetic_data_characteristics, st.session_state.df_synthetic)
        st.success("Data Card content generated successfully.")
    else:
        st.warning("Cannot generate Data Card: Synthetic DataFrame is empty or data characteristics are not defined. Please generate data and define data characteristics first.")
```
**Streamlit Implementation:**
*   `st.header("Section 11: Generating Data Card Content")`
*   `st.markdown("Similar to the Model Card, a Data Card provides...")`
*   The `generate_data_card_content` function will be defined with `@st.cache_data`.
*   The `run_data_card_generation()` function will be called, updating `st.session_state.data_card`.

---

### Section: Displaying the Data Card (Interactive Table)
**Notebook Content:**
```markdown
### Section 12: Displaying the Data Card (Interactive Table)
...
```
```python
def display_data_card():
    st.header("Section 12: Displaying the Data Card (Interactive Table)")
    st.markdown("The generated Data Card content is now ready for display...")
    if st.session_state.data_card:
        data_card_flat = {
            'Dataset Name': st.session_state.data_card['dataset_name'],
            'Number of Samples': st.session_state.data_card['n_samples'],
            'Number of Features': st.session_state.data_card['n_features'],
            'Number of Categorical Features': st.session_state.data_card['n_categorical_features'],
            'Data Provenance': st.session_state.data_card['data_provenance'],
            'Collection Method': st.session_state.data_card['collection_method'],
            'Identified Biases Description': st.session_state.data_card['identified_biases_description'],
            'Privacy Notes': st.session_state.data_card['privacy_notes']
        }
        feature_stats_list = []
        for feature, stats in st.session_state.data_card['feature_statistics'].items():
            stats_summary = f"Dtype: {stats.get('dtype')}, Missing: {stats.get('missing_count')}, Unique: {stats.get('unique_count', 'N/A')}"
            if 'mean' in stats and stats['mean'] is not None:
                stats_summary += f", Mean: {stats['mean']:.2f}, Std: {stats['std']:.2f}"
            if 'top' in stats and stats['top'] is not None:
                stats_summary += f", Top: {stats['top']} ({stats['top_frequency']})"
            feature_stats_list.append({'Feature': feature, 'Summary Statistics': stats_summary})

        data_card_df_top = pd.DataFrame(list(data_card_flat.items()), columns=['Attribute', 'Value'])
        data_card_df_features = pd.DataFrame(feature_stats_list)

        display_interactive_dataframe(data_card_df_top, "Synthetic Dataset Data Card (Overview)")
        st.markdown("\nDetailed Feature Statistics:")
        display_interactive_dataframe(data_card_df_features, "Synthetic Dataset Data Card (Feature Statistics)")
        st.success("Interactive Data Card displayed.")
    else:
        st.warning("Data Card is empty. Please generate Data Card content first.")
```
**Streamlit Implementation:**
*   `st.header("Section 12: Displaying the Data Card (Interactive Table)")`
*   `st.markdown("The generated Data Card content is now ready...")`
*   The `display_data_card()` function will be called, displaying two tables using `display_interactive_dataframe` (which uses `st.dataframe`).

---

### Section: User Input: Identifying AI Risks for the Risk Register
**Notebook Content:**
```markdown
### Section 13: User Input: Identifying AI Risks for the Risk Register
...
```
```python
def add_risk_to_register(current_risks, category, description, impact, mitigation):
    # ... (exact code from notebook) ...
    # Appends to current_risks list in session state

def create_risk_entry(risk_id, category, description, impact_rating, mitigation_strategy):
    # ... (exact code from notebook) ...
    return {...}

def run_risk_input_form():
    st.header("Section 13: User Input: Identifying AI Risks for the Risk Register")
    st.markdown("A Risk Register is a living document... Use the form below to add synthetic risks.")

    with st.expander("Add New AI Risk", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            category = st.selectbox(
                "Category:",
                options=RISK_CATEGORIES,
                help="Select the category that best describes this risk."
            )
            impact = st.selectbox(
                "Impact Rating:",
                options=IMPACT_RATINGS,
                help="Rate the potential impact of this risk (Low, Medium, High)."
            )
        with col2:
            description = st.text_area(
                "Description:",
                placeholder='Enter a detailed description of the AI risk...',
                height=100,
                help="Provide a clear and concise description of the identified risk."
            )
            mitigation = st.text_area(
                "Mitigation Strategies:",
                placeholder='Enter strategies to mitigate the identified risk...',
                height=100,
                help="Outline the steps or strategies to reduce or eliminate this risk."
            )

        if st.button("Add AI Risk (Simulated)", help="Click to add this risk to the register."):
            add_risk_to_register(st.session_state.risk_register_entries, category, description, impact, mitigation)
            st.success(f"Risk '{category}' (Impact: {impact}) added to register.")
            # Pre-populate some initial risks if the list is empty at start
            if not st.session_state.risk_register_entries: # Add simulated risks only once if empty
                add_risk_to_register(st.session_state.risk_register_entries, 'Data Quality', 'Synthetic data contains outliers affecting model robustness.', 'Medium', 'Implement robust data cleansing and outlier detection mechanisms.')
                add_risk_to_register(st.session_state.risk_register_entries, 'Algorithmic Bias', 'Simulated model shows bias towards certain synthetic demographic features.', 'High', 'Re-evaluate feature engineering; implement fairness-aware training techniques; monitor bias metrics.')
                add_risk_to_register(st.session_state.risk_register_entries, 'Human Over-reliance', 'Users may over-trust synthetic model predictions without understanding limitations.', 'Low', 'Provide clear documentation on model limitations and uncertainty; conduct user training.')
                add_risk_to_register(st.session_state.risk_register_entries, 'Privacy/Security', 'Risk of unintended data exposure if synthetic data generation is not fully controlled.', 'Medium', 'Ensure synthetic data generation environments are isolated and audited; review data anonymization techniques.')
                st.info("A few simulated risks have been pre-filled for demonstration purposes.")

    st.markdown("Currently added risks (from session state):")
    if st.session_state.risk_register_entries:
        temp_df = pd.DataFrame(st.session_state.risk_register_entries)
        st.dataframe(temp_df)
    else:
        st.info("No risks added yet.")

```
**Streamlit Implementation:**
*   `st.header("Section 13: User Input: Identifying AI Risks for the Risk Register")`
*   `st.markdown("A Risk Register is a living document...")`
*   The `add_risk_to_register` and `create_risk_entry` functions will be defined.
*   An input form will be created using `st.selectbox` and `st.text_area`.
*   A `st.button` will trigger `add_risk_to_register`, appending to `st.session_state.risk_register_entries`.
*   Initial simulated risks will be added to `st.session_state` if the list is empty, ensuring the feature works.
*   The current list of risks in `st.session_state` will be displayed using `st.dataframe`.

---

### Section: Generating Risk Register Content
**Notebook Content:**
```markdown
### Section 14: Generating Risk Register Content
...
```
```python
@st.cache_data
def compile_risk_register(risk_entries_list):
    # ... (exact code from notebook) ...
    return pd.DataFrame(risk_entries_list)

def run_risk_register_compilation():
    st.header("Section 14: Generating Risk Register Content")
    st.markdown("With the identified risks, we can now formally compile the Risk Register...")
    if st.session_state.risk_register_entries:
        st.session_state.risk_register_df = compile_risk_register(st.session_state.risk_register_entries)
        st.success("Risk Register DataFrame compiled.")
        st.dataframe(st.session_state.risk_register_df.head())
    else:
        st.warning("Cannot compile Risk Register: No risks have been added yet.")
```
**Streamlit Implementation:**
*   `st.header("Section 14: Generating Risk Register Content")`
*   `st.markdown("With the identified risks, we can now formally compile...")`
*   The `compile_risk_register` function will be defined with `@st.cache_data`.
*   The `run_risk_register_compilation()` function will be called, updating `st.session_state.risk_register_df`.
*   `st.dataframe(st.session_state.risk_register_df.head())` will show a preview.

---

### Section: Core Visual: Risk Register Aggregated Comparison (Bar Chart)
**Notebook Content:**
```markdown
### Section 15: Core Visual: Risk Register Aggregated Comparison (Bar Chart)
...
```
```python
@st.cache_data
def aggregate_risks_by_category_and_impact(risk_register_dataframe):
    # ... (exact code from notebook) ...
    return aggregated_df

def plot_risk_aggregation(risk_dataframe, category_column, impact_column, title, save_path=None):
    # ... (exact code from notebook) ...
    # Replace plt.show() with st.pyplot(plt)
    # Ensure plt.close() is called after st.pyplot()
    if not risk_dataframe.empty:
        aggregated_data = aggregate_risks_by_category_and_impact(risk_dataframe)
        plt.figure(figsize=(12, 7))
        ax = sns.barplot(
            data=aggregated_data, x=category_column, y='Count of Risks',
            hue=impact_column, palette='viridis', errorbar=None
        )
        ax.set_title(title, fontsize=16)
        ax.set_xlabel('Risk Category', fontsize=12)
        ax.set_ylabel('Count of Risks', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Impact Rating', bbox_to_anchor=(1.05, 1), loc='upper left')
        for container in ax.containers:
            ax.bar_label(container, fmt='%d', label_type='center', color='white', fontsize=10)
        plt.tight_layout()
        st.pyplot(plt)
        plt.close()

def run_risk_aggregation_plot():
    st.header("Section 15: Core Visual: Risk Register Aggregated Comparison (Bar Chart)")
    st.markdown("Visualizing the Risk Register helps in quickly identifying high-priority areas...")
    if not st.session_state.risk_register_df.empty:
        plot_risk_aggregation(st.session_state.risk_register_df, 'Category', 'Impact Rating', 'Aggregated AI Risks by Category and Impact')
        st.success("Aggregated AI risks bar chart generated.")
    else:
        st.warning("Cannot generate aggregated risk comparison plot: Risk Register is empty. Please add risks first.")
```
**Streamlit Implementation:**
*   `st.header("Section 15: Core Visual: Risk Register Aggregated Comparison (Bar Chart)")`
*   `st.markdown("Visualizing the Risk Register helps...")`
*   The `aggregate_risks_by_category_and_impact` and `plot_risk_aggregation` functions will be defined, with `aggregate_risks_by_category_and_impact` using `@st.cache_data`.
*   `plot_risk_aggregation` will use `st.pyplot(plt)` and `plt.close()`.
*   The `run_risk_aggregation_plot()` function will be called.

---

### Section: Displaying the Risk Register (Interactive Table)
**Notebook Content:**
```markdown
### Section 16: Displaying the Risk Register (Interactive Table)
...
```
```python
def display_risk_register():
    st.header("Section 16: Displaying the Risk Register (Interactive Table)")
    st.markdown("To provide a detailed view of each identified risk, we will display...")
    if not st.session_state.risk_register_df.empty:
        display_interactive_dataframe(st.session_state.risk_register_df, "AI System Risk Register")
        st.success("Interactive Risk Register table displayed.")
    else:
        st.warning("Cannot display Risk Register: DataFrame is empty. Please add risks first.")
```
**Streamlit Implementation:**
*   `st.header("Section 16: Displaying the Risk Register (Interactive Table)")`
*   `st.markdown("To provide a detailed view of each identified risk...")`
*   The `display_risk_register()` function will be called, using `display_interactive_dataframe` (which uses `st.dataframe`).

---

### Section: Discussion: Connecting Artifacts to AI Assurance
**Notebook Content:**
```markdown
### Section 17: Discussion: Connecting Artifacts to AI Assurance

The Model Card, Data Card, and Risk Register, when used together, form a powerful set of "Evidence Artifacts" for AI assurance [3].
*   The **Model Card** provides transparency into the model's behavior and limitations, which is critical for an "effective challenge" [1].
*   The **Data Card** ensures accountability regarding data provenance and potential biases, addressing the "data" dimension of AI risk [1].
*   The **Risk Register** explicitly maps identified hazards to mitigation strategies, directly supporting proactive risk management and compliance with frameworks like SR 11-7 and NIST AI RMF [1, 4].

These structured documents not only improve auditability but also foster trust and responsible development of AI systems by ensuring all key aspects are thoroughly documented and reviewed. They facilitate stakeholder review and can be mapped to NIST AI RMF functions (Govern, Map, Measure, Manage) [4].
```
**Streamlit Implementation:**
*   `st.header("Section 17: Discussion: Connecting Artifacts to AI Assurance")`
*   `st.markdown("The Model Card, Data Card, and Risk Register, when used together...")`
*   All bullet points and additional descriptive text will be rendered using `st.markdown`.

---

### Section: Conclusion
**Notebook Content:**
```markdown
### Section 18: Conclusion

This lab has provided a hands-on experience in generating crucial AI assurance documentation. By creating synthetic Model Cards, Data Cards, and Risk Registers, you've gained practical insight into how these artifacts:
*   Make AI models and data transparent.
*   Facilitate auditability and stakeholder review.
*   Provide essential evidence for robust AI risk management.

The ability to produce and maintain such artifacts is fundamental to building trustworthy and responsible AI systems.
```
**Streamlit Implementation:**
*   `st.header("Section 18: Conclusion")`
*   `st.markdown("This lab has provided a hands-on experience...")`
*   Bullet points will be rendered using `st.markdown`.

---

### Section: References
**Notebook Content:**
```markdown
### Section 19: References

#### References
[1] "Unit 1: Principles of AI Risk and Assurance," including "Model Risk Management" and "Risk Taxonomy," [Provided Document, Page 1]. These sections introduce the need for documentation and the classification of AI risks.
[2] "Unit 2: Large Language Models and Agentic Architectures," [Provided Document, Page 2]. General context for AI models.
[3] "Evidence Artifacts," [Provided Document, Page 2]. This section specifically describes model cards, data cards, and risk registers as evidence.
[4] "SR 11-7 $\leftrightarrow$ NIST AI RMF," [Provided Document, Page 1]. Mentions SR 11-7's call for thorough documentation.
[5] Python Software Foundation. Python Language Reference, version 3.x. Available at python.org.
[6] The Pandas Development Team. Pandas: a Python Data Analysis Library. Available at pandas.pydata.org.
[7] Hunter, J. D. (2007). Matplotlib: A 2D Graphics Environment. Computing in Science & Engineering, 9(3), 90-95.
[8] Waskom, M. L. (2021). Seaborn: statistical data visualization. Journal of Open Source Software, 6(60), 3021.
[9] Pedregosa et al., (2011). Scikit-learn: Machine Learning in Python. JMLR 12, pp. 2825-2830.
[10] The Project Jupyter Team. ipywidgets: Interactive HTML Widgets for Jupyter. Available at ipywidgets.readthedocs.io.
```
**Streamlit Implementation:**
*   `st.header("Section 19: References")`
*   `st.subheader("References")`
*   All references will be rendered using `st.markdown`. The LaTeX for SR 11-7 $\leftrightarrow$ NIST AI RMF will be `st.markdown("SR 11-7 $ \\leftrightarrow $ NIST AI RMF ...")`.

