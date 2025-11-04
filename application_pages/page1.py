import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def run_page1():
    st.header("2. Environment Setup (Overview)")
    st.markdown(r"""
    Before diving into the specifics of AI assurance artifacts, we need to ensure our environment is set up. This application utilizes several Python libraries for data generation, manipulation, visualization, and interactive elements.
    """)

    st.header("3. Data/Inputs Overview")
    st.markdown(r"""
    In this application, we will leverage **synthetic data** to simulate a realistic AI development and deployment scenario. This approach allows us to demonstrate the creation of AI assurance artifacts without relying on sensitive or proprietary real-world datasets.

    Our synthetic data will feature:
    *   **Numeric features:** Representing various measurable attributes.
    *   **Categorical features:** Simulating discrete variables.
    *   **A timestamp column:** Enabling the simulation of time-series trends and performance monitoring over time.
    *   **A binary target variable:** For a classification task, which is common in many AI applications.

    This simulated environment is critical for showcasing how Model Cards, Data Cards, and Risk Registers can be generated and utilized to promote transparency, auditability, and responsible AI practices, aligning with regulatory expectations for thorough documentation (e.g., SR 11-7 [4]). The business goal here is to establish a robust framework for documenting AI systems, even in early development stages or when sensitive data is not available for direct demonstration.
    """)

    st.header("4. Methodology Overview")
    st.markdown(r"""
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

    **1. Logistic Regression Probability (for Model Simulation):**
    $$ P(Y=1|X) = \frac{1}{1 + e^{-(\beta_0 + \sum_{i=1}^k \beta_i X_i)}} $$
    Where:
    *   $P(Y=1|X)$ is the probability of the positive class (e.g., a customer being `high-value`).
    *   $X_i$ represents the input features.
    *   $\beta_0$ is the intercept term.
    *   $\beta_i$ are the coefficients (weights) assigned to each feature.

    **Business Rationale:** Understanding how a model arrives at its predictions, especially in classification tasks, is crucial for interpretability and trust. Logistic Regression is a widely used algorithm that provides probabilistic outputs, making it a good candidate for demonstrating model behavior in a simulated environment. The formula shows how a linear combination of features is transformed into a probability using the sigmoid function.

    **2. Classification Performance Metrics (for Model Cards):**

    *   **Accuracy**:
        $$ \text{Accuracy} = \frac{\text{Number of Correct Predictions}}{\text{Total Number of Predictions}} $$
    *   **Precision**:
        $$ \text{Precision} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}} $$
    *   **Recall (Sensitivity)**:
        $$ \text{Recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}} $$
    *   **F1-Score**:
        $$ \text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} $$

    **Business Rationale:** These metrics are critical for evaluating whether an AI model meets its intended performance targets and for identifying potential issues like bias or poor generalization.
    *   **Accuracy** provides a general sense of correctness.
    *   **Precision** is important when the cost of false positives is high (e.g., incorrectly flagging a healthy patient with a disease).
    *   **Recall** is crucial when the cost of false negatives is high (e.g., failing to detect a fraudulent transaction).
    *   **F1-Score** offers a balance between precision and recall, especially useful for imbalanced datasets.
    Understanding these metrics helps stakeholders assess model reliability and suitability for deployment.
    """)

    st.header("Section 3: Defining Synthetic AI Model Parameters")
    define_model_parameters_streamlit()

    st.header("Section 4: Defining Synthetic Data Characteristics")
    define_data_characteristics_streamlit()

    st.header("Section 5: Synthetic Data Generation and Model Simulation")
    run_data_generation_and_simulation()


# Functions for Page 1

def define_model_parameters_streamlit():
    """
    Streamlit-compatible function to capture synthetic AI model parameters.
    """
    with st.expander("Define Synthetic AI Model Parameters", expanded=True):
        st.markdown(r"""A Model Card provides a structured overview of an AI model's purpose, characteristics, performance, and ethical considerations. It serves as a crucial document for transparency, accountability, and responsible deployment. Use the widgets below to specify parameters for our synthetic AI model.""")

        col1, col2 = st.columns(2)
        with col1:
            model_name = st.text_input(
                "Model Name:",
                value=st.session_state.synthetic_model_parameters.get('model_name', "Synthetic AI Analyst Assistant"),
                placeholder="e.g., Customer Sentiment Analyzer",
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

def define_data_characteristics_streamlit():
    """
    Streamlit-compatible function to capture synthetic dataset characteristics.
    """
    with st.expander("Define Synthetic Data Characteristics", expanded=True):
        st.markdown(r"""A Data Card documents the dataset's characteristics, provenance, collection methods, and any identified biases or privacy considerations. This is vital for data governance and ensuring fair and responsible AI development. Use the widgets below to specify parameters for our synthetic dataset.""")

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

@st.cache_data
def generate_synthetic_data(n_samples, n_features, n_classes=2, random_state=42):
    """
    Generates a synthetic dataset with numeric, categorical features, and a timestamp.
    """
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

@st.cache_data
def simulate_model_predictions(features, labels, model_type='Logistic Regression', random_state=42):
    """
    Simulates model training and prediction based on the specified model type.
    """
    if features.empty or labels.empty:
        return pd.Series([], dtype=int), pd.Series([], dtype=float)

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=random_state)

    if model_type == 'Logistic Regression':
        model = LogisticRegression(random_state=random_state, solver='liblinear')
    else:
        # Default or extend with other model types
        model = LogisticRegression(random_state=random_state, solver='liblinear')

    model.fit(X_train, y_train)
    predictions = model.predict(features)
    prediction_scores = model.predict_proba(features)[:, 1] # Probability of the positive class

    return pd.Series(predictions, index=features.index), pd.Series(prediction_scores, index=features.index)

def run_data_generation_and_simulation():
    if st.button("Generate Synthetic Data & Simulate Model"):
        if st.session_state.synthetic_data_characteristics.get('n_samples', 0) == 0:
            st.warning("Number of samples is 0. Dataframe will be empty. Please set 'Num Samples' > 0.")
            st.session_state.df_synthetic = pd.DataFrame()
            return
        if st.session_state.synthetic_data_characteristics.get('n_features', 0) == 0:
            st.warning("Number of features is 0. Data generation might not work as expected. Please set 'Num Features' > 0.")
            # Proceed with 0 features, make_classification will handle it, but it's a user warning.


        with st.spinner("Generating synthetic data and simulating model predictions..."):
            df_synthetic_local = generate_synthetic_data(
                st.session_state.synthetic_data_characteristics['n_samples'],
                st.session_state.synthetic_data_characteristics['n_features'],
                random_state=42
            )

            # Ensure we only select numeric features for the model training
            X = df_synthetic_local.select_dtypes(include=np.number).drop(['true_label'], axis=1, errors='ignore')
            y = df_synthetic_local['true_label']

            if not X.empty and not y.empty:
                # Use the model_type from session state to influence simulation if desired,
                # currently, simulate_model_predictions defaults to Logistic Regression.
                predicted_label, prediction_score = simulate_model_predictions(
                    X, y,
                    model_type=st.session_state.synthetic_model_parameters.get('model_type', 'Classification') # Pass model type
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
