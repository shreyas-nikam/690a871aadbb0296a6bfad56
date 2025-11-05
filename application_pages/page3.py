
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import plotly.express as px
import plotly.graph_objects as go

# Helper function to display dataframes (defined in app.py but repeated here for self-containment for testing purposes if needed)
def display_interactive_dataframe(dataframe, title):
    st.subheader(title)
    st.dataframe(dataframe)

# Define global risk categories and impact ratings for consistency (repeated here for self-containment)
RISK_CATEGORIES = ['Data Quality', 'Algorithmic Bias', 'Hallucination', 'Integration Flaws', 'Human Over-reliance', 'Governance', 'Privacy/Security']
IMPACT_RATINGS = ['Low', 'Medium', 'High']


@st.cache_data
def calculate_model_metrics(true_labels, predicted_labels, prediction_scores):
    if true_labels.empty or predicted_labels.empty or len(true_labels) == 0:
        return {
            'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0,
            'f1_score': 0.0, 'roc_auc': 0.0
        }

    # Ensure inputs are numpy arrays or lists for scikit-learn functions
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
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc
    }
    return metrics

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

def run_model_card_generation():
    st.header("Section 9: Generating Model Card Content")
    st.markdown(r"""Now we will compile all the relevant information for our Model Card. A Model Card provides a structured overview of an AI model's purpose, characteristics, performance, and ethical considerations. It serves as a crucial document for transparency, accountability, and responsible deployment. This section calculates key performance metrics and integrates them into the Model Card.""")
    st.markdown("We will calculate common classification metrics:")
    st.markdown(r"*   **Accuracy**: $$ \text{Accuracy} = \frac{\text{Number of Correct Predictions}}{\text{Total Number of Predictions}} $$")
    st.markdown(r"*   **Precision**: $$ \text{Precision} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}} $$")
    st.markdown(r"*   **Recall (Sensitivity)**: $$ \text{Recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}} $$")
    st.markdown(r"*   **F1-Score**: The harmonic mean of Precision and Recall. $$ \text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} $$")
    st.markdown(r"*   **ROC AUC**: The Area Under the Receiver Operating Characteristic Curve. It measures the ability of a classifier to distinguish between classes. A higher AUC indicates a better model performance at distinguishing between positive and negative classes. ")

    if not st.session_state.df_synthetic.empty and st.session_state.synthetic_model_parameters:
        if 'predicted_label' not in st.session_state.df_synthetic.columns or 'prediction_score' not in st.session_state.df_synthetic.columns:
            st.warning("Model predictions or scores are missing in the synthetic DataFrame. Please ensure model simulation was run successfully on the 'Lab Overview & Model/Data Definition' page.")
            return

        st.session_state.model_card = generate_model_card_content(
            st.session_state.synthetic_model_parameters,
            st.session_state.df_synthetic['true_label'],
            st.session_state.df_synthetic['predicted_label'],
            st.session_state.df_synthetic['prediction_score']
        )
        st.success("Model Card content generated successfully.")
    else:
        st.warning("Cannot generate Model Card: Synthetic DataFrame is empty or model parameters are not defined. Please generate data and define model parameters first on the 'Lab Overview & Model/Data Definition' page.")

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
        display_interactive_dataframe(model_card_df, "Synthetic AI Model Card")
        st.success("Interactive Model Card displayed.")
    else:
        st.warning("Model Card is empty. Please generate Model Card content first.")

@st.cache_data
def generate_data_card_content(data_params, dataframe):
    if dataframe.empty:
        return {
            'dataset_name': data_params.get('dataset_name', 'N/A'),
            'n_samples': 0,
            'n_features': 0,
            'n_categorical_features': 0,
            'data_provenance': data_params.get('data_provenance', 'N/A'),
            'collection_method': data_params.get('collection_method', 'N/A'),
            'identified_biases_description': data_params.get('identified_biases_description', 'N/A'),
            'privacy_notes': data_params.get('privacy_notes', 'N/A'),
            'feature_statistics': {}
        }

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

def run_data_card_generation():
    st.header("Section 11: Generating Data Card Content")
    st.markdown(r"""Similar to the Model Card, a Data Card provides critical documentation about the dataset used to train and evaluate the AI model. This includes information on data provenance, collection methods, and any identified biases, contributing significantly to data governance and responsible AI.""")
    if not st.session_state.df_synthetic.empty and st.session_state.synthetic_data_characteristics:
        st.session_state.data_card = generate_data_card_content(st.session_state.synthetic_data_characteristics, st.session_state.df_synthetic)
        st.success("Data Card content generated successfully.")
    else:
        st.warning("Cannot generate Data Card: Synthetic DataFrame is empty or data characteristics are not defined. Please generate data and define data characteristics first on the 'Lab Overview & Model/Data Definition' page.")

def display_data_card():
    st.header("Section 12: Displaying the Data Card (Interactive Table)")
    st.markdown(r"""The generated Data Card content is now ready for display. This section provides an interactive overview of the dataset and detailed statistics for each feature, enhancing transparency and aiding in data quality assessment.""")
    if st.session_state.data_card:
        data_card_flat = {
            'Dataset Name': st.session_state.data_card.get('dataset_name', 'N/A'),
            'Number of Samples': st.session_state.data_card.get('n_samples', 'N/A'),
            'Number of Features': st.session_state.data_card.get('n_features', 'N/A'),
            'Number of Categorical Features': st.session_state.data_card.get('n_categorical_features', 'N/A'),
            'Data Provenance': st.session_state.data_card.get('data_provenance', 'N/A'),
            'Collection Method': st.session_state.data_card.get('collection_method', 'N/A'),
            'Identified Biases Description': st.session_state.data_card.get('identified_biases_description', 'N/A'),
            'Privacy Notes': st.session_state.data_card.get('privacy_notes', 'N/A')
        }
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

        display_interactive_dataframe(data_card_df_top, "Synthetic Dataset Data Card (Overview)")
        st.markdown("\nDetailed Feature Statistics:")
        display_interactive_dataframe(data_card_df_features, "Synthetic Dataset Data Card (Feature Statistics)")
        st.success("Interactive Data Card displayed.")
    else:
        st.warning("Data Card is empty. Please generate Data Card content first.")

def create_risk_entry(risk_id, category, description, impact_rating, mitigation_strategy):
    return {
        'Risk ID': risk_id,
        'Category': category,
        'Description': description,
        'Impact Rating': impact_rating,
        'Mitigation Strategy': mitigation_strategy,
        'Status': 'Open'
    }

def add_risk_to_register(current_risks_list, category, description, impact, mitigation):
    risk_id = len(current_risks_list) + 1
    new_risk = create_risk_entry(risk_id, category, description, impact, mitigation)
    current_risks_list.append(new_risk)


def run_risk_input_form():
    st.header("Section 13: User Input: Identifying AI Risks for the Risk Register")
    st.markdown(r"A Risk Register is a living document that captures, assesses, and tracks potential risks associated with an AI system. It is a critical tool for proactive risk management and compliance. Use the form below to add synthetic risks to our register.")

    if not st.session_state.risk_register_entries:
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
            category = st.selectbox(
                "Category:",
                options=RISK_CATEGORIES,
                key="risk_category_input",
                help="Select the category that best describes this risk."
            )
            impact = st.selectbox(
                "Impact Rating:",
                options=IMPACT_RATINGS,
                key="risk_impact_input",
                help="Rate the potential impact of this risk (Low, Medium, High)."
            )
        with col2:
            description = st.text_area(
                "Description:",
                placeholder='Enter a detailed description of the AI risk...',
                height=100,
                key="risk_description_input",
                help="Provide a clear and concise description of the identified risk."
            )
            mitigation = st.text_area(
                "Mitigation Strategies:",
                placeholder='Enter strategies to mitigate the identified risk...',
                height=100,
                key="risk_mitigation_input",
                help="Outline the steps or strategies to reduce or eliminate this risk."
            )

        if st.button("Add AI Risk (Simulated)", help="Click to add this risk to the register."):
            if description and mitigation:
                add_risk_to_register(st.session_state.risk_register_entries, category, description, impact, mitigation)
                st.success(f"Risk '{category}' (Impact: {impact}) added to register.")
            else:
                st.warning("Please fill in both the Description and Mitigation Strategies before adding a risk.")

    st.markdown("Currently added risks (from session state):")
    if st.session_state.risk_register_entries:
        temp_df = pd.DataFrame(st.session_state.risk_register_entries)
        temp_df['Risk ID'] = range(1, len(temp_df) + 1)
        st.dataframe(temp_df)
    else:
        st.info("No risks added yet. Add new risks using the form above.")

@st.cache_data
def compile_risk_register(risk_entries_list):
    if not risk_entries_list:
        return pd.DataFrame(columns=['Risk ID', 'Category', 'Description', 'Impact Rating', 'Mitigation Strategy', 'Status'])
    df = pd.DataFrame(risk_entries_list)
    return df

def run_risk_register_compilation():
    st.header("Section 14: Generating Risk Register Content")
    st.markdown(r"""With the identified risks, we can now formally compile the Risk Register. This process converts the individual risk entries into a structured tabular format, making it easier to review, analyze, and manage all documented risks associated with the AI system.""")
    if st.session_state.risk_register_entries:
        st.session_state.risk_register_df = compile_risk_register(st.session_state.risk_register_entries)
        st.success("Risk Register DataFrame compiled.")
        st.dataframe(st.session_state.risk_register_df.head())
    else:
        st.warning("Cannot compile Risk Register: No risks have been added yet. Please add risks first on this page (Section 13).")

@st.cache_data
def aggregate_risks_by_category_and_impact(risk_register_dataframe):
    if risk_register_dataframe.empty:
        return pd.DataFrame(columns=['Category', 'Impact Rating', 'Count of Risks'])
    
    if 'Category' not in risk_register_dataframe.columns or 'Impact Rating' not in risk_register_dataframe.columns:
        st.error("Missing 'Category' or 'Impact Rating' columns in the risk register for aggregation.")
        return pd.DataFrame(columns=['Category', 'Impact Rating', 'Count of Risks'])

    aggregated_df = risk_register_dataframe.groupby(['Category', 'Impact Rating']).size().reset_index(name='Count of Risks')
    
    all_categories = pd.CategoricalDtype(RISK_CATEGORIES, ordered=True)
    all_impacts = pd.CategoricalDtype(IMPACT_RATINGS, ordered=True)

    aggregated_df['Category'] = aggregated_df['Category'].astype(all_categories)
    aggregated_df['Impact Rating'] = aggregated_df['Impact Rating'].astype(all_impacts)
    
    full_index = pd.MultiIndex.from_product([RISK_CATEGORIES, IMPACT_RATINGS], names=['Category', 'Impact Rating'])
    aggregated_df = aggregated_df.set_index(['Category', 'Impact Rating']).reindex(full_index, fill_value=0).reset_index()

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

        fig.update_layout(
            title_font_size=16,
            font_size=12,
            xaxis_title_font_size=12,
            yaxis_title_font_size=12,
            legend_title_text='Impact Rating',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="x unified"
        )
        
        fig.update_traces(texttemplate='%{y}', textposition='outside')

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Risk DataFrame is empty, cannot generate risk aggregation plot.")

def run_risk_aggregation_plot():
    st.header("Section 15: Core Visual: Risk Register Aggregated Comparison (Bar Chart)")
    st.markdown(r"""Visualizing the Risk Register helps in quickly identifying high-priority areas and understanding the distribution of risks across different categories and impact levels. This bar chart aggregates the identified risks, providing a clear overview for risk management and strategic planning.""")
    if not st.session_state.risk_register_df.empty:
        plot_risk_aggregation(st.session_state.risk_register_df, 'Category', 'Impact Rating', 'Aggregated AI Risks by Category and Impact')
        st.success("Aggregated AI risks bar chart generated.")
    else:
        st.warning("Cannot generate aggregated risk comparison plot: Risk Register is empty. Please add and compile risks first on this page (Sections 13 & 14).")

def display_risk_register():
    st.header("Section 16: Displaying the Risk Register (Interactive Table)")
    st.markdown(r"""To provide a detailed view of each identified risk, we will display the complete AI System Risk Register. This interactive table allows for easy review of all risks, their descriptions, impact ratings, mitigation strategies, and current status.""")
    if not st.session_state.risk_register_df.empty:
        display_interactive_dataframe(st.session_state.risk_register_df, "AI System Risk Register")
        st.success("Interactive Risk Register table displayed.")
    else:
        st.warning("Cannot display Risk Register: DataFrame is empty. Please add and compile risks first on this page (Sections 13 & 14).")

def run_discussion_section():
    st.header("Section 17: Discussion: Connecting Artifacts to AI Assurance")
    st.markdown(r"""\
The Model Card, Data Card, and Risk Register, when used together, form a powerful set of "Evidence Artifacts" for AI assurance [3]. These documents are not merely bureaucratic exercises but fundamental tools for building and maintaining trustworthy and responsible AI systems.

*   The **Model Card** provides transparency into the model's behavior, intended use, and limitations, which is critical for an "effective challenge" [1]. It helps ensure that stakeholders understand what the model does, how it performs, and where its boundaries lie.
*   The **Data Card** ensures accountability regarding data provenance, characteristics, and potential biases, directly addressing the "data" dimension of AI risk [1]. By meticulously documenting the dataset, organizations can better manage data quality, privacy, and fairness issues.
*   The **Risk Register** explicitly maps identified hazards to mitigation strategies, directly supporting proactive risk management and compliance with frameworks like SR 11-7 and NIST AI RMF [1, 4]. It serves as a dynamic record of potential pitfalls and the actions taken to address them.

These structured documents not only improve auditability but also foster trust and responsible development of AI systems by ensuring all key aspects are thoroughly documented and reviewed. They facilitate stakeholder review and can be mapped to NIST AI RMF functions (Govern, Map, Measure, Manage) [4]. Their systematic application ensures a comprehensive approach to AI governance, moving from abstract principles to concrete, actionable evidence.
    """)

def run_conclusion_section():
    st.header("Section 18: Conclusion")
    st.markdown(r"""\
This lab has provided a hands-on experience in generating crucial AI assurance documentation. By creating synthetic Model Cards, Data Cards, and Risk Registers, you've gained practical insight into how these artifacts:\n
*   Make AI models and data transparent, fostering greater understanding and trust among users and stakeholders.
*   Facilitate auditability and stakeholder review, enabling rigorous scrutiny of AI systems and their components.
*   Provide essential evidence for robust AI risk management, ensuring that potential issues are identified, assessed, and mitigated proactively.

The ability to produce and maintain such artifacts is fundamental to building trustworthy and responsible AI systems. These practices are not just about compliance, but about instilling confidence and enabling the safe and ethical deployment of AI technologies in real-world scenarios.
    """)

def run_references_section():
    st.header("Section 19: References")
    st.subheader("References")
    st.markdown(r"""\
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
    """)


def run_page3():
    run_model_card_generation()
    display_model_card()
    run_data_card_generation()
    display_data_card()
    run_risk_input_form()
    run_risk_register_compilation()
    run_risk_aggregation_plot()
    display_risk_register()
    run_discussion_section()
    run_conclusion_section()
    run_references_section()
