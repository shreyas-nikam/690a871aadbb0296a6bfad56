
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Helper function to display dataframes (defined in app.py but repeated here for self-containment for testing purposes if needed)
def display_interactive_dataframe(dataframe, title):
    st.subheader(title)
    st.dataframe(dataframe)

def run_page2():
    st.header("Section 6: Data Validation and Exploration")
    st.markdown("Before proceeding, it's crucial to validate the generated synthetic data to ensure its quality and completeness. This step helps identify any missing values, incorrect data types, or unexpected patterns that could affect model performance or lead to biased outcomes.")

    if not st.session_state.df_synthetic.empty:
        with st.expander("Perform Data Validation", expanded=True):
            critical_fields = [f'feature_{i+1}' for i in range(st.session_state.synthetic_data_characteristics.get('n_features', 0))] + ['true_label', 'predicted_label', 'prediction_score', 'timestamp']
            perform_data_validation(st.session_state.df_synthetic, critical_fields)
        st.success("Synthetic data validated.")
    else:
        st.warning("Cannot perform data validation: Synthetic DataFrame is empty. Please generate data first on the 'Lab Overview & Model/Data Definition' page.")

    st.header("Section 7: Core Visual: Relationship Plot")
    st.markdown("Understanding relationships between different features within the data is essential for gaining insights into the underlying patterns and potential correlations that the AI model might learn. A scatter plot helps visualize these relationships, often revealing clusters or trends related to the target variable.")
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
            st.warning("Not enough numeric features to generate a relationship plot. Please ensure your synthetic data has at least two numeric features.")
    else:
        st.warning("Cannot generate relationship plot: Synthetic DataFrame is empty. Please generate data first on the 'Lab Overview & Model/Data Definition' page.")

    st.header("Section 8: Core Visual: Trend Plot for Simulated Performance")
    st.markdown("Monitoring model performance over time is a critical aspect of AI governance, allowing us to detect performance degradation, concept drift, or other issues that might arise after deployment. A trend plot helps visualize how a key metric, like accuracy, evolves over a simulated timeline.")
    if not st.session_state.df_synthetic.empty and 'true_label' in st.session_state.df_synthetic.columns and 'predicted_label' in st.session_state.df_synthetic.columns and 'timestamp' in st.session_state.df_synthetic.columns:
        df_synthetic_copy = st.session_state.df_synthetic.copy()
        df_synthetic_copy['simulated_accuracy'] = (df_synthetic_copy['true_label'] == df_synthetic_copy['predicted_label']).astype(int)
        daily_performance = df_synthetic_copy.groupby(df_synthetic_copy['timestamp'].dt.date)['simulated_accuracy'].mean().reset_index()
        daily_performance.rename(columns={'timestamp': 'date_key'}, inplace=True)
        daily_performance['date_key'] = pd.to_datetime(daily_performance['date_key'])

        plot_performance_trend(daily_performance, 'date_key', 'simulated_accuracy', 'Simulated Model Accuracy Over Time')
        st.success("Simulated model accuracy trend plot generated.")
    else:
        st.warning("Cannot generate simulated performance trend plot: DataFrame is empty or missing 'true_label'/'predicted_label'/'timestamp'. Please generate data and ensure these columns exist on the 'Lab Overview & Model/Data Definition' page.")


# Functions for Page 2

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
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey', title_font_size=12, range=[0,1.05]) # Ensure y-axis for accuracy is 0-1
        fig.update_traces(mode='lines+markers', marker=dict(size=6, line=dict(width=0)))

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("DataFrame is empty, cannot generate performance trend plot.")
