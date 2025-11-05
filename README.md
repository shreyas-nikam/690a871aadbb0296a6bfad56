# QuLab: AI Assurance Artifact Generator

![QuLab Logo](https://www.quantuniversity.com/assets/img/logo5.jpg)

This repository hosts "QuLab," an interactive Streamlit application designed as a laboratory project to explore and generate essential AI assurance artifacts: **Model Cards**, **Data Cards**, and **Risk Registers**. Through a simulated scenario, users will learn to document AI models, their data, and associated risks, gaining practical experience in promoting transparency, auditability, and responsible AI practices.

The application focuses on the business goal of establishing a robust framework for documenting AI systems, even in early development stages or when sensitive data is not available for direct demonstration. It provides a hands-on experience in creating crucial AI assurance documentation, aligning with regulatory expectations for thorough documentation (e.g., SR 11-7 [4]).

---

## 🚀 Features

This interactive Streamlit application offers the following key functionalities:

*   **Interactive Parameter Definition**: Define parameters for a synthetic AI model and its dataset through intuitive Streamlit widgets.
*   **Synthetic Data Generation**: Generate a customizable synthetic dataset featuring numeric, categorical, and timestamp features, along with a binary target variable.
*   **Model Simulation**: Simulate a simple classification model (Logistic Regression) on the synthetic data to generate predictions and prediction scores.
*   **Data Validation & Exploration**: Perform basic data quality checks and visualize data distributions and relationships using interactive Plotly charts.
*   **Performance Trend Visualization**: Monitor simulated model accuracy over time, helping to understand potential performance degradation or drift.
*   **Model Card Generation**: Automatically compile a comprehensive Model Card detailing the model's purpose, type, limitations, usage, and key performance metrics (Accuracy, Precision, Recall, F1-Score, ROC AUC).
*   **Data Card Generation**: Document the dataset's characteristics, provenance, collection methods, identified biases, privacy notes, and detailed feature statistics in a Data Card.
*   **Interactive Risk Register**: Add and manage AI-related risks in a dynamic Risk Register, including risk categories, descriptions, impact ratings, and mitigation strategies. Includes pre-filled sample risks.
*   **Risk Aggregation Visualization**: Visualize the distribution of identified risks across categories and impact levels using an aggregated bar chart.
*   **Interactive Artifact Display**: All generated Model Cards, Data Cards, and Risk Registers are presented in user-friendly, interactive table formats.
*   **Educational Content**: In-app explanations of key concepts, formulae, and business rationales behind AI assurance artifacts and metrics.

---

## 🛠️ Getting Started

Follow these instructions to get a copy of the project up and running on your local machine.

### Prerequisites

Ensure you have Python installed (version 3.8 or higher is recommended). You can download Python from [python.org](https://www.python.org/downloads/).

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/QuLab-AI-Assurance.git
    cd QuLab-AI-Assurance
    ```
    *(Note: Replace `your-username/QuLab-AI-Assurance` with the actual repository path if this is hosted publicly.)*

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    ```

3.  **Activate the virtual environment:**

    *   **On Windows:**
        ```bash
        .\venv\Scripts\activate
        ```
    *   **On macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```

4.  **Install the required dependencies:**
    Create a `requirements.txt` file in the root directory with the following content:
    ```
    streamlit>=1.30.0
    pandas>=2.1.0
    numpy>=1.26.0
    plotly>=5.18.0
    scikit-learn>=1.3.0
    ```
    Then, install them:
    ```bash
    pip install -r requirements.txt
    ```

---

## 🚀 Usage

To run the Streamlit application:

1.  **Ensure your virtual environment is active** (as described in the Installation section).
2.  **Navigate to the project's root directory** (where `app.py` is located).
3.  **Run the Streamlit application:**

    ```bash
    streamlit run app.py
    ```

    This command will open the application in your default web browser. If it doesn't open automatically, look for a URL (usually `http://localhost:8501`) in your terminal output.

### Application Flow

The application is structured into three main pages, navigable via the sidebar:

1.  **Lab Overview & Model/Data Definition**:
    *   Provides an overview of the lab, data, and methodology.
    *   Allows interactive definition of synthetic AI model parameters and synthetic data characteristics.
    *   Initiate synthetic data generation and model simulation.
2.  **Data & Model Insights**:
    *   Perform data validation on the generated synthetic dataset.
    *   Visualize relationships between features using scatter plots.
    *   Plot the simulated model's accuracy trend over time.
3.  **Artifacts & Risk Management**:
    *   Generates and displays the **Model Card** (including calculated performance metrics).
    *   Generates and displays the **Data Card** (including detailed feature statistics).
    *   Offers an interactive form to add new AI risks to the **Risk Register**.
    *   Visualizes aggregated risks by category and impact.
    *   Displays the complete AI System Risk Register.
    *   Includes a discussion on connecting artifacts to AI assurance, a conclusion, and references.

---

## 📁 Project Structure

The project is organized as follows:

```
QuLab-AI-Assurance/
├── app.py                     # Main Streamlit application entry point
├── requirements.txt           # Python dependencies
├── README.md                  # This README file
└── application_pages/         # Directory for individual application pages
    ├── __init__.py            # Makes 'application_pages' a Python package
    ├── page1.py               # Lab overview, model/data definition, data generation
    ├── page2.py               # Data validation, data/model insights visualizations
    └── page3.py               # Artifact generation (Model/Data Cards), Risk Register, discussion
```

---

## 💻 Technology Stack

*   **Python**: The core programming language.
*   **Streamlit**: For building the interactive web application user interface.
*   **Pandas**: For data manipulation and analysis, especially with DataFrames.
*   **NumPy**: For numerical operations and array processing.
*   **Plotly Express & Plotly Graph Objects**: For creating interactive and publication-quality visualizations.
*   **Scikit-learn**: Used for synthetic data generation (`make_classification`), model training (`LogisticRegression`), and performance metric calculation (`accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `roc_auc_score`).

---

## 👋 Contributing

Contributions are welcome! If you have suggestions for improvements, new features, or bug fixes, please feel free to:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/YourFeature`).
3.  Make your changes and commit them (`git commit -m 'Add some feature'`).
4.  Push to the branch (`git push origin feature/YourFeature`).
5.  Open a Pull Request.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
*(Note: You will need to create a `LICENSE` file if it doesn't exist yet.)*

---

## 📧 Contact

For any questions or inquiries, please reach out:

*   **Quant University:** [https://www.quantuniversity.com](https://www.quantuniversity.com)
*   **Email:** info@quantuniversity.com *(Placeholder; replace with actual contact if needed)*

---

## License

## QuantUniversity License

© QuantUniversity 2025  
This notebook was created for **educational purposes only** and is **not intended for commercial use**.  

- You **may not copy, share, or redistribute** this notebook **without explicit permission** from QuantUniversity.  
- You **may not delete or modify this license cell** without authorization.  
- This notebook was generated using **QuCreate**, an AI-powered assistant.  
- Content generated by AI may contain **hallucinated or incorrect information**. Please **verify before using**.  

All rights reserved. For permissions or commercial licensing, contact: [info@qusandbox.com](mailto:info@qusandbox.com)
