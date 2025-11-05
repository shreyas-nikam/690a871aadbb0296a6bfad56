id: 690a871aadbb0296a6bfad56_user_guide
summary: AI Analyst Assistant for Model Governance User Guide
feedback link: https://docs.google.com/forms/d/e/1FAIpQLSfWkOK-in_bMMoHSZfcIvAeO58PAH9wrDqcxnJABHaxiDqhSA/viewform?usp=sf_link
environments: Web
status: Published
# QuLab: Understanding AI Assurance Artifacts with an Interactive Streamlit App

## 1. Introduction to QuLab and AI Assurance Artifacts
Duration: 00:05:00

Welcome to QuLab! This interactive Streamlit application is designed to demystify the crucial world of **AI assurance artifacts**. In an era where AI is rapidly integrating into various aspects of our lives and businesses, ensuring its safety, fairness, transparency, and accountability is paramount. This application provides a hands-on simulation to understand and generate three essential documents:

*   **Model Cards**: These act like nutrition labels for AI models, detailing their purpose, performance, limitations, and ethical considerations. They are vital for promoting transparency and helping users understand what a model does and doesn't do.
*   **Data Cards**: Just as important as model documentation, Data Cards provide a comprehensive overview of the dataset used to train an AI model. They describe data provenance, collection methods, characteristics, and any identified biases or privacy concerns, ensuring data quality and ethical sourcing.
*   **Risk Registers**: A living document that identifies, assesses, and tracks potential risks associated with an AI system throughout its lifecycle. It's a proactive tool for managing everything from data quality issues to algorithmic bias and operational failures.

<aside class="positive">
<b>Why are these artifacts important?</b> These documents serve as crucial "evidence artifacts" for AI assurance. They align with regulatory expectations, such as those highlighted in **SR 11-7 [4]** and the **NIST AI RMF [1, 4]**, by providing thorough documentation that promotes transparency, auditability, and responsible AI practices. The business goal is to establish a robust framework for documenting AI systems, even in early development or when sensitive data isn't available for direct demonstration.
</aside>

**What You Will Learn:**

*   The core functions and components of Model Cards, Data Cards, and Risk Registers.
*   How these documents serve as essential evidence for AI assurance.
*   To interactively define AI model parameters and data characteristics.
*   To explore synthetic data generation, validation, and simulated model predictions.
*   To visualize data relationships and model performance.
*   To identify and document AI risks, including categories, impact, and mitigation strategies.
*   To interpret generated artifacts in an interactive, user-friendly interface.

This codelab will guide you step-by-step through the application, focusing on the concepts and functionalities without delving into the underlying code.

## 2. Navigating the Application and Defining AI Model Parameters
Duration: 00:07:00

Upon launching the application, you'll see a sidebar on the left. This sidebar is your main navigation tool.

1.  **Navigation:** Locate the `Navigation` dropdown in the sidebar. Ensure you are on the **"Lab Overview & Model/Data Definition"** page. This page contains the initial setup for our synthetic AI scenario.

2.  **Environment Setup & Methodology Overview (Sections 2-4):**
    *   The first few sections provide an overview of the application's **Environment Setup** (Section 2), explaining that we use Python libraries for data generation, manipulation, and visualization.
    *   **Data/Inputs Overview** (Section 3) explains that the application uses **synthetic data** to simulate a real-world scenario without sensitive information. Our synthetic data will have numeric, categorical, and timestamp features, plus a binary target variable for a classification task.
    *   **Methodology Overview** (Section 4) outlines the eight-step process the application follows, from defining model/data characteristics to visualizing artifacts. This section also introduces key mathematical formulas for Logistic Regression probability and classification performance metrics like Accuracy, Precision, Recall, and F1-Score, along with their business rationale. These formulas are crucial for understanding how model performance is evaluated.

    $$ P(Y=1|X) = \frac{1}{1 + e^{-(\beta_0 + \sum_{i=1}^k \beta_i X_i)}} $$
    *This formula, for example, represents the probability a Logistic Regression model assigns to the positive class based on input features.*

3.  **Defining Synthetic AI Model Parameters (Section 5):**
    Scroll down to **"Section 3: Defining Synthetic AI Model Parameters."**
    This section is where you interactively define the characteristics of our simulated AI model, which will form the basis of our Model Card.

    <aside class="positive">
    A <b>Model Card</b> provides a structured overview of an AI model's purpose, characteristics, and performance. It's a foundational document for promoting transparency and accountability.
    </aside>

    *   **Model Name:** Give your synthetic AI model a descriptive name (e.g., "Synthetic AI Analyst Assistant").
    *   **Purpose:** Describe the primary objective of this model. For instance, "To classify synthetic customer data for analytical insights." This helps stakeholders understand the model's intended use.
    *   **Model Type:** Select the type of AI task (e.g., "Classification").
    *   **Min F1-score:** This is a key performance threshold. Enter a value (e.g., 0.75). If the simulated model's F1-score falls below this, it might trigger a review.
    *   **Limitations:** Document any known weaknesses or scenarios where the model might not perform well (e.g., "Performance may degrade on highly imbalanced datasets.").
    *   **Usage Notes:** Provide guidance on who should use the model and for what purposes (e.g., "Intended for internal analytical use by data science teams.").

    The values you input here are saved in the application's session state and will be used to generate the Model Card later. Once you interact with the inputs, the application will display a `st.success` message indicating that the parameters have been captured.

## 3. Defining Synthetic Data Characteristics
Duration: 00:05:00

Continuing on the **"Lab Overview & Model/Data Definition"** page, scroll down to **"Section 4: Defining Synthetic Data Characteristics."**

This section is dedicated to describing the dataset that our synthetic AI model will use. The information gathered here will populate our Data Card.

<aside class="positive">
A <b>Data Card</b> is crucial for data governance. It documents the dataset's characteristics, how it was collected, and any identified biases or privacy concerns, ensuring responsible AI development.
</aside>

*   **Dataset Name:** Provide a name for your synthetic dataset (e.g., "Synthetic Customer Data").
*   **Num Samples:** Specify the number of data points (rows) to generate. A default of `1000` is usually a good start.
*   **Num Features (for classification):** This defines the number of numeric features for the classification task. A default of `5` is provided.
*   **Num Categorical Features (added later):** This sets the number of non-numeric features. A default of `2` is provided.
*   **Data Provenance:** Explain where this synthetic data originates (e.g., "Generated by internal Python script for simulation.").
*   **Collection Method:** Describe how the data was hypothetically collected (e.g., "Simulated algorithmic collection based on predefined rules.").
*   **Identified Biases:** Document any known or simulated biases within the data (e.g., "Potential class imbalance and simulated feature correlations.").
*   **Privacy Notes:** Crucial for any dataset, especially in real-world scenarios. Here, state that "All data is synthetic and contains no personal identifiable information."

Like the model parameters, these data characteristics are saved, and you'll see a `st.success` message once your inputs are captured.

## 4. Generating Synthetic Data and Simulating Model Predictions
Duration: 00:03:00

Still on the **"Lab Overview & Model/Data Definition"** page, proceed to **"Section 5: Synthetic Data Generation and Model Simulation."**

This is the interactive part where we create our dataset and simulate an AI model's operation.

1.  **Click the Button:** Locate and click the "**Generate Synthetic Data & Simulate Model**" button.

    <aside class="negative">
    If you set 'Num Samples' or 'Num Features' to 0 in the previous step, you might receive a warning. Please ensure these values are greater than 0 for successful data generation and model simulation.
    </aside>

2.  **What Happens Next?**
    *   The application will display a spinner, indicating that it's "Generating synthetic data and simulating model predictions..."
    *   **Synthetic Data Generation:** Based on the characteristics you defined (number of samples, numeric features, categorical features, timestamp), a dataset is created. This data mimics real-world data but is entirely artificial.
    *   **Model Simulation:** A simple classification model (Logistic Regression) is then "trained" and "predicts" labels on this synthetic data. This gives us 'predicted_label' and 'prediction_score' columns, which are essential for evaluating the model later. Logistic Regression is chosen because it's a common classification algorithm that outputs probabilities.
    *   Once complete, a `st.success` message confirms the process, and a preview of the generated DataFrame's first few rows will be displayed. This preview shows your synthetic features, a `true_label`, a `predicted_label`, and `prediction_score`.

You have now successfully simulated an AI system's data and predictions, setting the stage for generating our assurance artifacts!

## 5. Data Validation and Exploration
Duration: 00:08:00

Now that we have our synthetic data and simulated predictions, let's switch gears to understand and validate this data.

1.  **Navigate to "Data & Model Insights":** In the sidebar, select the **"Data & Model Insights"** page from the Navigation dropdown.

2.  **Data Validation and Exploration (Section 6):**
    Scroll to **"Section 6: Data Validation and Exploration."**
    This step highlights the importance of data validation in any AI project. Before relying on data for training or evaluation, it's crucial to check for quality and completeness.

    *   Expand the "**Perform Data Validation**" section.
    *   Here, you'll see:
        *   **Data Overview:** Basic information like the number of rows and columns, and a preview of the DataFrame.
        *   **Missing Values:** A check for any missing data. In our synthetic data, ideally, there should be none, but in real-world scenarios, this is a critical check.
        *   **Data Types:** The detected data types for each column, ensuring they are as expected (e.g., numeric features are numbers, timestamp is a datetime object).
        *   **Critical Field Check:** Confirms that essential columns like `true_label`, `predicted_label`, and `timestamp` are present.

    <aside class="info">
    Data validation helps identify issues like missing values or incorrect data types that could impact model performance or lead to biased outcomes.
    </aside>

3.  **Core Visual: Relationship Plot (Section 7):**
    Move to **"Section 7: Core Visual: Relationship Plot."**
    Understanding how different features in your data relate to each other and to the target variable is key to gaining insights and detecting potential patterns or issues.

    *   **Select Features:** Use the dropdown menus to choose two numeric features from your dataset for the X and Y axes.
    *   **Observe the Plot:** An interactive scatter plot will be generated. The points are colored by the `true_label` (0 or 1), allowing you to visually inspect if there are clear separations or clusters based on the true outcome. This helps in understanding the data's structure and how well different features might distinguish between classes.

4.  **Core Visual: Trend Plot for Simulated Performance (Section 8):**
    Finally, proceed to **"Section 8: Core Visual: Trend Plot for Simulated Performance."**
    Monitoring an AI model's performance over time is crucial for detecting issues like performance degradation or "concept drift" (where the relationship between input data and the target variable changes over time).

    *   **Observe the Plot:** The application generates a line plot showing "Simulated Model Accuracy Over Time." This plot visualizes how the model's accuracy (the percentage of correct predictions) changes over the simulated timestamp. In a real-world scenario, this would be a critical dashboard to detect when a model needs retraining or intervention.

This page provided critical insights into the data and simulated model's behavior through validation and visualizations.

## 6. Generating and Displaying the Model Card
Duration: 00:08:00

It's time to compile our first AI assurance artifact: the Model Card!

1.  **Navigate to "Artifacts & Risk Management":** In the sidebar, select the **"Artifacts & Risk Management"** page from the Navigation dropdown.

2.  **Generating Model Card Content (Section 9):**
    Scroll to **"Section 9: Generating Model Card Content."**
    This section brings together all the model parameters you defined and calculates the simulated model's performance metrics.

    *   **Performance Metrics Explained:** The application will display and calculate several key classification metrics, which are crucial for evaluating how well our simulated model performs:
        *   **Accuracy:** $$ \text{Accuracy} = \frac{\text{Number of Correct Predictions}}{\text{Total Number of Predictions}} $$ Measures the overall correctness of the model.
        *   **Precision:** $$ \text{Precision} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}} $$ Important when the cost of incorrect positive predictions (false positives) is high.
        *   **Recall (Sensitivity):** $$ \text{Recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}} $$ Important when the cost of missing actual positive cases (false negatives) is high.
        *   **F1-Score:** $$ \text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} $$ Provides a balance between precision and recall, especially useful for imbalanced datasets.
        *   **ROC AUC (Area Under the Receiver Operating Characteristic Curve):** Measures the ability of a classifier to distinguish between classes. A higher AUC indicates better discriminatory power.

    <aside class="info">
    These metrics are critical for evaluating whether an AI model meets its intended performance targets and for identifying potential issues like bias or poor generalization.
    </aside>

    *   The application automatically generates the Model Card content based on the session state. You'll see a `st.success` message indicating it's done.

3.  **Displaying the Model Card (Interactive Table) (Section 10):**
    Move to **"Section 10: Displaying the Model Card (Interactive Table)."**
    This section presents your complete Model Card in an easy-to-read, interactive table format.

    *   **Review the Table:** The table summarizes all the model attributes you defined (name, purpose, limitations, etc.) and displays the calculated performance metrics. This provides a clear, comprehensive overview for any stakeholder to understand the model at a glance.

You've successfully created and reviewed a Model Card for your synthetic AI model!

## 7. Generating and Displaying the Data Card
Duration: 00:06:00

Next up is compiling the Data Card, an equally important artifact for AI assurance.

1.  **Generating Data Card Content (Section 11):**
    On the **"Artifacts & Risk Management"** page, scroll to **"Section 11: Generating Data Card Content."**
    This process compiles all the dataset characteristics you defined earlier and adds detailed statistics about each feature present in the synthetic data.

    <aside class="info">
    A <b>Data Card</b> enhances transparency about the data, aiding in data quality assessment, bias detection, and overall data governance.
    </aside>

    *   The application automatically gathers all the data characteristics and computes statistics (like data type, missing counts, unique values, mean, standard deviation, top values for categorical features) for each column in your generated synthetic dataset.
    *   A `st.success` message will confirm that the Data Card content has been generated.

2.  **Displaying the Data Card (Interactive Table) (Section 12):**
    Move to **"Section 12: Displaying the Data Card (Interactive Table)."**
    Here, the Data Card is presented in two interactive tables.

    *   **Synthetic Dataset Data Card (Overview):** This first table provides a high-level summary of your dataset, including the name, number of samples, number of features, data provenance, collection method, identified biases, and privacy notes—all based on your earlier inputs.
    *   **Detailed Feature Statistics:** The second table provides a granular view of each feature. For every column in your synthetic data, you'll see its data type, how many missing values it has (if any), the number of unique values, and relevant descriptive statistics (e.g., mean, standard deviation for numeric features, or top values and their frequencies for categorical features).

This detailed Data Card is crucial for understanding the dataset's composition and potential issues.

## 8. Identifying AI Risks and Building the Risk Register
Duration: 00:07:00

Now we'll move to identifying and managing risks associated with our AI system, culminating in the creation of a Risk Register.

1.  **User Input: Identifying AI Risks for the Risk Register (Section 13):**
    On the **"Artifacts & Risk Management"** page, scroll to **"Section 13: User Input: Identifying AI Risks for the Risk Register."**
    A Risk Register is a critical tool for proactive risk management. It captures, assesses, and tracks potential risks throughout the AI system's lifecycle.

    *   **Pre-filled Risks:** The application initially pre-fills a few simulated risks (e.g., 'Data Quality', 'Algorithmic Bias') to demonstrate the concept.
    *   **Add New AI Risk:** Use the provided form to add your own synthetic risks.
        *   **Category:** Select a risk category (e.g., 'Data Quality', 'Algorithmic Bias', 'Privacy/Security').
        *   **Description:** Enter a detailed description of the risk.
        *   **Impact Rating:** Rate the potential impact (Low, Medium, High).
        *   **Mitigation Strategies:** Outline the steps to reduce or eliminate this risk.
    *   **Click "Add AI Risk (Simulated)":** After filling in the details, click this button to add your risk to the temporary list.
    *   **Review Current Risks:** The "Currently added risks" section will update to show all risks, including any you've just added.

2.  **Generating Risk Register Content (Section 14):**
    Proceed to **"Section 14: Generating Risk Register Content."**
    Once you have identified a set of risks, this step formally compiles them into a structured tabular format, making it easier to review and manage.

    *   The application converts all the risks stored in the session state (including the pre-filled ones and any you added) into a pandas DataFrame, which is suitable for display and analysis.
    *   A `st.success` message will confirm the compilation, and a preview of the Risk Register DataFrame will be shown.

You now have a structured list of potential AI risks!

## 9. Visualizing and Displaying the Risk Register
Duration: 00:05:00

With our Risk Register compiled, let's visualize and review its contents.

1.  **Core Visual: Risk Register Aggregated Comparison (Bar Chart) (Section 15):**
    On the **"Artifacts & Risk Management"** page, scroll to **"Section 15: Core Visual: Risk Register Aggregated Comparison (Bar Chart)."**
    Visualizing risks helps in quickly identifying high-priority areas and understanding the distribution of risks across different categories and impact levels.

    *   **Observe the Bar Chart:** This interactive bar chart aggregates the identified risks. Each bar represents a risk category, and segments within the bar (or separate bars in a group) show the count of risks for different impact ratings (Low, Medium, High) within that category. This allows for a quick assessment of where the most significant risks lie.

2.  **Displaying the Risk Register (Interactive Table) (Section 16):**
    Move to **"Section 16: Displaying the Risk Register (Interactive Table)."**
    To provide a comprehensive, detailed view of each identified risk, the complete AI System Risk Register is displayed here.

    *   **Review the Table:** The interactive table lists every risk, including its ID, Category, Description, Impact Rating, Mitigation Strategy, and Status (`Open` by default in this simulation). This provides a single source of truth for all documented risks, facilitating detailed review and tracking.

You've successfully created, visualized, and reviewed your AI System Risk Register.

## 10. Understanding AI Assurance Concepts and Conclusion
Duration: 00:06:00

We've now gone through the practical steps of generating all three core AI assurance artifacts. Let's conclude by tying these practical exercises back to the broader concepts of AI assurance and responsible AI.

1.  **Discussion: Connecting Artifacts to AI Assurance (Section 17):**
    On the **"Artifacts & Risk Management"** page, scroll to **"Section 17: Discussion: Connecting Artifacts to AI Assurance."**
    This section explains how the Model Card, Data Card, and Risk Register are not just standalone documents but form a powerful suite of "Evidence Artifacts" for AI assurance [3].

    *   **Model Card's Role:** It provides transparency, enabling "effective challenge" [1] by stakeholders who can understand the model's behavior, intended use, and limitations.
    *   **Data Card's Role:** It ensures accountability regarding data provenance, characteristics, and potential biases, directly addressing the "data" dimension of AI risk [1]. This helps manage data quality, privacy, and fairness.
    *   **Risk Register's Role:** It supports proactive risk management and compliance with frameworks like SR 11-7 and NIST AI RMF [1, 4] by explicitly mapping hazards to mitigation strategies.

    <aside class="positive">
    These structured documents significantly improve auditability, foster trust, and promote responsible development of AI systems. They align with the NIST AI RMF functions (Govern, Map, Measure, Manage), providing a systematic approach to AI governance.
    </aside>

2.  **Conclusion (Section 18):**
    Scroll to **"Section 18: Conclusion."**
    This section summarizes the key takeaways from the lab:

    *   These artifacts make AI models and data transparent, building understanding and trust.
    *   They facilitate auditability and stakeholder review.
    *   They provide essential evidence for robust AI risk management.

    The ability to produce and maintain such artifacts is fundamental to building trustworthy and responsible AI systems, moving beyond abstract principles to concrete, actionable evidence for safe and ethical AI deployment.

3.  **References (Section 19):**
    Finally, **"Section 19: References"** lists the sources that inspired and inform the concepts presented in this QuLab application, providing further reading for those interested in delving deeper into AI assurance and risk management.

Congratulations! You have successfully completed the QuLab codelab, gaining a comprehensive understanding of AI Model Cards, Data Cards, and Risk Registers, and their crucial role in building responsible and trustworthy AI systems.
