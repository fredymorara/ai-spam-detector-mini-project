# AI Mini-Project - Spam Email Detector

## Group Members
*   Fredrick M. Morara - INTE/MG/2814/09/22 - Group Leader
*   Cleophas Kiama - INTE/MG/2834/09/22
*   Trevor Maosa - INTE/MG/2907/09/22
*   Morris Mwangi - INTE/MG/3110/09/22
*   Noelah Amoni - INTE/N/1988/09/23

## Brief Description
This project aims to develop a machine learning model capable of accurately classifying email/SMS messages as either "spam" (unsolicited, often malicious or commercial messages) or "ham" (legitimate messages). Spam messages are a persistent nuisance, wasting time, consuming resources, and posing security risks. This project tackles this problem by building an automated spam detection system.

## Dataset Used
*   **Source:** [SMS Spam Collection Dataset from Kaggle](https://www.kaggle.com/uciml/sms-spam-collection-dataset)
*   **Description:** The dataset contains 5,572 SMS messages in English, tagged according to being "ham" (legitimate) or "spam". It consists of two columns: `v1` (the label: ham/spam) and `v2` (the raw text of the SMS message). This project uses the `spam.csv` file from this collection.

## Methodology Overview
The project follows a standard machine learning pipeline:
1.  **Data Acquisition:** Loading the `spam.csv` dataset.
2.  **Data Cleaning & Preprocessing:**
    *   Handling missing values and duplicates.
    *   Text normalization: Lowercasing.
    *   Noise removal: Removing punctuation.
    *   Tokenization: Splitting text into individual words.
    *   Stopword removal: Eliminating common, non-informative words.
    *   Lemmatization: Reducing words to their base/dictionary form.
3.  **Exploratory Data Analysis (EDA):**
    *   Analyzing message length distributions for spam vs. ham.
    *   Identifying and visualizing the most frequent words in spam and ham messages using bar charts and word clouds.
4.  **Feature Engineering:**
    *   Converting the preprocessed text data into numerical vectors using **TF-IDF (Term Frequency-Inverse Document Frequency)**. A vocabulary of the top 5000 features (words) was considered.
5.  **Model Building:**
    *   Training two common classification models:
        *   **Multinomial Naive Bayes (MNB)**
        *   **Logistic Regression (LR)**
    *   Utilizing `scikit-learn` Pipelines to streamline the process of TF-IDF vectorization and model training.
6.  **Model Evaluation:**
    *   Assessing model performance on an unseen test set using metrics like Accuracy, Precision (for spam), Recall (for spam), F1-score (for spam), and ROC AUC.
    *   Comparing the performance of MNB and LR.
7.  **Results Interpretation:**
    *   Analyzing the coefficients of the Logistic Regression model to identify words that are strong indicators of spam or ham.
8.  **Deployment (Demonstration):**
    *   Saving the trained Logistic Regression model pipeline using `pickle`.
    *   Providing a demonstration of the model's prediction capabilities within the Colab notebook and (optionally) via a local Streamlit web application.

## Key Results
The **Logistic Regression** model was selected as the preferred model due to its [**mention reason, e.g., slightly better F1-score for spam and interpretability / or state which model was best and why**].
Its performance on the test set was as follows:
*   **Accuracy:** [**Fill in your Accuracy, e.g., 0.9750**]
*   **Precision (Spam):** [**Fill in your Precision for Spam, e.g., 0.9800**]
*   **Recall (Spam):** [**Fill in your Recall for Spam, e.g., 0.8500**]
*   **F1-score (Spam):** [**Fill in your F1-score for Spam, e.g., 0.9100**]
*   **ROC AUC:** [**Fill in your ROC AUC, e.g., 0.9850**]

The model demonstrated a strong ability to distinguish spam from ham messages, with a high precision for spam, which is crucial for minimizing the misclassification of legitimate messages.

## How to Run

### 1. Google Colab Notebook
The primary development and experimentation were done in a Google Colab notebook (`[Your_Notebook_Name].ipynb`).
1.  **Prerequisites:** A Google account.
2.  **Setup:**
    *   Open the `[Your_Notebook_Name].ipynb` file in Google Colab.
    *   Download the `spam.csv` dataset from [Kaggle](https://www.kaggle.com/uciml/sms-spam-collection-dataset).
    *   In the Colab environment, upload the `spam.csv` file to the session storage (usually via the file browser pane on the left).
3.  **Execution:**
    *   Run the notebook cells sequentially from top to bottom.
    *   The notebook covers all stages from data loading to model evaluation and a simple prediction demo.

### 2. (Optional) Local Streamlit Web Application (`app.py`)
A simple web application (`app.py`) is provided for interactive spam detection.
1.  **Prerequisites:**
    *   Python 3.7+ installed.
    *   Git (for cloning the repository).
2.  **Setup:**
    *   Clone this GitHub repository to your local machine:
        ```bash
        git clone [URL_OF_YOUR_GITHUB_REPO]
        cd [NAME_OF_YOUR_REPO_FOLDER]
        ```
    *   Create a virtual environment (recommended):
        ```bash
        python -m venv venv
        source venv/bin/activate  # On Windows: venv\Scripts\activate
        ```
    *   Install the required Python packages:
        ```bash
        pip install -r requirements.txt
        ```
    *   Ensure the trained model file (`spam_detector_lr_pipeline.pkl`) is in the same directory as `app.py`.
3.  **Running the App:**
    *   Navigate to the project directory in your terminal.
    *   Run the Streamlit application:
        ```bash
        streamlit run app.py
        ```
    *   The application should open in your web browser. You can then enter text messages to classify them as spam or ham.

## Link to Deployed App (Optional)
*   [**If you deploy it, e.g., on Streamlit Community Cloud, Heroku, etc., put the link here. Otherwise, remove this section or state "Not applicable for this mini-project."**]

---