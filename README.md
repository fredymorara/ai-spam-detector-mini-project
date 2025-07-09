# AI Mini-Project: Spam Email & SMS Detector

## Group Members
*   **Fredrick M. Morara** - `INTE/MG/2814/09/22` - *Group Leader*
*   **Cleophas Kiama** - `INTE/MG/2834/09/22`
*   **Trevor Maosa** - `INTE/MG/2907/09/22`
*   **Morris Mwangi** - `INTE/MG/3110/09/22`
*   **Noelah Amoni** - `INTE/N/1988/09/23`

---

## Live Application
A live, interactive version of our best model is deployed on Streamlit Community Cloud.

**➡️ [Access the Live Spam Detector App Here](https://aispamdetector.streamlit.app/)**

This project's code is also mirrored in the [deployment repository](https://github.com/fredymorara/AI-Spam-Detector-App).

---

## 1. Project Overview & Objective

This project tackles the persistent problem of spam messages (unsolicited emails and SMS). Spam is not only a nuisance but also a significant security risk, often used for phishing and malware distribution.

The primary goal was to build and evaluate a machine learning model capable of accurately classifying messages as either **"Spam"** or **"Ham"** (legitimate). The focus was on achieving a high F1-score, which represents a strong balance between high precision (minimizing false positives) and high recall (catching as much spam as possible).

This task is a **binary text classification** problem.

## 2. Dataset Used

*   **Source:** [SMS Spam Collection Dataset from Kaggle](https://www.kaggle.com/uciml/sms-spam-collection-dataset)
*   **Description:** The dataset contains 5,572 SMS messages in English. After cleaning and removing 403 duplicates, we worked with a dataset of 5,169 unique messages. The dataset is imbalanced, with approximately 87% ham and 13% spam.

## 3. Methodology & Model Development

The project followed a standard end-to-end machine learning workflow, detailed in the `SpamEmailDetector_AI_MiniProject.ipynb` notebook.

1.  **Data Preprocessing:**
    *   Text was normalized by converting to lowercase.
    *   Noise such as URLs, email addresses, numbers, and punctuation was removed using regular expressions.
    *   Standard English stopwords were removed, and words were reduced to their root form using **lemmatization** with NLTK.

2.  **Feature Engineering:**
    *   The cleaned text data was converted into numerical vectors using the **TF-IDF (Term Frequency-Inverse Document Frequency)** technique.
    *   To capture more contextual information, we used an `ngram_range` of `(1, 2)`, which creates features for both individual words and pairs of adjacent words (e.g., "free entry").

3.  **Model Building & Evaluation:**
    *   Three models were trained and compared using `scikit-learn` Pipelines:
        1.  Multinomial Naive Bayes (as a baseline)
        2.  Logistic Regression
        3.  **Linear Support Vector Machine (SVM)**
    *   The data was split into an 80% training set and a 20% testing set.

## 4. Key Results

The models were evaluated on the unseen test set. The **Linear SVM** emerged as the best-performing model.

| Model                     | Accuracy | Precision (Spam) | Recall (Spam) | F1-score (Spam) |
| ------------------------- | -------- | ---------------- | ------------- | --------------- |
| Multinomial Naive Bayes   | 0.9652   | 0.9897           | 0.7328        | 0.8421          |
| Logistic Regression       | 0.9584   | 1.0000           | 0.6718        | 0.8037          |
| **Linear SVM**            | **0.9816**   | **0.9912**           | **0.8626**        | **0.9224**          |

The **Linear SVM** achieved the best balance with an excellent precision of **99.1%** and a strong recall of **86.3%**, resulting in the highest F1-score of **92.2%**.

## 5. How to Run This Project

### Using the Google Colab Notebook
The notebook `SpamEmailDetector_AI_MiniProject.ipynb` contains the full development process.
1.  **Prerequisites:** A Google account.
2.  **Setup:**
    *   Open the notebook file in Google Colab.
    *   Upload the `spam.csv` dataset to the Colab session storage.
3.  **Execution:**
    *   Run the notebook cells sequentially to reproduce the entire analysis, training, and evaluation pipeline.