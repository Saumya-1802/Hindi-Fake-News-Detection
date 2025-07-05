# 📰 Hindi Fake News Detection (Python, scikit-learn, Flask, Bootstrap)

A machine learning web app to detect fake news articles in Hindi using Logistic Regression and TF-IDF vectorization.

### Screenshots

![WhatsApp Image 2025-07-05 at 12 24 45_6514a74a](https://github.com/user-attachments/assets/c5e9e3b4-d91f-4de1-b4e0-fe0c09d1043c)

![WhatsApp Image 2025-07-05 at 12 25 21_1c79dbf5](https://github.com/user-attachments/assets/f831b824-f59b-40a2-944c-f8341021655c)

![WhatsApp Image 2025-07-05 at 12 26 17_4eeffe18](https://github.com/user-attachments/assets/160cd2da-8694-4df3-b8d9-d627576b9cc5)

![WhatsApp Image 2025-07-05 at 12 26 47_3f8ba7be](https://github.com/user-attachments/assets/c7b3fb41-e266-47e5-9231-110f76d06d6b)


### 🚀 Features

Logistic Regression model trained on real + synthetic fake Hindi news data

~98% test accuracy on validation set

Supports Hindi-English mixed news (code-mixed text)

Uses TF-IDF vectorization (unigrams, bigrams) with 3000 features

Enhanced generalization with synthetic health/science/government fake news samples

Flask web app with Bootstrap-based gradient UI (soft pastel + nature-inspired themes)

### 📌 Technologies Used

Python (Flask, scikit-learn, pandas, nltk, Indic NLP)

HTML / CSS / Bootstrap (frontend)

Logistic Regression + TF-IDF

Joblib (model persistence)

### ⚙️ Setup Instructions

1️⃣ Clone the repo

git clone https://github.com/Saumya-1802/Hindi-Fake-News-Detection.git

cd Hindi-Fake-News-Detection

2️⃣ Install dependencies

pip install -r requirements.txt

3️⃣ Download NLTK stopwords (if not done already)

import nltk

nltk.download('stopwords')

4️⃣ Run the app

python app.py

### 📝 Project Structure

Hindi-Fake-News-Detection/

├── app.py              # Flask backend

├── hindi_fake_news_model.pkl  # Trained model

├── tfidf_vectorizer.pkl       # Saved vectorizer

├── templates/

│   ├── index.html

│   └── result.html

└── static/

    └── (optional CSS / images if any)

### 📊 Model Performance

| Metric        | Value                                      |
| ------------- | ------------------------------------------ |
| Test Accuracy | \~98%                                      |
| Classifier    | Logistic Regression                        |
| Features      | TF-IDF (1-gram, 2-gram), 3000 max features |

### 📌 Notes

This project was developed for local deployment and demonstration.

The model may need further fine-tuning for production use on unseen sources.
