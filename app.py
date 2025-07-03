from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load your model and vectorizer
model = joblib.load('hindi_fake_news_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

from indicnlp.tokenize import indic_tokenize
import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

stemmer = PorterStemmer()
hindi_stopwords = set("""
और क्या के को में कि है वह पर से यह जो थे था कर हो हैं लिए हम आप भी नहीं दिया इसलिए
कोई साथ कहा किया करना इसकी जैसे उनके कुछ बिना रहे रहा ऐसे यदि
""".strip().split())
english_stopwords = set(stopwords.words('english'))
stop_words = hindi_stopwords.union(english_stopwords)

def clean_text(text):
    text = re.sub(r'[^a-zA-Z\u0900-\u097F\s]', ' ', str(text))
    tokens = list(indic_tokenize.trivial_tokenize(text))
    filtered = [t.lower() for t in tokens if t.lower() not in stop_words and len(t) > 1]
    stemmed = [stemmer.stem(w) if re.match('[a-zA-Z]+', w) else w for w in filtered]
    return ' '.join(stemmed)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        news_text = request.form['news']
        cleaned = clean_text(news_text)
        vec = vectorizer.transform([cleaned])
        proba = model.predict_proba(vec)[0]
        label = "Fake" if proba[1] > 0.3 else "Real"
        return render_template('result.html', news=news_text, label=label, proba=proba)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
