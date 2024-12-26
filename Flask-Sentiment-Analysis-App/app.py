from flask import Flask, request, render_template
from transformers import pipeline
import nltk
from nltk.corpus import stopwords

# Download stopwords if not already available
nltk.download('stopwords')

# Initialize Flask app
app = Flask(__name__)

# Initialize the RoBERTa sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

@app.route('/')
def my_form():
    return render_template('form.html', title="Sentiment Analysis")

@app.route('/', methods=['POST'])
def my_form_post():
    stop_words = stopwords.words('english')  # Load stopwords
    text1 = request.form['text1'].strip()  # Get input

    if not text1:
        return render_template('form.html', title="Sentiment Analysis", error="Please enter text.")

    # Remove stopwords
    processed_doc1 = ' '.join([word for word in text1.split() if word.lower() not in stop_words])

    # Perform sentiment analysis using RoBERTa
    results = sentiment_pipeline(processed_doc1)
    label = results[0]['label']
    confidence = round(results[0]['score'], 4)

    # Map RoBERTa's labels to corresponding positive/negative/neutral
    sentiment_map = {
        "LABEL_0": "Negative",
        "LABEL_1": "Neutral",
        "LABEL_2": "Positive"
    }
    sentiment_label = sentiment_map.get(label, label)

    # Calculate probabilities for positive, negative, and neutral
    score_positive = confidence * 100 if sentiment_label == "Positive" else 0
    score_negative = confidence * 100 if sentiment_label == "Negative" else 0
    score_neutral = (1 - confidence) * 100 if sentiment_label == "Neutral" else 0

    # Return results to the template
    return render_template(
        'form.html',
        title="Sentiment Analysis",
        text1=text1,
        sentiment_label=sentiment_label,
        score_positive=round(score_positive, 2),
        score_negative=round(score_negative, 2),
        score_neutral=round(score_neutral, 2)
    )

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5002, threaded=True)

