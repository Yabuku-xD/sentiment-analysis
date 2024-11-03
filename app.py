from flask import Flask, render_template, request, jsonify
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import requests
from bs4 import BeautifulSoup
import re

nltk.download('stopwords')
nltk.download('punkt')

app = Flask(__name__)

# Load positive and negative words
def load_word_list(file_path):
    with open(file_path, 'r') as file:
        return [word.strip() for word in file.readlines()]

positive_words = load_word_list('positive-words.txt')
negative_words = load_word_list('negative-words.txt')

# Retrieve text from URL
def retrieve_text_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        return ' '.join(p.get_text() for p in paragraphs)
    except requests.exceptions.RequestException:
        return None

# Remove stopwords from text
def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

# Calculate sentiment percentages
def calculate_sentiment_percentage(text, positive_words, negative_words):
    words = text.lower().split()
    total_words = len(words)
    positive_count = sum(1 for word in words if word in positive_words)
    negative_count = sum(1 for word in words if word in negative_words)
    positive_percentage = (positive_count / total_words) * 100 if total_words > 0 else 0
    negative_percentage = (negative_count / total_words) * 100 if total_words > 0 else 0
    return positive_percentage, negative_percentage

# Calculate polarity and subjectivity scores
def calculate_polarity_score(positive_percentage, negative_percentage):
    return positive_percentage - negative_percentage

def calculate_subjectivity_score(positive_percentage, negative_percentage):
    return positive_percentage + negative_percentage

# Define the route
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    url = request.json.get("url")
    if url:
        article_text = retrieve_text_from_url(url)
        if article_text:
            article_text = remove_stopwords(article_text)
            positive_percentage, negative_percentage = calculate_sentiment_percentage(article_text, positive_words, negative_words)
            polarity_score = calculate_polarity_score(positive_percentage, negative_percentage)
            subjectivity_score = calculate_subjectivity_score(positive_percentage, negative_percentage)
            
            return jsonify({
                'POSITIVE SCORE': positive_percentage,
                'NEGATIVE SCORE': negative_percentage,
                'POLARITY SCORE': polarity_score,
                'SUBJECTIVITY SCORE': subjectivity_score
            })
    return jsonify({"error": "URL not accessible or text extraction failed."})

if __name__ == "__main__":
    app.run(debug=True)