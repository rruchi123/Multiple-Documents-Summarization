import numpy as np
from flask import Flask, request, render_template
from bs4 import BeautifulSoup
import requests
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import logging
from transformers import pipeline
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():

    # Get user input
    query = request.form.get('query')
    num_results = request.form.get('doccount')
    word_count = request.form.get('wordcount')

    # Hardcoded URLs (for now)
    urls = [
        'https://www.cnet.com/tech/mobile/best-iphone/',
        'https://www.techradar.com/news/best-iphone',
        'https://www.tomsguide.com/us/best-apple-iphone,review-6348.html'
    ]

    # Scrape paragraph text
    p_contents = []
    for url in urls:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        p_contents.append([p.get_text() for p in paragraphs])

    # Load stopwords
    stop_words = set(stopwords.words('english'))

    # Text preprocessing function
    def preprocess(text):
        cleaned_text = " ".join(text)
        cleaned_text = cleaned_text.replace('\n', ' ').replace('\r', '')
        tokens = word_tokenize(cleaned_text)
        tokens = [t.lower() for t in tokens if t.lower() not in stop_words]
        return " ".join(tokens)

    # Preprocess scraped data
    preprocessed_texts = [preprocess(p) for p in p_contents]
    combined_text = " ".join(preprocessed_texts)

    # Word Cloud (local display)
    wordcloud = WordCloud(width=800, height=800, background_color='white').generate(combined_text)
    plt.figure(figsize=(8, 8))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()

    # LSA Summarization
    parser = PlaintextParser.from_string(combined_text, Tokenizer("english"))
    lsa_summarizer = LsaSummarizer()
    lsa_summary = lsa_summarizer(parser.document, 7)
    lsa_sentences = [str(sentence) for sentence in lsa_summary]
    paragraph = " ".join(lsa_sentences)

    # Heatmap visualization
    vectorizer = CountVectorizer(stop_words='english')
    matrix = vectorizer.fit_transform(lsa_sentences)

    sns.heatmap(
        matrix.toarray(),
        annot=True,
        xticklabels=vectorizer.get_feature_names_out(),
        yticklabels=lsa_sentences,
        cmap='Blues'
    )
    plt.figure(figsize=(12, 8))
    sns.set(font_scale=1.2)
    plt.show()

    # Transformer-based summarization (T5)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    summarizer = pipeline("summarization", model="t5-small")
    final_summary = summarizer(paragraph, do_sample=False)[0]['summary_text']

    # Render SAME page with result
    return render_template('index.html', result=final_summary)


if __name__ == "__main__":
    app.run(debug=True)

   
