# app.py
from flask import Flask, render_template, request
import os
import joblib
import numpy as np
from sklearn.metrics import pairwise_distances_argmin_min

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load pre-trained models
lda = joblib.load(os.path.join(BASE_DIR, 'lda_model.pkl'))
vectorizer = joblib.load(os.path.join(BASE_DIR, 'vectorizer.pkl'))
ahc = joblib.load(os.path.join(BASE_DIR, 'ahc_model.pkl'))

# Load training LDA topics for nearest-cluster assignment (AHC)
train_lda_topics_path = os.path.join(BASE_DIR, 'train_lda_topics.npy')
train_lda_topics = np.load(train_lda_topics_path)

app = Flask(__name__)

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    # Get user responses
    responses = request.form.getlist('responses')
    if not responses:
        return "No responses submitted."

    #Vectorize text
    X_text = vectorizer.transform(responses)

    #LDA - topic distributions
    lda_topics = lda.transform(X_text)

    #Assign AHC clusters using nearest neighbor to training LDA topics
    nearest_idx, _ = pairwise_distances_argmin_min(lda_topics, train_lda_topics)
    ahc_clusters = ahc.labels_[nearest_idx]

    # Prepare results for rendering
    results = []
    for i, response in enumerate(responses):
        topic_dist = ", ".join([f"{p:.2f}" for p in lda_topics[i]])
        results.append({
            "response": response,
            "ahc_cluster": int(ahc_clusters[i]),
            "topics": topic_dist
        })

    return render_template('results.html', results=results)

import os

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Use PORT from Render, fallback to 5000 locally
    app.run(host='0.0.0.0', port=port, debug=True)