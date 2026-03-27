import os
import random
from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib

app = Flask(__name__)
# Load dataset & models
df = pd.read_excel("ALL_with_features.xlsx") 
lda = joblib.load("lda_model.pkl")            
vectorizer = joblib.load("vectorizer.pkl")    

# Home page
@app.route("/")
def home():
    return render_template("index.html")  

# Predict topic for a user question

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    question = data.get("question", "")
    if not question:
        return jsonify({"error": "No question provided"}), 400

    X = vectorizer.transform([question])
    topic_distribution = lda.transform(X)
    topic_id = topic_distribution.argmax()

    return jsonify({"predicted_topic": int(topic_id)})

# Random question feature
@app.route("/random", methods=["GET"])
def random_question():
    question = random.choice(df['Item'].tolist())
    X = vectorizer.transform([question])
    topic_distribution = lda.transform(X)
    topic_id = topic_distribution.argmax()

    return jsonify({
        "question": question,
        "predicted_topic": int(topic_id)
    })
# Run app with Render port
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)