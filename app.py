from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load model
lda = joblib.load("lda_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Load dataset
df = pd.read_csv("ALL_with_features.xlsx")  # Make sure this is uploaded

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/random")
def random_question():
    # Pick a random question
    question = df['Item'].sample(n=1).iloc[0]
    
    # Get topic prediction
    X = vectorizer.transform([question])
    topic = lda.transform(X)
    topic_id = topic.argmax()
    
    return render_template("index.html",
                           random_question=question,
                           predicted_topic=topic_id)

@app.route("/predict", methods=["POST"])
def predict():
    question = request.form["question"]
    X = vectorizer.transform([question])
    topic = lda.transform(X)
    topic_id = topic.argmax()
    
    return render_template("index.html",
                           user_question=question,
                           predicted_topic=topic_id)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)