from flask import Flask, request, jsonify
import joblib
import os

app = Flask(__name__)

# Load model correctly using joblib
lda = joblib.load("lda_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.route("/")
def home():
    return "LDA Competency Model Running"

@app.route("/predict", methods=["POST"])
def predict():
    question = request.json["question"]

    X = vectorizer.transform([question])
    topic = lda.transform(X)

    topic_id = topic.argmax()

    return jsonify({
        "predicted_topic": int(topic_id)
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)