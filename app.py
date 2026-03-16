from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load model
lda = pickle.load(open("lda_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

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
    app.run()