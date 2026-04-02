from flask import Flask, render_template, request
import os
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances_argmin_min

# Setup paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load pre-trained models
lda = joblib.load(os.path.join(BASE_DIR, 'lda_model.pkl'))
vectorizer = joblib.load(os.path.join(BASE_DIR, 'vectorizer.pkl'))
ahc = joblib.load(os.path.join(BASE_DIR, 'ahc_model.pkl'))

train_lda_topics = np.load(os.path.join(BASE_DIR, 'train_lda_topics.npy'))

# Initialize Flask app
app = Flask(__name__)

# Extract main concept (last word) for contradiction detection
def get_main_concept(answer):
    words = answer.lower().strip().split()
    return words[-1] if words else ""

# Competency Function
def get_competency(student_ans, correct_ans):
    student = student_ans.lower().strip()
    correct = correct_ans.lower().strip()

    # Direct match
    if student == correct:
        return "High Competency"

    # Main concept check
    student_concept = get_main_concept(student)
    correct_concept = get_main_concept(correct)

    if student_concept != correct_concept:
        return "Low Competency"

    # Similarity and keyword match
    student_vec = vectorizer.transform([student])
    correct_vec = vectorizer.transform([correct])
    similarity = cosine_similarity(student_vec, correct_vec)[0][0]

    correct_keywords = set(correct.split())
    student_words = set(student.split())
    keyword_match = len(correct_keywords & student_words) / max(len(correct_keywords), 1)

    # Final competency decision
    if similarity >= 0.75 or keyword_match >= 0.6:
        return "High Competency"
    elif similarity >= 0.4 or keyword_match >= 0.3:
        return "Moderate Competency"
    else:
        return "Low Competency"

# Feedback Function
def generate_feedback(competency):
    if competency == "High Competency":
        return "Excellent response. Demonstrates strong understanding."
    elif competency == "Moderate Competency":
        return "Partial understanding. Needs improvement."
    else:
        return "Incorrect answer. Review the concept."

# Explanation Function
# Simplified Explanation Function
def generate_explanation(competency, topic_dist, student_response, correct_ans):
    main_topic_idx = int(np.argmax(topic_dist))
    confidence = topic_dist[main_topic_idx]

    # Only show topic relevance and similarity/keyword metrics
    student_vec = vectorizer.transform([student_response])
    correct_vec = vectorizer.transform([correct_ans])
    similarity = cosine_similarity(student_vec, correct_vec)[0][0]

    correct_keywords = set(correct_ans.lower().split())
    student_words = set(student_response.lower().split())
    keyword_match = len(correct_keywords & student_words) / max(len(correct_keywords),1)

    return f"Relates mainly to Topic {main_topic_idx + 1} ({round(confidence*100,1)}% confidence). Similarity: {round(similarity*100,1)}%, Keyword match: {round(keyword_match*100,1)}%."
# Format topic distribution
def format_topics(topic_array):
    main_idx = int(np.argmax(topic_array))
    formatted = []
    for i, val in enumerate(topic_array):
        percentage = round(val * 100, 1)
        if i == main_idx:
            formatted.append(f"Topic {i+1}: {percentage}% (Main)")
        else:
            formatted.append(f"Topic {i+1}: {percentage}%")
    return ", ".join(formatted)

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    question = request.form.get('question')
    correct_answer = request.form.get('correct_answer')
    responses = request.form.getlist('responses')

    if not question or not correct_answer:
        return "Question and correct answer are required."
    if not responses or all(r.strip() == "" for r in responses):
        return "No valid student responses submitted."

    # Vectorize
    X_text = vectorizer.transform(responses)

    # LDA topics
    lda_topics = lda.transform(X_text)

    # AHC clusters (for internal use, not returned)
    nearest_idx, _ = pairwise_distances_argmin_min(lda_topics, train_lda_topics)
    _ = ahc.labels_[nearest_idx]  # cluster info not used in results

    results = []

    for i, response in enumerate(responses):
        competency = get_competency(response, correct_answer)
        topic_dist = lda_topics[i]
        explanation = generate_explanation(competency, topic_dist, response, correct_answer)
        feedback = generate_feedback(competency)

        results.append({
            "response": response,
            "competency": competency,
            "topics": format_topics(topic_dist),
            "explanation": explanation,
            "feedback": feedback
        })

    return render_template('results.html', results=results, question=question)

# Run app
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)