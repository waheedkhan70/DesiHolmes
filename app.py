from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__, template_folder=r'D:\My Project\Bittu\new\Desi_Holmes\template')

# Load the trained model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Load the dataset
file_name = r"D:\My Project\Bittu\new\Desi_Holmes\sherlock_holmes_cases.csv"
df = pd.read_csv(file_name)


@app.route('/')
def home():
    print("Current working directory:", os.getcwd())
    print("Template path:", os.path.join(os.getcwd(), 'templates', 'index.html'))
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    description = data['description']

    # Transform the description
    features = vectorizer.transform([description])

    # Make prediction
    prediction = model.predict(features)

    # Generate detailed case analysis
    analysis = generate_case_analysis(description)

    response = {
        'prediction': prediction[0],
        'analysis': analysis
    }

    return jsonify(response)


def generate_case_analysis(new_case_description):
    # Transform the new case description
    new_case_vector = vectorizer.transform([new_case_description])

    # Compute similarity with all previous cases
    case_vectors = vectorizer.transform(df['description'])
    similarities = cosine_similarity(new_case_vector, case_vectors).flatten()

    # Get top matching cases
    top_indices = similarities.argsort()[-5:][::-1]  # Top 5 most similar cases
    similar_cases = df.iloc[top_indices]

    # Generate unique observations and leads
    observations = []
    leads = []
    for i, case in similar_cases.iterrows():
        # Extract common keywords or themes from the cases
        common_keywords = set(new_case_description.split()).intersection(set(case['description'].split()))
        common_keywords_str = ', '.join(common_keywords) if common_keywords else "No significant keywords found"

        # Formulate observations based on specific insights
        observations.append(
            f"Similar case: '{case['description']}' with status '{case['status']}'. Common keywords: {common_keywords_str}.")

        # Generate leads with actionable insights
        leads.append({
            "Lead": f"Derived from case '{case['description']}'",
            "Confidence Score": round(similarities[i] * 100, 2),
            "Suggested Action": f"Investigate patterns or evidence similar to case '{case['description']}'."
        })

    # Default next steps
    next_steps = [
        "Compare forensic evidence with similar past cases.",
        "Conduct additional witness interviews to gather insights.",
        "Leverage AI tools for deeper pattern analysis in unresolved leads."
    ]

    return {
        "Observations": observations,
        "Leads": leads,
        "Next Steps": next_steps
    }


if __name__ == "__main__":
    app.run(debug=True)
