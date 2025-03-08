# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

# Upload your dataset (run this cell to upload the CSV file)
#from google.colab import files
#uploaded = files.upload()

# Read the uploaded file
#file_name = list(uploaded.keys())[0]
#file_name = (r"C:\Users\shsha\OneDrive\Desktop\Desi_Holmes\sherlock_holmes_cases.csv")
file_name=os.path.join(current_directory, 'sherlock_holmes_cases.csv')
df = pd.read_csv(file_name)

# Preprocessing the data
X = df['description']  # Features (case descriptions)
y = df['status']  # Target variable (status of the case: Solved/Unsolved)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline to vectorize the text and train a Random Forest Classifier
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train_tfidf, y_train)

# Evaluate the model
y_pred = classifier.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy*100:.2f}%")

# Function to predict the solution for a new case
def predict_case_solution(new_case_description):
    new_case_tfidf = vectorizer.transform([new_case_description])
    prediction = classifier.predict(new_case_tfidf)
    return prediction[0]

# Function to generate detailed observations, leads, and actions
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
        observations.append(f"Similar case: '{case['description']}' with status '{case['status']}'. Common keywords: {common_keywords_str}.")

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

# Input the case description
new_case_description = input("\nPlease enter the case description: ")

# Predict the solution
predicted_status = predict_case_solution(new_case_description)
print(f"\nThe predicted status for this case is: {predicted_status}")

# Generate detailed case analysis
analysis = generate_case_analysis(new_case_description)

# Output the detailed observations, leads, and next steps
print("\nObservations:")
for observation in analysis["Observations"]:
    print(f"- {observation}")

print("\nLeads:")
for lead in analysis["Leads"]:
    print(f"Lead: {lead['Lead']}")
    print(f"Confidence Score: {lead['Confidence Score']}")
    print(f"Suggested Action: {lead['Suggested Action']}")
    print()

print("Next Steps:")
for step in analysis["Next Steps"]:
    print(f"- {step}")

import pickle
pickle.dump(classifier,open("model.pkl", "wb"))

import pickle
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
