import streamlit as st
import joblib
import pandas as pd

# -------------------------
# Load the trained models
# -------------------------
model_A = joblib.load("models/Model_A_Balanced_NB.pkl")
model_B = joblib.load("models/Model_B_Imbalanced_LR.pkl")

# Load the corresponding TF-IDF vectorizers
vectorizer_A = joblib.load("models/tfidf_balanced.pkl")      # for balanced dataset
vectorizer_B = joblib.load("models/tfidf_imbalanced.pkl")    # for imbalanced dataset

# -------------------------
# Prediction function
# -------------------------
def predict_score(text, model, vectorizer):
    """
    Predict the review score using the given model and vectorizer.
    """
    X = vectorizer.transform([text])  # Transform raw text to numeric features
    prediction = model.predict(X)
    return prediction[0]

# -------------------------
# Streamlit UI
# -------------------------
st.title("Review Score Prediction (Model A vs Model B)")

# Text input for review
review_text = st.text_area("Enter a review text:")

# Prediction button
if st.button("Predict"):
    if review_text.strip():
        # Get predictions from both models
        prediction_A = predict_score(review_text, model_A, vectorizer_A)
        prediction_B = predict_score(review_text, model_B, vectorizer_B)

        # Display predictions side by side
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Model A (Balanced - Naive Bayes)")
            st.success(f"Predicted Score: **{prediction_A}**")

        with col2:
            st.subheader("Model B (Imbalanced - Logistic Regression)")
            st.info(f"Predicted Score: **{prediction_B}**")
    else:
        st.warning("Please enter some review text.")
