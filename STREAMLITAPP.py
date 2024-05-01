import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the pre-trained models
@st.cache(allow_output_mutation=True)
def load_models():
    # Load SVM model
    svm_model = joblib.load('C:/Users/durga/Desktop/SPU/DS 600/PROJECT/FINAL/svm_model.pkl')  # Update with your SVM model file path

    # Load Random Forest model
    rf_model = joblib.load('C:/Users/durga/Desktop/SPU/DS 600/PROJECT/FINAL/rf_model.pkl')  # Update with your Random Forest model file path

    return svm_model, rf_model

# Function to preprocess text input
def preprocess_text(text):
    # Perform any preprocessing steps here (e.g., tokenization, cleaning)
    return text

# Function to predict sentiment using SVM model
def predict_sentiment_svm(text, model):
    # Preprocess the text
    text = preprocess_text(text)
    # Feature hashing
    vectorizer = HashingVectorizer(n_features=1000)
    text_vectorized = vectorizer.transform([text])
    # Predict sentiment
    prediction = model.predict(text_vectorized)[0]
    return prediction

# Function to predict sentiment using Random Forest model
def predict_sentiment_rf(text, model):
    # Preprocess the text
    text = preprocess_text(text)
    # Feature hashing
    vectorizer = HashingVectorizer(n_features=1000)
    text_vectorized = vectorizer.transform([text])
    # Predict sentiment
    prediction = model.predict(text_vectorized)[0]
    return prediction

def main():
    # Load the models
    svm_model, rf_model = load_models()

    # Set title and description
    st.title("HATE SPEECH DETECTION")
    st.write("This app detects hate speech and categorizes it as either negative or neutral.")

    # Text input for user to enter comment
    comment = st.text_input("Enter your comment:")

    # Button to analyze the comment with SVM model
    if st.button("Analyze with SVM"):
        if comment:
            # Predict sentiment using SVM model
            prediction_svm = predict_sentiment_svm(comment, svm_model)
            st.write("Sentiment (SVM):", prediction_svm)
        else:
            st.warning("Please enter a comment.")

    # Button to analyze the comment with Random Forest model
    if st.button("Analyze with Random Forest"):
        if comment:
            # Predict sentiment using Random Forest model
            prediction_rf = predict_sentiment_rf(comment, rf_model)
            st.write("Sentiment (Random Forest):", prediction_rf)
        else:
            st.warning("Please enter a comment.")

if __name__ == "__main__":
    main()
