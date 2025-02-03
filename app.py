import streamlit as st
import tensorflow as tf
import joblib
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = tf.keras.models.load_model("improved_spam_classifier.h5")
tokenizer = joblib.load("tokenizer.pkl")

st.title("📩 Spam SMS Classifier")
st.write("Enter your SMS message below to classify it as **Spam** or **Ham**.")

sms_text = st.text_area("Enter SMS Text:", "")

if st.button("Check Spam"):
    if sms_text:
        # Preprocess input text
        sequence = tokenizer.texts_to_sequences([sms_text])
        padded_sequence = pad_sequences(sequence, maxlen=50, padding='post', truncating='post')

        # Predict whether it's spam or ham
        prediction = model.predict(padded_sequence)[0][0]
        spam_probability = float(prediction)
        label = "Spam" if spam_probability > 0.5 else "Ham"

        # Display result
        st.write(f"Prediction: **{label}**")
        st.write(f"Spam Probability: {spam_probability:.2f}")
    else:
        st.error("Please enter a message to classify!")
