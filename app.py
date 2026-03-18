import streamlit as st
import pickle

# Title
st.title("Ganesh Spam Message Detector 📩")

# Input box
message = st.text_input("Enter your message")

# Button
if st.button("Check"):
    if message == "":
        st.write("Please enter a message")
    else:
        # Dummy logic (replace with your model)
        if "free" in message.lower():
            st.write("Spam ❌")
        else:
            st.write("Not Spam ✅")

# Step 1: Import libraries
import pandas as pd
import numpy as np

# Step 2: Load dataset
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
data = pd.read_csv(url, sep='\t', names=["label", "message"])

# Step 3: Convert labels (ham=0, spam=1)
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Step 4: Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    data['message'], data['label'], test_size=0.2, random_state=42
)

# Step 5: Convert text to numbers
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Step 6: Train model
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Step 7: Accuracy check
accuracy = model.score(X_test_vec, y_test)
print("Model Accuracy:", accuracy)

# Step 8: Test with custom message
while True:
    msg = input("\nEnter a message (or type 'exit' to stop): ")
    
    if msg.lower() == 'exit':
        break
        
    msg_vec = vectorizer.transform([msg])
    prediction = model.predict(msg_vec)
    
    if prediction[0] == 1:
        print("🚫 This is SPAM message")
    else:
        print("✅ This is NOT SPAM")
