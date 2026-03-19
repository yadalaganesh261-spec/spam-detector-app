import streamlit as st
import pickle

# Load ML model & vectorizer
model = pickle.load(open("spam_model.pkl","rb"))
vectorizer = pickle.load(open("vectorizer.pkl","rb"))

# App title
st.title("Ganesh Spam Message Detector 🚀")

# Add instructions
st.markdown("📩 **Instructions:** Type any SMS message below and click the button to check if it’s spam or not.")

# Input box
message = st.text_input("Enter your message here ✏️")

# Check button
if st.button("Check Spam ✅"):
    if message.strip() == "":
        st.warning("⚠️ Please enter a message!")
    else:
        vect = vectorizer.transform([message])
        prediction = model.predict(vect)
        if prediction[0] == "spam":
            st.error("🚨 This message is SPAM!")
        else:
            st.success("✅ This message is HAM (Not Spam).")
