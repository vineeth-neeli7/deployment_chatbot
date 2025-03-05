import streamlit as st
import requests

# FastAPI backend URL
API_URL = "http://127.0.0.1:8000/chat"  

# Streamlit UI elements
st.title("Chat with Me ✌️")

# Initialize chat history in session state (limit to last 5 messages)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User input
user_input = st.text_input("You: ", "")

if st.button("Send") and user_input:
    # Send user message to FastAPI backend
    response = requests.post(API_URL, json={"message": user_input})

    # Get bot response
    if response.status_code == 200:
        bot_response = response.json()["response"]
    else:
        bot_response = "Error: Unable to get response."

    # Append the latest chat to history
    st.session_state.chat_history.append(f"You: {user_input}")
    st.session_state.chat_history.append(f"Bot: {bot_response}")

    # Keep only the last 5 messages
    st.session_state.chat_history = st.session_state.chat_history[-10:]  # (5 User + 5 Bot messages)

# Display only the last 5 exchanges
for chat in st.session_state.chat_history:
    st.write(chat)
