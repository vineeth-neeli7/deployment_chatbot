# Import necessary libraries
from fastapi import FastAPI
from pydantic import BaseModel
import google.generativeai as genai
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import requests
import gdown
import os

# Initialize FastAPI (Framework for serving chatbot API)
app = FastAPI()

# Configure Google Gemini AI API (Used for generating chatbot responses)
genai.configure(api_key="AIzaSyCDpnFZCiMT2el2NU5ww342n7w2aMWvbFA")  # API key for Gemini AI

# Load the trained BERT model and tokenizer
model_path = "bert_intent_model"  # Path where the model was saved
model_file = f"{model_path}/model.safetensors"
drive_file_id = "144yc79do3_c4u1ZVbLEAiX31_y6MSCy8" # Google Drive file ID for the model file https://drive.google.com/file/d/144yc79do3_c4u1ZVbLEAiX31_y6MSCy8/view?usp=drive_link

os.makedirs(model_path, exist_ok=True)

# Download the model file from Google Drive
if not os.path.exists(model_file):
    url = f"https://drive.google.com/uc?id={drive_file_id}"
    gdown.download(url, model_file, quiet=False)


tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
model.eval()  # Set the model to evaluation mode

# Define intent label mapping (Should match labels used in training)
label_map = {
   "joke": 0, "general_info": 1, "bot identity": 2, "weather": 3, "greeting": 4, "farewell": 5
}

# Initialize chat history to provide conversation context
chat_history = []

# Define the API request format
class ChatRequest(BaseModel):
    message: str  # User input message

# Function to predict user intent using the trained BERT model
def classify_intent(user_input, threshold=0.7):
    encoded_input = tokenizer(user_input, padding=True, truncation=True, return_tensors="pt")

    with torch.no_grad():
        output = model(**encoded_input)

    scores = torch.softmax(output.logits, dim=1)  # Convert to probability scores
    confidence, predicted_label = torch.max(scores, dim=1)  # Get the highest probability intent

    if confidence.item() < threshold:
        return "unknown"  # If confidence is too low, return "unknown" intent

    # Convert label ID back to intent name
    intent = list(label_map.keys())[list(label_map.values()).index(predicted_label.item())]
    return intent

# Function to fetch general knowledge from Wikipedia API
def get_wikipedia_info(query):
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{query.replace(' ', '_')}"
    response = requests.get(url)
    data = response.json()
    
    if "extract" in data:
        return data["extract"]  # Return the Wikipedia summary
    
    return "I couldn't find any information on that."

# Function to generate chatbot responses using Google Gemini AI
def chat_with_gemini(user_input, intent):
    model = genai.GenerativeModel("models/gemini-1.5-flash")  # Select Gemini AI model

    # Store conversation history for context-aware responses
    chat_history.append({"role": "user", "parts": [{"text": user_input}]})

    # Maintain a limited history of the last 10 messages
    if len(chat_history) > 10:
        chat_history.pop(0)

    # If the intent is "general_info", fetch details from Wikipedia instead of AI
    if intent == "general_info":
        bot_response = get_wikipedia_info(user_input)
    else:
        response = model.generate_content(chat_history)  # Get AI-generated response
        bot_response = response.text

    # Store the bot's response in conversation history
    chat_history.append({"role": "assistant", "parts": [{"text": bot_response}]})
    return bot_response

# Define the FastAPI endpoint for chatbot interaction
@app.post("/chat")
async def chat(request: ChatRequest):
    predicted_intent = classify_intent(request.message)  # Detect user intent
    response = chat_with_gemini(request.message, predicted_intent)  # Generate a response
    return {"response": response}


