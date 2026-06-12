import nltk
import os
nltk_data_path = os.path.join(os.path.expanduser('~'), "nltk_data")
if not os.path.exists(nltk_data_path):
    nltk.download("punkt", quiet = True)
    nltk.download("stopwords", quiet = True)
    nltk.download("wordnet", quiet= True)
    nltk.download("punkt_tab", quiet = True)

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import json
import random
import sys #internal list of folder paths

#>> Api directory path
API_DIR = os.path.dirname(os.path.abspath(__file__))

#>> Importing nlp file 
sys.path.append(os.path.join(API_DIR, ".."))
from nlp_processor import NLPProcessor, processor

#>>Chat UI 
INTERFACE_PATH = os.path.join(API_DIR,"..","interface")


app = Flask(__name__, template_folder=INTERFACE_PATH)

# -- Cors lets the chat.html talk to flask server without being blocked by the browser.

CORS(app) # Enable cors for all routes 

#>> Loading the Models 
MODEL_PATH = os.path.join(API_DIR, "..", "models", "intent_classifier.pkl")
VECTORIZER_PATH = os.path.join(API_DIR, "..", "models", "tfidf_vectorizer.pkl")
INTENTS_PATH = os.path.join(API_DIR, "..", "data", "intents.json")

print("Loading the models and vectorizer...")
model = joblib.load(MODEL_PATH)
tf_vectorizer = joblib.load(VECTORIZER_PATH)

with open(INTENTS_PATH, "r") as f:
    intents_data = json.load(f)
    
print("All files loaded successfully...")

#>> Get Response by Intents Tags
def get_response(intent_tag):
    
    for intent in intents_data["intents"]:
        if intent["tag"] == intent_tag:
            return random.choice(intent["responses"])
        
    return "I am sorry, I dindnt understand that"

#>> Chat HTML Get

@app.route("/", methods=["GET"])
def home():
    return render_template("chat.html")


#>> Health Endpoint 
@app.route("/health", methods = ["GET"])
def health():
    
    #-- Use to verify the server is running before the testing/chat
    
    return jsonify({
        "status": "ok",
        "message" : "Chatbot API is running"
    }), 200 # Success code

#>> Main Chat End point
@app.route("/chat", methods = ["POST"])
def chat():
    
    # Getting the incomming json body via post in messagefield
    data = request.get_json()
    
    # If empty message field
    if not data:
        return jsonify({
            "error" : "Request body is empty. Send Json with a 'message' field",
        }), 400 # Client side error
        
    message = data.get("message", '').strip()
    
    if not message:
        return jsonify({
            "error": "Message field is missing or empty"
        }), 400
    
    #-- nlp preprocessor    
    preprocessed = processor.main(message)
    
    #-- 500 words trained vectorizer
    #-- Passing [preprocessed] turns our single string into a 1-row dataset list,
    vector = tf_vectorizer.transform([preprocessed]) 
    
    #-- prediction intent
    predicted_intent = model.predict(vector)[0]
    
    #--confidence score
    confidence = float(model.predict_proba(vector).max())
    
    #-- If confidence is low
    if confidence < 0.15:
        return jsonify({
            "response" : "I am not quite sure I understood that. Could you rephrase your question?",
            "intent" : "uncertain",
            "confidence" : round(confidence, 4)
        }), 200
    
    #-- getting response frim intent.json
    response_text = get_response(predicted_intent)
    
    # Returning the Json response
    return jsonify({
        "response": response_text,
        "intent" : predicted_intent,
        "confidence": round(confidence,4)
    }), 200 # success code
    
#>> Running the Server

# block only runs when directly execute app.py.
if __name__ == "__main__":
    port = int(os.environ.get("PORT",5000))
    app.run(debug=False, host = '0.0.0.0', port = port)
    
    
    
    