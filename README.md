# 🤖 NovaMind — AI Chatbot for Customer Support

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![Flask](https://img.shields.io/badge/Flask-3.0-black?style=flat-square&logo=flask)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-orange?style=flat-square&logo=scikit-learn)
![NLTK](https://img.shields.io/badge/NLTK-3.8-green?style=flat-square)
![Render](https://img.shields.io/badge/Deployed-Render-purple?style=flat-square)
![Status](https://img.shields.io/badge/Status-Live-brightgreen?style=flat-square)

**A Rule-Based + ML AI Chatbot for Customer Support**  
*AI/ML Internship Project · NextGen Forge Technologies · NFGT/HR/INT/2026/138*

[🚀 Live Demo](https://novamind-chatbot.onrender.com) · [📡 API Health](https://novamind-chatbot.onrender.com/health) · [📂 Repository](https://github.com/Shravani-1325/Ai-Customer-Support-Chatbot)

</div>

---



## 1. Project Overview

**NovaMind** is a minimum viable AI-powered customer support chatbot built entirely with open-source Python tools. It simulates the core functionality of a real customer support system — understanding user queries through Natural Language Processing (NLP), classifying intent using a trained Machine Learning model, and returning context-appropriate automated responses through a REST API and a glassmorphism chat interface.



> **Internship Project** · NextGen Forge Technologies · AI/ML Program · May–June 2026  
> **Reference:** NFGT/HR/INT/2026/138  
> **Live URL:** https://novamind-chatbot.onrender.com

---

## 2. Problem Statement

Small businesses and startups often cannot afford dedicated customer support teams but still need to handle common customer queries efficiently. 
> This creates three core challenges:

- Repetitive Queries Consuming Support Time
- No 24/7 Support Availability
- 2.3 High Perceived Barrier to Entry


---

## 3. Project Objective

Build a **Rule-Based + ML AI Chatbot MVP** that:

| # | Goal | Description |
|---|------|-------------|
| G1 | Intent Classification | Classify incoming customer queries into one of 9 predefined intent categories |
| G2 | NLP Pipeline | Preprocess raw user text into clean, machine-readable tokens using NLTK |
| G3 | REST API | Expose ML predictions via a Flask REST API (POST /chat) |
| G4 | Chat Interface | Provide a glassmorphism chat UI connected to the live API |
| G5 | Cloud Deployment | Deploy the full system to a publicly accessible live URL |


---

## 4. Data Understanding

### 4.1 Dataset Source
The dataset was **manually created** — no external dataset was used. All training patterns were written specifically for this project to simulate realistic customer support queries across 9 intent categories.

### 4.2 Dataset Characteristics

| Property | Value |
|----------|-------|
| Total Training Patterns | 140 |
| Total Intent Classes | 9 |
| Average Patterns per Intent | ~15 |
| Responses per Intent | 4 |
| Data Format | JSON (intents.json) |
| Data Scale | MVP — production typically uses 500+ per intent |

### 4.3 Feature Breakdown

Each record in the dataset has three components:

```
intents.json structure:
{
  "tag":       string   → intent label (e.g. "order_status")
  "patterns":  list     → training example sentences (15–17 per intent)
  "responses": list     → bot reply templates (4 per intent)
}
```


---

## 5. Knowledge Base

The chatbot's knowledge base lives in `data/intents.json`. Each intent has example patterns the model trains on and response templates the bot randomly selects from.

### Intent Definitions

| Intent Tag | What It Handles | Example Trigger |
|------------|----------------|-----------------|
| `greetings` | Opening messages, conversation starters | "Hello I need help" |
| `order_status` | Order tracking, delivery queries | "Where is my order?" |
| `return_policy` | Returns, refunds, exchanges | "How do I return a product?" |
| `billing_inquiry` | Payment issues, wrong charges, invoices | "I was charged twice" |
| `store_hours` | Working hours, availability, holidays | "What time do you close?" |
| `product_pricing` | Pricing plans, discounts, subscriptions | "What is the premium plan cost?" |
| `escalate_to_human` | Requests for human agent | "I want to talk to a real person" |
| `chitchat` | Short acknowledgements, casual responses | "ok", "alright", "got it" |
| `shipping_delay` | Late deliveries, stuck packages | "My package is stuck in transit" |



---

## 6. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER BROWSER                             │
│                   chat.html (Frontend UI)                       │
│          Glassmorphism · Suggestion chips · Intent tags         │
└────────────────────────┬────────────────────────────────────────┘
                         │  POST /chat
                         │  {"message": "where is my order"}
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FLASK REST API (app.py)                      │
│                                                                 │
│  GET  /          → serve chat.html frontend                     │
│  POST /chat      → process message, return response             │
│  GET  /health    → server status (used by Render monitor)       │
│                                                                 │
│  Input Validation → confidence threshold (< 0.35 = fallback)    │
└──────────┬──────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────┐
│                     NLP PIPELINE (NLPProcessor)                 │
│                                                                 │
│  clean_text() → tokenise() → remove_stopwords() → lemmatise()   │
└──────────┬──────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ML INFERENCE PIPELINE                        │
│                                                                 │
│  tfidf_vectorizer.pkl  →  convert text to 500-dim vector        │
│  intent_classifier.pkl →  Logistic Regression predict intent    │
│  intents.json          →  lookup random response for intent     │
└──────────┬──────────────────────────────────────────────────────┘
           │
           ▼
┌────────────────────────────────────────────────────────────────────┐
│               RESPONSE                                             │
│  {"response": "...", "intent": "order_status", "confidence": 0.94} │
└────────────────────────────────────────────────────────────────────┘
```

**Deployment Architecture:**
```
GitHub Repo (main branch)
        ↓  Auto-deploy on commit
   Render Web Service
        ↓  gunicorn api.app:app
   Flask App (Python 3.11)
        ↓
   https://novamind-chatbot.onrender.com
```

---

## 7. Example Queries

Test these on the live chatbot at https://novamind-chatbot.onrender.com:

| User Input | Predicted Intent | Sample Response |
|------------|-----------------|-----------------|
| "Where is my package?" | order_status | "Please share your order ID and I'll check the status for you." |
| "My delivery is 2 weeks late" | shipping_delay | "I'm sorry for the delay! Could you share your order number?" |
| "I want to return this item" | return_policy | "We have a 30-day return policy. Items must be unused and in original packaging." |
| "I was charged twice" | billing_inquiry | "I'm sorry about the billing issue. Could you share your email or order ID?" |
| "Are you open on Sunday?" | store_hours | "Our support is available Monday to Friday, 9 AM to 6 PM IST." |
| "What is the premium plan price?" | product_pricing | "We have three plans: Basic ₹499, Standard ₹999, Premium ₹1999 per month." |
| "I want to talk to a human" | escalate_to_human | "I'll transfer you to our support team right away." |
| "ok got it" | chitchat | "Got it! Let me know if you have any other questions." |
| "Hello I need help" | greetings | "Hello! How can I help you today?" |

---


## 8. Project Structure

```
Ai-Customer-Support-Chatbot/
│
├── 📁 api/
│   └── app.py                    # Flask REST API — main application
│
├── 📁 data/
│   └── intents.json              # Knowledge base — 9 intents, 140 patterns
│
├── 📁 interface/
│   └── chat.html                 # Glassmorphism chat UI (served by Flask)
│
├── 📁 models/
│   ├── intent_classifier.pkl     # Trained Logistic Regression model
│   └── tfidf_vectorizer.pkl      # Fitted TF-IDF vectoriser (500 features)
│
├── 📁 notebooks/
│   ├── week1_setup_and_data.ipynb    # NLP pipeline + intents setup
│   └── week2_model_training.ipynb   # TF-IDF + model training (3 iterations)
│
├── nlp_processor.py              # Reusable NLPProcessor class
├── Procfile                      # Render deployment: gunicorn api.app:app
├── runtime.txt                   # Python version: python-3.11.9
├── requirements.txt              # All project dependencies
└── README.md                     # This file
```

---

## 9. Local Setup & Installation

### Prerequisites
- Python 3.11+
- Git

### Step 1 — Clone the Repository
```bash
git clone https://github.com/Shravani-1325/Ai-Customer-Support-Chatbot.git
cd Ai-Customer-Support-Chatbot
```

### Step 2 — Create Virtual Environment
```bash
# Windows
python -m venv myenv
myenv\Scripts\activate

# Mac/Linux
python3 -m venv myenv
source myenv/bin/activate
```

### Step 3 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Run the Flask API
```bash
python api/app.py
```

You should see:
```
Loading model and vectorizer...
All files loaded successfully ✅
 * Running on http://127.0.0.1:5000
```

### Step 5 — Open the Chat Interface
Visit `http://127.0.0.1:5000` in your browser — the chat UI loads directly.

Or open `interface/chat.html` directly in your browser (uses live Render API).

---

## 10. API Reference

### POST /chat
Send a customer message and receive an intent-classified response.

**Request:**
```json
POST /chat
Content-Type: application/json

{
  "message": "where is my order"
}
```

**Response (200 OK):**
```json
{
  "response": "Please share your order ID and I'll check the status for you.",
  "intent": "order_status",
  "confidence": 0.9421
}
```

**Error Response (400 Bad Request):**
```json
{
  "error": "Message field is missing or empty."
}
```

**Low Confidence Fallback (200 OK):**
```json
{
  "response": "I'm not quite sure I understood that. Could you rephrase your question?",
  "intent": "uncertain",
  "confidence": 0.2814
}
```

---

### GET /health
Check if the API server is running. Used by Render for uptime monitoring.

**Response (200 OK):**
```json
{
  "status": "ok",
  "message": "Chatbot API is running"
}
```

---

## 12. Deployment

The application is deployed on **Render** (free tier) at:

🌐 **https://novamind-chatbot.onrender.com**

### Deployment Stack
```
Platform  : Render Web Service (Free Tier)
Runtime   : Python 3.11.9
Server    : Gunicorn (production WSGI server)
Region    : Singapore
Branch    : main (auto-deploy on commit)
```

### Deployment Files

| File | Purpose |
|------|---------|
| `Procfile` | `web: gunicorn api.app:app` — tells Render how to start |
| `runtime.txt` | `python-3.11.9` — pins Python version |
| `requirements.txt` | All dependencies without strict version pins |

### ⚠️ Free Tier Note
Render's free tier spins down after 15 minutes of inactivity. The **first request after inactivity may take 30–60 seconds** to respond while the server wakes up. Subsequent requests are fast. This is a known free-tier limitation.

### Redeployment
Any push to the `main` branch triggers automatic redeployment via the "On Commit" auto-deploy setting.

---


## 13. Model Performance

### Training Summary

| Metric | Value |
|--------|-------|
| Algorithm | Logistic Regression |
| Training Samples | ~112 (80% of 140) |
| Test Samples | ~28 (20% of 140) |
| Accuracy | 75% |
| Weighted F1 Score | 0.70 |
| Intent Classes | 9 |
| Vectoriser | TF-IDF (500 features, ngram 1-2) |

### Model Selection Journey

| Iteration | Dataset | Naive Bayes | Logistic Regression | Winner |
|-----------|---------|-------------|--------------------:|--------|
| 1 — Initial 7 intents | 105 patterns |81% | 77% (class collapse) | Naive Bayes |
| 2 — Patterns rewritten | 140 patterns | 68% | **75%** | **Logistic Regression** |

Logistic Regression emerged as the winner after class balance was restored and overlapping vocabulary was removed from similar intents.


---

## 15. Author

<div align="center">

**Shravani More**  
B.Sc · St Francis De Sales College  
 
