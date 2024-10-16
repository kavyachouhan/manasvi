import spacy
from textblob import TextBlob
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render
from django.contrib.sessions.models import Session
import json
from .models import UserFeedback
from django.db import connection
from django.core.cache import cache
from decouple import config
import google.generativeai as genai

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

genai.configure(api_key=config("API_KEY"))

# State for conversation flow tracking
CONVERSATION_STATE = {}

GREETINGS = ["hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening"]

def index(request):
    request.session.flush()  # Clear session on page load
    request.session.create()  # Create a new session
    return render(request, 'index.html')

@csrf_exempt
def chatbot_response(request):
    # If there is no session or it's the first load, create a new session
    if not request.session.session_key:
        request.session.create()  # Create a new session

    if request.method == "POST":
        data = json.loads(request.body)
        user_id = request.session.session_key  # Assign a unique session ID per device/browser
        user_input = data.get("message", "").lower().strip()
        user_name = request.session.get('name', 'User')  # Retrieve name from session
        
        # Analyze sentiment of the user's input
        sentiment_score = analyze_sentiment(user_input)

        # Track conversation state
        if user_id not in CONVERSATION_STATE:
            CONVERSATION_STATE[user_id] = {"step": 1, "name": ""}

        state = CONVERSATION_STATE[user_id]

        # Response logic based on sentiment score
        if state["step"] == 1:
            if any(greeting in user_input for greeting in GREETINGS):
                response_text = "Hello! I'm Manasvi, your mental health companion. What's your name?"
                state["step"] = 2
            else:
                response_text = "I'm here to help. You can start by saying 'Hi' or 'Hello'."
        
        elif state["step"] == 2:
            if user_input:
                state["name"] = user_input
                response_text = f"Nice to meet you, {state['name']}! How are you feeling today?"
                state["step"] = 3
            else:
                response_text = "I didn't catch that. Could you please tell me your name?"

        elif state["step"] == 3:
            # Check sentiment for more nuanced responses
            if sentiment_score < -0.2:  # Negative sentiment
                response_text = "I'm sorry to hear that you're feeling down. Would you like to talk about it?"
                state["step"] = 4
            elif sentiment_score > 0.2:  # Positive sentiment
                response_text = f"That's great to hear, {state['name']}! What made you feel this way?"
                state["step"] = 4
            else:  # Neutral sentiment
                response_text = f"Can you tell me more about how you're feeling, {state['name']}?"
                state["step"] = 4

        elif state["step"] == 4:
            response_text = "Would you like to discuss more about this?"
            state["step"] = 5

        elif state["step"] == 5:
            if "yes" in user_input.lower():
                response_text = "I'm here to listen. Please go ahead."
                state["step"] = 4  # Keep the conversation going
            elif "no" in user_input.lower():
                response_text = "Alright, I'm glad we had this chat. Feel free to reach out anytime!"
                state["step"] = 1  # Reset for a new conversation
            else:
                response_text = "I didn't quite understand. Would you like to discuss more about this? Please say 'yes' or 'no'."
            
         # Detect greeting and handle flow with OpenAI response if not a greeting
        if any(greeting in user_input for greeting in GREETINGS):
            response_text = "Hello! I'm Manasvi, your mental health companion. What's your name?"
        else:
            # Call OpenAI API for general queries
            response_text = get_genai_response(user_input)

        return JsonResponse({"response": response_text})

    return JsonResponse({"error": "Invalid request"}, status=400)

def get_genai_response(user_input):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(user_input)
        return response.text
    except Exception as e:
        print(f"Error calling GenAI: {e}")
        return "Sorry, I am having trouble processing your request at the moment."

def analyze_sentiment(text):
    analysis = TextBlob(text)
    # Get the polarity score
    return analysis.sentiment.polarity

@csrf_exempt
def collect_feedback(request):
    if request.method == "POST":
        data = json.loads(request.body)
        user_input = data.get("user_input")
        response_text = data.get("response")
        feedback = data.get("feedback")
        feedback_context = data.get("feedback_context", "General conversation")

        # Save feedback to the database
        UserFeedback.objects.create(
            user_name=request.session.get('user_name', 'Anonymous'),
            user_input=user_input,
            response=response_text,
            feedback=feedback,
            feedback_context=feedback_context
        )

        connection.close() 

        return JsonResponse({"message": "Feedback saved successfully!"})

def my_view(request):
    data = cache.get('my_key')
    if not data:
        data = UserFeedback.objects.all()  # Fetch data if not cached
        cache.set('my_key', data, timeout=60*15)  # Cache data for 15 minutes
    return render(request, 'your_template.html', {'data': data})