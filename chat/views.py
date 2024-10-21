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

# Load the spaCy model for NER
nlp = spacy.load("en_core_web_sm")

# State for conversation flow tracking
CONVERSATION_STATE = {}

# Expanded knowledge base with categories and dynamic responses
KNOWLEDGE_BASE = {
    "stress": {
        "general": "Stress is a common issue. Would you like tips on managing stress?",
        "work": "Work-related stress can be challenging. How do you usually cope with it?",
        "academic": "Academic stress can feel overwhelming. Do you want strategies to handle it better?"
    },
    "anxiety": {
        "general": "Anxiety can be tough. Here are some resources: [Anxiety Resources Link]. Would you like to learn coping mechanisms?",
        "social": "Social anxiety is difficult, but manageable. I can share tips on handling it in social situations.",
        "panic_attack": "If you're experiencing panic attacks, deep breathing exercises might help. Want to try them?"
    },
    "exercise": {
        "general": "Exercise is great for mental health. Would you like advice on how to get started?",
        "yoga": "Yoga can help reduce anxiety and improve mindfulness. Are you interested in a few simple yoga exercises?",
        "outdoor": "Outdoor activities like walking or cycling can boost mood. Would you like outdoor activity suggestions?"
    }
}

GREETINGS = ["hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening"]

# Function to extract a person's name from user input using spaCy
# Function to extract a person's name from user input using spaCy and rule-based methods
def extract_name(user_input):
    # Process the input through spaCy
    doc = nlp(user_input)
    
    # Check if any entities are recognized as a person's name (using PERSON entity label)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text
    
    # If no entity was recognized as a name, fallback to simple rule-based extraction
    if "my name is" in user_input.lower():
        return user_input.split("my name is")[-1].strip()
    elif "i am" in user_input.lower():
        return user_input.split("i am")[-1].strip()
    elif "it's" in user_input.lower():
        return user_input.split("it's")[-1].strip()
    
    # If the input is a single word (likely a name), return it
    words = user_input.split()
    if len(words) == 1:
        return words[0].strip()

    # If no name is found, return None
    return None

# Function to handle user input based on expanded knowledge base
def handle_knowledge_base(user_input):
    # Search for keywords in the knowledge base and return dynamic responses
    for category, subtopics in KNOWLEDGE_BASE.items():
        if category in user_input:
            if "work" in user_input:
                return subtopics.get("work")
            elif "academic" in user_input:
                return subtopics.get("academic")
            elif "social" in user_input:
                return subtopics.get("social")
            elif "panic" in user_input:
                return subtopics.get("panic_attack")
            elif "yoga" in user_input:
                return subtopics.get("yoga")
            elif "outdoor" in user_input:
                return subtopics.get("outdoor")
            else:
                return subtopics.get("general")
    return None

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

        # Handle knowledge base responses first
        knowledge_response = handle_knowledge_base(user_input)
        if knowledge_response:
            return JsonResponse({"response": knowledge_response})

        # Response logic based on sentiment score
        if state["step"] == 1:
            if any(greeting in user_input for greeting in GREETINGS):
                response_text = "Hello! I'm Manasvi, your mental health companion. What's your name?"
                state["step"] = 2
            else:
                response_text = "I'm here to help. You can start by saying 'Hi' or 'Hello'."
        
        elif state["step"] == 2:
            # Extract name using the extract_name function
            extracted_name = extract_name(user_input)
            if extracted_name:
                state["name"] = extracted_name
                response_text = f"Nice to meet you, {state['name']}! How are you feeling today?"
                state["step"] = 3
            else:
                response_text = "I didn't catch that. Could you please tell me your name?"

        elif state["step"] == 3:
            # Check sentiment for more nuanced responses
            if sentiment_score < -0.5:  # Critical negative sentiment
                response_text = "It sounds like you're going through a tough time. It's important to talk to someone. Would you like resources on how to get help?"
                state["step"] = 4
            elif sentiment_score < -0.2:  # Negative sentiment
                response_text = "I'm sorry to hear that you're feeling down. Would you like to talk about it?"
                state["step"] = 4
            elif sentiment_score > 0.2:  # Positive sentiment
                response_text = f"That's great to hear, {state['name']}! What made you feel this way?"
                state["step"] = 4
            else:  # Neutral sentiment
                response_text = f"Can you tell me more about how you're feeling, {state['name']}?"
                state["step"] = 4

        elif state["step"] == 4:
            # Adding contextual follow-up questions for more complex conversations
            response_text = "Would you like to discuss more about this, or are you looking for advice on managing your feelings?"
            state["step"] = 5

        elif state["step"] == 5:
            if "yes" in user_input.lower():
                response_text = "I'm here to listen. Please go ahead."
                state["step"] = 4  # Keep the conversation going
            elif "no" in user_input.lower():
                response_text = "Alright, I'm glad we had this chat. If you need help, don't hesitate to ask."
                state["step"] = 1  # Reset for a new conversation
            else:
                response_text = "I didn't quite understand. Would you like to discuss more about this? Please say 'yes' or 'no'."

        return JsonResponse({"response": response_text})

    return JsonResponse({"error": "Invalid request"}, status=400)

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

