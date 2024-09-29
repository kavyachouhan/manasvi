import spacy
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render
from django.contrib.sessions.models import Session
import json
from .models import UserFeedback

# Load the spaCy model
nlp = spacy.load("en_core_web_md")

# State for conversation flow tracking
CONVERSATION_STATE = {}

GREETINGS = ["hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening"]

@csrf_exempt
def chatbot_response(request):
    if request.method == "POST":
        data = json.loads(request.body)
        user_input = data.get("message", "").lower().strip()

        # Use session to track the step and name
        step = request.session.get('step', 1)
        name = request.session.get('name', "User")  # Get name from session or default to "User"

        # Conversation flow based on state
        if step == 1:
            if any(greeting in user_input for greeting in GREETINGS):
                response_text = "Hello! I'm Manasvi, your mental health companion. What's your name?"
                request.session['step'] = 2
            else:
                response_text = "I'm here to help. You can start by saying 'Hi' or 'Hello.'"
        
        elif step == 2:
            if user_input:
                request.session['name'] = user_input
                response_text = f"Nice to meet you, {request.session['name']}! How are you feeling today?"  # Use the name stored in session
                request.session['step'] = 3
            else:
                response_text = "I didn't catch that. Could you please tell me your name?"

        elif step == 3:
            if "sad" in user_input or "depressed" in user_input or "bad" in user_input:
                response_text = "I'm sorry you're feeling down. Do you want to talk about what's been troubling you?"
                request.session['step'] = 4
            elif "happy" in user_input or "good" in user_input:
                response_text = f"That's great to hear, {request.session['name']}! What made you feel this way?"  # Use the name stored in session
                request.session['step'] = 4
            else:
                response_text = f"Can you tell me more about how you're feeling, {request.session['name']}?"
                request.session['step'] = 4
        elif step == 4:
            response_text = "Would you like to discuss more about this?"
            request.session['step'] = 5

        elif step == 5:
            if "yes" in user_input.lower() or "sure" in user_input.lower():
                response_text = "I'm here to listen. Please go ahead."
                request.session['step'] = 4  # Keep the conversation going
            elif "no" in user_input.lower():
                response_text = "Alright, I'm glad we had this chat. Feel free to reach out anytime!"
                request.session['step'] = 1  # Reset for a new conversation
            else:
                response_text = "I didn't quite understand. Would you like to discuss more about this? Please say 'yes' or 'no'."

        return JsonResponse({"response": response_text})

    return JsonResponse({"error": "Invalid request"}, status=400)


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

        return JsonResponse({"message": "Feedback saved successfully!"})

def index(request):
    return render(request, 'index.html')
