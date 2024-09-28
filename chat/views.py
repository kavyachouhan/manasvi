import spacy
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render
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
        user_name = data.get("name", "User")

        # Track conversation state
        user_id = request.session.session_key  # Session key to identify users
        if user_id not in CONVERSATION_STATE:
            CONVERSATION_STATE[user_id] = {"step": 1, "name": ""}

        state = CONVERSATION_STATE[user_id]

        # Conversation flow based on state
        if state["step"] == 1:
            # Check if the user started with a greeting
            if any(greeting in user_input for greeting in GREETINGS):
                response_text = "Hello! I'm Manasvi, your mental health companion. What's your name?"
                state["step"] = 2
            else:
                response_text = "I'm here to help. You can start by saying 'Hi' or 'Hello'."
        
        elif state["step"] == 2:
            # Expecting a name
            if user_input:
                state["name"] = user_input
                response_text = f"Nice to meet you, {state['name']}! How are you feeling today?"
                state["step"] = 3
            else:
                response_text = "I didn't catch that. Could you please tell me your name?"

        elif state["step"] == 3:
            # Expecting user to talk about their feelings
            if "sad" in user_input or "depressed" in user_input or "bad" in user_input:
                response_text = "I'm sorry you're feeling down. Do you want to talk about what's been troubling you?"
                state["step"] = 4
            elif "happy" in user_input or "good" in user_input:
                response_text = f"That's great to hear, {state['name']}! What made you feel this way?"
                state["step"] = 4
            else:
                response_text = f"Can you tell me more about how you're feeling, {state['name']}?"
                state["step"] = 4

        elif state["step"] == 4:
            # Expecting the user to continue discussing feelings
            response_text = "Would you like to discuss more about this?"
            state["step"] = 5

        elif state["step"] == 5:
            # Expecting the user to decide if they want to continue
            if "yes" in user_input.lower() or "sure" in user_input.lower():
                response_text = "I'm here to listen. Please go ahead."
                state["step"] = 4  # Keep the conversation going
            elif "no" in user_input.lower():
                response_text = "Alright, I'm glad we had this chat. Feel free to reach out anytime!"
                state["step"] = 1  # Reset for a new conversation
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
