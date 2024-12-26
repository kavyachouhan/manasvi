# app.py
import os
import random
import warnings
import json
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

load_dotenv()
warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)

# Global chat history to maintain conversation context
chat_history = []

# Define rule-based responses
initial_empathy_responses = {
    "keywords": {
        "hello": [
            "Hi there! I'm Manasvi, your mental health companion. How would you like me to call you?",
            "Hey, I’m Manasvi. Before we begin, may I know your name?",
            "Hello, I’m Manasvi. What should I call you?"
        ],
        "hi": [
            "Hi there! I'm Manasvi, your mental health companion. How would you like me to call you?",
            "Hey, I’m Manasvi. Before we begin, may I know your name?",
            "Hello, I’m Manasvi. What should I call you?"
        ],
        "hey": [
            "Hi there! I'm Manasvi, your mental health companion. How would you like me to call you?",
            "Hey, I’m Manasvi. Before we begin, may I know your name?",
            "Hello, I’m Manasvi. What should I call you?"
        ],
        "good morning": [
            "Good morning! I’m Manasvi. Before we start, may I ask your name?",
            "Good morning! I’m Manasvi. What's your name?",
            "Good morning! I’m Manasvi. I would love to know your name."
        ],
        "good afternoon": [
            "Good afternoon! I’m Manasvi. Before we start, may I ask your name?",
            "Good afternoon! I’m Manasvi. May I know your name before we continue?",
            "Good afternoon! I’m Manasvi. I would love to know your name."
        ],
        "good evening": [
            "Good evening! I’m Manasvi. Before we start, may I ask your name?",
            "Good evening! I’m Manasvi. Could you share your name with me?",
            "Good evening! I’m Manasvi. I would love to know your name."
        ],
        "bye": [
            "Thank you for sharing with me today. Take care and be kind to yourself.",
            "I’m glad we could talk. Wishing you well until next time.",
            "You’re not alone, and I’ll be here if you’d like to connect again. Take care."
        ]
    }
}

def find_empathetic_response(message):
    for category, keywords in initial_empathy_responses['keywords'].items():
        if any(keyword in message.lower() for keyword in category.split(',')):
            return random.choice(keywords)
    return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        # Parse incoming JSON data
        data = request.get_json()
        user_message = data.get('message', '')

        # Check if the user's message matches a rule-based response
        matched_response = find_empathetic_response(user_message)

        # If a rule-based empathetic response is found, return it
        if matched_response:
            return jsonify({
                'status': 'success',
                'message': matched_response,
                'type': 'empathetic_response'
            })

        # If no rule-based response, use the Gemini model
        embeddings = GoogleGenerativeAIEmbeddings(
            google_api_key=os.environ.get("GOOGLE_API_KEY"),
            model="models/embedding-001"
        )
        vectorstore = PineconeVectorStore(
            index_name=os.environ["INDEX_NAME"],
            embedding=embeddings
        )

        chat = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            temperature=0.5,
            max_tokens=None,
        )

        qa = ConversationalRetrievalChain.from_llm(
            llm=chat,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(),
            memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        )

        res = qa({
            "question": user_message,
            "chat_history": chat_history
        })

        history = (user_message, res['answer'])
        chat_history.append(history)

        return jsonify({
            'status': 'success',
            'message': res['answer'],
            'type': 'ai_response'
        })
    
    except Exception as ai_error:
        # Compassionate Error Handling
        error_responses = [
            "Something feels a bit off right now. Your feelings are still important to me.",
            "I'm having trouble connecting at the moment. Would you like to try again?",
            "Let's take a gentle pause. Would you like to reshare your thoughts?"
        ]
        
        return jsonify({
            'status': 'error',
            'message': random.choice(error_responses),
            'error_details': str(ai_error)
        }), 400

@app.route('/feedback', methods=['POST'])
def submit_feedback():
    try:
        # Parse JSON data
        data = request.get_json()
        
        # Extract feedback type from the JSON data
        feedback_type = data.get('feedbackType')
        
        # Map feedback type to a rating
        rating_map = {
            'thumbUp': 5,  # Positive feedback
            'thumbDown': 1  # Negative feedback
        }
        
        message_map = {
            'thumbUp': 'Thank you for your positive feedback!',
            'thumbDown': 'Thank you for your feedback. We will work on improving.'
        }
        
        feedback_message = message_map.get(feedback_type, 'Thank you for your feedback!')
        
        # Get the corresponding rating, default to 3 if not found
        rating = rating_map.get(feedback_type, 3)
        
        print(f"Feedback Received: Type {feedback_type}, Rating {rating}")
        
        return jsonify({
            'status': 'success',
            'message': feedback_message
        })
    
    except json.JSONDecodeError:
        # Handle JSON parsing errors
        return jsonify({'status': 'error', 'message': 'Invalid JSON'}), 400
    
    except Exception as e:
        # Catch any other unexpected errors
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)