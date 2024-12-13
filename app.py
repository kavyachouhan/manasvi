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
        "lonely": [
            "I understand loneliness can feel like a heavy cloud. You're not alone, even if it feels that way right now.",
            "Feeling isolated is painful. Your courage in sharing this means so much.",
            "Loneliness doesn't define you. Would you like to explore how you're feeling?"
        ],
        "anxiety": [
            "Anxiety can make the world feel overwhelming. Your feelings are valid and important.",
            "I hear the weight you're carrying. Let's take a moment to breathe together.",
            "Anxiety doesn't have to control your story. Would you like to talk through what you're experiencing?"
        ],
        "depressed": [
            "Depression can make everything feel colorless. Your pain is real, and you matter.",
            "Living with depression takes immense strength. I'm here, truly listening.",
            "Some days are harder than others. Would you like to share what's been difficult?"
        ],
        "stressed": [
            "Stress can be overwhelming. Would you like to explore what's causing you to feel this way?",
            "I'm hearing that you're under a lot of pressure. Can you tell me more about what's going on?",
            "It sounds like you're carrying a heavy load. I'm here to listen and support you."
        ],
        "heartbreak": [
            "Heartbreak is one of the most painful experiences. Would you like to share what happened?",
            "I'm so sorry you're going through this. Breakups can feel like an emotional earthquake.",
            "Healing takes time, and your feelings are completely valid. Would you like to talk about it?"
        ],
        "angry": [
            "Anger is a powerful emotion. Would you like to share what's making you feel this way?",
            "It sounds like you're carrying a lot of frustration. I'm here to listen without judgment.",
            "Your anger is valid. Can you tell me more about what's bringing up these feelings?"
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
            model="gemini-1.5-flash",
            temperature=0.6,
            max_tokens=None,
            system_instruction=f"""
            You are Maanasvi, a compassionate mental health companion.

            Emotional Support Principles:
            - Prioritize emotional validation
            - Create a safe, non-judgmental space
            - Respond with deep empathy and understanding
            - Focus on the human experience
            - Offer gentle, supportive guidance

            Communication Guidelines:
            - Listen actively and reflect feelings
            - Ask thoughtful, open-ended questions
            - Avoid giving direct medical advice
            - Maintain a warm, human-like tone
            - Respect the user's emotional boundaries

            Core Purpose:
            To provide a supportive, understanding presence 
            that helps users feel heard, validated, and 
            less alone in their emotional journey.
            """
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
        
        # Get the corresponding rating, default to 3 if not found
        rating = rating_map.get(feedback_type, 3)
        
        # In Flask, you'll need to handle database operations differently
        # This is a placeholder - replace with your actual database logic
        print(f"Feedback Received: Type {feedback_type}, Rating {rating}")
        
        return jsonify({'status': 'success'})
    
    except json.JSONDecodeError:
        # Handle JSON parsing errors
        return jsonify({'status': 'error', 'message': 'Invalid JSON'}), 400
    
    except Exception as e:
        # Catch any other unexpected errors
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)