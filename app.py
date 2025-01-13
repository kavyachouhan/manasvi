# app.py
import os
import random
import warnings
import json
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain.chains import ConversationalRetrievalChain, ConversationChain
from langchain.memory import ConversationSummaryMemory, ConversationBufferMemory, ConversationEntityMemory
import re
import psycopg2
from psycopg2.extras import DictCursor
from datetime import datetime

load_dotenv()
warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)

# Global chat history to maintain conversation context
context = []

DATABASE_URL = os.environ.get('DATABASE_URL')

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
            "Goodbye! Take care.",
            "I’m glad we could talk. Wishing you well until next time.",
            "See you soon! Stay well."
        ]
    }
}

def find_empathetic_response(message):
    # Lowercase and strip leading/trailing spaces
    message = message.lower().strip()
    
    for category, keywords in initial_empathy_responses['keywords'].items():
        category_words = category.split(',')

        for word in category_words:
            # Match only if the word appears at the start or as a standalone word
            pattern = rf'^(?:{re.escape(word)}[\s\.,!?]*|.*\b{re.escape(word)}\b\s*[\.,!?]?$)'
            
            if re.search(pattern, message):
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
        ip_address = request.remote_addr

        # Check if the user's message matches a rule-based response
        matched_response = find_empathetic_response(user_message)

        # If a rule-based empathetic response is found, return it
        if matched_response:
            try:
                conn = psycopg2.connect(DATABASE_URL, sslmode='require')
                cur = conn.cursor()
                cur.execute(
                    'INSERT INTO chat_messages (ip_address, user_message, bot_response, timestamp) VALUES (%s, %s, %s, %s)',
                    (ip_address, user_message, matched_response, datetime.now().isoformat())
                )
                conn.commit()
                cur.close()
                conn.close()
            except psycopg2.Error as e:
                print("Failed to connect to PostgreSQL:", e)
                return jsonify({'status': 'error', 'message': 'Database connection failed. Please try again later.'}), 500
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
            index_name="pdf-vectorized",
            embedding=embeddings
        )

        SYSTEM_PROMPT = """You are Manasvi, a compassionate mental health companion. Follow these guidelines:

        1. Conversation Style:
        - Address users by name and maintain a warm, personal tone
        - Mirror the user's emotional state without explicitly naming it
        - Keep responses conversational and natural, 1 sentence only
        - Share relevant personal anecdotes when appropriate to build connection

        2. Response Structure:
        - Acknowledge user's feelings and experiences in different contexts
        - Don't be repetitive; vary responses to show active listening
        - Personalize responses based on user's input and emotional state
        - Include one thoughtful follow-up question that builds on user's sharing

        3. Active Listening:
        - Note recurring themes in user's messages
        - Reference previous conversations to show continuity
        - Ask specific questions about mentioned experiences
        - Validate feelings through reflection rather than generic statements

        4. Example Responses:
        Bad: "I understand you're feeling anxious, [name]. Have you tried meditation?"
        Good: "Those racing thoughts sound exhausting, Sarah. What helps you feel most grounded when they start?"

        Bad: "Here's what the research says about depression..."
        Good: "You mentioned feeling low lately, Tom. How has this been affecting your daily routine?"


        Current conversation history: {history}
        User message: {input}

        Your Developer is Kavya Chouhan.

        Respond as Manasvi, focusing on building genuine connection while maintaining professionalism and brevity."""

        prompt_template = PromptTemplate(
            template=SYSTEM_PROMPT,
            input_variables=["history", "input"]
        )

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            temperature=0.5,
            max_tokens=None,
        )

        # repo_id = 'mistralai/Mistral-7B-Instruct-v0.3'
        # llm = HuggingFaceEndpoint(
        #     repo_id=repo_id,
        #     huggingfacehub_api_token=os.environ.get("HUGGINGFACEHUB_API_TOKEN"),
        #     temperature=0.7,
        #     model_kwargs={'max_length': 128}
        # )
        
        # qa = ConversationalRetrievalChain.from_llm(
        #     llm=llm,
        #     chain_type="stuff",
        #     retriever=vectorstore.as_retriever(),
        #     combine_docs_chain_kwargs={
        #         'prompt': prompt_template
        #     },
        #     memory = ConversationSummaryMemory(llm=llm, memory_key="chat_history", return_messages=True)
        # )

        # res = qa({
        #     "question": user_message,
        #     # "chat_history": context 
        # })

        conversation = ConversationChain(
            llm=llm,
            memory=ConversationSummaryMemory(llm=llm, memory_key="history", return_messages=True),
            prompt=prompt_template,
            verbose=False,
        )

        response = conversation.predict(input=user_message)


        history = (user_message, response)
        context.append(history)

        conn = psycopg2.connect(DATABASE_URL, sslmode='require')
        cur = conn.cursor()
        cur.execute(
            'INSERT INTO chat_messages (ip_address, user_message, bot_response, timestamp) VALUES (%s, %s, %s, %s)',
            (ip_address, user_message, response, datetime.now().isoformat())
        )
        conn.commit()
        cur.close()
        conn.close()

        return jsonify({
            'status': 'success',
            'message': response,
            'type': 'ai_response'
        })
    
    except Exception as ai_error:
        print("Unable to connect", ai_error)
        error_responses = [
            "I'm having trouble connecting at the moment. Would you like to try again?",
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

        conn = psycopg2.connect(DATABASE_URL, sslmode='require')
        cur = conn.cursor()
        cur.execute(
            'INSERT INTO feedback (feedback_type, rating, timestamp) VALUES (%s, %s, %s)',
            (feedback_type, rating, datetime.now().isoformat())
        )
        conn.commit()
        cur.close()
        conn.close()
        
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