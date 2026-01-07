from flask import Flask, render_template, jsonify, request
from dotenv import load_dotenv
from flask_cors import CORS
import os
import traceback

# Set langchain global settings BEFORE importing langchain modules
# This prevents AttributeError when langchain modules try to access langchain.debug/verbose
from langchain_core.globals import set_verbose, set_debug
import langchain

# Set globals early
set_verbose(False)
set_debug(False)

# Set attributes directly on langchain module for backward compatibility
# Some older code may still try to access langchain.debug, langchain.verbose, or langchain.llm_cache
try:
    if not hasattr(langchain, "debug"):
        langchain.debug = False
    if not hasattr(langchain, "verbose"):
        langchain.verbose = False
    if not hasattr(langchain, "llm_cache"):
        langchain.llm_cache = None
except Exception:
    pass  # If setting fails, continue anyway

# Now import langchain modules after globals are set
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import Pinecone
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from src.prompt import *
import google.generativeai as genai

app = Flask(__name__)

CORS(app, 
     origins=[
         "http://localhost:8080",
         "http://127.0.0.1:8080",
         "http://localhost:5174", 
         "http://127.0.0.1:5173",  
         "http://192.168.100.4:5173",          
     ], 
     methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
     allow_headers=["Content-Type", "Authorization", "Access-Control-Allow-Origin"],
     supports_credentials=True)

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
OPEN_AI_KEY = os.environ.get('OPEN_AI_KEY')
GOOGLE_AI_KEY = os.environ.get('GOOGLE_AI_KEY')

GEMINI_MODEL = os.environ.get('GEMINI_MODEL', 'gemini-1.5-pro')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPEN_AI_KEY"] = OPEN_AI_KEY
os.environ["GOOGLE_AI_KEY"] = GOOGLE_AI_KEY

embeddings = download_hugging_face_embeddings()

index_name = "dentaink-123"

docsearch = Pinecone.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)


def get_working_gemini_model(api_key):
    """Try to find a working Gemini model by testing common model names."""
   
    genai.configure(api_key=api_key)
    
   
    model_names_to_try = [
        "gemini-pro",  
        "models/gemini-pro",  
        "gemini-1.0-pro",  
        "models/gemini-1.0-pro",
        "gemini-1.5-flash",
        "models/gemini-1.5-flash",
        "gemini-1.5-pro",
        "models/gemini-1.5-pro",
        "gemini-2.0-flash-exp",
        "models/gemini-2.0-flash-exp",
    ]
    
    print("Attempting to find a working Gemini model...")
    
    
    try:
        print("Listing available models...")
        available_models = []
        for model in genai.list_models():
            if 'generateContent' in model.supported_generation_methods:
                model_name = model.name.replace('models/', '')  # Remove models/ prefix
                available_models.append(model_name)
                print(f"  Found: {model_name}")
        
        if available_models:
           
            working_model = available_models[0]
            print(f"Using model: {working_model}")
            return working_model
    except Exception as e:
        print(f"Could not list models: {e}")
        print("Trying common model names...")
    
    
    for model_name in model_names_to_try:
        try:
            print(f"Testing model: {model_name}")
         
            test_model = genai.GenerativeModel(model_name)
          
            response = test_model.generate_content(
                "Hi", 
                generation_config={"max_output_tokens": 1}
            )
            print(f" Successfully connected to model: {model_name}")
            
            return model_name.replace('models/', '')
        except Exception as e:
            error_msg = str(e)
            
            if "404" in error_msg:
                print(f"  ✗ Model not found")
            else:
                print(f"  ✗ Failed: {error_msg[:80]}")
            continue
    
    
    print(f"Warning: Could not find a working model, using default: {GEMINI_MODEL}")
    return GEMINI_MODEL

# Get a working model
WORKING_MODEL = get_working_gemini_model(GOOGLE_AI_KEY)
print(f"\nFinal model selection: {WORKING_MODEL}\n")

llm = ChatGoogleGenerativeAI(
    model=WORKING_MODEL,
    temperature=0.4,
    google_api_key=GOOGLE_AI_KEY  
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 5})
system_prompt = (
    "You are a dental assistant AI specialized in answering oral health and dentistry questions. "
    "Use the following pieces of retrieved context to answer the question when applicable. "
    "If you don't know the answer based on the context, say that you don't know. "
    "Your answer must be a maximum of three sentences and be concise. "
    "CRITICAL INSTRUCTIONS: "
    "1. When a question is about a dental condition or problem, your answer must first state the best general treatment or management advice based on the provided context, then conclude by strongly recommending they visit a dentist for definitive diagnosis and personalized treatment. "
    "2. For greetings (hello, hi, etc.), respond warmly and invite oral health questions. "
    "3. For acknowledgments (okay, thanks, etc.), respond politely and encourage further questions. "
    "4. For positive feedback about solutions, acknowledge appreciation and reinforce dental visit importance."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),  
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy", "message": "Medical chatbot API is running"})

@app.route("/api/chat", methods=["POST"])
def api_chat():
    try:
        data = request.get_json()
        message = data.get("message", "")
        
        if not message:
            return jsonify({"error": "No message provided"}), 400
        
        print(f"Received question: {message}")
        
        response = rag_chain.invoke({"input": message})
        answer = response["answer"]
        
        print(f"Response: {answer}")
        
        return jsonify({"response": answer})
        
    except Exception as e:
        error_msg = str(e)
        error_traceback = traceback.format_exc()
        print(f"Error processing request: {error_msg}")
        print(f"Full traceback:\n{error_traceback}")
        return jsonify({"error": f"An error occurred: {error_msg}", "details": error_traceback}), 500

@app.route("/chat", methods=["POST"])
def chat():
    try:
        print(f"Received request to /chat from {request.remote_addr}")
        print(f"Request headers: {dict(request.headers)}")
        
        data = request.get_json()
        if not data:
            print("No JSON data received")
            return jsonify({"error": "No JSON data provided"}), 400
            
        message = data.get("message", "")
        
        if not message:
            return jsonify({"error": "No message provided"}), 400
        
        print(f"Received question: {message}")
        
        response = rag_chain.invoke({"input": message})
        answer = response["answer"]
        
        print(f"Response: {answer}")
        
        return jsonify({"response": answer})
        
    except Exception as e:
        error_msg = str(e)
        error_traceback = traceback.format_exc()
        print(f"Error processing request: {error_msg}")
        print(f"Full traceback:\n{error_traceback}")
        return jsonify({"error": f"An error occurred: {error_msg}", "details": error_traceback}), 500

@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json()
        message = data.get("message", "")
        
        if not message:
            return jsonify({"error": "No message provided"}), 400
        
        print(f"Received question: {message}")
        
        response = rag_chain.invoke({"input": message})
        answer = response["answer"]
        
        print(f"Response: {answer}")
        
        return jsonify({"response": answer})
        
    except Exception as e:
        error_msg = str(e)
        error_traceback = traceback.format_exc()
        print(f"Error processing request: {error_msg}")
        print(f"Full traceback:\n{error_traceback}")
        return jsonify({"error": f"An error occurred: {error_msg}", "details": error_traceback}), 500

@app.route("/<path:path>", methods=["GET", "POST", "OPTIONS"])
def catch_all(path):
    print(f"Received request to: /{path}")
    print(f"Method: {request.method}")
    print(f"Headers: {dict(request.headers)}")
    if request.method == "OPTIONS":
        return "", 200
    return jsonify({"error": f"Endpoint /{path} not found"}), 404

if __name__ == '__main__':
    print("Starting Flask server on http://localhost:8080")
    # print("Available endpoints:")
    # print("  - GET  /health")
    # print("  - POST /api/chat")
    # print("  - POST /chat") 
    # print("  - POST /ask")
    app.run(host='0.0.0.0', port=8080, debug=True)