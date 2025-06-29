from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
from flask_cors import CORS
import os

app = Flask(__name__)


CORS(app, 
     origins=[
         "http://localhost:5173", 
         "http://127.0.0.1:5173",  
         "http://192.168.100.4:5173",  
         "*" 
     ], 
     methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
     allow_headers=["Content-Type", "Authorization", "Access-Control-Allow-Origin"],
     supports_credentials=True)

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
OPEN_AI_KEY = os.environ.get('OPEN_AI_KEY')
GOOGLE_AI_KEY = os.environ.get('GOOGLE_AI_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPEN_AI_KEY"] = OPEN_AI_KEY
os.environ["GOOGLE_AI_KEY"] = GOOGLE_AI_KEY

embeddings = download_hugging_face_embeddings()

index_name = "medbot"

docsearch = PineconeVectorStore.from_existing_index(   
    index_name=index_name,
    embedding=embeddings
)

llm = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-flash-latest",     
    temperature=0.4,
    max_output_tokens=500,
    google_api_key=GOOGLE_AI_KEY  
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

system_prompt = (
    "You are an assistant for question-answering task."
    "Use the following pieces of retrieved context to answer"
    "the question, If you don't know the answer, say that you don't know"
    "use three sentences maximum and keep the answer concise."
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
        print(f"Error processing request: {str(e)}")
        return jsonify({"error": "An error occurred while processing your request"}), 500

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
        print(f"Error processing request: {str(e)}")
        return jsonify({"error": "An error occurred while processing your request"}), 500

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
        print(f"Error processing request: {str(e)}")
        return jsonify({"error": "An error occurred while processing your request"}), 500

# Add a catch-all route for debugging
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
    print("Available endpoints:")
    print("  - GET  /health")
    print("  - POST /api/chat")
    print("  - POST /chat") 
    print("  - POST /ask")
    app.run(host='0.0.0.0', port=8080, debug=True)