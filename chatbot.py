import os
import fitz  # PyMuPDF is for extracting text from PDFs
from flask import Flask, request, jsonify
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

class InterviewChatbot:
    def __init__(self):
        self.app = Flask(__name__)
        self.groq_api_key = os.getenv("groq_api_key")   # Secure API key storage
        if not self.groq_api_key:
            raise ValueError("Groq API key is missing. Set the 'GROQ_API_KEY' environment variable.")
        
        self.chain = self.initialize_llm()
        self.interviews = {}  # Stores scheduled interviews
        self.setup_routes()

    def initialize_llm(self):
        """Initializes the LLM for interview scheduling and summarization."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful and intelligent assistant. Please provide accurate and well-structured responses to user queries in a clear and professional manner. "
                       "You are a text summarization assistant. Your task is to provide a concise, clear, and informative summary of the following text. Keep the key points intact while removing unnecessary details."),
            ("user", "Question: {question}")
        ])
        
        llm = ChatGroq(groq_api_key=self.groq_api_key, model_name="Llama3-8b-8192")
        output_parser = StrOutputParser()
        
        return prompt | llm | output_parser

    def generate_response(self, question):
        """Generates a response using the LLM chain."""
        return self.chain.invoke({'question': question})

    
    def summarize_pdf(self):
        """Extracts text from an uploaded PDF file and summarizes it."""
        try:
            if 'file' not in request.files:
                return jsonify({"error": "No file uploaded."}), 400
            
            file = request.files['file']
            if not file.filename.endswith('.pdf'):
                return jsonify({"error": "Invalid file format. Please upload a PDF file."}), 400

            # Extract text from PDF
            pdf_text = self.extract_text_from_pdf(file)

            if not pdf_text.strip():
                return jsonify({"error": "No extractable text found in the PDF."}), 400

            # Generate summary
            summary_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a text summarization assistant. Provide a concise summary of the given text."),
                ("user", "Text: {text}")
            ])
            
            summary_chain = summary_prompt | self.chain
            summary = summary_chain.invoke({'text': pdf_text})
            
            return jsonify({"summary": summary})
        
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    def extract_text_from_pdf(self, file):
        """Extracts text from a PDF file."""
        try:
            text = ""
            with fitz.open(stream=file.read(), filetype="pdf") as doc:
                for page in doc:
                    text += page.get_text("text") + "\n"
            return text.strip()
        except Exception as e:
            return f"Error extracting text from PDF: {str(e)}"

    def chat(self):
        """Handles the /questions endpoint."""
        try:
            data = request.get_json()
            question = data.get("question")  # Ensure correct key name

            if not question:
                return jsonify({"error": "Question is required."}), 400

            response = self.chain.invoke({"question": question})  # Correct format
            return jsonify({"response": response})
        except Exception as e:
            return jsonify({"error": str(e)}), 500


    def setup_routes(self):
        """Sets up Flask routes."""
        self.app.add_url_rule("/", "home", lambda: jsonify({"message": "Interview Chatbot is running!"}), methods=['GET'])
        self.app.add_url_rule("/summarize", "summarize_pdf", self.summarize_pdf, methods=['POST'])
        self.app.add_url_rule("/question", "question", self.chat, methods=['POST'])


    def run(self):
        """Runs the Flask application."""
        self.app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)

if __name__ == "__main__":
    chatbot = InterviewChatbot()
    chatbot.run()
