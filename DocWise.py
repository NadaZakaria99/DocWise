import base64
import requests
import gradio as gr
import PyPDF2
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer, util
import numpy as np
import os
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.documents import Document

# Configure Google Generative AI
api_key = "Replace with your actual API key"  
genai.configure(api_key=api_key)

# Create the Gemini model
generation_config = {
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 65536,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-thinking-exp-01-21",
    generation_config=generation_config,
)

chat_session = model.start_chat(history=[])

# Function to extract text from a PDF
def extract_text_from_pdf(file_path):
    try:
        with open(file_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            text = "".join(page.extract_text() for page in reader.pages)
        return text
    except Exception as e:
        return f"Error extracting text from PDF: {e}"

# Function to chunk the text
def chunk_text(text, chunk_size=500, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Function to embed the chunks
def embed_chunks(chunks, model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks, convert_to_tensor=True)
    return embeddings, model

# Function to retrieve relevant chunks
def retrieve_relevant_chunks(query, chunks, embeddings, model, top_k=3):
    query_embedding = model.encode(query, convert_to_tensor=True)
    similarities = util.cos_sim(query_embedding, embeddings)[0]
    top_k = min(top_k, len(chunks))
    top_indices = np.argsort(similarities.cpu().numpy())[-top_k:][::-1]
    relevant_chunks = [chunks[i] for i in top_indices]
    return relevant_chunks

# Function to summarize the agreement using Gemini
def summarize_agreement_with_gemini(text):
    try:
        # Create a prompt for summarization
        prompt = f"Summarize the following text in 3-5 sentences:\n\n{text}\n\nSummary:"
        
        # Send the prompt to the Gemini model
        response = chat_session.send_message(prompt)
        
        return response.text
    except Exception as e:
        return f"Error summarizing text with Gemini: {e}"

# Configure Tavily API
os.environ["TAVILY_API_KEY"] = "Replace with your Tavily API"
web_search_tool = TavilySearchResults(k=3)
def generate_response_with_rag(query, pdf_path, state):
    if "chunks" not in state or "embeddings" not in state or "embedding_model" not in state:
        text = extract_text_from_pdf(pdf_path)
        chunks = chunk_text(text)
        embeddings, embedding_model = embed_chunks(chunks)
        state["chunks"] = chunks
        state["embeddings"] = embeddings
        state["embedding_model"] = embedding_model
    else:
        chunks = state["chunks"]
        embeddings = state["embeddings"]
        embedding_model = state["embedding_model"]

    # Retrieve relevant chunks based on the query
    relevant_chunks = retrieve_relevant_chunks(query, chunks, embeddings, embedding_model, top_k=5)  # Increase top_k
    
    # Debug: Print relevant chunks
    print(f"Relevant Chunks: {relevant_chunks}")

    # Combine the relevant chunks into a single context
    context = "\n\n".join(relevant_chunks)

    # Debug: Print the context
    print(f"Context from PDF: {context}")

    # Create a prompt that instructs the model to answer only from the context
    prompt = f"""
    You are a helpful assistant that answers questions based on the provided context. 
    Use the context below to answer the question. If the context does not contain enough information to answer the question, respond with "I don't know."
    **Context:**
    {context}
    **Question:**
    {query}
    **Answer:**
    """

    # Debug: Print the prompt
    print(f"Prompt for Gemini: {prompt}")

    # Send the prompt to the Gemini model
    try:
        response = chat_session.send_message(prompt)
        initial_answer = response.text

        # Check if the initial answer is "I don't know"
        if "I don't know" in initial_answer or "i don't know" in initial_answer:
            print("Initial answer is 'I don't know'. Performing web search...")
            docs = web_search_tool.invoke({"query": query})
            web_results = "\n".join([d["content"] for d in docs])
            web_results = Document(page_content=web_results)
            
            # Debug: Print web search results
            print(f"Web Search Results: {web_results.page_content}")

            # Create a prompt that instructs the model to answer from the web search results
            web_prompt = f"""
            You are a helpful assistant that answers questions based on the provided context. 
            The context below is from a web search. Use the context to answer the question. If the context does not contain enough information to answer the question, respond with "I don't know."
            
            **Context:**
            {web_results.page_content}
            **Question:**
            {query}
            **Answer:**
            """
            
            # Debug: Print the prompt
            print(f"Prompt for Gemini (Web Search): {web_prompt}")

            # Send the prompt to the Gemini model
            web_response = chat_session.send_message(web_prompt)
            # Add a note indicating the answer is based on a web search
            return f"{web_response.text}\n\n*Note: This answer is based on a web search.*"
        else:
            return initial_answer
    except Exception as e:
        return f"Error generating response: {e}"
# Function to send document to DocuSign
def send_to_docusign(file_path, recipient_email, recipient_name):
    docusign_api_key = "Replace with your access_token"
    account_id = "Replace with your account_id"
    base_url = "Replace with your base_url"

    with open(file_path, "rb") as file:
        document_base64 = base64.b64encode(file.read()).decode()

    envelope_definition = {
        "emailSubject": "Please sign this document",
        "documents": [
            {
                "documentId": "1",
                "name": "document.pdf",
                "fileExtension": "pdf",
                "documentBase64": document_base64
            }
        ],
        "recipients": {
            "signers": [
                {
                    "email": recipient_email,
                    "name": recipient_name,
                    "recipientId": "1",
                    "tabs": {
                        "signHereTabs": [
                            {
                                "documentId": "1",
                                "pageNumber": "1",
                                "xPosition": "100",
                                "yPosition": "100"
                            }
                        ]
                    }
                }
            ]
        },
        "status": "sent"
    }

    headers = {
        "Authorization": f"Bearer {docusign_api_key}",
        "Content-Type": "application/json"
    }
    try:
        response = requests.post(
            f"{base_url}/v2.1/accounts/{account_id}/envelopes",
            headers=headers,
            json=envelope_definition
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

# Function to process the agreement
def process_agreement(file, recipient_email, recipient_name, state):
    try:
        text = extract_text_from_pdf(file.name)
        if text.startswith("Error"):
            return text, {}, {}, state

        # Use Gemini for summarization
        summary = summarize_agreement_with_gemini(text)
        if summary.startswith("Error"):
            return summary, {}, {}, state

        docusign_response = send_to_docusign(file.name, recipient_email, recipient_name)
        if "error" in docusign_response:
            return summary, {}, docusign_response, state

        return summary, {}, docusign_response, state
    except Exception as e:
        return f"Error: {e}", {}, {}, state

# Gradio interface
def main_interface(file, recipient_email, recipient_name, question, state):
    if file is not None:
        state["file"] = file
        state["text"] = extract_text_from_pdf(file.name)
        state["chat_history"] = []  # Initialize chat history

    summary_output = ""
    docusign_output = {}
    chatbot_output = ""

    if "file" in state:
        if recipient_email and recipient_name:
            summary_output, _, docusign_output, state = process_agreement(state["file"], recipient_email, recipient_name, state)

        if question:
            chatbot_output = generate_response_with_rag(question, state["file"].name, state)
            state["chat_history"].append((question, chatbot_output))  # Update chat history

    return summary_output, docusign_output, chatbot_output, state

# CSS for styling
css = """
.gradio-container {
    background-image: url('https://huggingface.co/spaces/Nadaazakaria/DocWise/resolve/main/DALL%C2%B7E%202025-01-26%2011.43.33%20-%20A%20futuristic%20and%20sleek%20magical%20animated%20GIF-style%20icon%20design%20for%20%27DocWise%27%2C%20representing%20knowledge%2C%20documents%2C%20and%20wisdom.%20The%20design%20includes%20a%20glow.jpg');
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}
.gradio-container h1, 
.gradio-container .tabs > .tab-nav > .tab-button {
    color: #FFF5E1 !important;
    text-shadow: 0 0 5px rgba(255, 245, 225, 0.5);
}
.tabs {
    background-color: #f0f0f0 !important;
    border-radius: 10px !important;
    padding: 20px !important;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
}
.tabs > .tab-nav {
    background-color: #e0e0e0 !important;
    border-radius: 5px !important;
    margin-bottom: 15px !important;
}
.tabs > .tab-nav > .tab-button {
    color: black !important;
    font-weight: bold !important;
}
.tabs > .tab-nav > .tab-button.selected {
    background-color: #d0d0d0 !important;
    color: black !important;
}
#process-button, #chatbot-button {
    background-color: white !important;
    color: black !important;
    border: 1px solid #ccc !important;
    padding: 10px 20px !important;
    border-radius: 5px !important;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1) !important;
    transition: background-color 0.3s ease !important;
}
#process-button:hover, #chatbot-button:hover {
    background-color: #f0f0f0 !important;
}
"""

# Gradio app
with gr.Blocks(css=css) as app:
    gr.Markdown(
        """
    <div style="text-align: center;">
        <h1 id="main-title">
            DocWise(Agreement Analyzer with Chatbot and Docusign Integration)
        </h1>
    </div>
        """,
    )

    state = gr.State({})
    file_input = gr.File(label="Upload Agreement (PDF)")

    with gr.Tab("Agreement Processing", elem_id="agreement-tab"):
        email_input = gr.Textbox(label="Recipient Email")
        name_input = gr.Textbox(label="Recipient Name")
        summary_output = gr.Textbox(label="Agreement Summary")
        docusign_output = gr.JSON(label="DocuSign Response")
        process_button = gr.Button("Process Agreement", elem_id="process-button")

    with gr.Tab("Chatbot", elem_id="chatbot-tab"):
        chatbot_question_input = gr.Textbox(label="Ask a Question")
        chatbot_answer_output = gr.Textbox(label="Answer")
        chatbot_button = gr.Button("Ask", elem_id="chatbot-button")

    process_button.click(
        main_interface,
        inputs=[file_input, email_input, name_input, chatbot_question_input, state],
        outputs=[summary_output, docusign_output, chatbot_answer_output, state]
    )
    chatbot_button.click(
        main_interface,
        inputs=[file_input, email_input, name_input, chatbot_question_input, state],
        outputs=[summary_output, docusign_output, chatbot_answer_output, state]
    )

app.launch(debug=True)
