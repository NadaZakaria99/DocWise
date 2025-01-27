import base64
import requests
import gradio as gr
import PyPDF2
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer, util
import numpy as np

# Configure Google Generative AI
api_key = "AIzaSyAt9uhJV-dIwqdndTz5V4K8ixW7__0C7Ao"  # Replace with your actual API key
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

# Function to generate response using RAG
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

    relevant_chunks = retrieve_relevant_chunks(query, chunks, embeddings, embedding_model)
    context = "\n\n".join(relevant_chunks)
    augmented_query = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    response = chat_session.send_message(augmented_query)
    return response.text

# Function to send document to DocuSign
def send_to_docusign(file_path, recipient_email, recipient_name):
    docusign_api_key = "eyJ0eXAiOiJNVCIsImFsZyI6IlJTMjU2Iiwia2lkIjoiNjgxODVmZjEtNGU1MS00Y2U5LWFmMWMtNjg5ODEyMjAzMzE3In0.AQoAAAABAAUABwCAceXAcD7dSAgAgLEIz7M-3UgCAOlO8Y4SurZKuV6S5XinyP8VAAEAAAAYAAIAAAAFAAAAHQAAAA0AJAAAADAwZTc4NzI1LTI5NzItNDkwMS1iOWQxLThhYWY3MjllN2JmZCIAJAAAADAwZTc4NzI1LTI5NzItNDkwMS1iOWQxLThhYWY3MjllN2JmZDAAgEY3U2w-3Ug3ANsGGOe0MuhFnvTaUOPDR9o.DzQmnFZZpXlwhKPpoNm64DXNbb6fKYElmTgE1k4Nxz2vJCgSpVzEYcen9U6th5rjh2J_HuUWQR2tjHT8IMoh4q-u93LhCMkhvb9_E7bEfwpO5m2-yR6jXsOEZBvCO8qhdVyU23SA0A1vDyVHOr1QUaSLk7ldgmRUz0vWp1W-L6lPCMDfE4uNd3GnAJT-eM6PZ7kWPtgiHehsEPOPeF4xTVvKY-_kBIDD2sQmB6SntSrqq5FzTbPcKjMqo_6yhN0ecTmvZ5RS-cxkH-GTVQptrxo8B6MgUNP99Muj7aJdSzyHdZF5Mih5qU3omLOtklAVThE_gradogh3Qq1ITKA0Hg"
    account_id = "184d0409-2626-4c48-98b5-d383b9854a47"
    base_url = "https://demo.docusign.net/restapi"

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
    background-image: url('https://github.com/NadaZakaria99/DocWise/blob/main/DALL%C2%B7E%202025-01-26%2011.43.33%20-%20A%20futuristic%20and%20sleek%20magical%20animated%20GIF-style%20icon%20design%20for%20'DocWise'%2C%20representing%20knowledge%2C%20documents%2C%20and%20wisdom.%20The%20design%20includes%20a%20glow%20(1).jpg');
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
            Agreement Analyzer with Chatbot and Docusign Integration
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
