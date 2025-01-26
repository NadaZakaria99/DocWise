import base64
import requests
import gradio as gr
import PyPDF2
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM

# Load BART model for summarization
summarization_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
summarization_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")

# Load DeepSeek model for chatbot
chatbot_tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
chatbot_model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")

# Function to extract text from a PDF
def extract_text_from_pdf(file_path):
    try:
        with open(file_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
        return text
    except Exception as e:
        return f"Error extracting text from PDF: {e}"

# Function to split text into chunks
def chunk_text(text, chunk_size=500):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

# Function to summarize the agreement using BART
def summarize_agreement(text):
    inputs = summarization_tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    outputs = summarization_model.generate(**inputs, max_new_tokens=200)
    summary = summarization_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary

# Function to retrieve the most relevant chunk for a question
def retrieve_relevant_chunk(chunks, question):
    # Simple retrieval: Find the chunk with the most overlapping words
    question_words = set(question.lower().split())
    best_chunk = None
    best_score = 0
    for chunk in chunks:
        chunk_words = set(chunk.lower().split())
        overlap = len(question_words.intersection(chunk_words))
        if overlap > best_score:
            best_chunk = chunk
            best_score = overlap
    return best_chunk

# Function to answer user questions using DeepSeek
def chatbot(text, question):
    # Step 1: Split the text into chunks
    chunks = chunk_text(text)

    # Step 2: Retrieve the most relevant chunk
    relevant_chunk = retrieve_relevant_chunk(chunks, question)

    # Step 3: Generate an answer using DeepSeek
    prompt = f"""
    Relevant Text:
    {relevant_chunk}

    Question:
    {question}

    Provide a concise and direct answer to the question. Do not add unnecessary commentary.

    Answer:
    """

    inputs = chatbot_tokenizer(prompt, return_tensors="pt", max_length=2048, truncation=True)
    outputs = chatbot_model.generate(**inputs, max_new_tokens=200)
    answer = chatbot_tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Post-process the answer to remove unnecessary commentary
    answer = answer.split("Answer:")[-1].strip()  # Remove the prompt
    return answer

# Function to send document to DocuSign
def send_to_docusign(file_path, recipient_email, recipient_name):
    # DocuSign API credentials (replace with your actual credentials)
    docusign_api_key = "eyJ0eXAiOiJNVCIsImFsZyI6IlJTMjU2Iiwia2lkIjoiNjgxODVmZjEtNGU1MS00Y2U5LWFmMWMtNjg5ODEyMjAzMzE3In0.AQoAAAABAAUABwCAUarwaj3dSAgAgJHN_q093UgCAOlO8Y4SurZKuV6S5XinyP8VAAEAAAAYAAIAAAAFAAAAHQAAAA0AJAAAADE5MjE2ZWM5LTBlYzYtNGUzNi1hZmRjLTI0YmYxYWMxZTlmNCIAJAAAADE5MjE2ZWM5LTBlYzYtNGUzNi1hZmRjLTI0YmYxYWMxZTlmNDAAgPgpoFg93Ug3ANsGGOe0MuhFnvTaUOPDR9o.av_Lw8n-1NzH4xLey3n-nCRmgsbN9hPHPMfd7FIRUemL5gIAYoYz5MWdGrF40b2hMnpqWzRjZwJmbFjwUaDujfF4307Tn7gSLeUHRDNSKVoUPasUEHdaZum_PY94s4jMnupQgT7LvU5aqqHFEdrOqBgjQfxMjpUT4HeMveeln4REjXXQrrQzDlmGgxIMGta3YKx5swmfey_bqwjf2ODmm7sviFXxePRmcNWIvyPoE8DkfogVVCWEwarz1p7yZ_OofPU8kUwCw2dJePiDaL0rVKdOGbDTSM5AxrZ7vXr5ap3xurs9yYpaqCOxk8jlNC7wQSULRZxv4qpAwAJTGvPiKA"  # Replace with your OAuth token
    account_id = "184d0409-2626-4c48-98b5-d383b9854a47"  # API Account ID
    base_url = "https://demo.docusign.net/restapi"  # Account Base URL + /restapi

    # Prepare the document
    with open(file_path, "rb") as file:
        document_base64 = base64.b64encode(file.read()).decode()

    # Create the envelope
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

    # Send the request
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
        response.raise_for_status()  # Raise an error for bad status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

# Function to process the agreement
def process_agreement(file, recipient_email, recipient_name):
    try:
        # Step 1: Extract text from PDF
        text = extract_text_from_pdf(file.name)
        if text.startswith("Error"):
            return text, {}, {}  # Return error message if text extraction fails

        # Step 2: Summarize the agreement
        summary = summarize_agreement(text)
        if summary.startswith("Error"):
            return summary, {}, {}  # Return error message if summarization fails

        # Step 3: Send the document to DocuSign
        docusign_response = send_to_docusign(file.name, recipient_email, recipient_name)
        if "error" in docusign_response:
            return summary, {}, docusign_response  # Return DocuSign error if API call fails

        return summary, {}, docusign_response
    except Exception as e:
        return f"Error: {e}", {}, {}

# Gradio interface
def main_interface(file, recipient_email, recipient_name, question, state):
    # Store the uploaded file and extracted text in the state
    if file is not None:
        state["file"] = file
        state["text"] = extract_text_from_pdf(file.name)

    # Initialize outputs
    summary_output = ""
    docusign_output = {}
    chatbot_output = ""

    # If the file is uploaded, process it
    if "file" in state:
        # Agreement Processing Tab
        if recipient_email and recipient_name:
            summary_output, _, docusign_output = process_agreement(state["file"], recipient_email, recipient_name)

        # Chatbot Tab
        if question:
            chatbot_output = chatbot(state["text"], question)

    return summary_output, docusign_output, chatbot_output, state

# Gradio app
with gr.Blocks() as app:
    gr.Markdown("# Agreement Analyzer with Chatbot and Docusign Integration")

    # State to store the uploaded file and extracted text
    state = gr.State({})

    # File upload (shared between tabs)
    file_input = gr.File(label="Upload Agreement (PDF)")

    with gr.Tab("Agreement Processing"):
        email_input = gr.Textbox(label="Recipient Email")
        name_input = gr.Textbox(label="Recipient Name")
        summary_output = gr.Textbox(label="Agreement Summary")
        docusign_output = gr.JSON(label="DocuSign Response")
        process_button = gr.Button("Process Agreement")

    with gr.Tab("Chatbot"):
        chatbot_question_input = gr.Textbox(label="Ask a Question")
        chatbot_answer_output = gr.Textbox(label="Answer")
        chatbot_button = gr.Button("Ask")

    # Link the inputs and outputs
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

# Launch the Gradio app
app.launch(debug=True)