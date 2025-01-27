# DocWise
![DALL·E 2025-01-26 11 43 33 - A futuristic and sleek magical animated GIF-style icon design for 'DocWise', representing knowledge, documents, and wisdom  The design includes a glow](https://github.com/user-attachments/assets/b3b50957-c732-4cbf-a745-fcc81282fb27)
# 💡Inspiration 
**DocWise** was inspired by a friend’s scholarship struggle. She faced unexpected issues and later discovered a hidden condition in her agreement that caused her to lose the scholarship. Her experience showed how easily critical details can be overlooked, leading to costly consequences. This inspired me to build DocWise a tool that uses AI to uncover hidden insights in agreements, helping users avoid similar pitfalls.

# 🎯What it does
**DocWise** is a comprehensive tool designed to enhance the agreement process from end to end. It offers the following key features:
- **Agreement Summarization:** Extracts and summarizes key points from uploaded PDF agreements using Google's Gemini AI, providing users with concise and actionable insights.
- **Chatbot Integration:** Allows users to ask questions about the agreement, with the chatbot providing answers based on the document's content. If the information is not available in the document, the chatbot performs a web search to provide relevant answers.
- **DocuSign Integration:** Enables users to send agreements directly to recipients for signing via DocuSign, streamlining the signing process.
- **AI-Powered Insights:** Leverages AI to identify and extract valuable data from agreements, helping users make informed decisions and avoid the "Agreement Trap."

# 🔌How we built it
**DocWise** was built using a combination of modern technologies and APIs:
- **Backend:** Python was used to handle the core logic, including PDF text extraction, text chunking, and embedding using Sentence Transformers.
- **AI Integration:** Google's Gemini AI was integrated for summarization and question-answering capabilities. The Tavily API was used for web searches when the document lacked sufficient information.
- **DocuSign API:** The DocuSign API was integrated to facilitate the seamless sending of agreements for signing.
- **Frontend:** The user interface was built using Gradio, providing an intuitive and interactive experience for users.

# ⚡Challenges we ran into
- **AI Integration:** Integrating Google's Gemini AI and ensuring it provided accurate and relevant responses required extensive prompt engineering and testing.
- **DocuSign API:** Configuring the DocuSign API to send agreements and handle responses involved navigating authentication and envelope creation complexities.
- **Performance Optimization:** Balancing the performance of text chunking, embedding, and retrieval while maintaining a responsive user experience was a significant challenge.

# 🌟Accomplishments that we're proud of
- **Seamless API Integration:** Successfully combined Google Gemini, DocuSign, and Tavily APIs into a unified, functional application, overcoming authentication and data consistency challenges.
- **User-Friendly Interface:** Built an intuitive Gradio interface that simplifies complex tasks like agreement summarization and signing, iterating based on user feedback.
**Robust Retrieval System:** Developed a retrieval mechanism that blends document-based answers with web search results for accurate, comprehensive responses.
**Solving the Agreement Trap:** Created a solution that unlocks the value of agreement data through AI and automation, empowering businesses to make smarter decisions.

# 🚀How to Use
- **Set Up API Keys:**
  - Obtain API keys for Google Gemini, DocuSign, and Tavily.
  - Replace the placeholders in the code with your actual API keys
- **Run the Application:**
  - Clone the repository and install dependencies:
    
    pip install -r requirements.txt
  - Run the Gradio app:
    python DocWise.py
- **Upload a PDF Agreement:**
  Use the file upload button to upload a PDF agreement.
- **Summarize the Agreement:**
  The system will automatically extract and summarize the key points using Google Gemini.
- **Ask Questions:**
  Use the chatbot tab to ask questions about the agreement. The system will provide answers based on the document content or perform a web search if needed.
- **Send for Signing:**
  Enter the recipient’s email and name, then click "Process Agreement" to send the document for signing via DocuSign.

# 📚What we learned
- The importance of prompt engineering when working with large language models like Gemini to achieve accurate and relevant outputs.
- How to effectively use embeddings and retrieval mechanisms to enhance question-answering systems.
- The intricacies of integrating third-party APIs like DocuSign and handling authentication, error handling, and response parsing.

# 🕒What's next for DocWise 
- **Leveraging DocuSign Webhooks:** I plan to utilize DocuSign's webhook capabilities to enable real-time notifications and updates about the status of agreements. This will allow users to stay informed about when documents are viewed, signed, or completed, further enhancing the automation and responsiveness of the platform.
- **Multi-Language Support:** Expanding DocWise's capabilities to support multiple languages, making it accessible to a global audience and enabling users to analyze agreements in their preferred language.
- **Risk Presentation:** Introducing a feature that identifies and presents potential risks within agreements, helping users make more informed decisions and mitigate potential issues before they arise.
- **Streaming Responses:** Providing streaming responses for real-time interaction with the chatbot, improving user engagement and satisfaction.

# 📝License
DocWise is licensed under the [MIT License](LICENSE).
