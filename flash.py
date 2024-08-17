import streamlit as st
import PyPDF2
from groq import Groq
import time

def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file."""
    text = ""
    reader = PyPDF2.PdfReader(pdf_file)
    for page in reader.pages:
        text += page.extract_text()
    return text

def generate_flashcard_response_groq(pdf_text, user_input, api_key):
    """Generate a flashcard response using Groq AI based on the extracted PDF text and user input."""
    client = Groq(api_key=api_key)
    
    # Limit the text length to avoid hitting the token limit
    max_text_length = 1500  # Adjust as necessary
    pdf_text = pdf_text[:max_text_length]
    
    prompt_template = (
        "You are a chatbot that helps users learn topics from a given document by creating flashcards. "
        "The document content is as follows:\n\n'{pdf_text}'\n\n"
        "User's question or topic: '{user_input}'\n"
        "Create a flashcard with a question and answer based on the document content."
    )
    
    prompt = prompt_template.format(pdf_text=pdf_text, user_input=user_input)
    
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "user", "content": prompt}
            ],
            model="llama3-8b-8192"
        )
        return chat_completion.choices[0].message.content.strip()
    
    except Exception as e:
        error_message = str(e)
        if "rate_limit_exceeded" in error_message:
            st.warning("Rate limit reached. Retrying after a short delay...")
            time.sleep(10)  # Wait for the rate limit to reset
            return generate_flashcard_response_groq(pdf_text, user_input, api_key)
        else:
            st.error(f"Error during Groq API call: {error_message}")
            return "Error generating response."

def main():
    # Custom CSS for chat-like UI with dark mode support
    st.markdown("""
        <style>
        .user-bubble {
            background-color: #BEE3F8;
            border-radius: 10px;
            padding: 10px;
            margin: 10px 0;
            text-align: left;
            max-width: 80%;
            margin-left: auto;
            color: #1A202C;
        }
        .bot-bubble {
            background-color: #CBD5E0;
            border-radius: 10px;
            padding: 10px;
            margin: 10px 0;
            text-align: left;
            max-width: 80%;
            margin-right: auto;
            color: #1A202C;
        }
        .dark-mode .user-bubble {
            background-color: #3182CE;
            color: white;
        }
        .dark-mode .bot-bubble {
            background-color: #4A5568;
            color: white;
        }
        .container {
            display: flex;
            flex-direction: column;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("Flashcard Learning App with Groq AI")
    st.write("Upload a PDF file, and learn its content through flashcards.")

    # File upload
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        pdf_text = extract_text_from_pdf(uploaded_file)
        st.write("PDF content loaded. The AI will now create flashcards for you.")
        
        # API Key input
        api_key = st.text_input("Enter your Groq API key:")
        
        if api_key:
            user_input = st.text_input("Enter a topic or question: ", "")

            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []
                st.session_state.current_flashcard = None

            if user_input:
                response = generate_flashcard_response_groq(pdf_text, user_input, api_key)
                st.session_state.chat_history.append({"role": "user", "content": user_input})
                st.session_state.chat_history.append({"role": "bot", "content": response})
                st.session_state.current_flashcard = response
            
            # Display the chat history
            for idx, chat in enumerate(st.session_state.chat_history):
                if chat["role"] == "user":
                    st.markdown(f'<div class="user-bubble">{chat["content"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="bot-bubble">{chat["content"]}</div>', unsafe_allow_html=True)
                    
                    if idx == len(st.session_state.chat_history) - 1:  # Only add 'understood' option to the latest response
                        understood = st.radio(
                            "Did you understand this flashcard?", 
                            ('Yes', 'No'), 
                            key=f"understood_{idx}"
                        )
                        if understood == 'No' and st.session_state.current_flashcard:
                            simplified_input = "Please simplify the last flashcard."
                            st.session_state.chat_history.append({"role": "user", "content": simplified_input})
                            simplified_response = generate_flashcard_response_groq(pdf_text, simplified_input, api_key)
                            st.session_state.chat_history.append({"role": "bot", "content": simplified_response})
                            st.session_state.current_flashcard = simplified_response

    st.write("Type 'exit' or 'quit' to end the session.")

if __name__ == "__main__":
    main()
