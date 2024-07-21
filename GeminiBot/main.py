import os
import streamlit as st
import google.generativeai as gen_ai
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

load_dotenv()

# Streamlit page configuration
st.set_page_config(
    page_title="Gemini Application",
    page_icon=":brain",
    layout="centered"
)

gen_ai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to translate roles between Gemini and Streamlit terminology
def translate_role_for_streamlit(user_role):
    return "assistant" if user_role == "model" else user_role

# Initialize chat history for chatbot
if "messages" not in st.session_state:
    st.session_state.messages = gen_ai.GenerativeModel("gemini-pro").start_chat(history=[])

# Read the uploaded PDF and extract the text
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Divide the text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    return chunks

# Convert the text chunks into vectors
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Create conversational chain
def get_conversational_chain():
    prompt_template = """
        Answer the question as detailed as possible from the provided context. Make sure to provide all the details. If the answer is not in the provided context,
        just say "Answer not available in the context". Don't provide the wrong answer.
        \n\nContext: \n {context}?\n
        Question: \n {question}\n
        Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Handle user input for chat with PDF
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain.invoke({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Answer:", response["output_text"])

# Main function to run the Streamlit app
def main():
    st.sidebar.title("Gemini Application")
    app_mode = st.sidebar.selectbox("Choose the feature to either chat with Gemini model for general queries or chat with your PDFs", ["Chat with Gemini", "Chat with PDF"])

    if app_mode == "Chat with Gemini":
        st.title("🤖 Gemini ChatBot")
        for message in st.session_state.messages.history:
            with st.chat_message(translate_role_for_streamlit(message.role)):
                st.markdown(message.parts[0].text)

        user_prompt = st.chat_input("Ask Gemini-Pro...")
        if user_prompt:
            with st.chat_message("user"):
                st.markdown(user_prompt)
            gemini_response = st.session_state.messages.send_message(user_prompt)
            with st.chat_message("assistant"):
                st.markdown(gemini_response.text)

    elif app_mode == "Chat with PDF":
        st.title("Chat with PDF using Gemini💁")
        user_question = st.text_input("Ask a Question from the PDF Files")

        if user_question:
            user_input(user_question)

        with st.sidebar:
            pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
            if st.button("Submit & Process"):
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Done")

if __name__ == "__main__":
    main()
