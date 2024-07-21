import os
import streamlit as st
import google.generativeai as gen_ai
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title = "Chat with Gemini Model!",
    page_icon = ":brain",
    layout = "centered"
)

gen_ai.configure(api_key = os.getenv("GOOGLE_API_KEY"))

model = gen_ai.GenerativeModel("gemini-pro")

# Function to translate roles between gemini and streamlit terminology
def translate_role_for_streamlit(user_role):
    if user_role == "model":
        return "assistant"
    else:
        return user_role

# initialize chat history - session state to store the chat history
if "messages" not in st.session_state:
    st.session_state.messages = model.start_chat(history = [])

st.title("🤖 Gemini ChatBot")

# Display chat history
for message in st.session_state.messages.history:
    with st.chat_message(translate_role_for_streamlit(message.role)):
        st.markdown(message.parts[0].text)
        
# Input field for user's message
user_prompt = st.chat_input("Ask Gemini-Pro...")
if user_prompt:
    # Add user's message to chat and display it
    with st.chat_message("user"):
        st.markdown(user_prompt)
        
    # Send user's message to Gemini-Pro and get the response
    gemini_response = st.session_state.messages.send_message(user_prompt)

    # Display Gemini-Pro's response
    with st.chat_message("assistant"):
        st.markdown(gemini_response.text)
       