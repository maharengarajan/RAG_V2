import streamlit as st
from streamlit_chat import message 
from query_data import query_rag


# Initialize session state for storing the conversation history
if 'messages' not in st.session_state:
    st.session_state['messages'] = []


def submit():
    user_input = st.session_state.input_text
    if user_input:
        # Add user message to session state
        st.session_state['messages'].append({"role": "user", "content": user_input})
        
        # Get response from the model
        bot_response = query_rag(user_input)
        
        # Add bot message to session state
        st.session_state['messages'].append({"role": "bot", "content": bot_response})

        # Clear the input field
        st.session_state.input_text = ""

# ---------------------- UI ENHANCEMENTS ---------------------- #

# Main header and description
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>RAG-Powered AI Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px;'>Ask questions based on your uploaded documents. The AI will retrieve relevant information and provide concise answers.</p>", unsafe_allow_html=True)

# Divider for a cleaner look
st.markdown("<hr style='border: 1px solid #D3D3D3;'>", unsafe_allow_html=True)

# Chat history display with enhanced aesthetics
st.markdown("<h3 style='color: #333;'>Conversation:</h3>", unsafe_allow_html=True)

# Display chat history (user and bot messages)
for msg in st.session_state['messages']:
    if msg['role'] == 'user':
        message(msg['content'], is_user=True)  # Display user messages on the right
    else:
        message(msg['content'], is_user=False)  # Display bot messages on the left

# Divider before input
st.markdown("<hr style='border: 1px solid #D3D3D3;'>", unsafe_allow_html=True)

# User input area with a more prominent call-to-action style
st.text_input("Type your question here:", key="input_text", on_change=submit, placeholder="Ask about your documents...")

# Footer with branding or further instructions
st.markdown("<p style='text-align: center; font-size: 14px; color: #808080;'>Powered by RAG-based Retrieval and Streamlit</p>", unsafe_allow_html=True)

# Footer with branding or further instructions
st.markdown("<p style='text-align: center; font-size: 14px; color: #808080;'>Developed by Renga Rajan K</p>", unsafe_allow_html=True)
