import streamlit as st
from rag_chatbot import chatbot, USER_PROFILE
import json
import os

MEMORY_FILE = "chat_memory.json"


def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    return []


def save_memory(data):
    with open(MEMORY_FILE, "w") as f:
        json.dump(data, f)


st.title("🇯🇵 AI Japanese Tutor")

if "messages" not in st.session_state:
    st.session_state.messages = load_memory()

# Sidebar
with st.sidebar:
    st.header("Profile")
    st.write(USER_PROFILE)

    if st.button("Clear Chat"):
        st.session_state.messages = []
        save_memory([])

# Show chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Input
user_input = st.chat_input("Ask something...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    response = chatbot(user_input, st.session_state.messages)

    st.session_state.messages.append({"role": "assistant", "content": response})

    save_memory(st.session_state.messages)

    st.rerun()