import streamlit.components.v1 as components
import streamlit as st
from PIL import Image
from dotenv import load_dotenv
from streamlit_chat import message
from streamlit_searchbox import st_searchbox
from main import generate_answer
import main as rtt
from langchain.chat_models import ChatOpenAI
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import create_sql_agent
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.llms.openai import OpenAI
import os
import warnings

warnings.filterwarnings("ignore")

import requests
import json
import pandas as pd
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)

# Function to initialize environment variables
def init():
    load_dotenv()
    if os.getenv("GOOGLE_API_KEY") is None:
        raise ValueError("Google API Key not provided. Please provide GOOGLE_API_KEY as an environment variable")

# Function to set up the Streamlit layout
def setup_layout():
    col1, col2 = st.columns(2)
    
    img = Image.open('bot.png')
    new_size = (150, 150)  # Specify the new size (width, height)
    resized_img = img.resize(new_size)
    col1.image(resized_img)
    col2.title("INSAAF AI")

    with open("ui/sidebar.md", "r") as sidebar_file:
       sidebar_content = sidebar_file.read()

    with open("ui/styles.md", "r") as styles_file:
        styles_content = styles_file.read()
    
    # Displays the content in the sidebar  
    st.sidebar.write(sidebar_content)

# Main function to handle user input and display chat
def main():
    init() 
    setup_layout()
    
    user_input = st.sidebar.text_input("Search your Query :", key="user_input")
    
    if "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content="You are a smart assistant for helping the users with their queries. My job is to help employees within the organization. My demeanour will be friendly yet professional.")
        ]

    if user_input:
        st.session_state.messages.append(HumanMessage(content=user_input))
        with st.spinner("Generating...."):
            try:
                response = generate_answer(rtt.db, user_input)
                st.session_state.messages.append(AIMessage(content=response))
            except Exception as e:
                st.session_state.messages.append(AIMessage(content=f"An error occurred: {e}"))
                st.error(f"An error occurred: {e}")

    messages = st.session_state.get('messages', [])
    for i, msg in enumerate(messages[1:]):
        if i % 2 == 0:
            message(msg.content, is_user=True, key=str(i) + '_user')
        else:
            message(msg.content, is_user=False, key=str(i) + '_ai')

if __name__ == '__main__':
    main()
