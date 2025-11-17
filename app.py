import streamlit as st
from dotenv import load_dotenv
from PIL import Image
from streamlit_chat import message  # Make sure you have this library installed
from main import generate_answer  # Import the function from your main script
import os
import warnings
import base64

warnings.filterwarnings("ignore")
load_dotenv()
# Function to set up the Streamlit layout
def setup_layout():
    col1, col2 = st.columns(2)

    img = Image.open('bot.png')  # Ensure 'bot.png' exists in your directory
    new_size = (150, 150)  # Specify the new size (width, height)
    resized_img = img.resize(new_size)
    col1.image(resized_img)
    col2.title("INSAAF INSIGHT")

# Main function to handle user input and display chat
def main():
    setup_layout()

    # Sidebar content with project description and PDF viewer
    with st.sidebar:
        st.write("""
        Insaaf Insight is an AI-driven legal assistant designed to help users with their legal queries in Indian courts.
        This application leverages advanced AI technologies to provide accurate and contextually relevant answers based on
        the documents related to employee relations and labor laws in India. Feel free to ask your questions!
        """)

        # Display the PDF directly in the sidebar
        st.write("### Reference Document")
        pdf_file = 'A Handbook on Employee Relations and Labour Laws in India.pdf'
        st.markdown(f'<iframe src="data:application/pdf;base64,{base64.b64encode(open(pdf_file, "rb").read()).decode()}" width="100%" height="400px"></iframe>', unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    user_input = st.sidebar.text_input("Search your Query:", key="user_input")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.spinner("Generating..."):
            try:
                response = generate_answer(user_input)  # Call your answer generation function
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.session_state.messages.append({"role": "assistant", "content": f"An error occurred: {e}"})
                st.error(f"An error occurred: {e}")

    # Display messages in the chat
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            message(msg["content"], is_user=True)
        else:
            message(msg["content"], is_user=False)

if __name__ == '__main__':
    main()
