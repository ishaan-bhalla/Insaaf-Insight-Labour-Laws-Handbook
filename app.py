import streamlit as st
from dotenv import load_dotenv
from PIL import Image
from streamlit_chat import message  # Make sure you have this library installed
from main import generate_answer  # Import the function from your main script
import os
import warnings

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
        # Browsers block PDFs embedded as data: URLs, so offer the file as a download instead.
        st.caption("A Handbook on Labour Laws of India, by Rajkumar S. Adukia (96 pages). "
                   "Every answer lists the pages it was drawn from.")
        with open(pdf_file, "rb") as f:
            st.download_button("Download the handbook (PDF)", f.read(), file_name=pdf_file, mime="application/pdf")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    user_input = st.sidebar.text_input("Search your Query:", key="user_input")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.spinner("Generating..."):
            try:
                answer, pages = generate_answer(user_input)
                if pages:
                    answer += "\n\nSources: handbook page" + ("s " if len(pages) > 1 else " ") + ", ".join(map(str, pages))
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                # Both models already failed (after retries), so say so plainly and keep the detail for the log.
                print(f"Answer failed: {e!r}")
                busy = "The language model is busy or unavailable right now. Please try again in a minute."
                st.session_state.messages.append({"role": "assistant", "content": busy})
                st.error(busy)

    # Display messages in the chat
    for i, msg in enumerate(st.session_state.messages):
        message(msg["content"], is_user=msg["role"] == "user", key=f"msg_{i}")

if __name__ == '__main__':
    main()
