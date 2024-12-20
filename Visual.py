import os
import base64
import streamlit as st
from SAbot import sabot
import time

# Configuration for the Streamlit app's theme
st.set_page_config(
    page_title="SAbot Assistant",
    page_icon=":robot_face:",
    layout="centered",
    initial_sidebar_state="auto",
)

@st.cache_data
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

background_image_path = "UI background.jpg" #must update with your path
icon_image_path = "robotic_icon.gif" #must update with your path

FILES_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "files"))
os.makedirs(FILES_DIR, exist_ok=True)

background_img = get_img_as_base64(background_image_path)
icon_img = get_img_as_base64(icon_image_path)

page_bg_img = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');

[data-testid="stAppViewContainer"] > .main {{
background-image: url("data:image/png;base64,{background_img}");
background-color: black;
background-size: cover;
background-position: top left;
background-repeat: no-repeat;
background-attachment: local;
color: white;
font-family: 'Poppins', sans-serif;
}}

[data-testid="stSidebar"] > div:first-child {{
background-image: url("data:image/png;base64,{background_img}");
background-color: black;
background-size: cover;
background-position: top left;
background-repeat: no-repeat;
background-attachment: fixed;
}}

[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}

[data-testid="stToolbar"] {{
right: 2rem;
}}

.stTitle {{
    color: white;
    font-family: 'Poppins', sans-serif;
}}

.stFileUploader {{
    background-color: #1a1a1a;
    color: white;
    border: 2px dashed #FFFFFF;
    border-radius: 10px;
    padding: 20px;
    font-family: 'Poppins', sans-serif;
}}

[data-testid="stFileUploader"] div {{
    color: white;
    font-family: 'Poppins', sans-serif;
}}

input[type="text"] {{
    background-color: #333333;
    color: white;
    font-family: 'Poppins', sans-serif;
    border: 2px solid #FFFFFF;
    padding: 10px;
    border-radius: 10px;
    box-shadow: 0 0 10px rgba(255, 255, 255, 0.5);
    backdrop-filter: blur(10px);
    width: 100%;
}}

textarea {{
    background-color: #333333;
    color: white;
    font-family: 'Poppins', sans-serif;
    border: 2px solid #FFFFFF;
    padding: 10px;
    border-radius: 10px;
    box-shadow: 0 0 10px rgba(255, 255, 255, 0.5);
    backdrop-filter: blur(10px);
    width: 100%;
}}

.stFileUploader div {{
    background-color: #1a1a1a;
    color: white;
    font-family: 'Poppins', sans-serif;
    border: 2px solid #FFFFFF;
    padding: 10px;
    box-shadow: 0 0 10px rgba(255, 255, 255, 0.5);
    backdrop-filter: blur(10px);
    border-radius: 10px;
}}

.stButton>button {{
    background-color: #333333;
    color: white;
    font-family: 'Poppins', sans-serif;
    border: 2px solid #FFFFFF;
    padding: 10px;
    border-radius: 10px;
    box-shadow: 0 0 10px rgba(255, 255, 255, 0.5);
    backdrop-filter: blur(10px);
}}

.answer-box {{
    display: block;
    align-items: flex-start;
    background-color: #333333;
    color: white;
    font-family: 'Poppins', sans-serif;
    border: 2px solid #FFFFFF;
    padding: 10px;
    margin-top: 10px;
    box-shadow: 0 0 10px rgba(255, 255, 255, 0.5);
    backdrop-filter: blur(10px);
    position: relative;
    border-radius: 10px;
    width: 100%;
    word-wrap: break-word;
    word-break: normal;
}}

.answer-box img {{
    margin-right: 10px;
    align-self: flex-start;
    float: left;
}}

.answer-box .time-taken {{
    position: absolute;
    bottom: 5px;
    right: 10px;
    font-size: 12px;
    color: white;
}}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)

st.markdown("<h1 class='stTitle'>SAbot an LLM RAG Assistant</h1>", unsafe_allow_html=True)

@st.cache_resource
def load_chatbot():
    return sabot()

chatbot = load_chatbot()

def save_file(uploaded_file):
    file_path = os.path.join(FILES_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

# Primary input for questions
question = st.text_input("Ask a question:")

# Optional file uploader
uploaded_file = st.file_uploader("Upload a file", type=["pdf"])

# New input sliders for temperature and max_tokens
temperature = st.select_slider("Select temperature:", options=[0.2, 0.5, 0.7, 1.0, 1.2, 1.5], value=0.2)
max_tokens = st.select_slider("Select max tokens:", options=[50, 100, 200, 300, 400], value=400)

if uploaded_file:
    file_path = save_file(uploaded_file)
    st.write(f"<span style='color:white;'>File saved to {file_path}</span>", unsafe_allow_html=True)

    pdf_paths = [file_path]
    docs = chatbot.load_documents(pdf_paths)
    chatbot.create_vector_db(docs)

if question:
    start_time = time.time()
    if chatbot.vector_db:
        retrieved_docs = chatbot.similarity_search(question, k=5)
        if retrieved_docs:
            context = " ".join([doc.page_content for doc in retrieved_docs])
            answer = chatbot.generate(question, context, temperature, max_tokens)
        else:
            answer = chatbot.generate(question, temperature=temperature, max_tokens=max_tokens)  # Fallback to built-in knowledge base
    else:
        answer = chatbot.generate(question, temperature=temperature, max_tokens=max_tokens)  # No document context available

    end_time = time.time()
    time_taken = end_time - start_time
    st.markdown(
        f'<div class="answer-box"><img src="data:image/png;base64,{icon_img}" width="50"/><span>{answer}</span><div class="time-taken">Time taken: {time_taken:.2f} seconds</div></div>',
        unsafe_allow_html=True
    )
