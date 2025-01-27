import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="Ask My Docs", layout="wide")
st.title("Ask My Docs")

st.sidebar.header("Upload File")
uploaded_file = st.sidebar.file_uploader("Upload a file (txt, pdf)", type=["txt", "pdf"])
submit_button = st.sidebar.button("Submit", key= "file_submit_button")

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if "file_uploaded" not in st.session_state:
    st.session_state.file_uploaded = False

# Process the uploaded file and create embeddings
if submit_button and uploaded_file:
    try:
        # Load the document
        file_text = ""
        print("Processing uploaded file...")
        print(f"Uploaded file type: {uploaded_file.type}. Uploaded file name: {uploaded_file.name}")
        if uploaded_file.type == "text/plain":
            file_text = uploaded_file.getvalue().decode("utf-8")
        elif uploaded_file.type == "application/pdf":
            from PyPDF2 import PdfReader
            pdf_reader = PdfReader(uploaded_file)
            for page in pdf_reader.pages:
                file_text += page.extract_text()

        # Split the document into chunks
        print("Splitting text into chunks...")
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_text(file_text)

        # Create embeddings for the chunks
        print("Creating embeddings for the chunks...")
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        db = FAISS.from_texts(texts, embeddings)

        retriever = db.as_retriever()
        st.session_state.qa_chain = ConversationalRetrievalChain.from_llm(
            retriever=retriever,
            llm=OpenAI(temperature=0, model="gpt-4o-mini")
        )
        st.session_state.file_uploaded = True
        st.success("File uploaded successfully")
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")

# Main Chat Interface
st.header("Chat with your documents")

if st.session_state.file_uploaded:
    user_input = st.text_input("Ask a question about the file:")

    if st.button("Send") and user_input:
        try:
            response = st.session_state.qa_chain.invoke(
                input=user_input, chat_history=st.session_state.chat_history
            )
            st.session_state.chat_history.append((user_input, response))
        except Exception as e:
            st.error(f"An error occurred while generating the response: {str(e)}")

    # Display chat history
    for question, answer in st.session_state.chat_history:
        st.write(f"**You:** {question}")
        st.write(f"**File:** {answer}")
else:
    st.info("Please upload a file to start chatting.")
