import os
import traceback
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()

OPENAI_CHAT_MODEL: str = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
OPENAI_EMBEDDINGS_MODEL: str = os.getenv(
    "OPENAI_EMBEDDINGS_MODEL", "text-embedding-3-small"
)

st.set_page_config(page_title="Ask My Docs", layout="wide")
st.title("Ask My Docs")

st.sidebar.header("Upload File")
uploaded_file = st.sidebar.file_uploader(
    "Upload a file (txt, pdf)", type=["txt", "pdf"]
)
submit_button = st.sidebar.button("Submit", key="file_submit_button")

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
        print(
            f"Uploaded file type: {uploaded_file.type}. Uploaded file name: {uploaded_file.name}"
        )
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
        embeddings = OpenAIEmbeddings(model=OPENAI_EMBEDDINGS_MODEL)
        db = FAISS.from_texts(texts, embeddings)

        retriever = db.as_retriever()
        llm = ChatOpenAI(temperature=0, model=OPENAI_CHAT_MODEL)

        # Contextualize question
        print("Creating conversational retrieval chain...")
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, just "
            "reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        history_aware_retriever = create_history_aware_retriever(
            llm,
            retriever,
            prompt=contextualize_q_prompt,
        )
        qa_system_prompt = (
            "You are an assistant for question-answering tasks. Use "
            "the following pieces of retrieved context to answer the "
            "question. If you don't know the answer, just say that you "
            "don't know. Use three sentences maximum and keep the answer "
            "concise."
            "\n\n"
            "{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(
            history_aware_retriever, question_answer_chain
        )
        st.session_state.qa_chain = rag_chain
        st.session_state.file_uploaded = True
        print("File uploaded successfully")
        st.success("File uploaded successfully")
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")

# Main Chat Interface
st.header("Chat with your documents")

if st.session_state.file_uploaded:
    user_input = st.text_input("Ask a question about the file:")

    if st.button("Send") and user_input:
        print("Generating response...")
        try:
            print(
                f"[DEBUG] User input: {user_input}\nChat history: {st.session_state.chat_history}"
            )
            response = st.session_state.qa_chain.invoke(
                {
                    "input": user_input,
                    "chat_history": st.session_state.chat_history,
                }
            )
            print(f"[DEBUG] Response: {response}")
            st.session_state.chat_history.append(("user", user_input))
            st.session_state.chat_history.append(("assistant", response["answer"]))
        except Exception as e:
            st.error(f"An error occurred while generating the response: {str(e)}")
            traceback.print_exc()

    # Display chat history
    for message_type, message in st.session_state.chat_history:
        st.write(f"**{message_type.capitalize()}:** {message}")
else:
    st.info("Please upload a file to start chatting.")
