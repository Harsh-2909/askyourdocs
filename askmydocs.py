import traceback
import streamlit as st
import base64
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

st.set_page_config(page_title="Ask My Docs", layout="wide")
st.title("Ask My Docs")

st.sidebar.header("OpenAI API Configuration")
st.sidebar.markdown("Your OpenAI API Key")
openai_api_key = st.sidebar.text_input(
    "API Key", type="password", key="openai_api_key")
st.sidebar.markdown("OpenAI Chat Model")
openai_chat_model = st.sidebar.selectbox(
    "Chat Model",
    options=[
        "gpt-4o-mini",
        "gpt-4o",
        "gpt-4o-turbo",
        "gpt-3.5-turbo",
        "o1-mini",
        "o1",
        "o3-mini",
    ],
    index=0,
    key="openai_chat_model",
)
st.sidebar.header("Upload File")
uploaded_file = st.sidebar.file_uploader(
    "Upload a file (txt, pdf, md)", type=["txt", "pdf", "md"]
)
submit_button = st.sidebar.button("Submit", key="file_submit_button")

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if "file_uploaded" not in st.session_state:
    st.session_state.file_uploaded = False


def display_file(file) -> None:
    """Function to display the uploaded file in Streamlit based on file type

    Args:
        file (BytesIO): Uploaded file to display
    """
    if file.type == "text/plain":
        display_text(file)
    elif file.type == "text/markdown":
        display_markdown(file)
    elif file.type == "application/pdf":
        print("PDF Display is currently disabled due to a bug.")
        # display_pdf(file)
    else:
        st.error("Unsupported file type")


def display_text(file) -> None:
    """Function to display text file in Streamlit

    Args:
        file (BytesIO): Text file to display
    """
    st.markdown("### Text Preview")
    st.text(file.getvalue().decode("utf-8"))


def display_markdown(file) -> None:
    """Function to display Markdown file in Streamlit

    Args:
        file (BytesIO): Markdown file to display
    """
    st.markdown("### Markdown Preview")
    st.markdown(file.getvalue().decode("utf-8"))


def display_pdf(file) -> None:
    """Function to display PDF file in Streamlit

    Args:
        file (BytesIO): PDF file to display
    """
    st.markdown("### PDF Preview")
    base64_pdf = base64.b64encode(file.read()).decode("utf-8")
    pdf_display = f"""<iframe src="data:application/pdf;base64,{base64_pdf}" width="400" height="100%" type="application/pdf"
                        style="height:100vh; width:100%"
                    >
                    </iframe>"""
    st.markdown(pdf_display, unsafe_allow_html=True)


# Process the uploaded file and create embeddings
if submit_button and uploaded_file and openai_api_key:
    try:
        # Load the document
        file_text = ""
        print("Processing uploaded file...")
        if uploaded_file.type in ["text/plain", "text/markdown"]:
            file_text = uploaded_file.getvalue().decode("utf-8")
        elif uploaded_file.type == "application/pdf":
            from PyPDF2 import PdfReader

            pdf_reader = PdfReader(uploaded_file)
            for page in pdf_reader.pages:
                file_text += page.extract_text()
        else:
            raise ValueError("Unsupported file type")

        with st.sidebar.expander("File Preview", expanded=True):
            display_file(uploaded_file)

        # Split the document into chunks
        print("Splitting text into chunks...")
        text_splitter = CharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_text(file_text)

        # Create embeddings for the chunks
        print("Creating embeddings for the chunks...")
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        db = FAISS.from_texts(texts, embeddings)

        retriever = db.as_retriever()
        llm = ChatOpenAI(temperature=0,
                         model=openai_chat_model,
                         api_key=openai_api_key)

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
        # Only pick the answer from the response
        st.session_state.qa_chain = rag_chain.pick("answer")
        st.session_state.file_uploaded = True
        print("File uploaded successfully")
        st.success("File uploaded successfully")
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")

# Main Chat Interface
st.header("Chat with your documents")

# Display chat history if available
# It ensures that the chat history is displayed even if the app is reloaded
for message_type, message in st.session_state.chat_history:
    with st.chat_message(message_type):
        st.markdown(message)

if st.session_state.file_uploaded:
    if user_input := st.chat_input("Ask a question about the file"):
        # Add user input to chat history and display it
        st.session_state.chat_history.append(("user", user_input))
        with st.chat_message("user"):
            st.markdown(user_input)
        # st.spinner("Generating response..."):
        print("Generating response...")
        try:
            with st.chat_message("assistant"):
                stream = st.session_state.qa_chain.stream(
                    {
                        "input": user_input,
                        "chat_history": st.session_state.chat_history,
                    }
                )
                response = st.write_stream(stream)
            st.session_state.chat_history.append(("assistant", response))
        except Exception as e:
            st.error(
                f"An error occurred while generating the response: {str(e)}")
            traceback.print_exc()

else:
    st.info("Please upload a file to start chatting.")
