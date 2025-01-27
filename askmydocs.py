import streamlit as st
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain

st.title("Ask My Docs")

st.sidebar.header("Upload a document")
uploaded_file = st.sidebar.file_uploader("Upload a document (txt, pdf)", type=["txt", "pdf"])

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Process the uploaded file and create embeddings
def process_file(uploaded_file):
    if uploaded_file is not None:
        # Load the document
        file_text = ""
        if uploaded_file.type == "text/plain":
            file_text = uploaded_file.getvalue().decode("utf-8")
        elif uploaded_file.type == "application/pdf":
            from PyPDF2 import PdfReader
            pdf_reader = PdfReader(uploaded_file)
            for page in pdf_reader.pages:
                file_text += page.extract_text()

        # Split the document into chunks
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = splitter.split_documents(file_text)

        # Create embeddings for the chunks
        # embeddings = OpenAIEmbeddings()
        # text_embeddings = embeddings.embed_documents(texts)
        # text_embedding_pairs = zip(texts, text_embeddings)
        # faiss = FAISS.from_embeddings(text_embedding_pairs, embeddings)
        embeddings = OpenAIEmbeddings()
        # db = FAISS.from_documents(texts, embeddings)
        db = FAISS.from_texts(texts, embeddings)

        return db
    return None

# Function to generate a response using RAG
def generate_response(query, db):
    qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=db.as_retriever())
    response = qa.run(query)
    return response

# Main Chat Interface
st.header("Chat with your documents")

# Display the chat history
for message in st.session_state.chat_history:
    st.write(f"{message['role']}: {message['content']}")

# Input for user message
user_query = st.text_input("You: ", key="user_query")

# Submit button
if st.button("Submit"):
    if uploaded_file is not None:
        db = process_file(uploaded_file)
        if db:
            response = generate_response(user_query, db)
            st.session_state.chat_history.append({"role": "You", "content": user_query})
            st.session_state.chat_history.append({"role": "AskMyDocs", "content": response})
            st.experimental_rerun()
    else:
        st.error("Please upload a document first")
