# AskYourDocs

AskYourDocs is a Streamlit-based application that allows users to upload a file and interact with it through a chat-like interface. The app uses Retrieval-Augmented Generation (RAG) to generate embeddings from the uploaded file and provides responses to user queries based on the content of the file.

## Features

- **File Upload**: Upload text-based files (e.g., `.txt`, `.pdf`, `.docx`).
- **Chat Interface**: Interact with the uploaded file using a conversational interface.
- **RAG Integration**: Leverages OpenAI's embeddings and GPT models to generate contextually relevant responses.
- **Session Management**: Maintains chat history during the session.

## Prerequisites

Before running the app, ensure you have the following installed:

- Python 3.8 or higher
- Streamlit
- OpenAI API key (for embeddings and GPT models)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Harsh-2909/askyourdocs.git
   cd askyourdocs
   ```

2. Install the required dependencies:

   ```bash
   pip install streamlit langchain openai faiss-cpu
   ```

3. Set up your OpenAI API key:

   Create a `.env` file in the project directory and add your OpenAI API key and model names (optional) as follows:

   ```
   OPENAI_API_KEY=your-api-key-here
   OPENAI_CHAT_MODEL=gpt-4o-mini (optional)
   OPENAI_EMBEDDING_MODEL=text-embedding-3-small (optional)
   ```

   Alternatively, you can set the environment variable directly in your terminal:

   ```bash
   export OPENAI_API_KEY=your-api-key-here
   ```

## Usage

1. Run the Streamlit app:

   ```bash
   streamlit run askyourdocs.py
   ```

2. Open your browser and navigate to the URL provided in the terminal (usually `http://localhost:8501`).

3. Upload a file using the sidebar. Supported file types include `.txt`, `.pdf`, and `.md`.

4. Once the file is uploaded, you can start chatting with it. Enter your query in the chat input box and click "Submit" to get a response.

5. The chat history will be displayed in the main interface, allowing you to continue the conversation.

## Example

1. Upload a text file containing information about a topic (e.g., a research paper or a book chapter).
2. Ask questions like:
   - "What is the main idea of this document?"
   - "Can you summarize the key points?"
   - "Explain the concept of X mentioned in the document."
3. The app will generate responses based on the content of the uploaded file.

## Customization

- **File Types**: Extend the app to support additional file types by integrating appropriate document loaders from LangChain.
- **UI Enhancements**: Customize the Streamlit UI to improve the user experience.
- **Advanced Models**: Replace the default OpenAI GPT model with other language models or fine-tune the RAG pipeline for specific use cases.

## Contributing

Contributions are welcome! If you'd like to contribute to this project, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Commit your changes and push to the branch.
4. Submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Streamlit](https://streamlit.io/) for the interactive web app framework.
- [LangChain](https://www.langchain.com/) for the RAG pipeline and document processing.
- [OpenAI](https://openai.com/) for the embeddings and GPT models.

---

Enjoy interacting with your files using AskYourDocs! 🚀
