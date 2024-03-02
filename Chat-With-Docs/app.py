import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from htmlTemplates import css, user_template, bot_template
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from mistral.ingest_final import run_ingest
from mistral.main import generate_response


def get_pdf_text(docs):
    print(docs)
    print(type(docs))
    text = ""

    storage_directory = "C:/Users/keyar/Documents/Sem 6/Chat-With-Docs/mistral/data"

    # Ensure the directory exists, if not, create it
    os.makedirs(storage_directory, exist_ok=True)

    # Check if files have been uploaded
    if docs is not None:
        for i, doc in enumerate(docs):
            # Get the filename
            # filename = os.path.join(storage_directory, f"{os.path.splitext(doc.name)[1]}")
            filename = os.path.join(storage_directory, doc.name)
    #     print('='*50)

            # Save the file to the specified directory
            with open(filename, "wb") as f:
                f.write(doc.getvalue())
    return 

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_text_embedding(text_chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vector_store = FAISS.from_texts(texts = text_chunks, embedding = embeddings)
    return vector_store

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", layout="wide")
    
    st.write(css, unsafe_allow_html=True)
    st.header("Chat with multiple PDFs :books:")

    # st.text_input("Ask a question about your documents:")
    user_query = st.text_input("Ask a question about your documents:")
        # Create an empty container to display the output
    output_container = st.empty()

    if user_query:
        # Process the user's query (you would replace this with your actual processing logic)
        # For demonstration, we'll simply echo back the query as the output
        output = generate_response(user_query)

        # Update the content of the output container with the generated output
        output_container.write(output)

    # st.write(user_template.replace(
    #             "{{MSG}}", "Hello human"), unsafe_allow_html=True)
    # st.write(bot_template.replace(
    #             "{{MSG}}", "Hello Bot"), unsafe_allow_html=True)


    with st.sidebar:
        st.subheader("Your documents")
        docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                text = get_pdf_text(docs)
                run_ingest()
                # # st.write(text)

                # text_chunks = get_text_chunks(text)
                # # st.write(text_chunks)

                # text_embedding = get_text_embedding(text_chunks)

if __name__ == '__main__':
    main()
