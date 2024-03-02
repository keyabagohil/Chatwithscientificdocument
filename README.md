The following are the required dependencies for running our RAG pipeline:

ctransformers
faiss-cpu
langchain==0.0.331
pypdf
python-box
sentence-transformers
python-docx
python-pptx
easyocr
bs4
streamlit

After installing these mentioned libraries we need to install a pretrained LLM for obtaining pretrained transformer. The link for Mistral AI model is: 
https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q2_K.gguf (The following model is modified by developer named TheBloke)

After placing the model in the directory(mistral\models\), run app.py using command "streamlit run app.py"

Running the commmand will open a web interface where you can upload the documents on the sidebar and click on the process button to inject the documents into the RAG vectorstore.

Once the processing is finished, the user can enter prompts to extract information from the provided documents.
------------------------------------------------------------------------------------------------------------------------

PROGRAM FLOW

1) After uploading the files, first the ingest.py is executed in which different document types are passed and are divided into chunks using RecursiveCharacterTextSplitter.
   These chunks are mapped in order to refer the sources once the response is fetched from the RAG pipeline and fed to LLM. These chunks are stored in FAISS vectorstore database
   which is created in local system in order to reduce latency.

2) The user prompt is given to the generate_response present in main.py file. This function then parses the query and then sends the request to RAG database from which vectors are retrieved
   and passed to LLM for generating response that can be interpreted.
