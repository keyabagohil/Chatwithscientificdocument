import box
# from langchain.chains import create_tagging_chain
import yaml
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Import config vars
with open('config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))


def run_ingest():
    loader = DirectoryLoader(cfg.DATA_PATH,
                             glob='*.pdf',
                             loader_cls=PyPDFLoader)
    print(loader)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=cfg.CHUNK_SIZE,
                                                   chunk_overlap=cfg.CHUNK_OVERLAP, 
                                                   length_function = len,
                                                   add_start_index=True
    )
    documents = loader.load()
    docs=[]
    ctr=0
    for i in documents:
      doc=i.page_content
      docs+=text_splitter.create_documents([doc])
      # print(text_splitter.create_documents([doc]))
    #   print(docs)

    # print(docs)
    # print(docs.shape)
    # print(docs)
    # texts = text_splitter.split_documents(documents)
    texts=[doc.page_content for doc in docs]
    # print(texts)
    embeddings = HuggingFaceEmbeddings(model_name=cfg.EMBEDDINGS,
                                       model_kwargs={'device': 'cpu'})
    # print(texts)
    metadata=[{"source":text} for text in texts]
    vectorstore = FAISS.from_texts(texts, embeddings,metadatas=metadata)
    vectorstore.save_local(cfg.DB_FAISS_PATH)

if __name__ == "__main__":
    run_ingest()
