from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
import re
import os


class DocumentProcessor:
    def __init__(self,
                 model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                 path_of_vectorDB="vectordb/chroma/"):
        self.model_name = model_name
        self.path_of_vectorDB = path_of_vectorDB
        self.all_documents = []
        self.docs_split = None
        self.vectordb = None

    def load_data(self, path):
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith('.pdf'):
                    pdf_path = os.path.join(root, file)
                    print(pdf_path)
                    try:
                        loader = PyPDFLoader(str(pdf_path))
                        docs = loader.load()
                        self.all_documents.extend(docs)
                    except Exception as e:
                        print(f"Error processing {pdf_path}: {e}")
                else:
                    print(f"Skipping non-PDF file: {file}")
        # print(self.all_documents)

    def clean_data(self):
        for doc in self.all_documents:
            # Delete \n and white space
            doc.page_content = doc.page_content.replace('\n', ' ')
            doc.page_content = re.sub(' +', ' ', doc.page_content)

    def split_documents(self, chunk_size=2000, chunk_overlap=300):
        splitter = CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator=" "
        )
        self.docs_split = splitter.split_documents(self.all_documents)

    def embedding(self):
        embedding = SentenceTransformerEmbeddings(model_name=self.model_name)
        self.vectordb = Chroma.from_documents(
            documents=self.docs_split,
            embedding=embedding,
            persist_directory=self.path_of_vectorDB
        )


doc_processor = DocumentProcessor()
doc_processor.load_data(path="../../Stock/Company/Visa/")
doc_processor.clean_data()
doc_processor.split_documents()
doc_processor.embedding()
