from langchain_community.llms import Ollama
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA


class Chatbot:
    def __init__(self, llm_name="llama3",
                 path_of_vectorDB="vectordb/chroma/",
                 embedding_model="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"):
        template = """
        Act as investor, your job is to analyze balance sheet, cash flow statement and income statement.
        Context: {context}
        Question: {question}
        """
        llm = Ollama(model=llm_name, temperature=0)
        embedding = SentenceTransformerEmbeddings(model_name=embedding_model)
        vectorDB = Chroma(persist_directory=path_of_vectorDB,
                          embedding_function=embedding)
        prompt = PromptTemplate(input_variables=["context", "question"],
                                template=template)
        retriever = vectorDB.as_retriever(search_kwargs={"k": 6})
        self.bot = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": prompt,
                "verbose": True
            })

    def answer(self, question):
        answer = self.bot({"query": question})
        return answer["result"]