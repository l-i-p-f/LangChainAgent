# -*- coding: utf-8 -*-
# @Author     : lipf702@gmail.com
# @Since      : 12/7/23
# @Description:

import os

from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def set_up_openai_api_key():
    config_path = "../config/openai_api_key.local"
    with open(config_path, "r") as fr:
        key = fr.read().strip()
    os.environ["OPENAI_API_KEY"] = key


class QAAgent:
    def __init__(self):
        filepath = "../data/pdf/WLAN 维护宝典（分销）.pdf"
        self.vector_store = self.set_up_vector_store(filepath)
        self.query = "hi"
        self.context = []
        self.response = ""

    @staticmethod
    def set_up_vector_store(filepath):
        # load file
        loader = PyPDFLoader(file_path=filepath)
        docs = loader.load()
        doc_text = " ".join([doc.page_content for doc in docs])
        # split text
        text_spliter = CharacterTextSplitter(
            separator="。\n",
            chunk_size=800,
            chunk_overlap=200,
        )
        docs_split = text_spliter.split_text(doc_text)
        # build embedding
        embedding_model = HuggingFaceEmbeddings(model_name="shibing624/text2vec-base-chinese")
        vector_store = FAISS.from_texts(docs_split, embedding_model)
        return vector_store

    def search_engine(self, query):
        print(f"Q: {query}")
        # search
        sim_docs = self.vector_store.similarity_search(query)
        search_res = [doc.page_content for doc in sim_docs]
        print("Searched:")
        print("===" * 3)
        for idx, doc in enumerate(search_res):
            print(f"{idx}. {doc}\n")
        print("===" * 3)
        # prompt
        prompt = PromptTemplate(
            input_variables=["context", "query"],
            template="已知信息:\n{context}\n根据已知信息回答问题:\n{query}",
        )
        # llm
        llm = ChatOpenAI()
        chain = LLMChain(llm=llm, prompt=prompt)
        response = chain.run({
            "context": search_res,
            "query": query,
        })
        print("A:")
        print(response)
        self.query = query
        self.context = search_res
        self.response = response


def main():
    set_up_openai_api_key()
    query = "终端连不上Wi-Fi信号"
    qa_agent = QAAgent()
    qa_agent.search_engine(query)
    print("Finished.")


if __name__ == '__main__':
    main()
