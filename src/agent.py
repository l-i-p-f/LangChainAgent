# -*- coding: utf-8 -*-
# @Author     : lipf702@gmail.com
# @Since      : 12/7/23
# @Description:

import os
from pathlib import Path

from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS

from src import _set_up_openai_api_key

os.environ["TOKENIZERS_PARALLELISM"] = "false"

LOCAL_INDEX_PATH = "../data/index"


class QAAgent:
    def __init__(self):
        filepath = "../data/pdf/WLAN 维护宝典（分销）.pdf"
        self.vector_store = self.set_up_vector_store(filepath)
        self.query = "hi"
        self.context = []
        self.response = ""

    @staticmethod
    def set_up_vector_store(filepath) -> FAISS:
        """ 初始化FAISS向量库 """
        # embedding model
        embedding_model = HuggingFaceEmbeddings(model_name="shibing624/text2vec-base-chinese")
        # load index from disk
        index_name = Path(filepath).stem
        try:
            print(f"加载向量库...")
            return FAISS.load_local(LOCAL_INDEX_PATH, embedding_model, index_name=index_name)
        except RuntimeError:
            print(f">> 向量加载错误，从文件生成... ")
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
        # embedding
        vector_store = FAISS.from_texts(docs_split, embedding_model)
        # save index to disk
        vector_store.save_local(LOCAL_INDEX_PATH, index_name)
        print(f"向量生成完毕，保存至: {LOCAL_INDEX_PATH}/{index_name}.faiss")
        return vector_store

    def search_engine(self, query: str) -> None:
        """ 对查询进行向量检索和回答 """
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
    _set_up_openai_api_key()
    query = "终端连不上Wi-Fi信号"
    qa_agent = QAAgent()
    qa_agent.search_engine(query)
    print("Finished.")


if __name__ == '__main__':
    main()
