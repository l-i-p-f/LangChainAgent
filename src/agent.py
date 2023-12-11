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

from agent_utils import load_config
from src import _set_up_openai_api_key

os.environ["TOKENIZERS_PARALLELISM"] = "false"

CONFIG = load_config()


class QAAgent:
    def __init__(self):
        filepath = CONFIG["data_path"]["input_path"]
        self.vector_store = self.set_up_vector_store(filepath)
        self.query = "hi"
        self.context = []
        self.response = ""

    @staticmethod
    def set_up_vector_store(filepath) -> FAISS:
        index_path = CONFIG["data_path"]["index_path"]
        model_name = CONFIG["embedding"]["model_name"]
        """ 初始化FAISS向量库 """
        # embedding model
        embedding_model = HuggingFaceEmbeddings(model_name=model_name)
        # load index from disk
        index_name = Path(filepath).stem
        try:
            print(f"加载向量库...")
            return FAISS.load_local(index_path, embedding_model, index_name=index_name)
        except RuntimeError:
            print(f">> 向量加载错误，从文件生成... ")
        # load file
        # 按页读取
        loader = PyPDFLoader(file_path=filepath)
        # 整个文档/拆分十分细粒度元素读取
        # loader = UnstructuredFileLoader(file_path=filepath, mode="elements")
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
        vector_store.save_local(index_path, index_name)
        print(f"向量生成完毕，保存至: {index_path}/{index_name}.faiss")
        return vector_store

    def search_engine(self, query: str) -> None:
        """ 对查询进行向量检索和回答 """
        print(f"Q: {query}")
        # search
        top_k = CONFIG["retrieval"]["top_k"]
        fetch_k = CONFIG["retrieval"]["fetch_k"]
        sim_docs = self.vector_store.similarity_search(query, k=top_k, fetch_k=fetch_k)
        search_res = [doc.page_content for doc in sim_docs]
        print("Searched:")
        print("===" * 3)
        for idx, doc in enumerate(search_res):
            print(f"{idx}. {doc}")
            print("  -- 分割线 --  ")
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
