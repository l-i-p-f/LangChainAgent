# -*- coding: utf-8 -*-
# @Author     : lipf702@gmail.com
# @Since      : 12/10/23
# @Description:


from langchain.chat_models import ChatOpenAI

from src import _set_up_openai_api_key

if __name__ == '__main__':
    _set_up_openai_api_key()
    query = """
    python program runs error as below:
        TypeError: Wrong number or type of arguments for overloaded function 'write_index'.
          Possible C/C++ prototypes are:
        faiss::write_index(faiss::Index const *,char const *)
        faiss::write_index(faiss::Index const *,FILE *)
        faiss::write_index(faiss::Index const *,faiss::IOWriter *)
    how to fix this?
    """
    llm = ChatOpenAI()
    print(llm.predict(query))
