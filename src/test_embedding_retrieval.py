# -*- coding: utf-8 -*-
# @Author     : lipf702@gmail.com
# @Since      : 12/10/23
# @Description:

import re

from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS


class TestEmbeddingRetrieval:
    def __init__(self):
        self.test_set = []

    def set_up(self) -> None:
        """ 读取测试数据 """
        test_file_path = "../data/test_set/testset_WLAN 维护宝典（分销）_retrieval_v1.txt"
        with open(test_file_path, "r", encoding="utf-8") as fr:
            for line in fr.readlines():
                q, t = line.split("|||")
                self.test_set.append((q.strip(), t.strip()))

    @staticmethod
    def compare_two_sentence(short_x, long_y) -> bool:
        """ 比较两个字符串，只要短串内容出现（不需要相邻）在长串中即认为相似 """
        m, n = len(short_x), len(long_y)

        if m > n:
            return False

        dp = [[False] * (n + 1) for _ in range(m + 1)]

        dp[0][0] = True

        for i in range(m + 1):
            for j in range(n + 1):
                if i == 0 and j == 0:
                    continue
                elif i == 0:
                    dp[i][j] = True
                elif j == 0:
                    dp[i][j] = False
                elif short_x[i - 1] == long_y[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = dp[i][j - 1]

        return dp[m][n]

    def test_embedding_retrieval(self) -> None:
        if not self.test_set:
            return
        # 读取向量索引
        embeddings = HuggingFaceEmbeddings(model_name="shibing624/text2vec-base-chinese")
        index_path = "../data/index/"
        vector_store = FAISS.load_local(index_path, embeddings, index_name="WLAN 维护宝典（分销）")
        correct_cnt = 0
        for q, a in self.test_set:
            print(f"Q: {q}")
            # 去除空白符
            a_rm_blank = re.sub(r"\s+", "", a)
            # 召回top3文档
            sim_docs = vector_store.similarity_search(q, k=3, fetch_k=30)
            for doc in sim_docs:
                # 比较文本相似度
                if self.compare_two_sentence(a_rm_blank, doc.page_content):
                    correct_cnt += 1
                    print(f"A: {doc.page_content}")
                    break
            print("===" * 3)

        print(f"\n"
              f">> Correct: {correct_cnt}\n"
              f">> Recall accuracy(@top3): {correct_cnt / len(self.test_set) * 100:.2f}%"
              )


if __name__ == '__main__':
    ter = TestEmbeddingRetrieval()
    ter.set_up()
    ter.test_embedding_retrieval()
