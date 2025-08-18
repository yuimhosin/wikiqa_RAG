# retriever_test.py

def test_vector_retrieval_and_answer(db, qa, test_question="What is the capital of France?"):
    """
    验证向量数据库是否被调用，并打印模型的最终回答，同时返回检索到的相关文档。
    """
    print("\n======= RAG 向量检索验证 =======")
    print(f"【测试问题】：{test_question}\n")

    retriever = db.as_retriever(search_kwargs={"k": 3})  # 获取前3个最相关的文档
    docs = retriever.get_relevant_documents(test_question)

    if not docs:
        print("未检索到任何相关文档。请检查向量数据库是否构建成功。")
        return

    # 打印检索到的文档
    print("检索到的相关文档（前3条）：")
    for i, doc in enumerate(docs, 1):
        print(f"\n[文档 {i}]：\n{doc.page_content}")

    try:
        # 调用问答系统并传递相关文档
        result = qa.invoke({
            "query": test_question,
            "documents": docs  # 将检索到的文档传递给模型进行回答
        })
        print("\nLLM 最终回答：")
        print(result.get("result", result))
    except Exception as e:
        print("调用 LLM 失败：", str(e))
