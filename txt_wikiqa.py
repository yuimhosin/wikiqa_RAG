import os
import re
import pandas as pd
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.llms import HuggingFaceHub
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")  # 推荐中文场景可用 bge-large-zh

import difflib
# ========== 1. 设置 OpenAI API ==========
os.environ["OPENAI_BASE_URL"] = "https://api.deepseek.com/v1"
os.environ["OPENAI_API_KEY"] = "sk-51fad469bb434d7ca141a6d6e5533760"

# ========== 2. 加载与清洗 WikiQA 数据 ==========
def load_and_clean_data(tsv_path="WikiQA/WikiQA-train.tsv"):
    df = pd.read_csv(tsv_path, sep="\t")
    df = df[df["Label"] == 1]
    qa_pairs = df.apply(lambda row: f"问题：{row['Question']}\n答案：{row['Sentence']}", axis=1)
    cleaned = [re.sub(r'\s+', ' ', qa).strip() for qa in qa_pairs]
    return cleaned

# ========== 3. 构建向量数据库 ==========
def build_vector_db(qa_texts):
    docs = [Document(page_content=qa) for qa in qa_texts]
    splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    split_docs = [Document(page_content=chunk) for doc in docs for chunk in splitter.split_text(doc.page_content)]
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    db = FAISS.from_documents(split_docs, embeddings)
    return db

# ========== 4. 构建问答系统 ==========
def create_qa_chain(db):
    llm = ChatOpenAI(model_name="deepseek-chat")
    return RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())

# ========== 5. 模型评估模块 ==========
def get_embedding(text, model):
    return model.embed_query(text)

def is_semantic_match(pred, truths, embed_model, threshold=0.7):
    pred_emb = get_embedding(pred, embed_model)
    for ans in truths:
        ans_emb = get_embedding(ans, embed_model)
        sim = cosine_similarity([pred_emb], [ans_emb])[0][0]
        if sim >= threshold:
            return True
    return False

def evaluate_model(qa_chain, tsv_path="WikiQA/WikiQA-test.tsv", max_q=50):
    print(f" 正在加载测试集：{tsv_path}")
    df = pd.read_csv(tsv_path, sep="\t")
    grouped = df.groupby("Question")

    embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    total, correct = 0, 0

    for i, (question, group) in enumerate(grouped):
        if i >= max_q:
            break
        true_answers = group[group["Label"] == 1]["Sentence"].tolist()
        if not true_answers:
            continue
        try:
            prediction = qa_chain.invoke({"query": question})
            pred_text = prediction.get("result", "")
        except Exception as e:
            print(f" 问题出错：{question} | {e}")
            continue

        matched = is_semantic_match(pred_text, true_answers, embed_model, threshold=0.8)
        total += 1
        if matched:
            correct += 1

        result = "correct" if matched else "false"
        print(f"{result} [{i+1}] Q: {question}\n→ 预测: {pred_text}\n→ 正确: {true_answers[0]}\n")

    if total == 0:
        print("评估失败：无有效问题被处理。")
    else:
        print(f"评估完成：共 {total} 个问题，准确回答 {correct} 个，准确率 = {correct / total:.2%}")

# ========== 6. 主程序（循环问答） ==========
def main():
    path = "WikiQA/WikiQA-train.tsv"
    if not os.path.exists(path):
        raise FileNotFoundError("请确保 WikiQA 数据集已解压并放在 WikiQA/ 文件夹中")

    print("正在加载 WikiQA 数据...")
    qa_texts = load_and_clean_data(path)

    print("正在构建向量数据库...")
    db = build_vector_db(qa_texts)

    print("正在创建问答系统...")
    qa = create_qa_chain(db)

    print(" 正在评估模型准确率...")
    evaluate_model(qa, tsv_path="WikiQA/WikiQA-test.tsv", max_q=50)

    print("你可以开始提问了（输入 'exit' 或 'quit' 退出）：")
    while True:
        question = input("你的问题： ").strip()
        if question.lower() in {"exit", "quit"}:
            print("已退出问答系统，再见！")
            break
        if not question:
            print("问题不能为空，请重新输入。")
            continue
        try:
            answer = qa.invoke({"query": question})
            print("答案：", answer.get("result", answer))
        except Exception as e:
            print("发生错误：", str(e))

if __name__ == "__main__":
    main()
