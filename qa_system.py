# qa_system.py

from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

def create_qa_chain(db):
    """
    创建问答系统链
    """
    llm = ChatOpenAI(model_name="deepseek-chat")
    return RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever(), return_source_documents=True)
