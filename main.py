# main.py

import os
from data_processing import load_and_clean_data
from vector_db import build_vector_db
from qa_system import create_qa_chain
from evaluation import evaluate_model
from retriever_test import test_vector_retrieval_and_answer  # 导入新的验证模块
from feedback import get_user_feedback, save_feedback, feedback_data  # 导入反馈相关函数
from reranking import (
    create_reranker,
    show_retrieved_documents_with_rerank,
    analyze_retrieval_quality_with_rerank,
    show_contrastive_learning_stats,
    demo_reranking_functionality,
    test_reranking_performance
)  # 导入重排序模块
from wikiqa_rag.model_manager import model_manager  # 导入模型管理器


def main():
    # 配置环境变量
    os.environ["OPENAI_BASE_URL"] = "https://api.deepseek.com/v1"
    os.environ["OPENAI_API_KEY"] = "your_api_key"

    # 加载并清洗 WikiQA 数据（包含负样本对比学习）
    path = "WikiQA/WikiQA-train.tsv"
    if not os.path.exists(path):
        raise FileNotFoundError("请确保 WikiQA 数据集已解压并放在 WikiQA/ 文件夹中")

    print("正在加载和清洗数据，包含负样本生成...")
    qa_texts, sample_info = load_and_clean_data(path, include_negative_samples=True, negative_ratio=0.3)

    # 创建对比学习数据
    from data_processing import create_contrastive_learning_data
    contrastive_data = create_contrastive_learning_data(qa_texts, sample_info)

    print(f"数据加载完成:")
    print(f"   - 总文档数: {len(qa_texts)}")
    print(f"   - 正样本: {len(contrastive_data['positive_samples'])}")
    print(f"   - 负样本: {len(contrastive_data['negative_samples'])}")
    print(f"   - 对比学习三元组: {len(contrastive_data['triplets'])}")

    # 选择分块策略
    print("\n请选择分块策略:")
    print("1. 基于字符分块 (Character-based) - 适合一般用途")
    print("2. 基于句子分块 (Sentence-based) - 保持语义完整性")
    print("3. 基于主题分块 (Topic-based) - 根据主题聚类")
    print("4. 使用默认策略 (Character-based, 200字符)")

    while True:
        choice = input("请输入选择（1/2/3/4）: ").strip()

        if choice == "1":
            chunk_size = int(input("请输入分块大小（默认200）: ") or 200)
            chunk_overlap = int(input("请输入分块重叠大小（默认20）: ") or 20)
            print(f"使用字符分块策略构建向量数据库（块大小: {chunk_size}, 重叠: {chunk_overlap}）...")
            db = build_vector_db(qa_texts, chunking_strategy="character",
                                 chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            break

        elif choice == "2":
            print("使用句子分块策略构建向量数据库...")
            db = build_vector_db(qa_texts, chunking_strategy="sentence")
            break

        elif choice == "3":
            num_topics = int(input("请输入主题数量（默认5）: ") or 5)
            print(f"使用主题分块策略构建向量数据库（主题数: {num_topics}）...")
            db = build_vector_db(qa_texts, chunking_strategy="topic", num_topics=num_topics)
            break

        elif choice == "4":
            print("使用默认分块策略构建向量数据库...")
            db = build_vector_db(qa_texts)
            break

        else:
            print("无效选择，请输入 1、2、3 或 4")
            continue

    print(f"向量数据库构建完成！分块数量: {len(db.index_to_docstore_id)}")

    # 预加载嵌入模型以避免后续缓存未命中
    print("\n预加载嵌入模型...")
    model_manager.get_embedding_model()

    # 创建问答系统
    qa = create_qa_chain(db)  # Ensure this line is executed to initialize `qa`

    # 创建重排序器
    print("正在初始化重排序器...")
    try:
        # 使用对比学习重排序器
        reranker = create_reranker("contrastive")
        print("对比学习重排序器初始化成功")
    except Exception as e:
        print(f"对比学习重排序器初始化失败，使用简单重排序器: {e}")
        reranker = create_reranker("simple")
        print("简单重排序器初始化成功")

    # 评估模型
    print("\n" + "=" * 60)
    print("模型性能评估")
    print("=" * 60)

    eval_choice = input("是否进行模型评估？(y/n，默认n): ").strip().lower()
    if eval_choice == 'y':
        try:
            max_questions = int(input("请输入评估问题数量（默认50）: ") or 50)
            test_path = input("请输入测试集路径（默认WikiQA/WikiQA-test.tsv）: ").strip() or "WikiQA/WikiQA-test.tsv"

            print(f"正在评估模型准确率（测试{max_questions}个问题）...")
            evaluate_model(qa, tsv_path=test_path, max_q=max_questions)
        except FileNotFoundError:
            print("测试集文件未找到，跳过评估")
        except Exception as e:
            print(f"评估过程出错: {e}")
    else:
        print("跳过模型评估")

    # 调用检索验证函数，进行外部知识验证
    print("\n" + "=" * 60)
    print("外部知识验证测试")
    print("=" * 60)
    test_vector_retrieval_and_answer(db, qa, test_question="What was `Freedom Summer`")

    # 显示对比学习数据统计
    show_contrastive_learning_stats(contrastive_data)

    # 演示重排序功能
    print("\n" + "=" * 60)
    print("重排序功能演示")
    print("=" * 60)
    demo_reranking_functionality(db, reranker, contrastive_data)

    # 性能测试（可选）
    print("\n" + "=" * 60)
    print("重排序性能测试")
    print("=" * 60)
    test_reranking_performance(db, reranker, contrastive_data)

    print("\n" + "=" * 60)
    print("RAG问答系统已启动！")
    print("每次回答后会显示相关的知识库文档")
    print("请对每次回答进行评价以帮助改进系统")
    print("输入 'exit' 或 'quit' 退出系统")
    print("=" * 60)

    while True:
        question = input("\n你的问题： ").strip()
        if question.lower() in {"exit", "quit"}:
            print("\n本次会话反馈统计：")
            if feedback_data:
                correct_count = sum(1 for item in feedback_data if item['feedback'] == 1)
                total_count = len(feedback_data)
                accuracy = (correct_count / total_count) * 100
                print(f"总问题数：{total_count}")
                print(f"正确回答：{correct_count}")
                print(f"准确率：{accuracy:.1f}%")
            else:
                print("本次会话无反馈数据")
            print("已退出问答系统，再见！")
            break

        if not question:
            print("问题不能为空，请重新输入。")
            continue

        try:
            # 显示检索到的相关文档（包含重排序）
            retrieved_docs = show_retrieved_documents_with_rerank(db, question, contrastive_data, reranker)

            # 显示简化的相关文档信息
            print("\n检索到的相关文档概览:")
            retriever = db.as_retriever(search_kwargs={"k": 3})
            docs = retriever.get_relevant_documents(question)

            if docs:
                for i, doc in enumerate(docs[:3], 1):
                    print(f"\n[文档 {i}]:")
                    print(f"{doc.page_content[:100]}...")
                print("-" * 50)
            else:
                print("未检索到相关文档")

            # 获取AI回答
            print("\nAI正在思考...")
            answer = qa.invoke({"query": question})
            final_answer = answer.get("result", answer)

            print(f"\nAI回答：")
            print(f"{final_answer}")

            # 分析检索质量（包含重排序效果）
            analyze_retrieval_quality_with_rerank(question, retrieved_docs, contrastive_data)

            # 获取用户反馈
            print("\n" + "-" * 40)
            try:
                feedback = get_user_feedback()
                save_feedback(question, final_answer, feedback)

                if feedback == 1:
                    print("感谢您的正面反馈！")
                else:
                    print("感谢您的反馈，我们会持续改进！")
                    print("提示：您可以尝试更具体的问题描述")

            except ValueError:
                print("输入无效，跳过反馈环节")
            except KeyboardInterrupt:
                print("\n用户中断，退出系统")
                break

        except Exception as e:
            print(f"发生错误：{str(e)}")
            print("请检查网络连接或稍后重试")


if __name__ == "__main__":
    main()