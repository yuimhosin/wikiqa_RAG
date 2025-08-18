from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
import re
from nltk import sent_tokenize
import nltk
from wikiqa_rag.model_manager import model_manager  # 导入模型管理器

# 导入scikit-learn用于优化的主题分块
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans

    SKLEARN_AVAILABLE = True
except ImportError:
    print("警告：scikit-learn不可用，主题分块将使用简化版本")
    SKLEARN_AVAILABLE = False


# 下载所需的NLTK数据
def download_nltk_data():
    """下载NLTK所需的数据包"""
    try:
        # 尝试下载punkt_tab（新版本）
        nltk.download('punkt_tab', quiet=True)
        print("NLTK punkt_tab 下载成功")
    except:
        try:
            # 如果punkt_tab不可用，尝试下载punkt（旧版本）
            nltk.download('punkt', quiet=True)
            print("NLTK punkt 下载成功")
        except Exception as e:
            print(f"NLTK数据下载失败: {e}")
            print("将使用简单的句子分割方法")


# 初始化时下载NLTK数据
download_nltk_data()


def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def simple_sentence_split(text):
    """简单的句子分割方法，作为NLTK的备选方案"""
    # 使用基本的标点符号分割句子
    sentences = re.split(r'[.!?]+', text)
    # 过滤空句子并添加句号
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def build_vector_db(qa_texts, chunking_strategy="character", chunk_size=200, chunk_overlap=20, num_topics=5):
    # 处理输入数据 - 确保是字符串列表
    processed_texts = []

    if isinstance(qa_texts, list):
        for qa in qa_texts:
            if isinstance(qa, list):
                # 如果是列表的列表，将内部列表转换为字符串
                processed_texts.append(' '.join(str(item) for item in qa))
            elif isinstance(qa, str):
                # 如果已经是字符串，直接使用
                processed_texts.append(qa)
            else:
                # 其他类型转换为字符串
                processed_texts.append(str(qa))
    else:
        # 如果不是列表，转换为单元素列表
        processed_texts = [str(qa_texts)]

    # 创建初始文档
    docs = [Document(page_content=text) for text in processed_texts]

    # 初始化分块结果
    split_docs = []

    # 根据选择的分块策略进行分块
    if chunking_strategy == "character":
        # 固定字符数分块
        splitter = CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        for doc in docs:
            chunks = splitter.split_text(doc.page_content)
            split_docs.extend([Document(page_content=chunk) for chunk in chunks])

    elif chunking_strategy == "sentence":
        # 基于句子分块
        for doc in docs:
            try:
                # 尝试使用NLTK分句
                sentences = sent_tokenize(doc.page_content)
            except LookupError:
                # 如果NLTK数据不可用，使用简单分句方法
                print("警告：NLTK数据不可用，使用简单分句方法")
                sentences = simple_sentence_split(doc.page_content)
            except Exception as e:
                print(f"句子分割出错，使用简单分句方法: {e}")
                sentences = simple_sentence_split(doc.page_content)

            split_docs.extend([Document(page_content=sentence) for sentence in sentences if sentence.strip()])

    elif chunking_strategy == "topic":
        # 基于主题分块 - 使用完全优化的方法
        print("使用优化的主题分块策略...")
        split_docs = optimized_topic_chunking(processed_texts, num_topics)

        if not split_docs:
            print("主题分块失败，改用字符分块")
            return build_vector_db(qa_texts, chunking_strategy="character",
                                   chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    else:
        raise ValueError(f"不支持的分类策略: {chunking_strategy}")

    # 确保有分块文档
    if not split_docs:
        print("警告：分块结果为空，使用原始文档")
        split_docs = docs

    # 创建向量数据库 - 使用统一的模型管理器
    try:
        print("正在创建向量数据库...")
        embeddings = model_manager.get_embedding_model()  # 使用单例模型
        db = FAISS.from_documents(split_docs, embeddings)
        print(f"向量数据库创建成功，包含 {len(split_docs)} 个文档块")
        return db
    except Exception as e:
        print(f"向量数据库创建失败: {e}")
        raise


def demo_chunking_strategies(qa_texts):
    """演示不同的分块策略"""
    print("请选择分块策略：")
    print("1. 基于字符分块 (Character-based) - 快速，适合一般用途")
    print("2. 基于句子分块 (Sentence-based) - 保持语义完整性")
    print("3. 基于主题分块 (Topic-based) - 智能分组，适合长文档")

    choice = input("请输入选择的分块策略（1/2/3）：").strip()

    strategies = {
        "1": "character",
        "2": "sentence",
        "3": "topic"
    }

    if choice not in strategies:
        print("无效的选择，请输入 1、2 或 3。")
        return

    selected_strategy = strategies[choice]
    print(f"你选择了 {selected_strategy} 策略")

    # 根据策略获取参数
    if selected_strategy == "topic":
        num_topics = int(input("请输入主题数量（默认为5）：") or 5)
        print(f"开始使用优化的主题分块策略（主题数: {num_topics}）...")
        db = build_vector_db(
            qa_texts,
            chunking_strategy=selected_strategy,
            num_topics=num_topics
        )
    else:
        # 字符和句子分块策略
        if selected_strategy == "character":
            chunk_size = int(input("请输入分块大小（默认为200）：") or 200)
            chunk_overlap = int(input("请输入分块重叠大小（默认为20）：") or 20)
            db = build_vector_db(
                qa_texts,
                chunking_strategy=selected_strategy,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        else:
            db = build_vector_db(
                qa_texts,
                chunking_strategy=selected_strategy
            )

    print(f"\n=== 分块策略 '{selected_strategy}' 结果 ===")
    print(f"分块数量: {len(db.index_to_docstore_id)}")

    # 打印前几个分块内容和元数据
    print("\n前3个分块预览:")
    for i, doc_id in enumerate(list(db.index_to_docstore_id.values())[:3]):
        doc = db.docstore.search(doc_id)
        print(f"\n--- 分块 #{i + 1} ---")
        print(f"内容: {doc.page_content[:150]}...")

        if doc.metadata:
            print(f"元数据: {doc.metadata}")
        print("-" * 50)

    # 如果是主题分块，显示主题统计
    if selected_strategy == "topic":
        print("\n=== 主题分块统计 ===")
        topic_stats = {}
        total_sentences = 0

        for doc_id in db.index_to_docstore_id.values():
            doc = db.docstore.search(doc_id)
            if doc.metadata:
                topic_id = doc.metadata.get('topic', 'unknown')
                sentence_count = doc.metadata.get('sentence_count', 0)
                method = doc.metadata.get('method', 'unknown')

                if topic_id not in topic_stats:
                    topic_stats[topic_id] = {'sentences': 0, 'method': method}
                topic_stats[topic_id]['sentences'] += sentence_count
                total_sentences += sentence_count

        print(f"总句子数: {total_sentences}")
        print(f"主题数量: {len(topic_stats)}")
        print(f"分块方法: {list(topic_stats.values())[0]['method'] if topic_stats else 'unknown'}")

        for topic_id, stats in sorted(topic_stats.items()):
            percentage = (stats['sentences'] / total_sentences * 100) if total_sentences > 0 else 0
            print(f"主题 {topic_id}: {stats['sentences']} 句子 ({percentage:.1f}%)")

    return db


# 示例用法
if __name__ == "__main__":
    # 示例数据
    sample_qa_texts = [
        "What is machine learning? Machine learning is a subset of artificial intelligence that focuses on algorithms.",
        "How does deep learning work? Deep learning uses neural networks with multiple layers to learn patterns.",
        "What are the applications of AI? AI is used in healthcare, finance, transportation, and many other fields."
    ]

    demo_chunking_strategies(sample_qa_texts)