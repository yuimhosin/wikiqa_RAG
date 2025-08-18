# data_processing.py 清洗数据

import pandas as pd
import re
import random
from langchain.embeddings import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
import nltk
import html
from bs4 import BeautifulSoup
import unicodedata
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 下载必要的NLTK数据
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
except Exception as e:
    logger.warning(f"NLTK下载失败: {e}")
    from nltk.corpus import stopwords


class RAGDataCleaner:
    """
    RAG知识库标准数据清洗器
    """

    def __init__(self, min_answer_length=100, max_answer_length=5000,
                 min_question_length=5, similarity_threshold=0.95):
        """
        初始化数据清洗器

        Args:
            min_answer_length: 答案最小长度（字符数）
            max_answer_length: 答案最大长度（字符数）
            min_question_length: 问题最小长度（字符数）
            similarity_threshold: 相似度阈值，用于去重
        """
        self.min_answer_length = min_answer_length
        self.max_answer_length = max_answer_length
        self.min_question_length = min_question_length
        self.similarity_threshold = similarity_threshold
        self.stop_words = set(stopwords.words('english'))

        # 预编译正则表达式提高性能
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.phone_pattern = re.compile(r'(\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})')
        self.special_chars_pattern = re.compile(r'[^\w\s.,!?;:()\-\'\"]+')
        self.multiple_spaces_pattern = re.compile(r'\s+')
        self.multiple_punctuation_pattern = re.compile(r'([.!?]){2,}')

    def remove_html_tags(self, text):
        """
        去除HTML标签和HTML实体
        """
        if not text:
            return ""

        # 解码HTML实体
        text = html.unescape(text)

        # 使用BeautifulSoup去除HTML标签
        try:
            soup = BeautifulSoup(text, 'html.parser')
            # 去除script和style标签及其内容
            for script in soup(["script", "style"]):
                script.decompose()
            text = soup.get_text()
        except Exception as e:
            logger.warning(f"BeautifulSoup解析失败，使用正则表达式: {e}")
            # 备用方案：使用正则表达式
            text = re.sub(r'<[^>]+>', '', text)

        return text

    def normalize_unicode(self, text):
        """
        Unicode标准化
        """
        if not text:
            return ""

        # 标准化Unicode字符
        text = unicodedata.normalize('NFKC', text)

        # 去除零宽字符
        text = re.sub(r'[\u200b\u200c\u200d\ufeff]', '', text)

        return text

    def remove_urls_emails_phones(self, text):
        """
        去除URL、邮箱和电话号码
        """
        if not text:
            return ""

        # 去除URL
        text = self.url_pattern.sub('[URL]', text)

        # 去除邮箱
        text = self.email_pattern.sub('[EMAIL]', text)

        # 去除电话号码
        text = self.phone_pattern.sub('[PHONE]', text)

        return text

    def clean_text_content(self, text):
        """
        清理文本内容
        """
        if not text:
            return ""

        # 去除制表符和换行符
        text = re.sub(r'[\t\n\r\f\v]+', ' ', text)

        # 去除过多的特殊字符，但保留基本标点
        text = self.special_chars_pattern.sub(' ', text)

        # 标准化标点符号
        text = self.multiple_punctuation_pattern.sub(r'\1', text)

        # 去除多余空格
        text = self.multiple_spaces_pattern.sub(' ', text)

        return text.strip()

    def is_valid_text(self, text, min_length):
        """
        检查文本是否有效
        """
        if not text or len(text.strip()) < min_length:
            return False

        # 检查是否主要是特殊字符或数字
        word_chars = re.sub(r'[^\w]', '', text)
        if len(word_chars) < min_length * 0.6:  # 至少60%是单词字符
            return False

        # 检查是否有意义的内容（不全是重复字符）
        if len(set(text.lower())) < 5:  # 至少5个不同字符
            return False

        return True

    def remove_duplicates(self, qa_pairs):
        """
        去除重复和高度相似的问答对
        """
        if not qa_pairs:
            return []

        unique_pairs = []
        seen_questions = set()
        seen_answers = set()

        for question, answer in qa_pairs:
            # 简单去重：基于精确匹配
            q_lower = question.lower().strip()
            a_lower = answer.lower().strip()

            if q_lower in seen_questions or a_lower in seen_answers:
                continue

            seen_questions.add(q_lower)
            seen_answers.add(a_lower)
            unique_pairs.append((question, answer))

        logger.info(f"去重后保留 {len(unique_pairs)} 个问答对（原有 {len(qa_pairs)} 个）")
        return unique_pairs

    def clean_single_text(self, text):
        """
        清洗单个文本
        """
        if not text:
            return ""

        # 1. Unicode标准化
        text = self.normalize_unicode(text)

        # 2. 去除HTML标签
        text = self.remove_html_tags(text)

        # 3. 去除URL、邮箱、电话
        text = self.remove_urls_emails_phones(text)

        # 4. 清理文本内容
        text = self.clean_text_content(text)

        # 5. 基本文本处理
        text = text.lower()

        # 6. 可选：拼写纠错
        # try:
        #     text = str(TextBlob(text).correct())
        # except:
        #     pass

        # 7. 去除停用词（保留问号等重要标点）
        words = text.split()
        filtered_words = []
        for word in words:
            if word not in self.stop_words or word in ['?', '!', '.']:
                filtered_words.append(word)

        text = ' '.join(filtered_words)

        # 8. 最终清理
        text = self.multiple_spaces_pattern.sub(' ', text).strip()

        return text


def load_and_clean_data(tsv_path="WikiQA/WikiQA-train.tsv", include_negative_samples=True, negative_ratio=0.3):
    """
    加载并清洗 WikiQA 数据集，应用标准RAG数据清洗流程，并生成负样本用于对比学习

    Args:
        tsv_path: 数据集路径
        include_negative_samples: 是否包含负样本
        negative_ratio: 负样本与正样本的比例
    """
    logger.info(f"开始加载数据集: {tsv_path}")

    # 初始化清洗器
    cleaner = RAGDataCleaner(
        min_answer_length=15,  # 答案至少15个字符
        max_answer_length=800,  # 答案最多800个字符
        min_question_length=8,  # 问题至少8个字符
        similarity_threshold=0.95
    )

    try:
        # 读取数据集
        df = pd.read_csv(tsv_path, sep="\t", encoding='utf-8')
        logger.info(f"原始数据集大小: {len(df)}")

        # 数据预处理
        df = df.dropna(subset=['Question', 'Sentence'])  # 去除空值

        # 分别处理正样本和负样本
        positive_df = df[df["Label"] == 1]  # 正确答案
        negative_df = df[df["Label"] == 0]  # 错误答案
        logger.info(f"正样本数量: {len(positive_df)}, 负样本数量: {len(negative_df)}")

        # 处理正样本
        logger.info("开始处理正样本...")
        positive_pairs = []

        for idx, row in positive_df.iterrows():
            clean_question = cleaner.clean_single_text(row['Question'])
            clean_answer = cleaner.clean_single_text(row['Sentence'])

            if (cleaner.is_valid_text(clean_question, cleaner.min_question_length) and
                    cleaner.is_valid_text(clean_answer, cleaner.min_answer_length) and
                    len(clean_answer) <= cleaner.max_answer_length):
                positive_pairs.append((clean_question, clean_answer, 1))  # 添加标签

        logger.info(f"正样本清洗后保留: {len(positive_pairs)} 个")

        # 处理负样本
        negative_pairs = []
        if include_negative_samples and len(negative_df) > 0:
            logger.info("开始生成负样本...")
            negative_pairs = generate_negative_samples(
                positive_pairs, negative_df, cleaner, negative_ratio
            )
            logger.info(f"生成负样本: {len(negative_pairs)} 个")

        # 合并正负样本
        all_pairs = positive_pairs + negative_pairs

        # 去除重复
        all_pairs = cleaner.remove_duplicates([(pair[0], pair[1]) for pair in all_pairs])

        # 组合问题和答案用于向量化
        qa_texts = []
        sample_info = []  # 记录样本信息用于后续分析

        for question, answer in all_pairs:
            # 使用特殊分隔符连接问题和答案
            combined_text = f"Question: {question} Answer: {answer}"
            qa_texts.append(combined_text)

            # 判断是否为负样本（简单启发式方法）
            is_negative = any(neg_q == question for neg_q, _, _ in negative_pairs)
            sample_info.append({
                'text': combined_text,
                'question': question,
                'answer': answer,
                'is_negative': is_negative
            })

        logger.info(f"最终清洗完成，生成 {len(qa_texts)} 个文档用于RAG")

        # 数据质量报告
        if qa_texts:
            avg_length = sum(len(text) for text in qa_texts) / len(qa_texts)
            positive_count = len(positive_pairs)
            negative_count = len(negative_pairs)

            logger.info(f"平均文档长度: {avg_length:.1f} 字符")
            logger.info(f"正样本数量: {positive_count} ({positive_count / len(qa_texts) * 100:.1f}%)")
            logger.info(f"负样本数量: {negative_count} ({negative_count / len(qa_texts) * 100:.1f}%)")
            logger.info(f"最短文档: {min(len(text) for text in qa_texts)} 字符")
            logger.info(f"最长文档: {max(len(text) for text in qa_texts)} 字符")

        return qa_texts, sample_info  # 返回文本和样本信息

    except Exception as e:
        logger.error(f"数据处理失败: {e}")
        raise


def clean_text(text):
    """
    兼容性函数：简单文本清洗
    """
    cleaner = RAGDataCleaner()
    return cleaner.clean_single_text(text)


# 额外的数据质量检查函数
def analyze_data_quality(qa_texts):
    """
    分析清洗后数据的质量
    """
    if not qa_texts:
        return "没有数据可分析"

    total_docs = len(qa_texts)
    lengths = [len(text) for text in qa_texts]

    # 统计信息
    stats = {
        "总文档数": total_docs,
        "平均长度": sum(lengths) / total_docs,
        "最短长度": min(lengths),
        "最长长度": max(lengths),
        "中位数长度": sorted(lengths)[total_docs // 2],
    }

    # 长度分布
    short_docs = sum(1 for l in lengths if l < 50)
    medium_docs = sum(1 for l in lengths if 50 <= l < 200)
    long_docs = sum(1 for l in lengths if l >= 200)

    stats.update({
        "短文档(<50字符)": f"{short_docs} ({short_docs / total_docs * 100:.1f}%)",
        "中等文档(50-200字符)": f"{medium_docs} ({medium_docs / total_docs * 100:.1f}%)",
        "长文档(>=200字符)": f"{long_docs} ({long_docs / total_docs * 100:.1f}%)",
    })

    return stats


def generate_negative_samples(positive_pairs, negative_df, cleaner, negative_ratio=0.3):
    """
    生成高质量的负样本用于对比学习

    Args:
        positive_pairs: 正样本对列表
        negative_df: 包含错误答案的DataFrame
        cleaner: 数据清洗器实例
        negative_ratio: 负样本比例

    Returns:
        负样本列表，格式为[(question, answer, label), ...]
    """
    logger.info("开始生成负样本...")

    negative_pairs = []
    target_negative_count = int(len(positive_pairs) * negative_ratio)

    # 策略1: 使用数据集中标记为错误的答案
    logger.info("策略1: 使用标记的错误答案")
    dataset_negatives = []

    for idx, row in negative_df.iterrows():
        clean_question = cleaner.clean_single_text(row['Question'])
        clean_answer = cleaner.clean_single_text(row['Sentence'])

        if (cleaner.is_valid_text(clean_question, cleaner.min_question_length) and
                cleaner.is_valid_text(clean_answer, cleaner.min_answer_length) and
                len(clean_answer) <= cleaner.max_answer_length):
            dataset_negatives.append((clean_question, clean_answer, 0))

    # 随机选择一部分数据集负样本
    if dataset_negatives:
        selected_count = min(len(dataset_negatives), target_negative_count // 2)
        selected_negatives = random.sample(dataset_negatives, selected_count)
        negative_pairs.extend(selected_negatives)
        logger.info(f"从数据集中选择了 {len(selected_negatives)} 个负样本")

    # 策略2: 随机配对生成困难负样本
    logger.info("策略2: 生成随机配对负样本")
    remaining_count = target_negative_count - len(negative_pairs)

    if remaining_count > 0:
        questions = [pair[0] for pair in positive_pairs]
        answers = [pair[1] for pair in positive_pairs]

        # 确保问题和答案不匹配
        random_negatives = []
        attempts = 0
        max_attempts = remaining_count * 3  # 避免无限循环

        while len(random_negatives) < remaining_count and attempts < max_attempts:
            q_idx = random.randint(0, len(questions) - 1)
            a_idx = random.randint(0, len(answers) - 1)

            # 确保不是原本的正确配对
            if q_idx != a_idx:
                question = questions[q_idx]
                answer = answers[a_idx]

                # 检查语义相似度，避免生成意外的正样本
                if not are_semantically_similar(question, answer):
                    random_negatives.append((question, answer, 0))

            attempts += 1

        negative_pairs.extend(random_negatives)
        logger.info(f"生成了 {len(random_negatives)} 个随机配对负样本")

    # 策略3: 生成困难负样本（部分正确但不完整的答案）
    logger.info("策略3: 生成困难负样本")
    hard_negatives = generate_hard_negatives(positive_pairs, cleaner)

    # 限制困难负样本数量
    max_hard_negatives = min(len(hard_negatives), target_negative_count // 4)
    if hard_negatives:
        selected_hard = random.sample(hard_negatives, max_hard_negatives)
        negative_pairs.extend(selected_hard)
        logger.info(f"生成了 {len(selected_hard)} 个困难负样本")

    # 随机打乱负样本
    random.shuffle(negative_pairs)

    logger.info(f"总共生成 {len(negative_pairs)} 个负样本")
    return negative_pairs


def generate_hard_negatives(positive_pairs, cleaner):
    """
    生成困难负样本（语义相关但不正确的答案）
    """
    hard_negatives = []

    try:
        # 导入用于文本相似度计算的库
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        # 提取所有答案
        answers = [pair[1] for pair in positive_pairs]
        questions = [pair[0] for pair in positive_pairs]

        if len(answers) < 2:
            return hard_negatives

        # 计算答案之间的TF-IDF相似度
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        answer_vectors = vectorizer.fit_transform(answers)
        similarity_matrix = cosine_similarity(answer_vectors)

        # 为每个问题找到相似但不正确的答案
        for i, (question, correct_answer) in enumerate(positive_pairs[:50]):  # 限制处理数量以提高性能
            # 找到与正确答案相似度在0.3-0.7之间的答案
            similarities = similarity_matrix[i]

            for j, sim_score in enumerate(similarities):
                if i != j and 0.3 <= sim_score <= 0.7:  # 相似但不完全相同
                    similar_answer = answers[j]

                    # 确保不是同一个答案
                    if similar_answer != correct_answer:
                        hard_negatives.append((question, similar_answer, 0))

                        # 限制每个问题生成的困难负样本数量
                        if len([hn for hn in hard_negatives if hn[0] == question]) >= 2:
                            break

        logger.info(f"基于语义相似度生成了 {len(hard_negatives)} 个困难负样本")

    except ImportError:
        logger.warning("无法导入sklearn，跳过困难负样本生成")
    except Exception as e:
        logger.warning(f"困难负样本生成失败: {e}")

    return hard_negatives


def are_semantically_similar(question, answer, threshold=0.3):
    """
    简单的语义相似度检查，避免生成意外的正样本
    """
    try:
        # 简单的词汇重叠检查
        q_words = set(question.lower().split())
        a_words = set(answer.lower().split())

        if len(q_words) == 0 or len(a_words) == 0:
            return False

        # 计算词汇重叠率
        overlap = len(q_words.intersection(a_words))
        union = len(q_words.union(a_words))

        jaccard_similarity = overlap / union if union > 0 else 0

        return jaccard_similarity > threshold

    except Exception:
        return False


def create_contrastive_learning_data(qa_texts, sample_info):
    """
    为对比学习创建训练数据格式

    Args:
        qa_texts: 清洗后的文本列表
        sample_info: 样本信息列表

    Returns:
        对比学习数据格式
    """
    contrastive_data = {
        'positive_samples': [],
        'negative_samples': [],
        'triplets': []  # (anchor, positive, negative) 三元组
    }

    # 分离正负样本
    for info in sample_info:
        if info['is_negative']:
            contrastive_data['negative_samples'].append({
                'question': info['question'],
                'answer': info['answer'],
                'text': info['text']
            })
        else:
            contrastive_data['positive_samples'].append({
                'question': info['question'],
                'answer': info['answer'],
                'text': info['text']
            })

    # 生成三元组用于对比学习
    positive_samples = contrastive_data['positive_samples']
    negative_samples = contrastive_data['negative_samples']

    for pos_sample in positive_samples[:100]:  # 限制数量以提高性能
        question = pos_sample['question']
        positive_answer = pos_sample['answer']

        # 找到相同问题的负样本作为困难样本
        hard_negatives = [
            neg for neg in negative_samples
            if neg['question'] == question
        ]

        if hard_negatives:
            # 选择一个困难负样本
            negative_sample = random.choice(hard_negatives)
            contrastive_data['triplets'].append({
                'anchor': question,
                'positive': positive_answer,
                'negative': negative_sample['answer']
            })

    logger.info(f"创建了 {len(contrastive_data['triplets'])} 个对比学习三元组")

    return contrastive_data


def evaluate_negative_sample_quality(positive_pairs, negative_pairs):
    """
    评估负样本的质量
    """
    logger.info("评估负样本质量...")

    if not negative_pairs:
        logger.warning("没有负样本可评估")
        return

    # 统计负样本类型
    stats = {
        'total_negatives': len(negative_pairs),
        'avg_question_length': 0,
        'avg_answer_length': 0,
        'unique_questions': 0,
        'unique_answers': 0
    }

    questions = [pair[0] for pair in negative_pairs]
    answers = [pair[1] for pair in negative_pairs]

    stats['avg_question_length'] = sum(len(q) for q in questions) / len(questions)
    stats['avg_answer_length'] = sum(len(a) for a in answers) / len(answers)
    stats['unique_questions'] = len(set(questions))
    stats['unique_answers'] = len(set(answers))

    logger.info(f"负样本质量评估: {stats}")

    return stats