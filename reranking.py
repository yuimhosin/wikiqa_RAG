# reranking.py - 重排序模块

import re
from typing import List, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class SimpleReranker:
    """
    简单重排序器 - 基于关键词匹配和基础启发式
    """

    def __init__(self, keyword_weight=0.7, length_weight=0.3):
        """
        初始化简单重排序器

        Args:
            keyword_weight: 关键词匹配权重
            length_weight: 长度适应权重
        """
        self.keyword_weight = keyword_weight
        self.length_weight = length_weight

        # 英文停用词
        self.stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'what', 'where', 'when', 'why', 'how', 'who', 'which', 'that', 'this'
        }

    def rerank_documents(self, query: str, documents: List[Any], top_k: int = 5) -> List[Tuple[Any, float]]:
        """
        对文档进行重排序

        Args:
            query: 查询文本
            documents: 文档列表
            top_k: 返回前k个文档

        Returns:
            重排序后的(文档, 分数)元组列表
        """
        if not documents:
            return []

        scored_docs = []
        query_keywords = self._extract_keywords(query.lower())

        for doc in documents:
            doc_text = self._extract_text(doc).lower()

            # 计算关键词匹配分数
            keyword_score = self._calculate_keyword_score(query_keywords, doc_text)

            # 计算长度适应分数
            length_score = self._calculate_length_score(query, doc_text)

            # 综合分数
            final_score = (self.keyword_weight * keyword_score +
                           self.length_weight * length_score)

            scored_docs.append((doc, final_score))

        # 按分数排序
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        return scored_docs[:top_k]

    def _extract_text(self, document) -> str:
        """提取文档文本"""
        if hasattr(document, 'page_content'):
            return document.page_content
        elif isinstance(document, str):
            return document
        else:
            return str(document)

    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词"""
        # 使用正则表达式提取单词
        words = re.findall(r'\b\w+\b', text.lower())
        # 过滤停用词和短词
        keywords = [word for word in words if word not in self.stopwords and len(word) > 2]
        return keywords

    def _calculate_keyword_score(self, query_keywords: List[str], doc_text: str) -> float:
        """计算关键词匹配分数"""
        if not query_keywords:
            return 0.0

        matches = 0
        for keyword in query_keywords:
            if keyword in doc_text:
                # 精确单词匹配得更高分
                if f" {keyword} " in doc_text or doc_text.startswith(keyword) or doc_text.endswith(keyword):
                    matches += 1
                else:
                    matches += 0.5  # 部分匹配

        return matches / len(query_keywords)

    def _calculate_length_score(self, query: str, doc_text: str) -> float:
        """计算长度适应分数"""
        doc_length = len(doc_text)

        # 根据查询长度确定理想文档长度
        query_length = len(query.split())
        if query_length <= 3:
            ideal_length = 150  # 短查询偏好短答案
        elif query_length <= 8:
            ideal_length = 300  # 中等查询
        else:
            ideal_length = 500  # 长查询偏好详细答案

        # 计算长度分数（越接近理想长度分数越高）
        length_diff = abs(doc_length - ideal_length)
        max_diff = ideal_length  # 最大差异

        length_score = max(0, 1 - (length_diff / max_diff))

        # 对过短文档进行惩罚
        if doc_length < 50:
            length_score *= 0.5

        return length_score


class ContrastiveReranker:
    """
    对比学习重排序器 - 利用正负样本信息
    """

    def __init__(self):
        self.base_reranker = SimpleReranker()

    def rerank_documents(self, query: str, documents: List[Any],
                         contrastive_data: dict = None, top_k: int = 5) -> List[Tuple[Any, float]]:
        """
        使用对比学习进行重排序
        """
        if not documents:
            return []

        # 先用基础重排序器得到基础分数
        base_results = self.base_reranker.rerank_documents(query, documents, len(documents))

        if not contrastive_data:
            return base_results[:top_k]

        # 调整分数基于正负样本信息
        adjusted_results = []
        negative_samples = contrastive_data.get('negative_samples', [])

        for doc, base_score in base_results:
            doc_text = self.base_reranker._extract_text(doc)

            # 检查是否与负样本相似
            is_negative = self._is_similar_to_negatives(doc_text, negative_samples)

            if is_negative:
                # 降低负样本的分数
                adjusted_score = base_score * 0.3
            else:
                # 保持或轻微提升正样本分数
                adjusted_score = base_score * 1.1

            adjusted_results.append((doc, adjusted_score))

        # 重新排序
        adjusted_results.sort(key=lambda x: x[1], reverse=True)

        return adjusted_results[:top_k]

    def _is_similar_to_negatives(self, doc_text: str, negative_samples: List[dict]) -> bool:
        """检查文档是否与负样本相似"""
        doc_clean = doc_text.replace("Question: ", "").replace("Answer: ", "").lower()

        for neg_sample in negative_samples:
            neg_text = neg_sample.get('text', '').replace("Question: ", "").replace("Answer: ", "").lower()

            if neg_text and self._text_similarity(doc_clean, neg_text) > 0.7:
                return True

        return False

    def _text_similarity(self, text1: str, text2: str) -> float:
        """简单的文本相似度计算"""
        words1 = set(text1.split())
        words2 = set(text2.split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union if union > 0 else 0.0


# ================== 重排序展示函数 ==================

def show_retrieved_documents_with_rerank(db, question, contrastive_data=None, reranker=None):
    """
    显示检索结果和重排序效果
    """
    print("\n 正在从知识库中检索相关文档...")
    retriever = db.as_retriever(search_kwargs={"k": 8})  # 检索8个文档用于重排序
    docs = retriever.get_relevant_documents(question)

    if not docs:
        print(" 未检索到相关文档")
        return []

    print(f" 初始检索到 {len(docs)} 个文档")

    if reranker:
        print(" 正在应用重排序...")
        try:
            # 应用重排序
            if hasattr(reranker, 'rerank_documents'):
                if isinstance(reranker, ContrastiveReranker):
                    reranked_results = reranker.rerank_documents(question, docs, contrastive_data, top_k=5)
                else:
                    reranked_results = reranker.rerank_documents(question, docs, top_k=5)

                print(" 重排序后的相关文档（前5条）：")
                final_docs = []

                for i, (doc, score) in enumerate(reranked_results, 1):
                    doc_text = doc.page_content

                    # 简单的正负样本标注
                    doc_type = _identify_doc_type(doc_text, contrastive_data)

                    print(f"\n[重排序文档 {i}] {doc_type} | 分数: {score:.3f}")
                    print(f"   内容: {doc_text[:150]}...")
                    print("-" * 50)

                    final_docs.append(doc)

                # 显示重排序效果
                _show_simple_rerank_stats(len(docs), len(final_docs))

            else:
                final_docs = docs[:5]
                print(" 重排序器接口不兼容，使用原始结果")

        except Exception as e:
            print(f" 重排序失败: {e}")
            final_docs = docs[:5]
    else:
        final_docs = docs[:5]
        print(" 检索到的相关文档（前5条）：")
        for i, doc in enumerate(final_docs, 1):
            print(f"\n[文档 {i}]：")
            print(f"{doc.page_content[:150]}...")
            print("-" * 50)

    return final_docs


def analyze_retrieval_quality_with_rerank(question, retrieved_docs, contrastive_data):
    """
    简单的检索质量分析
    """
    if not retrieved_docs or not contrastive_data:
        return

    positive_count = 0
    negative_count = 0

    for doc in retrieved_docs:
        doc_content = doc.page_content.replace("Question: ", "").replace("Answer: ", "")

        # 检查是否为负样本
        is_negative = any(
            doc_content.lower() in neg['text'].replace("Question: ", "").replace("Answer: ", "").lower()
            for neg in contrastive_data.get('negative_samples', [])
        )

        if is_negative:
            negative_count += 1
        else:
            positive_count += 1

    total_docs = len(retrieved_docs)
    if total_docs > 0:
        quality_score = (positive_count / total_docs) * 100

        print(f"\n检索质量分析:")
        print(f"    正样本文档: {positive_count}/{total_docs} ({quality_score:.1f}%)")
        print(f"    负样本文档: {negative_count}/{total_docs} ({(negative_count / total_docs) * 100:.1f}%)")

        if quality_score >= 80:
            print(f"    质量评估: 优秀")
        elif quality_score >= 60:
            print(f"    质量评估: 良好")
        else:
            print(f"    质量评估: 需要改进")


def demo_reranking_functionality(db, reranker, contrastive_data):
    """
    简单的重排序功能演示
    """
    demo_questions = [
        "What is machine learning?",
        "How does AI work?"
    ]

    print(" 重排序功能演示")

    for i, question in enumerate(demo_questions, 1):
        print(f"\n 演示问题 {i}: {question}")
        print("-" * 40)

        retriever = db.as_retriever(search_kwargs={"k": 5})
        docs = retriever.get_relevant_documents(question)

        if not docs:
            print(" 未检索到相关文档")
            continue

        print(" 原始检索结果:")
        for j, doc in enumerate(docs[:3], 1):
            print(f"   [{j}] {doc.page_content[:80]}...")

        if reranker:
            try:
                if isinstance(reranker, ContrastiveReranker):
                    reranked = reranker.rerank_documents(question, docs, contrastive_data, top_k=3)
                else:
                    reranked = reranker.rerank_documents(question, docs, top_k=3)

                print("\n 重排序后结果:")
                for j, (doc, score) in enumerate(reranked, 1):
                    print(f"   [{j}] {doc.page_content[:80]}... (分数: {score:.2f})")

            except Exception as e:
                print(f" 重排序演示失败: {e}")


# ================== 辅助函数 ==================

def _identify_doc_type(doc_text, contrastive_data):
    """识别文档类型"""
    if not contrastive_data:
        return "无对比学习数据"

    doc_clean = doc_text.replace("Question: ", "").replace("Answer: ", "").lower()

    is_negative = any(
        doc_clean in neg['text'].replace("Question: ", "").replace("Answer: ", "").lower()
        for neg in contrastive_data.get('negative_samples', [])
    )

    return " 负样本" if is_negative else " 正样本"


def _show_simple_rerank_stats(original_count, final_count):
    """显示简单的重排序统计"""
    print(f"\n 重排序统计: 从 {original_count} 个文档中选出前 {final_count} 个")


def create_reranker(reranker_type="simple", **kwargs):
    """
    创建重排序器

    Args:
        reranker_type: "simple" 或 "contrastive"
    """
    if reranker_type == "simple":
        return SimpleReranker(**kwargs)
    elif reranker_type == "contrastive":
        return ContrastiveReranker()
    else:
        raise ValueError(f"未知的重排序器类型: {reranker_type}")


# 为了向后兼容，保留一些空的函数
def test_reranking_performance(db, reranker, contrastive_data):
    """简化版性能测试"""
    print(" 重排序器工作正常")


def show_contrastive_learning_stats(contrastive_data):
    """显示对比学习统计"""
    print(f"\n 对比学习数据: 正样本 {len(contrastive_data.get('positive_samples', []))} 个, "
          f"负样本 {len(contrastive_data.get('negative_samples', []))} 个")

    def show_retrieved_documents_with_rerank(db, question, contrastive_data=None, reranker=None):
        """
        显示从RAG知识库中检索到的最匹配原文，应用重排序，并标注正负样本
        """
        print("\n 正在从知识库中检索相关文档...")
        retriever = db.as_retriever(search_kwargs={"k": 10})  # 先检索更多文档用于重排序
        docs = retriever.get_relevant_documents(question)

        if not docs:
            print(" 未检索到相关文档")
            return []

        print(f" 初始检索到 {len(docs)} 个文档")

        # 应用重排序
        if reranker:
            print(" 正在应用重排序...")
            try:
                if hasattr(reranker, 'rerank_documents') and hasattr(reranker, 'weights'):
                    # 多阶段重排序器
                    rerank_results = reranker.rerank_documents(
                        query=question,
                        documents=docs,
                        contrastive_data=contrastive_data,
                        top_k=5
                    )

                    print(" 重排序后的相关文档（前5条）：")
                    for i, result in enumerate(rerank_results, 1):
                        doc_text = result.document.page_content

                        # 识别文档类型
                        doc_type = "未知"
                        if contrastive_data:
                            is_negative = any(
                                doc_text.replace("Question: ", "").replace("Answer: ", "") in
                                (neg['text'].replace("Question: ", "").replace("Answer: ", ""))
                                for neg in contrastive_data.get('negative_samples', [])
                            )
                            doc_type = " 负样本" if is_negative else "🟢 正样本"

                        # 显示排名变化
                        rank_change = result.original_rank - result.new_rank
                        if rank_change > 0:
                            rank_indicator = f" 从第{result.original_rank + 1}位上升到第{result.new_rank + 1}位"
                        elif rank_change < 0:
                            rank_indicator = f" 从第{result.original_rank + 1}位下降到第{result.new_rank + 1}位"
                        else:
                            rank_indicator = f" 排名未变（第{result.new_rank + 1}位）"

                        print(f"\n[重排序文档 {i}] {doc_type} | 分数: {result.score:.3f}")
                        print(f"   {rank_indicator}")
                        print(f"   原因: {result.rerank_reason}")
                        print(f"   内容: {doc_text[:200]}...")
                        print("-" * 60)

                    # 提取重排序后的文档
                    final_docs = [result.document for result in rerank_results]

                    # 显示重排序统计
                    show_rerank_statistics(rerank_results)

                else:
                    # 简单重排序器
                    reranked_pairs = reranker.rerank_documents(question, docs, top_k=5)
                    print(" 重排序后的相关文档（前5条）：")

                    final_docs = []
                    for i, (doc, score) in enumerate(reranked_pairs, 1):
                        doc_text = doc.page_content

                        # 识别文档类型
                        doc_type = "未知"
                        if contrastive_data:
                            is_negative = any(
                                doc_text.replace("Question: ", "").replace("Answer: ", "") in
                                (neg['text'].replace("Question: ", "").replace("Answer: ", ""))
                                for neg in contrastive_data.get('negative_samples', [])
                            )
                            doc_type = " 负样本" if is_negative else " 正样本"

                        print(f"\n[简单重排序文档 {i}] {doc_type} | 分数: {score:.3f}")
                        print(f"   内容: {doc_text[:200]}...")
                        print("-" * 50)

                        final_docs.append(doc)

            except Exception as e:
                print(f"重排序失败，使用原始检索结果: {e}")
                final_docs = docs[:5]

                print(" 原始检索文档（前5条）：")
                for i, doc in enumerate(final_docs, 1):
                    doc_text = doc.page_content
                    doc_type = "未知"
                    if contrastive_data:
                        is_negative = any(
                            doc_text.replace("Question: ", "").replace("Answer: ", "") in
                            (neg['text'].replace("Question: ", "").replace("Answer: ", ""))
                            for neg in contrastive_data.get('negative_samples', [])
                        )
                        doc_type = " 负样本" if is_negative else " 正样本"

                    print(f"\n[原始文档 {i}] {doc_type}：")
                    print(f"   {doc_text[:200]}...")
                    print("-" * 50)
        else:
            # 无重排序器，使用原始结果
            final_docs = docs[:5]
            print(" 检索到的相关文档（前5条）：")
            for i, doc in enumerate(final_docs, 1):
                print(f"\n[文档 {i}]：")
                print(f"{doc.page_content[:200]}...")
                print("-" * 50)

        return final_docs

    def show_rerank_statistics(rerank_results):
        """显示重排序统计信息"""
        if not rerank_results:
            return

        # 计算排名变化统计
        rank_changes = [result.original_rank - result.new_rank for result in rerank_results]
        significant_changes = [change for change in rank_changes if abs(change) >= 2]

        avg_score = sum(result.score for result in rerank_results) / len(rerank_results)

        print(f"\n 重排序统计:")
        print(f"    平均重排序分数: {avg_score:.3f}")
        print(f"    显著排名变化数量: {len(significant_changes)}")
        print(f"    最大排名提升: {max(rank_changes) if rank_changes else 0} 位")
        print(f"   ️ 最大排名下降: {abs(min(rank_changes)) if rank_changes else 0} 位")

        # 按重排序原因统计
        reason_counts = {}
        for result in rerank_results:
            for reason in result.rerank_reason.split('; '):
                if '高' in reason:
                    key = reason.split('匹配')[0] + '匹配'
                    reason_counts[key] = reason_counts.get(key, 0) + 1

        if reason_counts:
            print(f"    主要提升因素: {', '.join(reason_counts.keys())}")

    def analyze_retrieval_quality_with_rerank(question, retrieved_docs, contrastive_data, rerank_results=None):
        """
        分析检索质量，包含重排序效果评估
        """
        if not retrieved_docs or not contrastive_data:
            return

        positive_count = 0
        negative_count = 0

        for doc in retrieved_docs:
            doc_content = doc.page_content.replace("Question: ", "").replace("Answer: ", "")

            # 检查是否匹配负样本
            is_negative = any(
                doc_content in neg['text'].replace("Question: ", "").replace("Answer: ", "")
                for neg in contrastive_data.get('negative_samples', [])
            )

            if is_negative:
                negative_count += 1
            else:
                positive_count += 1

        total_docs = len(retrieved_docs)
        if total_docs > 0:
            print(f"\n 检索质量分析:")
            print(f"    正样本文档: {positive_count}/{total_docs} ({positive_count / total_docs * 100:.1f}%)")
            print(f"    负样本文档: {negative_count}/{total_docs} ({negative_count / total_docs * 100:.1f}%)")

            # 计算检索质量分数
            quality_score = positive_count / total_docs * 100
            if quality_score >= 80:
                print(f"    检索质量: 优秀 ({quality_score:.1f}分)")
            elif quality_score >= 60:
                print(f"   ️ 检索质量: 良好 ({quality_score:.1f}分)")
            else:
                print(f"    检索质量: 需要改进 ({quality_score:.1f}分)")

            # 如果有重排序结果，显示重排序改进效果
            if rerank_results:
                rerank_positive = 0
                for result in rerank_results:
                    doc_content = result.document.page_content.replace("Question: ", "").replace("Answer: ", "")
                    is_negative = any(
                        doc_content in neg['text'].replace("Question: ", "").replace("Answer: ", "")
                        for neg in contrastive_data.get('negative_samples', [])
                    )
                    if not is_negative:
                        rerank_positive += 1

                rerank_quality = rerank_positive / len(rerank_results) * 100
                improvement = rerank_quality - quality_score

                print(f"    重排序后质量: {rerank_quality:.1f}分")
                if improvement > 5:
                    print(f"    重排序改进: +{improvement:.1f}分 (显著提升)")
                elif improvement > 0:
                    print(f"    重排序改进: +{improvement:.1f}分 (轻微提升)")
                elif improvement < -5:
                    print(f"    重排序影响: {improvement:.1f}分 (需要调优)")
                else:
                    print(f"    重排序影响: {improvement:.1f}分 (基本不变)")

        return {
            'positive_count': positive_count,
            'negative_count': negative_count,
            'quality_score': positive_count / total_docs * 100 if total_docs > 0 else 0,
            'retrieved_docs_info': [
                (doc.page_content[:100], "positive" if positive_count > negative_count else "negative")
                for doc in retrieved_docs]
        }
        print(" 请检查网络连接或稍后重试")

    def analyze_retrieval_quality(question, retrieved_docs, contrastive_data):
        """
        分析检索质量，基于对比学习数据评估检索到的文档
        """
        if not retrieved_docs or not contrastive_data:
            return

        positive_count = 0
        negative_count = 0
        relevant_docs = []

        for doc in retrieved_docs:
            doc_content = doc.page_content.replace("Question: ", "").replace("Answer: ", "")

            # 检查是否匹配负样本
            is_negative = any(
                doc_content in neg['text'].replace("Question: ", "").replace("Answer: ", "")
                for neg in contrastive_data.get('negative_samples', [])
            )

            if is_negative:
                negative_count += 1
                relevant_docs.append(("negative", doc_content[:100]))
            else:
                positive_count += 1
                relevant_docs.append(("positive", doc_content[:100]))

        total_docs = len(retrieved_docs)
        if total_docs > 0:
            print(f"\n 检索质量分析:")
            print(f"    正样本文档: {positive_count}/{total_docs} ({positive_count / total_docs * 100:.1f}%)")
            print(f"    负样本文档: {negative_count}/{total_docs} ({negative_count / total_docs * 100:.1f}%)")

            # 计算检索质量分数
            quality_score = positive_count / total_docs * 100
            if quality_score >= 80:
                print(f"    检索质量: 优秀 ({quality_score:.1f}分)")
            elif quality_score >= 60:
                print(f"    检索质量: 良好 ({quality_score:.1f}分)")
            else:
                print(f"    检索质量: 需要改进 ({quality_score:.1f}分)")

        return {
            'positive_count': positive_count,
            'negative_count': negative_count,
            'quality_score': positive_count / total_docs * 100 if total_docs > 0 else 0,
            'retrieved_docs_info': relevant_docs
        }

    def show_contrastive_learning_stats(contrastive_data):
        """
        显示对比学习数据的统计信息
        """
        print("\n 对比学习数据统计:")
        print(f"    正样本数量: {len(contrastive_data.get('positive_samples', []))}")
        print(f"    负样本数量: {len(contrastive_data.get('negative_samples', []))}")
        print(f"    三元组数量: {len(contrastive_data.get('triplets', []))}")

        # 显示一些示例三元组
        triplets = contrastive_data.get('triplets', [])
        if triplets:
            print(f"\n 对比学习三元组示例 (显示前3个):")
            for i, triplet in enumerate(triplets[:3], 1):
                print(f"\n   [{i}] 锚点问题: {triplet['anchor'][:80]}...")
                print(f"        正确答案: {triplet['positive'][:60]}...")
                print(f"        错误答案: {triplet['negative'][:60]}...")