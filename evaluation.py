# evaluation.py

from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from sklearn.metrics import recall_score, f1_score
from wikiqa_rag.model_manager import model_manager  # 导入模型管理器


def get_embedding(text, model):
    """
    获取文本的嵌入向量
    """
    return model.embed_query(text)


def is_semantic_match(pred, truths, embed_model, threshold=0.7):
    """
    使用余弦相似度验证语义匹配
    """
    pred_emb = get_embedding(pred, embed_model)
    for ans in truths:
        ans_emb = get_embedding(ans, embed_model)
        sim = cosine_similarity([pred_emb], [ans_emb])[0][0]
        if sim >= threshold:
            return True
    return False


def evaluate_model(qa_chain, tsv_path="WikiQA/WikiQA-test.tsv", max_q=50):
    """
    评估模型准确率、召回率和 F1 分数
    """
    try:
        df = pd.read_csv(tsv_path, sep="\t")
    except FileNotFoundError:
        print(f"错误：找不到测试文件 {tsv_path}")
        return
    except Exception as e:
        print(f"错误：读取测试文件失败 {e}")
        return

    grouped = df.groupby("Question")

    # 使用统一的模型管理器获取嵌入模型
    embed_model = model_manager.get_embedding_model()

    total, correct = 0, 0
    y_true, y_pred = [], []  # 用于计算召回率和 F1 分数

    print(f"开始评估，预计处理 {min(max_q, len(grouped))} 个问题...")

    for i, (question, group) in enumerate(grouped):
        if i >= max_q:
            break

        # 显示进度
        if (i + 1) % 10 == 0:
            print(f"已处理 {i + 1}/{min(max_q, len(grouped))} 个问题...")

        true_answers = group[group["Label"] == 1]["Sentence"].tolist()
        if not true_answers:
            continue

        try:
            prediction = qa_chain.invoke({"query": question})
            pred_text = prediction.get("result", "")

            if not pred_text.strip():
                print(f"警告：问题 [{i + 1}] 获得空回答")
                continue

        except Exception as e:
            print(f"问题出错：{question} | {e}")
            continue

        matched = is_semantic_match(pred_text, true_answers, embed_model)
        total += 1
        correct += int(matched)
        y_true.append(1)  # 真实标签，假设真答案是正确的
        y_pred.append(1 if matched else 0)  # 预测标签，匹配则为 1 否则为 0

        status = "✓ correct" if matched else "✗ false"
        print(f"{status} [{i + 1}] Q: {question[:60]}...")
        print(f"   → 预测: {pred_text[:100]}...")
        print(f"   → 正确: {true_answers[0][:100]}...")
        print()

    if total == 0:
        print("评估失败：无有效问题被处理。")
        return None
    else:
        accuracy = correct / total

        # 避免除零错误
        if len(set(y_pred)) > 1:
            recall = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
        else:
            recall = 0.0
            f1 = 0.0
            print("警告：所有预测结果相同，无法计算召回率和F1分数")

        print("=" * 60)
        print("评估结果汇总：")
        print(f"总问题数: {total}")
        print(f"正确回答: {correct}")
        print(f"准确率: {accuracy:.2%}")
        print(f"召回率: {recall:.2%}")
        print(f"F1 分数: {f1:.3f}")
        print("=" * 60)

        return {
            'total': total,
            'correct': correct,
            'accuracy': accuracy,
            'recall': recall,
            'f1': f1
        }