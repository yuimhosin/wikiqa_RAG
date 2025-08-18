# feedback.py

feedback_data = []


def get_user_feedback():
    """
    询问用户对模型答案的反馈，1 为正确，0 为错误
    """
    while True:
        try:
            feedback = input("系统的回答是否正确？（1 = 正确，0 = 错误）：").strip()
            if feedback in ['1', '0']:
                return int(feedback)
            else:
                print("请输入 1（正确）或 0（错误）")
        except (ValueError, KeyboardInterrupt):
            raise


def save_feedback(question, answer, feedback):
    """
    保存用户反馈数据，可以用于后续分析和模型优化
    """
    feedback_entry = {
        "question": question,
        "answer": answer,
        "feedback": feedback,
        "timestamp": __import__('datetime').datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    feedback_data.append(feedback_entry)
    print(f"反馈已保存")


def get_feedback_statistics():
    """
    获取反馈统计信息
    """
    if not feedback_data:
        return "暂无反馈数据"

    total_count = len(feedback_data)
    correct_count = sum(1 for item in feedback_data if item['feedback'] == 1)
    accuracy = (correct_count / total_count) * 100

    return {
        "total": total_count,
        "correct": correct_count,
        "accuracy": accuracy
    }


def export_feedback_to_file(filename="feedback_log.txt"):
    """
    将反馈数据导出到文件
    """
    if not feedback_data:
        print("没有反馈数据可导出")
        return

    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("RAG问答系统反馈日志\n")
            f.write("=" * 50 + "\n\n")
            for i, entry in enumerate(feedback_data, 1):
                f.write(f"条目 {i}:\n")
                f.write(f"时间: {entry['timestamp']}\n")
                f.write(f"问题: {entry['question']}\n")
                f.write(f"回答: {entry['answer']}\n")
                f.write(f"反馈: {'正确' if entry['feedback'] == 1 else '错误'}\n")
                f.write("-" * 30 + "\n\n")

        stats = get_feedback_statistics()
        print(f"反馈数据已导出到 {filename}")
        print(f"统计：总计 {stats['total']} 条，准确率 {stats['accuracy']:.1f}%")
    except Exception as e:
        print(f" 导出失败：{str(e)}")