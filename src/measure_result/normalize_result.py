# 决策函数
import numpy as np


def calculate_decision_score(predicted_returns: list, risk_tolerance: str = 'moderate') -> float:
    """
    根据预测的预期收益和风险偏好，计算决策分数。

    Args:
        predicted_returns (list): 预测的 4 个预期收益值，分别对应 1 天、3 天、5 天和 10 天的涨跌幅
        risk_tolerance (str, optional): 用户的风险偏好，可选 'aggressive', 'moderate', 'conservative'。
            默认为 'moderate'。

    Returns:
        float: 决策分数，分数越高代表越倾向于买入。
    """

    short_term_return = predicted_returns[1]  # 3 天的预测值
    medium_term_return = predicted_returns[2] # 5 天的预测值
    long_term_return = predicted_returns[3] # 10 天的预测值

    # 设定不同的权重，考虑风险偏好
    if risk_tolerance == 'aggressive':
        short_weight, medium_weight, long_weight = 0.2, 0.3, 0.5
    elif risk_tolerance == 'moderate':
        short_weight, medium_weight, long_weight = 0.3, 0.4, 0.3
    else:  # conservative
        short_weight, medium_weight, long_weight = 0.4, 0.5, 0.1

    # 计算加权平均得分
    decision_score = short_term_return * short_weight + \
                     medium_term_return * medium_weight + \
                     long_term_return * long_weight

    return decision_score


def make_decision(decision_score: float) -> str:
    """
    根据决策分数，返回操作建议。

    Args:
        decision_score (float): 决策分数，由 `calculate_decision_score` 函数计算得到。

    Returns:
        str: 操作建议，包含 '买入（Buy）', '卖出（Sell）', '持有（Hold）', '观望（Wait）'。
    """

    # 根据决策分数确定操作建议
    if decision_score >= 1:
        return "买入（Buy）"
    elif decision_score >= 0.5:
        return "持有（Hold）"
    elif decision_score >= 0:
        return "观望（Wait）"
    else:
        return "卖出（Sell）"


# 多数投票法
def majority_vote_decision(predicted_returns: list) -> str:
    decisions = [make_decision(r) for r in predicted_returns]
    return max(set(decisions), key=decisions.count)







# # 假设预测结果为：
# predicted_returns =[0.9197320938110352, 0.8447604179382324, 0.6181734800338745, 0.05447008088231087]  # 1天，3天，5天，10天
#
# # 用户的风险偏好
# risk_tolerance = 'moderate'
#
# # 调用决策函数
# decision = make_decision(predicted_returns, risk_tolerance)
#
# # 输出决策结果
# print(decision)  # 输出结果: 买入（Buy）

# predicted_returns = [predicted_return_1, predicted_return_3, predicted_return_5]

# 获取综合决策
# average_dec = average_decision(predicted_returns)
# weighted_average_dec = weighted_average_decision(predicted_returns, weights=[0.5, 0.3, 0.2])
# majority_vote_dec = majority_vote_decision(predicted_returns)